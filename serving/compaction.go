// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package serving

import (
	"fmt"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gollmx/kvcache"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

// compactKVCache performs KV cache compaction for a single request after prefill.
// It reads KV cache variables from the model context, compacts them on CPU,
// and writes the compacted data back. This reduces the effective cache length
// from req.position to config.TargetLen.
//
// Only works with flat KV cache (unified mode with FlatKVCacheAccessor).
func (e *Engine) compactKVCache(req *engineRequest) error {
	cfg := e.config.Compaction
	if cfg == nil {
		return nil
	}

	position := req.position
	if position <= cfg.TargetLen {
		return nil // Nothing to compact.
	}
	if cfg.MinSeqLen > 0 && position < cfg.MinSeqLen {
		return nil // Sequence too short to compact.
	}

	// Find all KV cache variable pairs (and optional ref queries) by scanning the context.
	keySuffix := fmt.Sprintf("/%s/%s", attention.KVCacheScopeName, "key")
	valueSuffix := fmt.Sprintf("/%s/%s", attention.KVCacheScopeName, "value")
	refQueriesSuffix := fmt.Sprintf("/%s/%s", attention.KVCacheScopeName, "ref_queries")

	type kvPair struct {
		keyVar      *mlctx.Variable
		valueVar    *mlctx.Variable
		refQueryVar *mlctx.Variable // optional: captured reference queries
		scope       string          // scope prefix (e.g., "layer_0/MultiHeadAttention")
	}

	// Collect variables by scope and suffix.
	keyVars := make(map[string]*mlctx.Variable)
	valueVars := make(map[string]*mlctx.Variable)
	refQueryVars := make(map[string]*mlctx.Variable)

	for v := range e.modelCtx.IterVariablesInScope() {
		name := v.ScopeAndName()
		if scope, ok := strings.CutSuffix(name, keySuffix); ok {
			keyVars[scope] = v
		} else if scope, ok := strings.CutSuffix(name, valueSuffix); ok {
			valueVars[scope] = v
		} else if scope, ok := strings.CutSuffix(name, refQueriesSuffix); ok {
			refQueryVars[scope] = v
		}
	}

	var pairs []kvPair
	for scope, keyVar := range keyVars {
		if valueVar, ok := valueVars[scope]; ok {
			pairs = append(pairs, kvPair{
				keyVar:      keyVar,
				valueVar:    valueVar,
				refQueryVar: refQueryVars[scope], // may be nil
				scope:       scope,
			})
		}
	}

	if len(pairs) == 0 {
		return nil // No KV cache variables found.
	}

	slot := req.slot

	// Process each layer's KV cache.
	for _, pair := range pairs {
		// Extract ref queries tensor if captured during prefill.
		var refQueries *tensors.Tensor
		if pair.refQueryVar != nil {
			refQueries = pair.refQueryVar.MustValue()
		}

		if err := e.compactOneLayer(pair.keyVar, pair.valueVar, pair.scope, slot, position, cfg, refQueries); err != nil {
			return fmt.Errorf("compaction layer %s: %w", pair.scope, err)
		}
	}

	// Update the request's position to the compacted length.
	req.position = cfg.TargetLen
	return nil
}

// compactOneLayer compacts the KV cache for a single attention layer.
// refQueries is optional: if non-nil, it provides captured projected queries
// [numKVHeads, numRefQueries, headDim] for higher-quality compaction.
func (e *Engine) compactOneLayer(
	keyVar, valueVar *mlctx.Variable,
	scope string,
	slot, position int,
	cfg *kvcache.CompactionConfig,
	refQueries *tensors.Tensor,
) error {
	keyShape := keyVar.Shape()
	// Expected shape: [batchSize, numKVHeads, maxSeqLen, headDim]
	if keyShape.Rank() != 4 {
		return fmt.Errorf("unexpected KV cache rank %d, want 4", keyShape.Rank())
	}

	numKVHeads := keyShape.Dimensions[1]
	maxSeqLen := keyShape.Dimensions[2]
	headDim := keyShape.Dimensions[3]

	// Extract this slot's data from the full cache tensors.
	keyData := tensors.MustCopyFlatData[float32](keyVar.MustValue())
	valData := tensors.MustCopyFlatData[float32](valueVar.MustValue())

	slotStride := numKVHeads * maxSeqLen * headDim
	headStride := maxSeqLen * headDim
	slotOffset := slot * slotStride

	// Build [numKVHeads, position, headDim] tensors for compaction input.
	slotKeys := make([]float32, numKVHeads*position*headDim)
	slotVals := make([]float32, numKVHeads*position*headDim)

	for h := range numKVHeads {
		srcOffset := slotOffset + h*headStride
		dstOffset := h * position * headDim
		copy(slotKeys[dstOffset:dstOffset+position*headDim], keyData[srcOffset:srcOffset+position*headDim])
		copy(slotVals[dstOffset:dstOffset+position*headDim], valData[srcOffset:srcOffset+position*headDim])
	}

	keysTensor := tensors.FromFlatDataAndDimensions(slotKeys, numKVHeads, position, headDim)
	valsTensor := tensors.FromFlatDataAndDimensions(slotVals, numKVHeads, position, headDim)

	// Run compaction with optional captured reference queries.
	result, err := kvcache.Compact(*cfg, keysTensor, valsTensor, refQueries)
	if err != nil {
		return err
	}

	// Write compacted data back to the cache at positions [0, targetLen).
	targetLen := cfg.TargetLen
	compKeys := tensors.MustCopyFlatData[float32](result.CompactedKeys)
	compVals := tensors.MustCopyFlatData[float32](result.CompactedValues)

	for h := range numKVHeads {
		dstOffset := slotOffset + h*headStride
		srcOffset := h * targetLen * headDim

		// Write compacted data.
		copy(keyData[dstOffset:dstOffset+targetLen*headDim], compKeys[srcOffset:srcOffset+targetLen*headDim])
		copy(valData[dstOffset:dstOffset+targetLen*headDim], compVals[srcOffset:srcOffset+targetLen*headDim])

		// Zero out positions [targetLen, maxSeqLen).
		for i := targetLen * headDim; i < maxSeqLen*headDim; i++ {
			keyData[dstOffset+i] = 0
			valData[dstOffset+i] = 0
		}
	}

	// Write updated cache tensors back.
	keyVar.SetValue(tensors.FromFlatDataAndDimensions(keyData, keyShape.Dimensions...))
	valueVar.SetValue(tensors.FromFlatDataAndDimensions(valData, keyShape.Dimensions...))

	// Write bias variable.
	biasData := tensors.MustCopyFlatData[float32](result.Biases)

	// Build full bias tensor [batchSize, numKVHeads, maxSeqLen] with this slot's bias.
	batchSize := keyShape.Dimensions[0]
	biasShape := shapes.Make(keyShape.DType, batchSize, numKVHeads, maxSeqLen)

	// Navigate to the bias variable scope by walking each element.
	// scope is e.g. "layer_0/attn", and we need to reach "layer_0/attn/kv_cache/bias".
	biasCtx := e.modelCtx
	for _, elem := range strings.Split(scope, "/") {
		if elem != "" {
			biasCtx = biasCtx.In(elem)
		}
	}
	biasCtx = biasCtx.In(attention.KVCacheScopeName).Reuse().Checked(false).WithInitializer(initializers.Zero)
	biasVar := biasCtx.VariableWithShape("bias", biasShape)

	// If bias variable was just created, it has no CPU-side value yet. Initialize with zeros.
	if _, err := biasVar.Value(); err != nil {
		biasVar.SetValue(tensors.FromShape(biasShape))
	}

	// Read existing bias data (may have other slots' biases).
	fullBias := tensors.MustCopyFlatData[float32](biasVar.MustValue())
	biasSlotStride := numKVHeads * maxSeqLen
	biasSlotOffset := slot * biasSlotStride

	for h := range numKVHeads {
		dstOffset := biasSlotOffset + h*maxSeqLen
		srcOffset := h * targetLen

		// Write compacted bias for positions [0, targetLen).
		copy(fullBias[dstOffset:dstOffset+targetLen], biasData[srcOffset:srcOffset+targetLen])

		// Zero out positions [targetLen, maxSeqLen).
		for i := targetLen; i < maxSeqLen; i++ {
			fullBias[dstOffset+i] = 0
		}
	}

	biasVar.SetValue(tensors.FromFlatDataAndDimensions(fullBias, biasShape.Dimensions...))

	return nil
}
