// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package serving

import (
	"context"
	"testing"

	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/decode/sample"
	"github.com/ajroetker/gollmx/kvcache"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

// makeKVCacheModelFn creates a ModelFn that writes random-ish data to the KV cache
// through the accessor. This exercises the full compaction pipeline.
func makeKVCacheModelFn(vocabSize, numKVHeads, headDim int, outputToken int32) decode.ModelFn {
	return func(ctx *mlctx.Context, tokens *Node, positions *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs) *Node {
		g := tokens.Graph()
		batchSize := tokens.Shape().Dimensions[0]
		seqLen := tokens.Shape().Dimensions[1]

		// Generate synthetic key/value projections from token IDs.
		// Shape: [batchSize, numKVHeads, seqLen, headDim]
		tokFloat := ConvertDType(tokens, dtypes.Float32)
		tokFloat = ExpandDims(tokFloat, -1)                                                     // [batch, seq, 1]
		tokFloat = BroadcastToShape(tokFloat, shapes.Make(dtypes.Float32, batchSize, seqLen, headDim)) // [batch, seq, headDim]
		// Scale to avoid all-same values (use iota as offset).
		iota_ := Iota(g, shapes.Make(dtypes.Float32, headDim), 0)
		iota_ = BroadcastPrefix(iota_, batchSize, seqLen)
		tokFloat = Add(tokFloat, MulScalar(iota_, 0.01))

		// Expand to [batch, numKVHeads, seq, headDim]
		tokFloat = ExpandDims(tokFloat, 1) // [batch, 1, seq, headDim]
		newKeys := BroadcastToShape(tokFloat, shapes.Make(dtypes.Float32, batchSize, numKVHeads, seqLen, headDim))
		newValues := MulScalar(newKeys, 0.5)

		// Write to and read from KV cache.
		cacheCtx := ctx.In("layer_0").In("attn").In("kv_cache").Reuse().Checked(false)
		kv.WriteRead(cacheCtx, g, newKeys, newValues)

		// Return constant logits (compaction doesn't affect logit computation in this mock).
		logits := AddScalar(Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize)), -1.0)
		oneHot := OneHot(Const(g, outputToken), vocabSize, dtypes.Float32)
		oneHot = MulScalar(oneHot, 11.0)
		oneHot = BroadcastPrefix(oneHot, batchSize, seqLen)
		return Add(logits, oneHot)
	}
}

// makeKVCacheModelFnWithQueryCapture is like makeKVCacheModelFn but also stores
// reference queries via the context, simulating WithQueryCapture.
func makeKVCacheModelFnWithQueryCapture(vocabSize, numKVHeads, headDim, numRefQueries int, outputToken int32) decode.ModelFn {
	return func(ctx *mlctx.Context, tokens *Node, positions *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs) *Node {
		g := tokens.Graph()
		batchSize := tokens.Shape().Dimensions[0]
		seqLen := tokens.Shape().Dimensions[1]

		// Generate synthetic key/value projections.
		tokFloat := ConvertDType(tokens, dtypes.Float32)
		tokFloat = ExpandDims(tokFloat, -1)
		tokFloat = BroadcastToShape(tokFloat, shapes.Make(dtypes.Float32, batchSize, seqLen, headDim))
		iota_ := Iota(g, shapes.Make(dtypes.Float32, headDim), 0)
		iota_ = BroadcastPrefix(iota_, batchSize, seqLen)
		tokFloat = Add(tokFloat, MulScalar(iota_, 0.01))

		tokFloat = ExpandDims(tokFloat, 1)
		newKeys := BroadcastToShape(tokFloat, shapes.Make(dtypes.Float32, batchSize, numKVHeads, seqLen, headDim))
		newValues := MulScalar(newKeys, 0.5)

		cacheCtx := ctx.In("layer_0").In("attn").In("kv_cache").Reuse().Checked(false)
		kv.WriteRead(cacheCtx, g, newKeys, newValues)

		// Capture reference queries (simulates WithQueryCapture).
		// Only during prefill (seqLen > 1).
		if seqLen > 1 {
			numCapture := min(numRefQueries, seqLen)
			// Create synthetic queries: [numKVHeads, numCapture, headDim]
			refQ := Iota(g, shapes.Make(dtypes.Float32, numKVHeads, numCapture, headDim), 2)
			refQ = MulScalar(refQ, 0.1)

			refCtx := ctx.In("layer_0").In("attn").In("kv_cache").Reuse().Checked(false)
			refShape := shapes.Make(dtypes.Float32, numKVHeads, numCapture, headDim)
			refVar := refCtx.VariableWithShape("ref_queries", refShape)
			refVar.SetValueGraph(refQ)
		}

		// Constant logits.
		logits := AddScalar(Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize)), -1.0)
		oneHot := OneHot(Const(g, outputToken), vocabSize, dtypes.Float32)
		oneHot = MulScalar(oneHot, 11.0)
		oneHot = BroadcastPrefix(oneHot, batchSize, seqLen)
		return Add(logits, oneHot)
	}
}

func TestCompactionEndToEnd(t *testing.T) {
	vocabSize := 10
	numKVHeads := 2
	headDim := 4
	outputToken := int32(3)
	maxSeqLen := 64
	targetLen := 8
	promptLen := 20

	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: -1}

	config := DefaultConfig()
	config.MaxSeqLen = maxSeqLen
	config.MaxBatchSize = 1
	config.Compaction = &kvcache.CompactionConfig{
		TargetLen:     targetLen,
		NumRefQueries: 16,
	}

	modelFn := makeKVCacheModelFn(vocabSize, numKVHeads, headDim, outputToken)
	eng := NewEngine(backend, ctx, modelFn, tok, config, numKVHeads, headDim, dtypes.Float32)
	defer eng.Stop()

	// Build a prompt longer than targetLen to trigger compaction.
	prompt := make([]int32, promptLen)
	for i := range prompt {
		prompt[i] = int32(i%vocabSize + 1)
	}

	maxTokens := 5
	outCh, errCh, err := eng.Submit(
		context.Background(),
		prompt,
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
		nil,
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var deltas []SequenceDelta
	for d := range outCh {
		deltas = append(deltas, d)
	}
	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(deltas) != maxTokens {
		t.Fatalf("Expected %d deltas, got %d", maxTokens, len(deltas))
	}

	for _, d := range deltas {
		if d.TokenID != outputToken {
			t.Errorf("Expected token %d, got %d", outputToken, d.TokenID)
		}
	}
}

func TestCompactionWithRefQueries(t *testing.T) {
	vocabSize := 10
	numKVHeads := 2
	headDim := 4
	numRefQueries := 8
	outputToken := int32(5)
	maxSeqLen := 64
	targetLen := 8
	promptLen := 20

	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: -1}

	config := DefaultConfig()
	config.MaxSeqLen = maxSeqLen
	config.MaxBatchSize = 1
	config.Compaction = &kvcache.CompactionConfig{
		TargetLen:     targetLen,
		NumRefQueries: numRefQueries,
	}

	modelFn := makeKVCacheModelFnWithQueryCapture(vocabSize, numKVHeads, headDim, numRefQueries, outputToken)
	eng := NewEngine(backend, ctx, modelFn, tok, config, numKVHeads, headDim, dtypes.Float32)
	defer eng.Stop()

	prompt := make([]int32, promptLen)
	for i := range prompt {
		prompt[i] = int32(i%vocabSize + 1)
	}

	maxTokens := 3
	outCh, errCh, err := eng.Submit(
		context.Background(),
		prompt,
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
		nil,
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var deltas []SequenceDelta
	for d := range outCh {
		deltas = append(deltas, d)
	}
	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(deltas) != maxTokens {
		t.Fatalf("Expected %d deltas, got %d", maxTokens, len(deltas))
	}

	for _, d := range deltas {
		if d.TokenID != outputToken {
			t.Errorf("Expected token %d, got %d", outputToken, d.TokenID)
		}
	}
}

func TestCompactionSkippedWhenShort(t *testing.T) {
	vocabSize := 10
	numKVHeads := 2
	headDim := 4
	outputToken := int32(3)

	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: -1}

	config := DefaultConfig()
	config.MaxSeqLen = 64
	config.MaxBatchSize = 1
	config.Compaction = &kvcache.CompactionConfig{
		TargetLen: 16,
		MinSeqLen: 100, // MinSeqLen higher than prompt — compaction should be skipped.
	}

	modelFn := makeKVCacheModelFn(vocabSize, numKVHeads, headDim, outputToken)
	eng := NewEngine(backend, ctx, modelFn, tok, config, numKVHeads, headDim, dtypes.Float32)
	defer eng.Stop()

	// Short prompt (shorter than MinSeqLen) — compaction should be skipped.
	prompt := []int32{1, 2, 3, 4, 5}
	maxTokens := 3
	outCh, errCh, err := eng.Submit(
		context.Background(),
		prompt,
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
		nil,
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var deltas []SequenceDelta
	for d := range outCh {
		deltas = append(deltas, d)
	}
	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(deltas) != maxTokens {
		t.Fatalf("Expected %d deltas, got %d", maxTokens, len(deltas))
	}
}

func TestCompactionBiasWritten(t *testing.T) {
	vocabSize := 10
	numKVHeads := 2
	headDim := 4
	outputToken := int32(3)
	maxSeqLen := 64
	targetLen := 8
	promptLen := 20

	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: -1}

	config := DefaultConfig()
	config.MaxSeqLen = maxSeqLen
	config.MaxBatchSize = 1
	config.Compaction = &kvcache.CompactionConfig{
		TargetLen:     targetLen,
		NumRefQueries: 16,
	}

	modelFn := makeKVCacheModelFn(vocabSize, numKVHeads, headDim, outputToken)
	eng := NewEngine(backend, ctx, modelFn, tok, config, numKVHeads, headDim, dtypes.Float32)
	defer eng.Stop()

	prompt := make([]int32, promptLen)
	for i := range prompt {
		prompt[i] = int32(i%vocabSize + 1)
	}

	// Run one request to completion so compaction runs.
	maxTokens := 1
	outCh, errCh, err := eng.Submit(
		context.Background(),
		prompt,
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
		nil,
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}
	for range outCh {
	}
	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Check that bias variable was created with the expected shape.
	var foundBias bool
	for v := range ctx.IterVariablesInScope() {
		name := v.ScopeAndName()
		if len(name) > 4 && name[len(name)-4:] == "bias" {
			foundBias = true
			biasShape := v.Shape()
			// Bias shape should be [batchSize, numKVHeads, maxSeqLen].
			if biasShape.Rank() != 3 {
				t.Errorf("Bias variable rank = %d, want 3", biasShape.Rank())
			}
			if biasShape.Dimensions[1] != numKVHeads {
				t.Errorf("Bias numKVHeads = %d, want %d", biasShape.Dimensions[1], numKVHeads)
			}
			// Verify it has a value (not uninitialized).
			_ = tensors.MustCopyFlatData[float32](v.MustValue())
		}
	}
	if !foundBias {
		t.Error("No bias variable found after compaction")
	}
}
