// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package kvcache

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// PagedKVCacheAccessor implements attention.KVCacheAccessor using a paged
// (block-based) KV cache. Physical blocks are allocated by a BlockManager
// and mapped to logical positions via page tables.
type PagedKVCacheAccessor struct {
	// Config is the paged cache configuration.
	Config PagedKVCacheConfig

	// PageTables is a [batchSize, maxBlocksPerRequest] int32 tensor mapping
	// logical block indices to physical block indices.
	PageTables *Node

	// Positions is a [batchSize] int32 tensor with per-element positions.
	Positions *Node

	// ReadNumBlocks is the number of blocks to read per request (compile-time constant).
	ReadNumBlocks int

	// newSeqLen is set by WriteRead to the number of tokens just written.
	// Used by Mask to include the newly written positions.
	newSeqLen int
}

// WriteRead implements attention.KVCacheAccessor.
func (a *PagedKVCacheAccessor) WriteRead(ctx *context.Context, g *Graph, newKey, newValue *Node) (cachedKeys, cachedValues *Node) {
	a.newSeqLen = newKey.Shape().Dimensions[2]
	PagedKVCacheWriteBatched(ctx, g, a.Config, a.PageTables, a.Positions, newKey, newValue)

	batchSize := newKey.Shape().Dimensions[0]

	// Read each batch element's blocks and stack.
	allKeys := make([]*Node, batchSize)
	allValues := make([]*Node, batchSize)
	for b := range batchSize {
		batchPT := Squeeze(Slice(a.PageTables, AxisElem(b), AxisRange()), 0)
		k, v := PagedKVCacheRead(ctx, g, a.Config, batchPT, a.ReadNumBlocks)
		allKeys[b] = k   // [1, numKVHeads, seqLen, headDim]
		allValues[b] = v  // [1, numKVHeads, seqLen, headDim]
	}

	cachedKeys = Concatenate(allKeys, 0)   // [batchSize, numKVHeads, seqLen, headDim]
	cachedValues = Concatenate(allValues, 0)
	return
}

// Mask implements attention.KVCacheAccessor.
// The mask includes all positions written during WriteRead: positions[b]..positions[b]+newSeqLen-1.
func (a *PagedKVCacheAccessor) Mask(g *Graph, querySeqLen int) *Node {
	keySeqLen := a.KeySeqLen()
	batchSize := a.PageTables.Shape().Dimensions[0]

	// Positions is the starting write position. After WriteRead, valid cache entries
	// span 0..positions+newSeqLen-1, so the mask boundary is positions+newSeqLen.
	posI32 := ConvertDType(a.Positions, dtypes.Int32)
	maskEnd := AddScalar(posI32, int32(a.newSeqLen))

	// effectivePositions = min(maskEnd, keySeqLen)
	effectivePositions := MinScalar(maskEnd, keySeqLen)

	// Key indices: [keySeqLen]
	keyPositions := Iota(g, shapes.Make(dtypes.Int32, keySeqLen), 0)

	// Compare: [batchSize, keySeqLen]
	effectivePositions = ExpandDims(effectivePositions, -1) // [batchSize, 1]
	keyPositions = ExpandDims(keyPositions, 0)               // [1, keySeqLen]
	mask := LessThan(keyPositions, effectivePositions)        // [batchSize, keySeqLen]

	// Reshape to [batchSize, 1, querySeqLen, keySeqLen]
	mask = ExpandDims(mask, 1)
	mask = ExpandDims(mask, 2)
	mask = BroadcastToShape(mask, shapes.Make(dtypes.Bool, batchSize, 1, querySeqLen, keySeqLen))
	return mask
}

// KeySeqLen implements attention.KVCacheAccessor.
func (a *PagedKVCacheAccessor) KeySeqLen() int {
	return a.ReadNumBlocks * a.Config.BlockSize
}
