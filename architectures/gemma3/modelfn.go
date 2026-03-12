package gemma3

import (
	"math"
	"strconv"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"

	"github.com/ajroetker/gollmx/architectures/common"
)

// BuildModelFn returns a decode.ModelFn that uses KVCacheAccessor for
// engine-managed KV cache. This enables the serving engine to control
// cache layout (flat, paged) and transparently apply compaction,
// prefix caching, and other optimizations.
//
// The returned function handles both prefill (seqLen > 1) and decode
// (seqLen == 1) in a single unified path.
func (b *Builder) BuildModelFn() decode.ModelFn {
	return func(ctx *context.Context, tokens *Node, positions *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs) *Node {
		cfg := b.config
		g := tokens.Graph()
		batchSize := tokens.Shape().Dimensions[0]
		seqLen := tokens.Shape().Dimensions[1]

		hidden := b.BuildEmbeddings(ctx, tokens)

		// Merge image features into embeddings during prefill.
		if aux != nil && aux.ImageFeatures != nil && seqLen > 1 {
			hidden = b.mergeImageFeatures(hidden, aux.ImageFeatures, tokens)
		}

		// Build position IDs [batch, seqLen] for RoPE.
		// positions is [batch] int32 (per-element start position).
		posI64 := ConvertDType(positions, dtypes.Int64)
		var positionIDs *Node
		if seqLen > 1 {
			// Prefill: positions + [0, 1, 2, ..., seqLen-1]
			posOffset := Reshape(posI64, batchSize, 1)
			positionIDs = Add(Iota(g, shapes.Make(dtypes.Int64, batchSize, seqLen), 1), posOffset)
		} else {
			// Decode: just the single position.
			positionIDs = Reshape(posI64, batchSize, 1)
		}

		for i := range cfg.NumHiddenLayers {
			layerCtx := ctx.In("layers").In(strconv.Itoa(i))
			hidden = b.decoderLayerWithKV(layerCtx, hidden, positionIDs, kv, aux, i, seqLen)
		}

		hidden = common.RMSNorm(ctx.In("norm"), hidden, cfg.RMSNormEps)
		return b.ApplyLMHead(ctx, hidden, g)
	}
}

// mergeImageFeatures replaces embeddings at image token positions with projected vision features.
// hidden: [batch, seqLen, hiddenSize], imageFeatures: [batch, numPatches, hiddenSize],
// tokens: [batch, seqLen] int32.
//
// Image tokens may not start at position 0 (e.g., they follow a chat template header).
// We use CumSum on the image mask to build a proper index into imageFeatures so that
// the 1st image token gets feature 0, the 2nd gets feature 1, etc.
func (b *Builder) mergeImageFeatures(hidden, imageFeatures, tokens *Node) *Node {
	g := hidden.Graph()
	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	hiddenSize := hidden.Shape().Dimensions[2]

	// Create a boolean mask: true where token == imageTokenID.
	imageTokenConst := Scalar(g, dtypes.Int32, int32(262144))
	isImage := Equal(tokens, imageTokenConst) // [batch, seqLen] bool

	// Build per-position indices into imageFeatures using CumSum.
	// CumSum uses SumPool internally which requires float types.
	// isImageF32: [batch, seqLen] with 1.0 at image positions, 0.0 elsewhere.
	// CumSum gives running count: 0, 0, ..., 1, 2, 3, ..., 4096, 4096, ...
	// Subtract 1 for 0-based indexing, clamp to [0, numPatches-1].
	isImageF32 := ConvertDType(isImage, dtypes.Float32)
	cumIdx := CumSum(isImageF32, 1)                              // [batch, seqLen] float32
	cumIdx = ConvertDType(cumIdx, dtypes.Int32)                  // back to int for indexing
	featureIdx := Sub(cumIdx, Scalar(g, dtypes.Int32, int32(1))) // 0-based
	numPatches := imageFeatures.Shape().Dimensions[1]
	featureIdx = MinScalar(MaxScalar(featureIdx, 0), int32(numPatches-1)) // clamp

	// Convert imageFeatures dtype to match hidden if needed.
	if imageFeatures.DType() != hidden.DType() {
		imageFeatures = ConvertDType(imageFeatures, hidden.DType())
	}

	// Gather: for each sequence position, fetch the corresponding image feature.
	// featureIdx: [batch, seqLen] → gathered: [batch, seqLen, hiddenSize]
	// We expand featureIdx to [batch, seqLen, 1] and use it to index axis 1 of imageFeatures.
	featureIdx3D := InsertAxes(featureIdx, -1) // [batch, seqLen, 1]
	featureIdx3D = BroadcastToDims(featureIdx3D, batchSize, seqLen, hiddenSize) // [batch, seqLen, hiddenSize]

	// Use Gather along axis 1: for each batch element, gather from [numPatches, hiddenSize].
	// Since GoMLX doesn't have a simple batched gather-by-index, we use OnehostAndDot approach:
	// Build one-hot [batch, seqLen, numPatches] from featureIdx, then matmul with imageFeatures.
	featureIdxFlat := Reshape(featureIdx, batchSize*seqLen)
	oneHot := OneHot(featureIdxFlat, numPatches, dtypes.Float32) // [batch*seqLen, numPatches]
	oneHot = Reshape(oneHot, batchSize, seqLen, numPatches)               // [batch, seqLen, numPatches]
	// Matmul: [batch, seqLen, numPatches] @ [batch, numPatches, hiddenSize] = [batch, seqLen, hiddenSize]
	gathered := Einsum("bsp,bph->bsh", oneHot, imageFeatures)

	// Expand mask and select: image positions get gathered features, others keep hidden.
	isImage3D := InsertAxes(isImage, -1) // [batch, seqLen, 1]
	isImage3D = BroadcastToDims(isImage3D, batchSize, seqLen, hiddenSize)
	return Where(isImage3D, gathered, hidden)
}

// decoderLayerWithKV runs a single decoder layer using KVCacheAccessor.
func (b *Builder) decoderLayerWithKV(ctx *context.Context, hidden, positionIDs *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs, layerIdx, seqLen int) *Node {
	cfg := b.config

	normalized := common.RMSNorm(ctx.In("input_norm"), hidden, cfg.RMSNormEps)
	attnOutput := b.attentionWithKV(ctx, normalized, positionIDs, kv, aux, layerIdx, seqLen)
	attnOutput = common.RMSNorm(ctx.In("post_attn_norm"), attnOutput, cfg.RMSNormEps)
	hidden = Add(hidden, attnOutput)

	normalized = common.RMSNorm(ctx.In("ffn_norm"), hidden, cfg.RMSNormEps)
	mlpOutput := b.BuildMLP(ctx, normalized)
	mlpOutput = common.RMSNorm(ctx.In("post_ffn_norm"), mlpOutput, cfg.RMSNormEps)
	hidden = Add(hidden, mlpOutput)

	return hidden
}

// attentionWithKV runs self-attention using KVCacheAccessor for KV storage
// and attention.Core for the scaled dot-product attention computation.
// attention.Core automatically uses the fused SDPA backend op when available,
// falling back to decomposed Einsum+Softmax otherwise.
func (b *Builder) attentionWithKV(ctx *context.Context, hidden, positionIDs *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs, layerIdx, seqLen int) *Node {
	cfg := b.config
	attnCtx := ctx.In("attention")
	g := hidden.Graph()

	batchSize := hidden.Shape().Dimensions[0]
	headDim := cfg.HeadDim
	kvHeads := cfg.KVHeads()

	// Q, K, V projections.
	query := b.denseOrQuantized(attnCtx.In("query"), hidden)
	key := b.denseOrQuantized(attnCtx.In("key"), hidden)
	value := b.denseOrQuantized(attnCtx.In("value"), hidden)

	// Reshape to [batch, heads, seq, head_dim].
	query = Reshape(query, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	query = Transpose(query, 1, 2)

	key = Reshape(key, batchSize, seqLen, kvHeads, headDim)
	key = Transpose(key, 1, 2)

	value = Reshape(value, batchSize, seqLen, kvHeads, headDim)
	value = Transpose(value, 1, 2)

	// QK-norm.
	query = common.RMSNorm(attnCtx.In("q_norm"), query, cfg.RMSNormEps)
	key = common.RMSNorm(attnCtx.In("k_norm"), key, cfg.RMSNormEps)

	// RoPE — use per-layer theta (local=10K, global=1M) and scaling.
	ropeTheta := cfg.RopeFreqBase(layerIdx)
	ropeScale := cfg.RopeScaling(layerIdx)
	query, key = common.RoPE(query, key, positionIDs, ropeTheta, seqLen, headDim, ropeScale)

	// Store K/V in cache and retrieve full cached K/V.
	// newKey/newValue: [batch, kvHeads, seqLen, headDim]
	// cachedKey/cachedValue: [batch, kvHeads, maxSeqLen, headDim]
	cachedKey, cachedValue := kv.WriteRead(attnCtx, g, key, value)

	// Build boolean attention mask and call attention.Core.
	// attention.Core handles GQA internally via Q-reshape (no RepeatKV needed)
	// and tries the fused SDPA backend op, falling back to decomposed ops.
	keySeqLen := kv.KeySeqLen()
	scale := 1.0 / math.Sqrt(float64(headDim))
	mask := b.buildBooleanMask(g, kv, aux, batchSize, seqLen, keySeqLen, layerIdx)

	output, _ := attention.Core(nil, query, cachedKey, cachedValue, scale, mask, 0, attention.LayoutBHSD, false, false)

	// Reshape: [batch, heads, seq, headDim] -> [batch, seq, heads*headDim]
	output = Transpose(output, 1, 2)
	output = Reshape(output, batchSize, seqLen, cfg.NumAttentionHeads*headDim)

	return b.denseOrQuantized(attnCtx.In("output"), output)
}

// buildBooleanMask builds a boolean attention mask combining the KV cache
// validity mask with causal masking and optional sliding window constraints.
// Returns a mask shaped [batch, 1, seqLen, keySeqLen] where true = attend.
func (b *Builder) buildBooleanMask(g *Graph, kv attention.KVCacheAccessor, aux *decode.AuxInputs, batchSize, seqLen, keySeqLen, layerIdx int) *Node {
	cfg := b.config

	if seqLen > 1 {
		// Prefill: use full causal mask for ALL layers.
		// Sliding window is only enforced during decode — during prefill, all
		// positions are processed simultaneously so local layers can safely
		// attend to the full causal context. The sliding window constraint
		// is restored during autoregressive decode via the KV cache mask.
		var causalMask *Node
		causalMask = LowerTriangular(g, seqLen)
		// Reshape to [1, 1, seqLen, seqLen] for broadcasting.
		causalMask = Reshape(causalMask, 1, 1, seqLen, seqLen)

		// Pad to keySeqLen — positions beyond seqLen are masked out.
		if keySeqLen > seqLen {
			padWidth := keySeqLen - seqLen
			falsePad := BroadcastPrefix(Const(g, false), 1, 1, seqLen, padWidth)
			causalMask = Concatenate([]*Node{causalMask, falsePad}, 3)
		}
		return causalMask
	}

	// Decode: use the KV cache accessor's validity mask.
	baseMask := kv.Mask(g, seqLen) // [batch, 1, 1, keySeqLen] bool

	if cfg.IsLocalAttentionLayer(layerIdx) && cfg.SlidingWindow > 0 {
		// Apply sliding window: only attend to [pos-window, pos].
		maskPos := aux.CacheWritePositions
		if maskPos == nil {
			maskPos = kv.(*attention.FlatKVCacheAccessor).Positions
		}
		posI32 := ConvertDType(maskPos, dtypes.Int32)
		windowStart := Max(SubScalar(posI32, int32(cfg.SlidingWindow)), Const(g, int32(0)))
		windowStart = Reshape(windowStart, batchSize, 1) // [batch, 1]

		keyPositions := Iota(g, shapes.Make(dtypes.Int32, keySeqLen), 0) // [keySeqLen]
		keyPositions = Reshape(keyPositions, 1, keySeqLen)               // [1, keySeqLen]
		windowMask := GreaterOrEqual(keyPositions, windowStart)          // [batch, keySeqLen]
		windowMask = Reshape(windowMask, batchSize, 1, 1, keySeqLen)    // [batch, 1, 1, keySeqLen]

		baseMask = And(baseMask, windowMask)
	}

	return baseMask
}
