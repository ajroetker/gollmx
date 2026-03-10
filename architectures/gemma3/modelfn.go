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

// attentionWithKV runs self-attention using KVCacheAccessor for KV storage.
func (b *Builder) attentionWithKV(ctx *context.Context, hidden, positionIDs *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs, layerIdx, seqLen int) *Node {
	cfg := b.config
	attnCtx := ctx.In("attention")
	g := hidden.Graph()

	batchSize := hidden.Shape().Dimensions[0]
	headDim := cfg.HeadDim
	kvHeads := cfg.KVHeads()
	headsPerGroup := cfg.HeadsPerKVGroup()

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

	// RoPE — use per-layer theta (local=10K, global=1M).
	ropeTheta := cfg.RopeFreqBase(layerIdx)
	query, key = common.RoPE(query, key, positionIDs, ropeTheta, seqLen, headDim)

	// Store K/V in cache and retrieve full cached K/V.
	// newKey/newValue: [batch, kvHeads, seqLen, headDim]
	// cachedKey/cachedValue: [batch, kvHeads, maxSeqLen, headDim]
	cachedKey, cachedValue := kv.WriteRead(attnCtx, g, key, value)

	// Expand KV heads for GQA.
	if headsPerGroup > 1 {
		cachedKey = common.RepeatKV(cachedKey, headsPerGroup)
		cachedValue = common.RepeatKV(cachedValue, headsPerGroup)
	}

	// Attention scores: [batch, heads, seqLen, keySeqLen].
	keySeqLen := kv.KeySeqLen()
	scores := Einsum("bhqd,bhkd->bhqk", query, cachedKey)
	scale := ConstAs(scores, 1.0/math.Sqrt(float64(headDim)))
	scores = Mul(scores, scale)

	// Apply mask.
	if seqLen > 1 {
		// Prefill: use causal mask (or sliding window causal mask).
		// The KV cache mask isn't useful here since we're writing all positions.
		if cfg.IsLocalAttentionLayer(layerIdx) && cfg.SlidingWindow > 0 {
			causalMask := common.CreateSlidingWindowCausalMask(g, seqLen, cfg.SlidingWindow, scores.DType())
			// Pad to keySeqLen (maxSeqLen) — positions beyond seqLen are zero in cache.
			if keySeqLen > seqLen {
				padWidth := keySeqLen - seqLen
				negInfPad := MulScalar(Ones(g, shapes.Make(scores.DType(), 1, 1, seqLen, padWidth)), -1e9)
				causalMask = Concatenate([]*Node{causalMask, negInfPad}, 3)
			}
			scores = Add(scores, causalMask)
		} else {
			causalMask := common.CreateCausalMask(g, seqLen, scores.DType())
			if keySeqLen > seqLen {
				padWidth := keySeqLen - seqLen
				negInfPad := MulScalar(Ones(g, shapes.Make(scores.DType(), 1, 1, seqLen, padWidth)), -1e9)
				causalMask = Concatenate([]*Node{causalMask, negInfPad}, 3)
			}
			scores = Add(scores, causalMask)
		}
	} else {
		// Decode: use accessor's mask (valid cache positions).
		baseMask := kv.Mask(g, seqLen) // [batch, 1, 1, keySeqLen] bool

		if cfg.IsLocalAttentionLayer(layerIdx) && cfg.SlidingWindow > 0 {
			// Also apply sliding window: only attend to [pos-window, pos].
			// Use cacheWritePositions if available (post-compaction), else positions.
			maskPos := aux.CacheWritePositions
			if maskPos == nil {
				maskPos = kv.(*attention.FlatKVCacheAccessor).Positions
			}
			posI32 := ConvertDType(maskPos, dtypes.Int32)
			windowStart := Max(SubScalar(posI32, int32(cfg.SlidingWindow)), Const(g, int32(0)))
			windowStart = Reshape(windowStart, batchSize, 1) // [batch, 1]

			keyPositions := Iota(g, shapes.Make(dtypes.Int32, keySeqLen), 0)          // [keySeqLen]
			keyPositions = Reshape(keyPositions, 1, keySeqLen)                         // [1, keySeqLen]
			windowMask := GreaterOrEqual(keyPositions, windowStart)                    // [batch, keySeqLen]
			windowMask = Reshape(windowMask, batchSize, 1, 1, keySeqLen)               // [batch, 1, 1, keySeqLen]

			baseMask = And(baseMask, windowMask)
		}

		// Convert boolean mask to additive float mask.
		floatMask := Where(baseMask, ZerosLike(scores), MulScalar(OnesLike(scores), -1e9))
		scores = Add(scores, floatMask)
	}

	attnWeights := Softmax(scores, -1)
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, cachedValue)

	// Reshape: [batch, heads, seq, headDim] -> [batch, seq, heads*headDim]
	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, seqLen, cfg.NumAttentionHeads*headDim)

	return b.denseOrQuantized(attnCtx.In("output"), attnOutput)
}
