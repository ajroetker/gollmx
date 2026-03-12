package phi3

import (
	"math"
	"strconv"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"

	"github.com/gomlx/gollmx/architectures/common"
)

// BuildModelFn returns a decode.ModelFn that uses KVCacheAccessor for
// engine-managed KV cache.
func (b *Builder) BuildModelFn() decode.ModelFn {
	return func(ctx *context.Context, tokens *Node, positions *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs) *Node {
		cfg := b.config
		g := tokens.Graph()
		batchSize := tokens.Shape().Dimensions[0]
		seqLen := tokens.Shape().Dimensions[1]

		hidden := b.BuildEmbeddings(ctx, tokens)

		// Merge image features into embeddings during prefill.
		if aux != nil && aux.ImageFeatures != nil && seqLen > 1 {
			hidden = common.MergeImageFeatures(hidden, aux.ImageFeatures, tokens, b.imageTokenID)
		}

		// Build position IDs [batch, seqLen] for RoPE.
		posI64 := ConvertDType(positions, dtypes.Int64)
		var positionIDs *Node
		if seqLen > 1 {
			posOffset := Reshape(posI64, batchSize, 1)
			positionIDs = Add(Iota(g, shapes.Make(dtypes.Int64, batchSize, seqLen), 1), posOffset)
		} else {
			positionIDs = Reshape(posI64, batchSize, 1)
		}

		for i := range cfg.NumHiddenLayers {
			layerCtx := ctx.In("layers").In(strconv.Itoa(i))
			hidden = b.decoderLayerWithKV(layerCtx, hidden, positionIDs, kv, aux, seqLen)
		}

		hidden = common.RMSNorm(ctx.In("norm"), hidden, cfg.RMSNormEps)
		return b.ApplyLMHead(ctx, hidden, g)
	}
}

// decoderLayerWithKV runs a single decoder layer using KVCacheAccessor.
func (b *Builder) decoderLayerWithKV(ctx *context.Context, hidden, positionIDs *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs, seqLen int) *Node {
	cfg := b.config

	normalized := common.RMSNorm(ctx.In("input_norm"), hidden, cfg.RMSNormEps)
	attnOutput := b.attentionWithKV(ctx, normalized, positionIDs, kv, aux, seqLen)
	hidden = Add(hidden, attnOutput)

	normalized = common.RMSNorm(ctx.In("post_attn_norm"), hidden, cfg.RMSNormEps)
	mlpOutput := b.BuildMLP(ctx, normalized)
	hidden = Add(hidden, mlpOutput)

	return hidden
}

// attentionWithKV runs self-attention using KVCacheAccessor for KV storage.
func (b *Builder) attentionWithKV(ctx *context.Context, hidden, positionIDs *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs, seqLen int) *Node {
	cfg := b.config
	attnCtx := ctx.In("attention")
	g := hidden.Graph()

	batchSize := hidden.Shape().Dimensions[0]
	headDim := cfg.KVHeadDim()
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

	// RoPE (supports partial rotary and LongRoPE).
	ropeCfg := common.RoPEConfig{
		Theta:         cfg.RopeTheta,
		HeadDim:       headDim,
		RotaryDim:     cfg.RotaryDim(),
		LongFactors:   cfg.LongRoPELongFactor,
		ShortFactors:  cfg.LongRoPEShortFactor,
		OrigMaxSeqLen: cfg.OrigMaxSeqLen,
	}
	query, key = common.RoPEWithConfig(query, key, positionIDs, seqLen, ropeCfg)

	// Store K/V in cache and retrieve full cached K/V.
	cachedKey, cachedValue := kv.WriteRead(attnCtx, g, key, value)

	// Build boolean attention mask and call attention.Core.
	// attention.Core handles GQA internally (no RepeatKV needed).
	keySeqLen := kv.KeySeqLen()
	scale := 1.0 / math.Sqrt(float64(headDim))
	mask := b.buildBooleanMask(g, kv, aux, seqLen, keySeqLen)

	output, _ := attention.Core(nil, query, cachedKey, cachedValue, scale, mask, 0, attention.LayoutBHSD, false, false)

	// Reshape: [batch, heads, seq, headDim] -> [batch, seq, heads*headDim]
	output = Transpose(output, 1, 2)
	output = Reshape(output, batchSize, seqLen, cfg.NumAttentionHeads*headDim)

	return b.denseOrQuantized(attnCtx.In("output"), output)
}

// buildBooleanMask builds a boolean attention mask combining the KV cache
// validity mask with causal masking.
// Returns a mask shaped [batch, 1, seqLen, keySeqLen] where true = attend.
func (b *Builder) buildBooleanMask(g *Graph, kv attention.KVCacheAccessor, aux *decode.AuxInputs, seqLen, keySeqLen int) *Node {
	if seqLen > 1 {
		// Prefill: use full causal mask.
		causalMask := LowerTriangular(g, seqLen)
		causalMask = Reshape(causalMask, 1, 1, seqLen, seqLen)

		if keySeqLen > seqLen {
			padWidth := keySeqLen - seqLen
			falsePad := BroadcastPrefix(Const(g, false), 1, 1, seqLen, padWidth)
			causalMask = Concatenate([]*Node{causalMask, falsePad}, 3)
		}
		return causalMask
	}

	// Decode: use the KV cache accessor's validity mask.
	return kv.Mask(g, seqLen)
}
