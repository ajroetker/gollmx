package llama

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
// engine-managed KV cache. This enables the serving engine to control
// cache layout (flat, paged) and transparently apply compaction,
// prefix caching, and other optimizations.
func (b *Builder) BuildModelFn() decode.ModelFn {
	return func(ctx *context.Context, tokens *Node, positions *Node, kv attention.KVCacheAccessor, aux *decode.AuxInputs) *Node {
		cfg := b.config
		g := tokens.Graph()
		batchSize := tokens.Shape().Dimensions[0]
		seqLen := tokens.Shape().Dimensions[1]

		hidden := b.BuildEmbeddings(ctx, tokens)

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
		return b.applyLMHead(ctx, hidden, g)
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
	headsPerGroup := cfg.HeadsPerKVGroup()

	// Q, K, V projections.
	query := common.DenseWeightOnly(attnCtx.In("query"), hidden)
	key := common.DenseWeightOnly(attnCtx.In("key"), hidden)
	value := common.DenseWeightOnly(attnCtx.In("value"), hidden)

	// Reshape to [batch, heads, seq, head_dim].
	query = Reshape(query, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	query = Transpose(query, 1, 2)

	key = Reshape(key, batchSize, seqLen, kvHeads, headDim)
	key = Transpose(key, 1, 2)

	value = Reshape(value, batchSize, seqLen, kvHeads, headDim)
	value = Transpose(value, 1, 2)

	// RoPE.
	query, key = common.RoPE(query, key, positionIDs, cfg.RopeTheta, seqLen, headDim)

	// Store K/V in cache and retrieve full cached K/V.
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
		// Prefill: use causal mask.
		causalMask := common.CreateCausalMask(g, seqLen, scores.DType())
		if keySeqLen > seqLen {
			padWidth := keySeqLen - seqLen
			negInfPad := MulScalar(Ones(g, shapes.Make(scores.DType(), 1, 1, seqLen, padWidth)), -1e9)
			causalMask = Concatenate([]*Node{causalMask, negInfPad}, 3)
		}
		scores = Add(scores, causalMask)
	} else {
		// Decode: use accessor's mask (valid cache positions).
		baseMask := kv.Mask(g, seqLen) // [batch, 1, 1, keySeqLen] bool
		floatMask := Where(baseMask, ZerosLike(scores), MulScalar(OnesLike(scores), -1e9))
		scores = Add(scores, floatMask)
	}

	attnWeights := Softmax(scores, -1)
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, cachedValue)

	// Reshape: [batch, heads, seq, headDim] -> [batch, seq, heads*headDim]
	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, seqLen, cfg.NumAttentionHeads*headDim)

	return common.DenseWeightOnly(attnCtx.In("output"), attnOutput)
}

// applyLMHead applies the language model head to produce logits.
func (b *Builder) applyLMHead(ctx *context.Context, hidden *Node, g *Graph) *Node {
	lmHeadCtx := ctx.In("lm_head")
	lmHeadVar := lmHeadCtx.GetVariableByScopeAndName(lmHeadCtx.Scope(), "weights")
	if lmHeadVar != nil {
		return common.DenseWeightOnly(lmHeadCtx, hidden)
	}

	// Tied embeddings: reuse token embedding weight.
	embCtx := ctx.In("embeddings")
	embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
	if embVar == nil {
		panic("applyLMHead: no lm_head weights and no embedding weights for tied embeddings")
	}
	embWeight := embVar.ValueGraph(g) // [vocabSize, hiddenSize]
	return Einsum("bsh,vh->bsv", hidden, embWeight)
}

// LlamaConfig returns the Llama-specific configuration.
func (b *Builder) LlamaConfig() *Config {
	return b.config
}
