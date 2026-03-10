// Package gemma3 provides a Gemma 3 architecture implementation for GoMLX.
//
// Gemma 3 uses:
//   - RoPE (Rotary Position Embedding) for positions
//   - RMSNorm with 4 norms per layer (pre/post attention, pre/post FFN)
//   - QK-norm (RMSNorm on Q and K per-head after projection, before RoPE)
//   - GeLU activation in gated MLP (not SiLU like Llama)
//   - Grouped Query Attention (GQA)
//   - Hybrid local/global attention (sliding window for local layers)
//   - Embedding scaling by sqrt(hidden_size)
//   - Explicit head_dim (not hidden_size/num_heads)
//
// Reference: https://arxiv.org/abs/2503.19786
package gemma3

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/nn"

	models "github.com/ajroetker/huggingface-gomlx"
	"github.com/ajroetker/huggingface-gomlx/architectures/common"
)

func init() {
	models.RegisterArchitecture("gemma3", func() models.ArchitectureBuilder { return &Builder{} })
}

// Config holds Gemma 3-specific configuration.
type Config struct {
	*models.BaseConfig

	// Attention.
	HeadDim      int `json:"head_dim"`       // Explicit head dimension (e.g., 256)
	NumKVHeads   int `json:"num_kv_heads"`    // Key-value heads for GQA
	SlidingWindow int `json:"sliding_window"` // Window size for local attention layers

	// RoPE.
	RopeTheta      float64 `json:"rope_theta"`       // Base frequency for global attention layers (default 1e6)
	RopeLocalTheta float64 `json:"rope_local_theta"`  // Base frequency for local (SWA) attention layers (default 1e4)

	// Normalization.
	RMSNormEps float64 `json:"rms_norm_eps"`
}

// KVHeads returns the number of key-value heads.
func (c *Config) KVHeads() int {
	if c.NumKVHeads > 0 {
		return c.NumKVHeads
	}
	return c.NumAttentionHeads
}

// HeadsPerKVGroup returns how many query heads share each KV head.
func (c *Config) HeadsPerKVGroup() int {
	return c.NumAttentionHeads / c.KVHeads()
}

// IsLocalAttentionLayer returns true if the given layer uses sliding window (local) attention.
// Gemma 3 uses a repeating pattern: 5 local layers followed by 1 global layer.
func (c *Config) IsLocalAttentionLayer(layerIdx int) bool {
	// Pattern: layers 0-4 local, layer 5 global, layers 6-10 local, layer 11 global, ...
	return layerIdx%6 != 5
}

// RopeTheta returns the RoPE base frequency for the given layer index.
func (c *Config) RopeFreqBase(layerIdx int) float64 {
	if c.IsLocalAttentionLayer(layerIdx) {
		return c.RopeLocalTheta
	}
	return c.RopeTheta
}

// Builder implements the Gemma 3 architecture.
type Builder struct {
	config    *Config
	isGGUF    bool
	quantInfo models.QuantInfo // scope path → GGML quant type for quantized weights
}

// Name returns the architecture name.
func (b *Builder) Name() string {
	return "Gemma3"
}

// ParseConfig extracts Gemma 3-specific configuration from BaseConfig.Raw.
func (b *Builder) ParseConfig(base *models.BaseConfig) error {
	b.config = &Config{BaseConfig: base}

	if v, ok := base.GetInt("head_dim"); ok {
		b.config.HeadDim = v
	} else {
		// Gemma 3 4B default.
		b.config.HeadDim = 256
	}

	if v, ok := base.GetInt("num_key_value_heads"); ok {
		b.config.NumKVHeads = v
	}

	if v, ok := base.GetInt("sliding_window"); ok {
		b.config.SlidingWindow = v
	} else {
		b.config.SlidingWindow = 1024
	}

	if v, ok := base.GetFloat("rope_theta"); ok {
		b.config.RopeTheta = v
	} else {
		b.config.RopeTheta = 1e6
	}

	if v, ok := base.GetFloat("rope_local_base_freq"); ok {
		b.config.RopeLocalTheta = v
	} else {
		// Gemma 3 uses 10K for local (sliding window) attention layers.
		b.config.RopeLocalTheta = 1e4
	}

	if v, ok := base.GetFloat("rms_norm_eps"); ok {
		b.config.RMSNormEps = v
	} else {
		b.config.RMSNormEps = 1e-6
	}

	return nil
}

// Config returns the base configuration.
func (b *Builder) Config() *models.BaseConfig {
	return b.config.BaseConfig
}

// LoadWeights loads weights into the GoMLX context.
// Selects the appropriate weight mapping based on the weight source type.
// For GGUF sources, tracks which weights are quantized for use with QuantizedDense.
func (b *Builder) LoadWeights(ctx *context.Context, weights models.WeightSource) error {
	var mapping map[string]string
	if _, ok := weights.(*models.GGUFSource); ok {
		mapping = b.ggufWeightMapping()
		b.isGGUF = true
	} else {
		mapping = b.hfWeightMapping()
	}
	quantInfo, err := models.LoadWeightsFromMapping(weights, mapping, ctx)
	if err != nil {
		return err
	}
	b.quantInfo = quantInfo
	return nil
}

// denseOrQuantized applies a weight-only dense layer, using QuantizedDense for
// GGML-quantized weights or regular DenseWeightOnly for float weights.
func (b *Builder) denseOrQuantized(ctx *context.Context, x *Node) *Node {
	scopePath := strings.TrimPrefix(ctx.Scope(), "/") + "/weights"
	if qt, ok := b.quantInfo[scopePath]; ok {
		return common.QuantizedDenseWeightOnly(ctx, x, qt)
	}
	return common.DenseWeightOnly(ctx, x)
}

// WeightMapping returns the GGUF weight mapping (primary target).
func (b *Builder) WeightMapping() map[string]string {
	return b.ggufWeightMapping()
}

// ggufWeightMapping returns the mapping from GGUF tensor names to context scope paths.
func (b *Builder) ggufWeightMapping() map[string]string {
	mapping := make(map[string]string)
	cfg := b.config

	// Embeddings.
	mapping["token_embd.weight"] = "embeddings/embeddings"

	// Layers.
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		blk := fmt.Sprintf("blk.%d", i)
		scope := fmt.Sprintf("layers/%d", i)

		// Pre-attention norm.
		mapping[blk+".attn_norm.weight"] = scope + "/input_norm/weight"

		// Attention projections.
		mapping[blk+".attn_q.weight"] = scope + "/attention/query/weights"
		mapping[blk+".attn_k.weight"] = scope + "/attention/key/weights"
		mapping[blk+".attn_v.weight"] = scope + "/attention/value/weights"
		mapping[blk+".attn_output.weight"] = scope + "/attention/output/weights"

		// QK-norm.
		mapping[blk+".attn_q_norm.weight"] = scope + "/attention/q_norm/weight"
		mapping[blk+".attn_k_norm.weight"] = scope + "/attention/k_norm/weight"

		// Post-attention norm.
		mapping[blk+".post_attention_norm.weight"] = scope + "/post_attn_norm/weight"

		// Pre-FFN norm.
		mapping[blk+".ffn_norm.weight"] = scope + "/ffn_norm/weight"

		// MLP (gated).
		mapping[blk+".ffn_gate.weight"] = scope + "/mlp/gate/weights"
		mapping[blk+".ffn_up.weight"] = scope + "/mlp/up/weights"
		mapping[blk+".ffn_down.weight"] = scope + "/mlp/down/weights"

		// Post-FFN norm.
		mapping[blk+".post_ffw_norm.weight"] = scope + "/post_ffn_norm/weight"
	}

	// Final norm.
	mapping["output_norm.weight"] = "norm/weight"

	// LM head (may be absent if tied to embeddings).
	mapping["output.weight"] = "lm_head/weights"

	return mapping
}

// hfWeightMapping returns the mapping from HuggingFace tensor names to context scope paths.
func (b *Builder) hfWeightMapping() map[string]string {
	mapping := make(map[string]string)
	cfg := b.config
	prefix := "model"

	// Embeddings.
	mapping[prefix+".embed_tokens.weight"] = "embeddings/embeddings"

	// Layers.
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		lp := fmt.Sprintf("%s.layers.%d", prefix, i)
		scope := fmt.Sprintf("layers/%d", i)

		// Pre-attention norm.
		mapping[lp+".input_layernorm.weight"] = scope + "/input_norm/weight"

		// Attention projections.
		mapping[lp+".self_attn.q_proj.weight"] = scope + "/attention/query/weights"
		mapping[lp+".self_attn.k_proj.weight"] = scope + "/attention/key/weights"
		mapping[lp+".self_attn.v_proj.weight"] = scope + "/attention/value/weights"
		mapping[lp+".self_attn.o_proj.weight"] = scope + "/attention/output/weights"

		// QK-norm.
		mapping[lp+".self_attn.q_norm.weight"] = scope + "/attention/q_norm/weight"
		mapping[lp+".self_attn.k_norm.weight"] = scope + "/attention/k_norm/weight"

		// Post-attention norm.
		mapping[lp+".post_attention_layernorm.weight"] = scope + "/post_attn_norm/weight"

		// Pre-FFN norm.
		mapping[lp+".pre_feedforward_layernorm.weight"] = scope + "/ffn_norm/weight"

		// MLP (gated).
		mapping[lp+".mlp.gate_proj.weight"] = scope + "/mlp/gate/weights"
		mapping[lp+".mlp.up_proj.weight"] = scope + "/mlp/up/weights"
		mapping[lp+".mlp.down_proj.weight"] = scope + "/mlp/down/weights"

		// Post-FFN norm.
		mapping[lp+".post_feedforward_layernorm.weight"] = scope + "/post_ffn_norm/weight"
	}

	// Final norm.
	mapping[prefix+".norm.weight"] = "norm/weight"

	// LM head.
	mapping["lm_head.weight"] = "lm_head/weights"

	return mapping
}

// BuildEmbeddings builds the embedding layer with Gemma 3 scaling.
// For quantized GGUF models, uses QuantizedEmbedding to dequantize only selected rows.
func (b *Builder) BuildEmbeddings(ctx *context.Context, inputIDs *Node) *Node {
	embCtx := ctx.In("embeddings")
	cfg := b.config

	var embeddings *Node
	if qt, ok := b.quantInfo["embeddings/embeddings"]; ok {
		// Quantized path: dequantize only the selected rows on-the-fly.
		embeddings = common.QuantizedEmbedding(embCtx, inputIDs, qt)
	} else {
		// Float path.
		vocabSize := cfg.VocabSize
		if v := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings"); v != nil {
			vocabSize = v.Shape().Dimensions[0]
		}
		embeddings = common.Embedding(embCtx, inputIDs, vocabSize, cfg.HiddenSize)
	}

	// Gemma 3 scales embeddings by sqrt(hidden_size).
	scale := ConstAs(embeddings, math.Sqrt(float64(cfg.HiddenSize)))
	embeddings = Mul(embeddings, scale)

	return embeddings
}

// BuildAttention builds the self-attention layer with QK-norm, RoPE, and optional sliding window.
func (b *Builder) BuildAttention(ctx *context.Context, hidden, positionIDs *Node, layerIdx int) *Node {
	out, _, _ := b.buildAttentionPrefill(ctx, hidden, positionIDs, layerIdx)
	return out
}


// BuildMLP builds the GeLU-gated MLP.
func (b *Builder) BuildMLP(ctx *context.Context, hidden *Node) *Node {
	mlpCtx := ctx.In("mlp")

	// Gated GeLU: gate_proj(x) * GeLU(up_proj(x)), then down_proj.
	gate := b.denseOrQuantized(mlpCtx.In("gate"), hidden)
	up := b.denseOrQuantized(mlpCtx.In("up"), hidden)

	activated := Mul(activations.GeluApproximate(gate), up)

	return b.denseOrQuantized(mlpCtx.In("down"), activated)
}

// BuildDecoderLayer builds a single decoder layer with 4 norms.
func (b *Builder) BuildDecoderLayer(ctx *context.Context, hidden, positionIDs *Node, layerIdx int) *Node {
	out, _, _ := b.buildDecoderLayerPrefill(ctx, hidden, positionIDs, layerIdx)
	return out
}

// BuildDecoder builds the full decoder stack.
func (b *Builder) BuildDecoder(ctx *context.Context, hidden, positionIDs *Node) *Node {
	for i := 0; i < b.config.NumHiddenLayers; i++ {
		hidden = b.BuildDecoderLayer(ctx.In("layers").In(strconv.Itoa(i)), hidden, positionIDs, i)
	}

	// Final normalization.
	hidden = common.RMSNorm(ctx.In("norm"), hidden, b.config.RMSNormEps)

	return hidden
}

// Forward runs the forward pass.
func (b *Builder) Forward(ctx *context.Context, inputIDs, positionIDs *Node) *Node {
	g := inputIDs.Graph()

	// Embeddings with scaling.
	hidden := b.BuildEmbeddings(ctx, inputIDs)

	// Create position IDs if not provided.
	if positionIDs == nil {
		batchSize := inputIDs.Shape().Dimensions[0]
		seqLen := inputIDs.Shape().Dimensions[1]
		positionIDs = common.GetPositionIDs(g, batchSize, seqLen)
	}

	// Decoder.
	hidden = b.BuildDecoder(ctx, hidden, positionIDs)

	// LM head (or tied embeddings).
	return b.ApplyLMHead(ctx, hidden, g)
}

// CreateExecGraphFn returns a function suitable for context.NewExec.
func (b *Builder) CreateExecGraphFn() func(*context.Context, *Node) *Node {
	return func(ctx *context.Context, inputIDs *Node) *Node {
		return b.Forward(ctx, inputIDs, nil)
	}
}

// GetVariableShape returns the expected shape for a variable.
func (b *Builder) GetVariableShape(name string) shapes.Shape {
	cfg := b.config

	switch {
	case strings.Contains(name, "embed_tokens") || strings.Contains(name, "token_embd"):
		return shapes.Make(dtypes.Float32, cfg.VocabSize, cfg.HiddenSize)
	case strings.Contains(name, "attn_q") || strings.Contains(name, "q_proj"):
		return shapes.Make(dtypes.Float32, cfg.NumAttentionHeads*cfg.HeadDim, cfg.HiddenSize)
	case strings.Contains(name, "attn_k") || strings.Contains(name, "k_proj"):
		return shapes.Make(dtypes.Float32, cfg.KVHeads()*cfg.HeadDim, cfg.HiddenSize)
	case strings.Contains(name, "attn_v") || strings.Contains(name, "v_proj"):
		return shapes.Make(dtypes.Float32, cfg.KVHeads()*cfg.HeadDim, cfg.HiddenSize)
	case strings.Contains(name, "ffn_gate") || strings.Contains(name, "gate_proj"):
		return shapes.Make(dtypes.Float32, cfg.IntermediateSize, cfg.HiddenSize)
	case strings.Contains(name, "ffn_up") || strings.Contains(name, "up_proj"):
		return shapes.Make(dtypes.Float32, cfg.IntermediateSize, cfg.HiddenSize)
	case strings.Contains(name, "ffn_down") || strings.Contains(name, "down_proj"):
		return shapes.Make(dtypes.Float32, cfg.HiddenSize, cfg.IntermediateSize)
	default:
		return shapes.Shape{}
	}
}

// ---------------------------------------------------------------------------
// KV-cached inference: ForwardPrefill + ForwardDecode
// ---------------------------------------------------------------------------

// ForwardPrefill runs the full prompt through the model and returns logits plus KV cache.
// inputIDs: [batch, seqLen], seqLenNode: scalar int32 (actual sequence length, for padded inputs).
// Returns [lastLogits, allKeys, allValues] where:
//   - lastLogits: [vocabSize]
//   - allKeys:    [numLayers, batch, kvHeads, seqLen, headDim]
//   - allValues:  same shape
func (b *Builder) ForwardPrefill(ctx *context.Context, inputIDs, seqLenNode *Node) []*Node {
	cfg := b.config
	g := inputIDs.Graph()

	hidden := b.BuildEmbeddings(ctx, inputIDs)

	batchSize := inputIDs.Shape().Dimensions[0]
	seqLen := inputIDs.Shape().Dimensions[1]
	positionIDs := common.GetPositionIDs(g, batchSize, seqLen)

	allKeys := make([]*Node, cfg.NumHiddenLayers)
	allValues := make([]*Node, cfg.NumHiddenLayers)

	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerCtx := ctx.In("layers").In(strconv.Itoa(i))
		var keys, values *Node
		hidden, keys, values = b.buildDecoderLayerPrefill(layerCtx, hidden, positionIDs, i)
		allKeys[i] = keys
		allValues[i] = values
	}

	hidden = common.RMSNorm(ctx.In("norm"), hidden, cfg.RMSNormEps)

	// LM head.
	logits := b.ApplyLMHead(ctx, hidden, g)

	// Extract last position logits.
	vocabSize := logits.Shape().Dimensions[2]
	lastPos := SubScalar(seqLenNode, int32(1))
	lastLogits := DynamicSlice(logits, []*Node{
		Const(g, int32(0)), lastPos, Const(g, int32(0)),
	}, []int{1, 1, vocabSize})
	lastLogits = Reshape(lastLogits, vocabSize)

	stackedKeys := Stack(allKeys, 0)
	stackedValues := Stack(allValues, 0)

	return []*Node{lastLogits, stackedKeys, stackedValues}
}

// ForwardDecode processes a single new token with KV cache.
// newTokenID: [batch, 1], positionID: [batch, 1],
// allKeys/allValues: [numLayers, batch, kvHeads, bufferLen, headDim],
// kvInsertPos: scalar int32 (position to insert new K/V).
// Returns [logits, updatedKeys, updatedValues] where logits is [vocabSize].
func (b *Builder) ForwardDecode(ctx *context.Context, newTokenID, positionID, allKeys, allValues, kvInsertPos *Node) []*Node {
	cfg := b.config
	g := newTokenID.Graph()

	hidden := b.BuildEmbeddings(ctx, newTokenID)

	batchSize := newTokenID.Shape().Dimensions[0]
	kvHeads := cfg.KVHeads()
	headDim := cfg.HeadDim
	bufferLen := allKeys.Shape().Dimensions[3]

	updatedLayerKeys := make([]*Node, cfg.NumHiddenLayers)
	updatedLayerValues := make([]*Node, cfg.NumHiddenLayers)

	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerCtx := ctx.In("layers").In(strconv.Itoa(i))

		// Extract this layer's KV from the stacked tensor.
		layerKeys := Slice(allKeys,
			AxisRange(i, i+1), AxisRange(), AxisRange(), AxisRange(), AxisRange())
		layerKeys = Reshape(layerKeys, batchSize, kvHeads, bufferLen, headDim)

		layerValues := Slice(allValues,
			AxisRange(i, i+1), AxisRange(), AxisRange(), AxisRange(), AxisRange())
		layerValues = Reshape(layerValues, batchSize, kvHeads, bufferLen, headDim)

		var updK, updV *Node
		hidden, updK, updV = b.buildDecoderLayerDecode(
			layerCtx, hidden, positionID, layerKeys, layerValues, kvInsertPos, i)

		updatedLayerKeys[i] = updK
		updatedLayerValues[i] = updV
	}

	hidden = common.RMSNorm(ctx.In("norm"), hidden, cfg.RMSNormEps)

	// LM head — single token, logits are [batch, 1, vocabSize].
	logits := b.ApplyLMHead(ctx, hidden, g)
	vocabSize := logits.Shape().Dimensions[2]
	logits = Reshape(logits, vocabSize)

	newAllKeys := Stack(updatedLayerKeys, 0)
	newAllValues := Stack(updatedLayerValues, 0)

	return []*Node{logits, newAllKeys, newAllValues}
}

// buildDecoderLayerPrefill runs one decoder layer and also returns the cached K/V.
func (b *Builder) buildDecoderLayerPrefill(ctx *context.Context, hidden, positionIDs *Node, layerIdx int) (*Node, *Node, *Node) {
	cfg := b.config

	normalized := common.RMSNorm(ctx.In("input_norm"), hidden, cfg.RMSNormEps)
	attnOutput, keys, values := b.buildAttentionPrefill(ctx, normalized, positionIDs, layerIdx)
	attnOutput = common.RMSNorm(ctx.In("post_attn_norm"), attnOutput, cfg.RMSNormEps)
	hidden = Add(hidden, attnOutput)

	normalized = common.RMSNorm(ctx.In("ffn_norm"), hidden, cfg.RMSNormEps)
	mlpOutput := b.BuildMLP(ctx, normalized)
	mlpOutput = common.RMSNorm(ctx.In("post_ffn_norm"), mlpOutput, cfg.RMSNormEps)
	hidden = Add(hidden, mlpOutput)

	return hidden, keys, values
}

// buildDecoderLayerDecode runs one decoder layer with KV cache.
func (b *Builder) buildDecoderLayerDecode(ctx *context.Context, hidden, positionIDs, prevKeys, prevValues, kvInsertPos *Node, layerIdx int) (*Node, *Node, *Node) {
	cfg := b.config

	normalized := common.RMSNorm(ctx.In("input_norm"), hidden, cfg.RMSNormEps)
	attnOutput, updKeys, updValues := b.buildAttentionDecode(
		ctx, normalized, positionIDs, prevKeys, prevValues, kvInsertPos, layerIdx)
	attnOutput = common.RMSNorm(ctx.In("post_attn_norm"), attnOutput, cfg.RMSNormEps)
	hidden = Add(hidden, attnOutput)

	normalized = common.RMSNorm(ctx.In("ffn_norm"), hidden, cfg.RMSNormEps)
	mlpOutput := b.BuildMLP(ctx, normalized)
	mlpOutput = common.RMSNorm(ctx.In("post_ffn_norm"), mlpOutput, cfg.RMSNormEps)
	hidden = Add(hidden, mlpOutput)

	return hidden, updKeys, updValues
}

// buildAttentionPrefill is like BuildAttention but also returns K/V after QK-norm + RoPE.
// Returns (attnOutput, keys, values) where keys/values are [batch, kvHeads, seqLen, headDim].
func (b *Builder) buildAttentionPrefill(ctx *context.Context, hidden, positionIDs *Node, layerIdx int) (*Node, *Node, *Node) {
	cfg := b.config
	attnCtx := ctx.In("attention")

	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	headDim := cfg.HeadDim
	kvHeads := cfg.KVHeads()
	headsPerGroup := cfg.HeadsPerKVGroup()

	query := b.denseOrQuantized(attnCtx.In("query"), hidden)
	key := b.denseOrQuantized(attnCtx.In("key"), hidden)
	value := b.denseOrQuantized(attnCtx.In("value"), hidden)

	query = Reshape(query, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	query = Transpose(query, 1, 2)

	key = Reshape(key, batchSize, seqLen, kvHeads, headDim)
	key = Transpose(key, 1, 2)

	value = Reshape(value, batchSize, seqLen, kvHeads, headDim)
	value = Transpose(value, 1, 2)

	query = common.RMSNorm(attnCtx.In("q_norm"), query, cfg.RMSNormEps)
	key = common.RMSNorm(attnCtx.In("k_norm"), key, cfg.RMSNormEps)

	ropeTheta := cfg.RopeFreqBase(layerIdx)
	query, key = common.RoPE(query, key, positionIDs, ropeTheta, seqLen, headDim)

	// Save K/V for cache (before GQA expansion).
	cachedKeys := key
	cachedValues := value

	if headsPerGroup > 1 {
		key = common.RepeatKV(key, headsPerGroup)
		value = common.RepeatKV(value, headsPerGroup)
	}

	scores := Einsum("bhqd,bhkd->bhqk", query, key)
	scale := ConstAs(scores, 1.0/math.Sqrt(float64(headDim)))
	scores = Mul(scores, scale)

	g := hidden.Graph()
	if cfg.IsLocalAttentionLayer(layerIdx) && cfg.SlidingWindow > 0 {
		mask := common.CreateSlidingWindowCausalMask(g, seqLen, cfg.SlidingWindow, scores.DType())
		scores = Add(scores, mask)
	} else {
		mask := common.CreateCausalMask(g, seqLen, scores.DType())
		scores = Add(scores, mask)
	}

	attnWeights := Softmax(scores, -1)
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, value)

	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, seqLen, cfg.NumAttentionHeads*headDim)

	attnOutput = b.denseOrQuantized(attnCtx.In("output"), attnOutput)

	return attnOutput, cachedKeys, cachedValues
}

// buildAttentionDecode processes a single new token with KV cache.
// hidden: [batch, 1, hiddenSize], positionIDs: [batch, 1],
// prevKeys/prevValues: [batch, kvHeads, bufferLen, headDim],
// kvInsertPos: scalar int32.
// Returns (attnOutput, updatedKeys, updatedValues).
func (b *Builder) buildAttentionDecode(ctx *context.Context, hidden, positionIDs, prevKeys, prevValues, kvInsertPos *Node, layerIdx int) (*Node, *Node, *Node) {
	cfg := b.config
	attnCtx := ctx.In("attention")
	g := hidden.Graph()

	batchSize := hidden.Shape().Dimensions[0]
	headDim := cfg.HeadDim
	kvHeads := cfg.KVHeads()
	headsPerGroup := cfg.HeadsPerKVGroup()
	bufferLen := prevKeys.Shape().Dimensions[2]

	// Q/K/V projections on single token.
	query := b.denseOrQuantized(attnCtx.In("query"), hidden)
	key := b.denseOrQuantized(attnCtx.In("key"), hidden)
	value := b.denseOrQuantized(attnCtx.In("value"), hidden)

	// Reshape: [batch, 1, proj_dim] -> [batch, heads, 1, headDim]
	query = Reshape(query, batchSize, 1, cfg.NumAttentionHeads, headDim)
	query = Transpose(query, 1, 2)

	key = Reshape(key, batchSize, 1, kvHeads, headDim)
	key = Transpose(key, 1, 2)

	value = Reshape(value, batchSize, 1, kvHeads, headDim)
	value = Transpose(value, 1, 2)

	// QK-norm.
	query = common.RMSNorm(attnCtx.In("q_norm"), query, cfg.RMSNormEps)
	key = common.RMSNorm(attnCtx.In("k_norm"), key, cfg.RMSNormEps)

	// RoPE with explicit position — use per-layer theta.
	ropeTheta := cfg.RopeFreqBase(layerIdx)
	query, key = common.RoPE(query, key, positionIDs, ropeTheta, 1, headDim)

	// Insert new K/V into buffer at kvInsertPos.
	updatedKeys := DynamicUpdateSlice(prevKeys, key, []*Node{
		Const(g, int32(0)), Const(g, int32(0)), kvInsertPos, Const(g, int32(0)),
	})
	updatedValues := DynamicUpdateSlice(prevValues, value, []*Node{
		Const(g, int32(0)), Const(g, int32(0)), kvInsertPos, Const(g, int32(0)),
	})

	// Build decode attention mask.
	mask := b.buildDecodeMask(g, bufferLen, kvInsertPos, layerIdx, hidden.DType())

	// Expand KV for GQA.
	fullKeys := common.RepeatKV(updatedKeys, headsPerGroup)
	fullValues := common.RepeatKV(updatedValues, headsPerGroup)

	// Attention scores: [batch, heads, 1, bufferLen]
	scores := Einsum("bhqd,bhkd->bhqk", query, fullKeys)
	scale := ConstAs(scores, 1.0/math.Sqrt(float64(headDim)))
	scores = Mul(scores, scale)

	scores = Add(scores, mask)

	attnWeights := Softmax(scores, -1)
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, fullValues)

	// Reshape: [batch, heads, 1, headDim] -> [batch, 1, heads*headDim]
	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, 1, cfg.NumAttentionHeads*headDim)

	attnOutput = b.denseOrQuantized(attnCtx.In("output"), attnOutput)

	return attnOutput, updatedKeys, updatedValues
}

// buildDecodeMask builds an attention mask for the decode step.
// Valid positions (< realLen) get 0; invalid positions get -1e9.
// For local attention layers, positions outside the sliding window are also masked.
func (b *Builder) buildDecodeMask(g *Graph, bufferLen int, kvInsertPos *Node, layerIdx int, dtype dtypes.DType) *Node {
	cfg := b.config

	// positions = [0, 1, 2, ..., bufferLen-1]
	positions := Iota(g, shapes.Make(dtypes.Int32, bufferLen), 0)

	// realLen = kvInsertPos + 1 (the new token is at kvInsertPos).
	realLen := AddScalar(kvInsertPos, int32(1))

	// inRange: 1 where position < realLen, 0 otherwise.
	inRange := Where(
		LessThan(positions, realLen),
		ConstAs(positions, int32(1)),
		ConstAs(positions, int32(0)),
	)

	validMask := inRange

	if cfg.IsLocalAttentionLayer(layerIdx) && cfg.SlidingWindow > 0 {
		// windowStart = max(realLen - slidingWindow, 0)
		windowStart := Max(
			SubScalar(realLen, int32(cfg.SlidingWindow)),
			Const(g, int32(0)),
		)
		// inWindow: 1 where position >= windowStart, 0 otherwise.
		// Equivalent to: NOT (position < windowStart).
		inWindow := Where(
			LessThan(positions, windowStart),
			ConstAs(positions, int32(0)),
			ConstAs(positions, int32(1)),
		)
		// Both conditions must hold.
		validMask = Mul(validMask, inWindow)
	}

	// Convert to float mask: 0 for valid, -1e9 for invalid.
	maskFloat := ConvertDType(validMask, dtype)
	one := ConstAs(maskFloat, 1.0)
	negInf := ConstAs(maskFloat, -1e9)
	mask := Mul(Sub(one, maskFloat), negInf)

	return Reshape(mask, 1, 1, 1, bufferLen)
}

// applyLMHead applies the language model head (or tied embeddings).
// hidden: [batch, seqLen, hiddenSize], returns [batch, seqLen, vocabSize].
func (b *Builder) ApplyLMHead(ctx *context.Context, hidden *Node, g *Graph) *Node {
	cfg := b.config

	lmHeadCtx := ctx.In("lm_head")
	lmHeadVar := lmHeadCtx.GetVariableByScopeAndName(lmHeadCtx.Scope(), "weights")
	if lmHeadVar != nil {
		return b.denseOrQuantized(lmHeadCtx, hidden)
	}

	// Tied embeddings: reuse token_embd.weight.
	embCtx := ctx.In("embeddings")
	embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
	if embVar == nil {
		panic("gemma3: neither lm_head nor embeddings weights found")
	}
	embWeights := embVar.ValueGraph(g)
	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]

	if qt, ok := b.quantInfo["embeddings/embeddings"]; ok {
		// Quantized tied embeddings: use QuantizedDense.
		// GGML weights are [vocabSize, bytesPerRow] — already in the layout QuantizedDense expects.
		quant := &Quantization{
			Scheme:   backends.QuantGGML,
			GGMLType: qt,
		}
		return nn.QuantizedDense(hidden, embWeights, quant, nil)
	}

	// Float tied embeddings.
	vocabSize := embVar.Shape().Dimensions[0]
	hiddenFlat := Reshape(hidden, batchSize*seqLen, cfg.HiddenSize)
	logits := Einsum("bh,vh->bv", hiddenFlat, embWeights)
	return Reshape(logits, batchSize, seqLen, vocabSize)
}

// Gemma3Config returns the Gemma 3-specific configuration.
// This is useful for examples that need access to architecture-specific parameters.
func (b *Builder) Gemma3Config() *Config {
	return b.config
}

