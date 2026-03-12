// Package llama provides Llama/Mistral architecture implementation for GoMLX.
//
// Llama uses:
//   - RoPE (Rotary Position Embedding) for positions
//   - RMSNorm instead of LayerNorm
//   - SiLU activation in MLP
//   - Grouped Query Attention (GQA) for efficiency
//
// Reference: https://arxiv.org/abs/2302.13971
package llama

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

	models "github.com/gomlx/gollmx"
	"github.com/gomlx/gollmx/architectures/common"
)

func init() {
	models.RegisterArchitecture("llama", func() models.ArchitectureBuilder { return &Builder{} })
	models.RegisterArchitecture("mistral", func() models.ArchitectureBuilder { return &Builder{} })
}

// Config holds Llama-specific configuration.
type Config struct {
	*models.BaseConfig

	// RoPE configuration.
	RopeTheta   float64 `json:"rope_theta"`   // Default 10000.0
	RopeScaling *struct {
		Type   string  `json:"type"`   // "linear", "dynamic"
		Factor float64 `json:"factor"` // Scaling factor
	} `json:"rope_scaling,omitempty"`

	// RMSNorm epsilon (different from LayerNormEps).
	RMSNormEps float64 `json:"rms_norm_eps"`

	// Grouped Query Attention.
	NumKeyValueHeads int `json:"num_key_value_heads"` // If different from NumAttentionHeads

	// MLP configuration.
	MLPBias bool `json:"mlp_bias"` // Whether MLP has bias (usually false for Llama)
}

// KVHeads returns the number of key-value heads (for GQA).
func (c *Config) KVHeads() int {
	if c.NumKeyValueHeads > 0 {
		return c.NumKeyValueHeads
	}
	return c.NumAttentionHeads
}

// KVHeadDim returns the dimension per KV head.
func (c *Config) KVHeadDim() int {
	return c.HiddenSize / c.NumAttentionHeads
}

// HeadsPerKVGroup returns how many query heads share each KV head.
func (c *Config) HeadsPerKVGroup() int {
	return c.NumAttentionHeads / c.KVHeads()
}

// Builder implements the Llama architecture.
type Builder struct {
	config          *Config
	visionConfig    *common.VisionConfig // nil for text-only models
	projectorLayers int                  // number of linear layers in MLP projector (default 2)
	imageTokenID    int32                // token ID for <image> placeholder (from tokenizer)
	isGGUF          bool
	quantInfo       models.QuantInfo // scope path → GGML quant type
}

// Name returns the architecture name.
func (b *Builder) Name() string {
	return "Llama"
}

// ParseConfig extracts Llama-specific configuration from BaseConfig.Raw.
func (b *Builder) ParseConfig(base *models.BaseConfig) error {
	b.config = &Config{BaseConfig: base}

	// Parse Llama-specific fields from Raw.
	if v, ok := base.GetFloat("rope_theta"); ok {
		b.config.RopeTheta = v
	} else {
		b.config.RopeTheta = 10000.0
	}
	if v, ok := base.GetFloat("rms_norm_eps"); ok {
		b.config.RMSNormEps = v
	} else {
		b.config.RMSNormEps = 1e-5
	}
	if v, ok := base.GetInt("num_key_value_heads"); ok {
		b.config.NumKeyValueHeads = v
	}
	if v, ok := base.GetBool("mlp_bias"); ok {
		b.config.MLPBias = v
	}

	// Parse vision config if present (LLaVA multimodal).
	b.parseVisionConfig(base)

	return nil
}

// parseVisionConfig extracts vision encoder configuration if available.
func (b *Builder) parseVisionConfig(base *models.BaseConfig) {
	if v, ok := base.GetInt("vision.block_count"); ok {
		vc := &common.VisionConfig{NumLayers: v, UseGELU: true}
		if v, ok := base.GetInt("vision.embedding_length"); ok {
			vc.HiddenSize = v
		}
		if v, ok := base.GetInt("vision.attention.head_count"); ok {
			vc.NumHeads = v
		}
		if v, ok := base.GetInt("vision.feed_forward_length"); ok {
			vc.MLPDim = v
		}
		if v, ok := base.GetInt("vision.image_size"); ok {
			vc.ImageSize = v
		}
		if v, ok := base.GetInt("vision.patch_size"); ok {
			vc.PatchSize = v
		}
		if v, ok := base.GetInt("vision.num_channels"); ok {
			vc.NumChannels = v
		} else {
			vc.NumChannels = 3
		}
		if v, ok := base.GetFloat("vision.attention.layer_norm_epsilon"); ok {
			vc.LayerNormEps = v
		} else {
			vc.LayerNormEps = 1e-6
		}
		b.visionConfig = vc
		b.projectorLayers = 2 // LLaVA default: 2-layer MLP (mlp2x_gelu)
		if v, ok := base.GetInt("image_token_id"); ok {
			b.imageTokenID = int32(v)
		}
	}
}

// Config returns the base configuration.
func (b *Builder) Config() *models.BaseConfig {
	return b.config.BaseConfig
}

// LoadWeights loads weights into the GoMLX context.
func (b *Builder) LoadWeights(ctx *context.Context, weights models.WeightSource) error {
	var mapping map[string]string
	if _, ok := weights.(*models.GGUFSource); ok {
		mapping = b.ggufWeightMapping()
		b.isGGUF = true
	} else {
		mapping = b.hfWeightMapping()
	}
	quantInfo, err := models.LoadWeightsFromMapping(weights, mapping, ctx, models.LoadWeightsOptions{
		ComputeDType: dtypes.Float32,
	})
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

		mapping[blk+".attn_norm.weight"] = scope + "/input_norm/weight"
		mapping[blk+".attn_q.weight"] = scope + "/attention/query/weights"
		mapping[blk+".attn_k.weight"] = scope + "/attention/key/weights"
		mapping[blk+".attn_v.weight"] = scope + "/attention/value/weights"
		mapping[blk+".attn_output.weight"] = scope + "/attention/output/weights"
		mapping[blk+".ffn_norm.weight"] = scope + "/post_attn_norm/weight"
		mapping[blk+".ffn_gate.weight"] = scope + "/mlp/gate/weights"
		mapping[blk+".ffn_up.weight"] = scope + "/mlp/up/weights"
		mapping[blk+".ffn_down.weight"] = scope + "/mlp/down/weights"
	}

	// Final norm.
	mapping["output_norm.weight"] = "norm/weight"

	// LM head (may be absent if tied to embeddings).
	mapping["output.weight"] = "lm_head/weights"

	// Vision encoder weights (CLIP).
	if b.visionConfig != nil {
		vc := b.visionConfig

		mapping["v.patch_embedding.weight"] = "vision/patch_embedding/weights"
		mapping["v.patch_embedding.bias"] = "vision/patch_embedding/biases"
		mapping["v.position_embedding.weight"] = "vision/position_embeddings"

		for i := range vc.NumLayers {
			src := fmt.Sprintf("v.blk.%d", i)
			dst := fmt.Sprintf("vision/layers/%d", i)

			mapping[src+".layer_norm1.weight"] = dst + "/layer_norm1/gain"
			mapping[src+".layer_norm1.bias"] = dst + "/layer_norm1/offset"
			mapping[src+".layer_norm2.weight"] = dst + "/layer_norm2/gain"
			mapping[src+".layer_norm2.bias"] = dst + "/layer_norm2/offset"

			mapping[src+".attn_q.weight"] = dst + "/attn_q/weights"
			mapping[src+".attn_q.bias"] = dst + "/attn_q/biases"
			mapping[src+".attn_k.weight"] = dst + "/attn_k/weights"
			mapping[src+".attn_k.bias"] = dst + "/attn_k/biases"
			mapping[src+".attn_v.weight"] = dst + "/attn_v/weights"
			mapping[src+".attn_v.bias"] = dst + "/attn_v/biases"
			mapping[src+".attn_output.weight"] = dst + "/attn_output/weights"
			mapping[src+".attn_output.bias"] = dst + "/attn_output/biases"

			mapping[src+".mlp.fc1.weight"] = dst + "/mlp/fc1/weights"
			mapping[src+".mlp.fc1.bias"] = dst + "/mlp/fc1/biases"
			mapping[src+".mlp.fc2.weight"] = dst + "/mlp/fc2/weights"
			mapping[src+".mlp.fc2.bias"] = dst + "/mlp/fc2/biases"
		}

		mapping["v.post_layernorm.weight"] = "vision/post_layernorm/gain"
		mapping["v.post_layernorm.bias"] = "vision/post_layernorm/offset"

		// MLP projector: mm.0, mm.2 (2-layer), or mm.0, mm.2, mm.4 (3-layer).
		for i := range b.projectorLayers {
			idx := i * 2
			mapping[fmt.Sprintf("mm.%d.weight", idx)] = fmt.Sprintf("mm/%d/weights", idx)
			mapping[fmt.Sprintf("mm.%d.bias", idx)] = fmt.Sprintf("mm/%d/biases", idx)
		}
	}

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
		layerPrefix := fmt.Sprintf("%s.layers.%d", prefix, i)
		layerScope := fmt.Sprintf("layers/%d", i)

		mapping[layerPrefix+".input_layernorm.weight"] = layerScope + "/input_norm/weight"
		mapping[layerPrefix+".self_attn.q_proj.weight"] = layerScope + "/attention/query/weights"
		mapping[layerPrefix+".self_attn.k_proj.weight"] = layerScope + "/attention/key/weights"
		mapping[layerPrefix+".self_attn.v_proj.weight"] = layerScope + "/attention/value/weights"
		mapping[layerPrefix+".self_attn.o_proj.weight"] = layerScope + "/attention/output/weights"
		mapping[layerPrefix+".post_attention_layernorm.weight"] = layerScope + "/post_attn_norm/weight"
		mapping[layerPrefix+".mlp.gate_proj.weight"] = layerScope + "/mlp/gate/weights"
		mapping[layerPrefix+".mlp.up_proj.weight"] = layerScope + "/mlp/up/weights"
		mapping[layerPrefix+".mlp.down_proj.weight"] = layerScope + "/mlp/down/weights"
	}

	// Final norm.
	mapping[prefix+".norm.weight"] = "norm/weight"

	// LM head (optional, may be tied to embeddings).
	mapping["lm_head.weight"] = "lm_head/weights"

	return mapping
}

// BuildEmbeddings builds the embedding layer.
// For quantized GGUF models, uses QuantizedEmbedding to dequantize only selected rows.
func (b *Builder) BuildEmbeddings(ctx *context.Context, inputIDs *Node) *Node {
	embCtx := ctx.In("embeddings")

	if qt, ok := b.quantInfo["embeddings/embeddings"]; ok {
		return common.QuantizedEmbedding(embCtx, inputIDs, qt)
	}
	if v := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings"); v != nil {
		return common.EmbeddingFromVar(embCtx, inputIDs, v)
	}
	return common.Embedding(embCtx, inputIDs, b.config.VocabSize, b.config.HiddenSize)
}

// BuildAttention builds the self-attention layer with RoPE.
func (b *Builder) BuildAttention(ctx *context.Context, hidden, attentionMask, positionIDs *Node) *Node {
	g := hidden.Graph()
	cfg := b.config
	attnCtx := ctx.In("attention")

	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	headDim := cfg.HeadDim()
	kvHeads := cfg.KVHeads()
	headsPerGroup := cfg.HeadsPerKVGroup()

	// Q, K, V projections (no bias in Llama).
	query := b.denseOrQuantized(attnCtx.In("query"), hidden)
	key := b.denseOrQuantized(attnCtx.In("key"), hidden)
	value := b.denseOrQuantized(attnCtx.In("value"), hidden)

	// Reshape for multi-head attention.
	// Query: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
	query = Reshape(query, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	query = Transpose(query, 1, 2)

	// Key/Value: [batch, seq, kv_hidden] -> [batch, kv_heads, seq, head_dim]
	key = Reshape(key, batchSize, seqLen, kvHeads, headDim)
	key = Transpose(key, 1, 2)

	value = Reshape(value, batchSize, seqLen, kvHeads, headDim)
	value = Transpose(value, 1, 2)

	// Apply RoPE to query and key.
	query, key = common.RoPE(query, key, positionIDs, cfg.RopeTheta, seqLen, headDim)

	// Expand KV heads for grouped query attention.
	// [batch, kv_heads, seq, head_dim] -> [batch, heads, seq, head_dim]
	if headsPerGroup > 1 {
		// Repeat KV heads to match query heads.
		key = common.RepeatKV(key, headsPerGroup)
		value = common.RepeatKV(value, headsPerGroup)
	}

	// Attention scores: Q @ K^T / sqrt(d_k)
	scores := Einsum("bhqd,bhkd->bhqk", query, key)
	scale := ConstAs(scores, 1.0/math.Sqrt(float64(headDim)))
	scores = Mul(scores, scale)

	// Apply causal mask.
	causalMask := common.CreateCausalMask(g, seqLen, scores.DType())
	scores = Add(scores, causalMask)

	// Apply attention mask if provided.
	if attentionMask != nil {
		mask := common.ExpandAttentionMask(attentionMask, scores.DType())
		scores = Add(scores, mask)
	}

	// Softmax and attention output.
	attnWeights := Softmax(scores, -1)
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, value)

	// Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, seqLen, cfg.HiddenSize)

	// Output projection.
	attnOutput = b.denseOrQuantized(attnCtx.In("output"), attnOutput)

	return attnOutput
}


// BuildMLP builds the SwiGLU MLP.
func (b *Builder) BuildMLP(ctx *context.Context, hidden *Node) *Node {
	mlpCtx := ctx.In("mlp")

	// SwiGLU: SiLU(gate) * up, then down projection.
	gate := b.denseOrQuantized(mlpCtx.In("gate"), hidden)
	up := b.denseOrQuantized(mlpCtx.In("up"), hidden)
	activated := Mul(activations.Swish(gate), up)

	// Down projection.
	return b.denseOrQuantized(mlpCtx.In("down"), activated)
}

// BuildDecoderLayer builds a single decoder layer.
func (b *Builder) BuildDecoderLayer(ctx *context.Context, hidden, attentionMask, positionIDs *Node) *Node {
	cfg := b.config

	// Input normalization (RMSNorm).
	normalized := common.RMSNorm(ctx.In("input_norm"), hidden, cfg.RMSNormEps)

	// Self-attention with residual.
	attnOutput := b.BuildAttention(ctx, normalized, attentionMask, positionIDs)
	hidden = Add(hidden, attnOutput)

	// Post-attention normalization (RMSNorm).
	normalized = common.RMSNorm(ctx.In("post_attn_norm"), hidden, cfg.RMSNormEps)

	// MLP with residual.
	mlpOutput := b.BuildMLP(ctx, normalized)
	hidden = Add(hidden, mlpOutput)

	return hidden
}

// BuildDecoder builds the full decoder stack.
func (b *Builder) BuildDecoder(ctx *context.Context, hidden, attentionMask, positionIDs *Node) *Node {
	for i := 0; i < b.config.NumHiddenLayers; i++ {
		hidden = b.BuildDecoderLayer(ctx.In("layers").In(strconv.Itoa(i)), hidden, attentionMask, positionIDs)
	}

	// Final normalization.
	hidden = common.RMSNorm(ctx.In("norm"), hidden, b.config.RMSNormEps)

	return hidden
}

// Forward runs the forward pass.
func (b *Builder) Forward(ctx *context.Context, inputIDs, attentionMask, positionIDs *Node) *Node {
	g := inputIDs.Graph()

	// Embeddings.
	hidden := b.BuildEmbeddings(ctx, inputIDs)

	// Create position IDs if not provided.
	if positionIDs == nil {
		batchSize := inputIDs.Shape().Dimensions[0]
		seqLen := inputIDs.Shape().Dimensions[1]
		positionIDs = common.GetPositionIDs(g, batchSize, seqLen)
	}

	// Decoder.
	hidden = b.BuildDecoder(ctx, hidden, attentionMask, positionIDs)

	return hidden
}

// CreateExecGraphFn returns a function suitable for context.NewExec.
func (b *Builder) CreateExecGraphFn() func(*context.Context, *Node, *Node) *Node {
	return func(ctx *context.Context, inputIDs, attentionMask *Node) *Node {
		return b.Forward(ctx, inputIDs, attentionMask, nil)
	}
}

// GetVariableShape returns the expected shape for a variable.
func (b *Builder) GetVariableShape(name string) shapes.Shape {
	cfg := b.config

	switch {
	case strings.Contains(name, "embed_tokens"):
		return shapes.Make(dtypes.Float32, cfg.VocabSize, cfg.HiddenSize)
	case strings.Contains(name, "q_proj"):
		return shapes.Make(dtypes.Float32, cfg.HiddenSize, cfg.HiddenSize)
	case strings.Contains(name, "k_proj") || strings.Contains(name, "v_proj"):
		kvDim := cfg.KVHeads() * cfg.HeadDim()
		return shapes.Make(dtypes.Float32, kvDim, cfg.HiddenSize)
	case strings.Contains(name, "gate_proj") || strings.Contains(name, "up_proj"):
		return shapes.Make(dtypes.Float32, cfg.IntermediateSize, cfg.HiddenSize)
	case strings.Contains(name, "down_proj"):
		return shapes.Make(dtypes.Float32, cfg.HiddenSize, cfg.IntermediateSize)
	default:
		return shapes.Shape{}
	}
}

// ApplyLMHead applies the language model head (or tied embeddings).
func (b *Builder) ApplyLMHead(ctx *context.Context, hidden *Node) *Node {
	g := hidden.Graph()

	lmHeadCtx := ctx.In("lm_head")
	lmHeadVar := lmHeadCtx.GetVariableByScopeAndName(lmHeadCtx.Scope(), "weights")
	if lmHeadVar != nil {
		return b.denseOrQuantized(lmHeadCtx, hidden)
	}

	// Tied embeddings: reuse token embedding weight.
	embCtx := ctx.In("embeddings")
	embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
	if embVar == nil {
		panic("llama: neither lm_head nor embeddings weights found")
	}
	embWeights := embVar.ValueGraph(g)
	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]

	if qt, ok := b.quantInfo["embeddings/embeddings"]; ok {
		quant := &Quantization{
			Scheme:   backends.QuantGGML,
			GGMLType: qt,
		}
		return nn.QuantizedDense(hidden, embWeights, quant, nil)
	}

	if embWeights.DType() != hidden.DType() {
		embWeights = ConvertDType(embWeights, hidden.DType())
	}
	vocabSize := embVar.Shape().Dimensions[0]
	hiddenFlat := Reshape(hidden, batchSize*seqLen, b.config.HiddenSize)
	logits := Einsum("bh,vh->bv", hiddenFlat, embWeights)
	return Reshape(logits, batchSize, seqLen, vocabSize)
}

// LlamaConfig returns the Llama-specific configuration.
func (b *Builder) LlamaConfig() *Config {
	return b.config
}

