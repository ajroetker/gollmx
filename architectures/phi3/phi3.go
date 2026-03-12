// Package phi3 provides Phi-3/Phi-3-Vision architecture implementation for GoMLX.
//
// Phi-3 uses:
//   - RoPE (Rotary Position Embedding) with su/longrope scaling
//   - RMSNorm
//   - SiLU activation in gated MLP
//   - Grouped Query Attention (GQA)
//   - Optional CLIP vision encoder with 3-layer MLP projector
//
// Reference: https://arxiv.org/abs/2404.14219
package phi3

import (
	"fmt"
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
	models.RegisterArchitecture("phi3", func() models.ArchitectureBuilder { return &Builder{} })
}

// Config holds Phi-3-specific configuration.
type Config struct {
	*models.BaseConfig

	// RoPE configuration.
	RopeTheta          float64   `json:"rope_theta"`          // Default 10000.0
	PartialRotaryFactor float64  `json:"partial_rotary_factor"` // Fraction of headDim to apply RoPE to (0 or 1 = all).
	OrigMaxSeqLen      int       `json:"original_max_position_embeddings"` // LongRoPE threshold.
	LongRoPELongFactor []float32 // Per-dimension scaling for long sequences.
	LongRoPEShortFactor []float32 // Per-dimension scaling for short sequences.

	// RMSNorm epsilon.
	RMSNormEps float64 `json:"rms_norm_eps"`

	// Grouped Query Attention.
	NumKeyValueHeads int `json:"num_key_value_heads"`
}

// RotaryDim returns the number of dimensions to apply RoPE to.
func (c *Config) RotaryDim() int {
	headDim := c.KVHeadDim()
	if c.PartialRotaryFactor > 0 && c.PartialRotaryFactor < 1.0 {
		return int(c.PartialRotaryFactor * float64(headDim))
	}
	return headDim
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

// Builder implements the Phi-3 architecture.
type Builder struct {
	config          *Config
	visionConfig    *common.VisionConfig // nil for text-only
	projectorLayers int                  // number of linear layers in projector (default 3)
	imageTokenID    int32                // token ID for <|image_1|> placeholder (from tokenizer)
	isGGUF          bool
	quantInfo       models.QuantInfo
}

// Name returns the architecture name.
func (b *Builder) Name() string {
	return "Phi3"
}

// ParseConfig extracts Phi-3-specific configuration from BaseConfig.Raw.
func (b *Builder) ParseConfig(base *models.BaseConfig) error {
	b.config = &Config{BaseConfig: base}

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

	// Partial rotary factor (Phi-4-mini uses 0.75).
	if v, ok := base.GetFloat("partial_rotary_factor"); ok {
		b.config.PartialRotaryFactor = v
	}

	// LongRoPE: original max sequence length and per-dimension factors.
	if v, ok := base.GetInt("original_max_position_embeddings"); ok {
		b.config.OrigMaxSeqLen = v
	}
	b.parseLongRoPEFactors(base)

	// Parse vision config if present.
	b.parseVisionConfig(base)

	return nil
}

// parseLongRoPEFactors extracts LongRoPE per-dimension factors from config.
// In HF config.json, these are in rope_scaling.long_factor / rope_scaling.short_factor.
// In GGUF, they are stored as tensors and loaded via LoadWeights.
func (b *Builder) parseLongRoPEFactors(base *models.BaseConfig) {
	rsRaw, ok := base.Raw["rope_scaling"]
	if !ok {
		return
	}
	rs, ok := rsRaw.(map[string]interface{})
	if !ok {
		return
	}
	// Check type is "longrope" or "su".
	if tp, ok := rs["type"].(string); ok {
		if tp != "longrope" && tp != "su" {
			return
		}
	}
	b.config.LongRoPELongFactor = extractFloat32Slice(rs, "long_factor")
	b.config.LongRoPEShortFactor = extractFloat32Slice(rs, "short_factor")
}

// extractFloat32Slice extracts a []float32 from a map value that may be []interface{} or []float64.
func extractFloat32Slice(m map[string]interface{}, key string) []float32 {
	v, ok := m[key]
	if !ok {
		return nil
	}
	switch arr := v.(type) {
	case []interface{}:
		result := make([]float32, len(arr))
		for i, item := range arr {
			if f, ok := item.(float64); ok {
				result[i] = float32(f)
			}
		}
		return result
	case []float64:
		result := make([]float32, len(arr))
		for i, f := range arr {
			result[i] = float32(f)
		}
		return result
	case []float32:
		return arr
	}
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
		b.projectorLayers = 3 // Phi-3-Vision uses 3-layer MLP projector
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

	// Extract LongRoPE factor tensors from GGUF if present.
	// These are stored as regular tensors (not metadata).
	if b.config.LongRoPELongFactor == nil {
		b.config.LongRoPELongFactor = b.loadRoPEFactorTensor(weights, "rope_factors_long.weight")
	}
	if b.config.LongRoPEShortFactor == nil {
		b.config.LongRoPEShortFactor = b.loadRoPEFactorTensor(weights, "rope_factors_short.weight")
	}

	return nil
}

// loadRoPEFactorTensor tries to load a LongRoPE factor tensor from the weight source.
// Returns nil if the tensor doesn't exist.
func (b *Builder) loadRoPEFactorTensor(weights models.WeightSource, name string) []float32 {
	lw, err := weights.GetTensor(name)
	if err != nil || lw == nil {
		return nil
	}
	var factors []float32
	lw.Tensor.ConstFlatData(func(data any) {
		switch d := data.(type) {
		case []float32:
			factors = make([]float32, len(d))
			copy(factors, d)
		}
	})
	return factors
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

// WeightMapping returns the GGUF weight mapping.
func (b *Builder) WeightMapping() map[string]string {
	return b.ggufWeightMapping()
}

// ggufWeightMapping returns the mapping from GGUF tensor names to context scope paths.
func (b *Builder) ggufWeightMapping() map[string]string {
	mapping := make(map[string]string)
	cfg := b.config

	mapping["token_embd.weight"] = "embeddings/embeddings"

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

	mapping["output_norm.weight"] = "norm/weight"
	mapping["output.weight"] = "lm_head/weights"

	// Vision encoder weights (CLIP).
	if b.visionConfig != nil {
		vc := b.visionConfig

		mapping["v.patch_embd.weight"] = "vision/patch_embedding/weights"
		mapping["v.patch_embd.bias"] = "vision/patch_embedding/biases"
		mapping["v.position_embd.weight"] = "vision/position_embeddings"

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

		mapping["v.post_ln.weight"] = "vision/post_layernorm/gain"
		mapping["v.post_ln.bias"] = "vision/post_layernorm/offset"

		// 3-layer MLP projector: indices 0, 2, 4.
		for i := range b.projectorLayers {
			idx := i * 2
			mapping[fmt.Sprintf("v.projection.%d.weight", idx)] = fmt.Sprintf("mm/%d/weights", idx)
			mapping[fmt.Sprintf("v.projection.%d.bias", idx)] = fmt.Sprintf("mm/%d/biases", idx)
		}
	}

	return mapping
}

// hfWeightMapping returns the mapping from HuggingFace tensor names to context scope paths.
func (b *Builder) hfWeightMapping() map[string]string {
	mapping := make(map[string]string)
	cfg := b.config
	prefix := "model"

	mapping[prefix+".embed_tokens.weight"] = "embeddings/embeddings"

	for i := 0; i < cfg.NumHiddenLayers; i++ {
		lp := fmt.Sprintf("%s.layers.%d", prefix, i)
		scope := fmt.Sprintf("layers/%d", i)

		mapping[lp+".input_layernorm.weight"] = scope + "/input_norm/weight"
		mapping[lp+".self_attn.qkv_proj.weight"] = scope + "/attention/qkv/weights" // Phi-3 fuses QKV
		mapping[lp+".self_attn.o_proj.weight"] = scope + "/attention/output/weights"
		mapping[lp+".post_attention_layernorm.weight"] = scope + "/post_attn_norm/weight"
		mapping[lp+".mlp.gate_up_proj.weight"] = scope + "/mlp/gate_up/weights" // Phi-3 fuses gate+up
		mapping[lp+".mlp.down_proj.weight"] = scope + "/mlp/down/weights"
	}

	mapping[prefix+".norm.weight"] = "norm/weight"
	mapping["lm_head.weight"] = "lm_head/weights"

	return mapping
}

// BuildEmbeddings builds the embedding layer.
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

// BuildMLP builds the SwiGLU MLP.
func (b *Builder) BuildMLP(ctx *context.Context, hidden *Node) *Node {
	mlpCtx := ctx.In("mlp")

	gate := b.denseOrQuantized(mlpCtx.In("gate"), hidden)
	up := b.denseOrQuantized(mlpCtx.In("up"), hidden)
	activated := Mul(activations.Swish(gate), up)
	return b.denseOrQuantized(mlpCtx.In("down"), activated)
}

// ApplyLMHead applies the language model head (or tied embeddings).
func (b *Builder) ApplyLMHead(ctx *context.Context, hidden *Node, g *Graph) *Node {
	lmHeadCtx := ctx.In("lm_head")
	lmHeadVar := lmHeadCtx.GetVariableByScopeAndName(lmHeadCtx.Scope(), "weights")
	if lmHeadVar != nil {
		return b.denseOrQuantized(lmHeadCtx, hidden)
	}

	// Tied embeddings.
	embCtx := ctx.In("embeddings")
	embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
	if embVar == nil {
		panic("phi3: neither lm_head nor embeddings weights found")
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

// GetVariableShape returns the expected shape for a variable.
func (b *Builder) GetVariableShape(name string) shapes.Shape {
	cfg := b.config

	switch {
	case strings.Contains(name, "embed_tokens") || strings.Contains(name, "token_embd"):
		return shapes.Make(dtypes.Float32, cfg.VocabSize, cfg.HiddenSize)
	case strings.Contains(name, "attn_q") || strings.Contains(name, "q_proj"):
		return shapes.Make(dtypes.Float32, cfg.HiddenSize, cfg.HiddenSize)
	case strings.Contains(name, "attn_k") || strings.Contains(name, "k_proj"):
		kvDim := cfg.KVHeads() * cfg.KVHeadDim()
		return shapes.Make(dtypes.Float32, kvDim, cfg.HiddenSize)
	case strings.Contains(name, "gate_proj") || strings.Contains(name, "up_proj"):
		return shapes.Make(dtypes.Float32, cfg.IntermediateSize, cfg.HiddenSize)
	case strings.Contains(name, "down_proj"):
		return shapes.Make(dtypes.Float32, cfg.HiddenSize, cfg.IntermediateSize)
	default:
		return shapes.Shape{}
	}
}

// Phi3Config returns the Phi-3-specific configuration.
func (b *Builder) Phi3Config() *Config {
	return b.config
}
