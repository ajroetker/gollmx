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
	"strings"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"

	"github.com/ajroetker/gollmx"
	"github.com/ajroetker/gollmx/architectures/common"
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
	config *Config
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

	return nil
}

// Config returns the base configuration.
func (b *Builder) Config() *models.BaseConfig {
	return b.config.BaseConfig
}

// LoadWeights loads weights into the GoMLX context.
func (b *Builder) LoadWeights(ctx *context.Context, weights models.WeightSource) error {
	_, err := models.LoadWeightsFromMapping(weights, b.WeightMapping(), ctx)
	return err
}

// WeightMapping returns the mapping from safetensors keys to context scope paths.
func (b *Builder) WeightMapping() map[string]string {
	mapping := make(map[string]string)
	cfg := b.config
	prefix := "model"

	// Embeddings.
	mapping[prefix+".embed_tokens.weight"] = "embeddings/embeddings"

	// Layers.
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("%s.layers.%d", prefix, i)
		layerScope := fmt.Sprintf("layers/%d", i)

		// Input LayerNorm (RMSNorm).
		mapping[layerPrefix+".input_layernorm.weight"] = layerScope + "/input_norm/weight"

		// Self-attention.
		mapping[layerPrefix+".self_attn.q_proj.weight"] = layerScope + "/attention/query/weights"
		mapping[layerPrefix+".self_attn.k_proj.weight"] = layerScope + "/attention/key/weights"
		mapping[layerPrefix+".self_attn.v_proj.weight"] = layerScope + "/attention/value/weights"
		mapping[layerPrefix+".self_attn.o_proj.weight"] = layerScope + "/attention/output/weights"

		// Post-attention LayerNorm (RMSNorm).
		mapping[layerPrefix+".post_attention_layernorm.weight"] = layerScope + "/post_attn_norm/weight"

		// MLP (gate-up-down projections).
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
func (b *Builder) BuildEmbeddings(ctx *context.Context, inputIDs *Node) *Node {
	embCtx := ctx.In("embeddings")

	// Word embeddings only (Llama uses RoPE for positions).
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
	query := common.DenseWeightOnly(attnCtx.In("query"), hidden)
	key := common.DenseWeightOnly(attnCtx.In("key"), hidden)
	value := common.DenseWeightOnly(attnCtx.In("value"), hidden)

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
	scale := ConstAs(scores, 1.0/float64(headDim))
	scores = Mul(scores, Sqrt(scale))

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
	attnOutput = common.DenseWeightOnly(attnCtx.In("output"), attnOutput)

	return attnOutput
}


// BuildMLP builds the SwiGLU MLP.
func (b *Builder) BuildMLP(ctx *context.Context, hidden *Node) *Node {
	mlpCtx := ctx.In("mlp")

	// SwiGLU: gate * SiLU(up) then down projection
	gate := common.DenseWeightOnly(mlpCtx.In("gate"), hidden)
	up := common.DenseWeightOnly(mlpCtx.In("up"), hidden)

	// SwiGLU activation: gate * SiLU(up)
	activated := Mul(activations.Swish(gate), up)

	// Down projection.
	return common.DenseWeightOnly(mlpCtx.In("down"), activated)
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
		hidden = b.BuildDecoderLayer(ctx.In("layers").In(itoa(i)), hidden, attentionMask, positionIDs)
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

func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}
