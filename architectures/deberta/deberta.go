// Package deberta provides DeBERTa v2/v3 architecture implementation for GoMLX.
//
// DeBERTa uses disentangled attention with separate content and position representations.
// Config-driven behavior based on:
//   - pos_att_type: ["c2p", "p2c"] - which attention components to use
//   - norm_rel_ebd: ["layer_norm"] - apply LayerNorm to relative embeddings
//   - share_att_key: true - reuse key projection for positions
//
// Reference: https://arxiv.org/abs/2006.03654
package deberta

import (
	"fmt"
	"slices"
	"strings"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"

	"github.com/ajroetker/huggingface-gomlx"
	"github.com/ajroetker/huggingface-gomlx/architectures/common"
)

func init() {
	models.RegisterArchitecture("deberta", func() models.ArchitectureBuilder { return &Builder{} })
	models.RegisterArchitecture("deberta-v2", func() models.ArchitectureBuilder { return &Builder{} })
	models.RegisterArchitecture("deberta-v3", func() models.ArchitectureBuilder { return &Builder{} })
}

// Config holds DeBERTa-specific configuration.
type Config struct {
	*models.BaseConfig

	// Disentangled attention config.
	RelativeAttention    bool     `json:"relative_attention"`
	PosAttType           []string `json:"pos_att_type"`           // ["c2p", "p2c"]
	MaxRelativePositions int      `json:"max_relative_positions"` // Default 256

	// Relative embedding normalization.
	NormRelEbd []string `json:"norm_rel_ebd"` // ["layer_norm"]

	// Key/Query sharing for positions.
	ShareAttKey bool `json:"share_att_key"`

	// Position embedding type.
	PositionBiasedInput bool `json:"position_biased_input"`
	PositionBuckets     int  `json:"position_buckets"` // For bucketed relative positions
}

// UsesC2P returns true if content-to-position attention is enabled.
func (c *Config) UsesC2P() bool {
	return slices.Contains(c.PosAttType, "c2p")
}

// UsesP2C returns true if position-to-content attention is enabled.
func (c *Config) UsesP2C() bool {
	return slices.Contains(c.PosAttType, "p2c")
}

// NormalizeRelativeEmbeddings returns true if relative embeddings should be normalized.
func (c *Config) NormalizeRelativeEmbeddings() bool {
	return slices.Contains(c.NormRelEbd, "layer_norm")
}

// NumAttentionComponents returns the number of disentangled attention components (1-3).
func (c *Config) NumAttentionComponents() int {
	count := 1 // c2c is always included
	if c.UsesC2P() {
		count++
	}
	if c.UsesP2C() {
		count++
	}
	return count
}

// Builder implements the DeBERTa architecture.
type Builder struct {
	config *Config
}

// Name returns the architecture name.
func (b *Builder) Name() string {
	return "DeBERTa"
}

// ParseConfig extracts DeBERTa-specific configuration from BaseConfig.Raw.
func (b *Builder) ParseConfig(base *models.BaseConfig) error {
	b.config = &Config{BaseConfig: base}

	// Parse DeBERTa-specific fields from Raw.
	if v, ok := base.GetBool("relative_attention"); ok {
		b.config.RelativeAttention = v
	}
	if v, ok := base.GetStringSlice("pos_att_type"); ok {
		b.config.PosAttType = v
	} else {
		// Default for DeBERTa v2/v3
		b.config.PosAttType = []string{"c2p", "p2c"}
	}
	if v, ok := base.GetInt("max_relative_positions"); ok {
		b.config.MaxRelativePositions = v
	} else {
		b.config.MaxRelativePositions = 256
	}
	if v, ok := base.GetStringSlice("norm_rel_ebd"); ok {
		b.config.NormRelEbd = v
	}
	if v, ok := base.GetBool("share_att_key"); ok {
		b.config.ShareAttKey = v
	}
	if v, ok := base.GetBool("position_biased_input"); ok {
		b.config.PositionBiasedInput = v
	}
	if v, ok := base.GetInt("position_buckets"); ok {
		b.config.PositionBuckets = v
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
	prefix := "deberta" // DeBERTa v2/v3 use "deberta" prefix

	// Word embeddings.
	mapping[prefix+".embeddings.word_embeddings.weight"] = "embeddings/embeddings"
	mapping[prefix+".embeddings.LayerNorm.weight"] = "embeddings/layer_norm/gain"
	mapping[prefix+".embeddings.LayerNorm.bias"] = "embeddings/layer_norm/offset"

	// Relative position embeddings.
	mapping[prefix+".encoder.rel_embeddings.weight"] = "encoder/rel_embeddings/embeddings"

	// Encoder LayerNorm for relative embeddings (if norm_rel_ebd is set).
	if cfg.NormalizeRelativeEmbeddings() {
		mapping[prefix+".encoder.LayerNorm.weight"] = "encoder/layer_norm/gain"
		mapping[prefix+".encoder.LayerNorm.bias"] = "encoder/layer_norm/offset"
	}

	// Encoder layers.
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("%s.encoder.layer.%d", prefix, i)
		layerScope := fmt.Sprintf("encoder/layer/%d", i)

		// Self-attention (DeBERTa uses *_proj naming).
		mapping[layerPrefix+".attention.self.query_proj.weight"] = layerScope + "/attention/query/weights"
		mapping[layerPrefix+".attention.self.query_proj.bias"] = layerScope + "/attention/query/biases"
		mapping[layerPrefix+".attention.self.key_proj.weight"] = layerScope + "/attention/key/weights"
		mapping[layerPrefix+".attention.self.key_proj.bias"] = layerScope + "/attention/key/biases"
		mapping[layerPrefix+".attention.self.value_proj.weight"] = layerScope + "/attention/value/weights"
		mapping[layerPrefix+".attention.self.value_proj.bias"] = layerScope + "/attention/value/biases"

		// Attention output.
		mapping[layerPrefix+".attention.output.dense.weight"] = layerScope + "/attention/output/dense/weights"
		mapping[layerPrefix+".attention.output.dense.bias"] = layerScope + "/attention/output/dense/biases"
		mapping[layerPrefix+".attention.output.LayerNorm.weight"] = layerScope + "/attention/output/layer_norm/gain"
		mapping[layerPrefix+".attention.output.LayerNorm.bias"] = layerScope + "/attention/output/layer_norm/offset"

		// Feed-forward.
		mapping[layerPrefix+".intermediate.dense.weight"] = layerScope + "/ff/intermediate/weights"
		mapping[layerPrefix+".intermediate.dense.bias"] = layerScope + "/ff/intermediate/biases"
		mapping[layerPrefix+".output.dense.weight"] = layerScope + "/ff/output/weights"
		mapping[layerPrefix+".output.dense.bias"] = layerScope + "/ff/output/biases"
		mapping[layerPrefix+".output.LayerNorm.weight"] = layerScope + "/ff/layer_norm/gain"
		mapping[layerPrefix+".output.LayerNorm.bias"] = layerScope + "/ff/layer_norm/offset"
	}

	return mapping
}

// BuildEmbeddings builds the embedding layer.
func (b *Builder) BuildEmbeddings(ctx *context.Context, inputIDs *Node) *Node {
	embCtx := ctx.In("embeddings")

	// Word embeddings only (DeBERTa doesn't use absolute position embeddings).
	embeddings := common.Embedding(embCtx, inputIDs, b.config.VocabSize, b.config.HiddenSize)

	// Layer normalization.
	embeddings = common.LayerNorm(embCtx.In("layer_norm"), embeddings, b.config.LayerNormEps)

	return embeddings
}

// BuildRelativeEmbeddings builds and optionally normalizes relative position embeddings.
// Note: This returns a function that takes a graph and returns the embeddings,
// since we need a graph to access variables but don't have one at this point.
func (b *Builder) BuildRelativeEmbeddings(ctx *context.Context, hidden *Node, seqLen int) *Node {
	g := hidden.Graph()
	encCtx := ctx.In("encoder")

	// Get relative embeddings: [512, hidden_size] (256 positions each direction).
	relEmbVar := encCtx.In("rel_embeddings").GetVariableByScopeAndName(
		encCtx.In("rel_embeddings").Scope(), "embeddings")
	relEmb := relEmbVar.ValueGraph(g)

	// Apply LayerNorm if configured.
	if b.config.NormalizeRelativeEmbeddings() {
		relEmb = common.LayerNorm(encCtx.In("layer_norm"), relEmb, b.config.LayerNormEps)
	}

	// Build relative position embedding matrix: [seq_len, seq_len, hidden]
	return common.BuildRelativePositionEmbeddings(g, relEmb, seqLen)
}

// BuildDisentangledAttention builds a single disentangled attention layer.
func (b *Builder) BuildDisentangledAttention(ctx *context.Context, hidden, attentionMask, relPosEmb *Node) *Node {
	g := hidden.Graph()
	cfg := b.config
	attnCtx := ctx.In("attention")

	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	headDim := cfg.HeadDim()

	// Q, K, V projections for content.
	query := common.DenseWithBias(attnCtx.In("query"), hidden)
	key := common.DenseWithBias(attnCtx.In("key"), hidden)
	value := common.DenseWithBias(attnCtx.In("value"), hidden)

	// Reshape for multi-head attention: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
	query = Reshape(query, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	key = Reshape(key, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	value = Reshape(value, batchSize, seqLen, cfg.NumAttentionHeads, headDim)

	query = Transpose(query, 1, 2)
	key = Transpose(key, 1, 2)
	value = Transpose(value, 1, 2)

	// Content-to-content attention (c2c).
	c2cScores := Einsum("bhqd,bhkd->bhqk", query, key)
	scores := c2cScores

	// Disentangled attention components based on config.
	if cfg.UsesC2P() || cfg.UsesP2C() {
		// Get projection weights for relative embeddings.
		queryWeights := attnCtx.In("query").GetVariableByScopeAndName(attnCtx.In("query").Scope(), "weights").ValueGraph(g)
		queryBiases := attnCtx.In("query").GetVariableByScopeAndName(attnCtx.In("query").Scope(), "biases").ValueGraph(g)
		keyWeights := attnCtx.In("key").GetVariableByScopeAndName(attnCtx.In("key").Scope(), "weights").ValueGraph(g)
		keyBiases := attnCtx.In("key").GetVariableByScopeAndName(attnCtx.In("key").Scope(), "biases").ValueGraph(g)

		// Project relative position embeddings.
		// relPosEmb: [seq_q, seq_k, hidden]
		relPosKey := Einsum("qkh,oh->qko", relPosEmb, keyWeights)
		relPosKey = Add(relPosKey, Reshape(keyBiases, 1, 1, keyBiases.Shape().Dimensions[0]))

		relPosQuery := Einsum("qkh,oh->qko", relPosEmb, queryWeights)
		relPosQuery = Add(relPosQuery, Reshape(queryBiases, 1, 1, queryBiases.Shape().Dimensions[0]))

		// Reshape for heads: [seq, seq, hidden] -> [seq, seq, heads, head_dim]
		relPosKey = Reshape(relPosKey, seqLen, seqLen, cfg.NumAttentionHeads, headDim)
		relPosQuery = Reshape(relPosQuery, seqLen, seqLen, cfg.NumAttentionHeads, headDim)

		// Content-to-position (c2p): query_content @ key_position^T
		if cfg.UsesC2P() {
			c2pScores := Einsum("bhqd,qkhd->bhqk", query, relPosKey)
			scores = Add(scores, c2pScores)
		}

		// Position-to-content (p2c): query_position @ key_content^T
		if cfg.UsesP2C() {
			p2cScores := Einsum("qkhd,bhkd->bhqk", relPosQuery, key)
			scores = Add(scores, p2cScores)
		}
	}

	// Scale by 1/sqrt(num_components * head_dim).
	numComponents := float64(cfg.NumAttentionComponents())
	scaleFactor := 1.0 / (float64(headDim) * numComponents)
	scale := ConstAs(scores, scaleFactor)
	scores = Mul(scores, Sqrt(scale))

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

	return attnOutput
}

// BuildEncoderLayer builds a single DeBERTa encoder layer.
func (b *Builder) BuildEncoderLayer(ctx *context.Context, hidden, attentionMask, relPosEmb *Node) *Node {
	cfg := b.config

	// Disentangled self-attention.
	residual := hidden
	attnOutput := b.BuildDisentangledAttention(ctx, hidden, attentionMask, relPosEmb)

	// Output projection and residual.
	attnCtx := ctx.In("attention")
	attnOutput = common.DenseWithBias(attnCtx.In("output").In("dense"), attnOutput)
	hidden = Add(residual, attnOutput)
	hidden = common.LayerNorm(attnCtx.In("output").In("layer_norm"), hidden, cfg.LayerNormEps)

	// Feed-forward network.
	ffCtx := ctx.In("ff")
	residual = hidden
	hidden = common.DenseWithBias(ffCtx.In("intermediate"), hidden)
	hidden = activations.GeluApproximate(hidden)
	hidden = common.DenseWithBias(ffCtx.In("output"), hidden)
	hidden = Add(residual, hidden)
	hidden = common.LayerNorm(ffCtx.In("layer_norm"), hidden, cfg.LayerNormEps)

	return hidden
}

// BuildEncoder builds the full encoder stack.
func (b *Builder) BuildEncoder(ctx *context.Context, hidden, attentionMask *Node) *Node {
	seqLen := hidden.Shape().Dimensions[1]

	// Build relative position embeddings once for all layers.
	relPosEmb := b.BuildRelativeEmbeddings(ctx, hidden, seqLen)

	// Process through encoder layers.
	encCtx := ctx.In("encoder")
	for i := 0; i < b.config.NumHiddenLayers; i++ {
		hidden = b.BuildEncoderLayer(encCtx.In("layer").In(itoa(i)), hidden, attentionMask, relPosEmb)
	}

	return hidden
}

// Forward runs the forward pass.
func (b *Builder) Forward(ctx *context.Context, inputIDs, attentionMask *Node) *Node {
	// Embeddings.
	hidden := b.BuildEmbeddings(ctx, inputIDs)

	// Encoder.
	hidden = b.BuildEncoder(ctx, hidden, attentionMask)

	return hidden
}

// CreateExecGraphFn returns a function suitable for context.NewExec.
func (b *Builder) CreateExecGraphFn() func(*context.Context, *Node, *Node) *Node {
	return func(ctx *context.Context, inputIDs, attentionMask *Node) *Node {
		return b.Forward(ctx, inputIDs, attentionMask)
	}
}

// GetVariableShape returns the expected shape for a variable.
func (b *Builder) GetVariableShape(name string) shapes.Shape {
	cfg := b.config

	switch {
	case strings.Contains(name, "word_embeddings"):
		return shapes.Make(dtypes.Float32, cfg.VocabSize, cfg.HiddenSize)
	case strings.Contains(name, "rel_embeddings"):
		return shapes.Make(dtypes.Float32, 512, cfg.HiddenSize) // 256 * 2 positions
	case strings.Contains(name, "query") || strings.Contains(name, "key") || strings.Contains(name, "value"):
		if strings.HasSuffix(name, "weights") {
			return shapes.Make(dtypes.Float32, cfg.HiddenSize, cfg.HiddenSize)
		}
		return shapes.Make(dtypes.Float32, cfg.HiddenSize)
	case strings.Contains(name, "intermediate"):
		if strings.HasSuffix(name, "weights") {
			return shapes.Make(dtypes.Float32, cfg.IntermediateSize, cfg.HiddenSize)
		}
		return shapes.Make(dtypes.Float32, cfg.IntermediateSize)
	default:
		return shapes.Shape{}
	}
}

func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}
