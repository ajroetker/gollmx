// Package bert provides a BERT architecture implementation for GoMLX.
//
// BERT (Bidirectional Encoder Representations from Transformers) uses
// absolute position embeddings and standard multi-head self-attention.
//
// Supported model types: bert, roberta, distilbert
package bert

import (
	"fmt"
	"strings"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"

	"github.com/gomlx/gollmx"
	"github.com/gomlx/gollmx/architectures/common"
)

func init() {
	models.RegisterArchitecture("bert", func() models.ArchitectureBuilder { return &Builder{} })
	models.RegisterArchitecture("roberta", func() models.ArchitectureBuilder { return &Builder{} })
	models.RegisterArchitecture("distilbert", func() models.ArchitectureBuilder { return &Builder{isDistilBert: true} })
}

// Builder implements the BERT architecture.
type Builder struct {
	config       *models.BaseConfig
	isDistilBert bool
}

// Name returns the architecture name.
func (b *Builder) Name() string {
	if b.isDistilBert {
		return "DistilBERT"
	}
	return "BERT"
}

// ParseConfig extracts BERT-specific configuration.
func (b *Builder) ParseConfig(base *models.BaseConfig) error {
	b.config = base
	return nil
}

// Config returns the base configuration.
func (b *Builder) Config() *models.BaseConfig {
	return b.config
}

// LoadWeights loads weights into the GoMLX context.
func (b *Builder) LoadWeights(ctx *context.Context, weights models.WeightSource) error {
	_, err := models.LoadWeightsFromMapping(weights, b.WeightMapping(), ctx)
	return err
}

// WeightMapping returns the mapping from safetensors keys to context scope paths.
func (b *Builder) WeightMapping() map[string]string {
	mapping := make(map[string]string)
	prefix := "bert"

	// Word embeddings.
	mapping[prefix+".embeddings.word_embeddings.weight"] = "embeddings/embeddings"
	mapping[prefix+".embeddings.position_embeddings.weight"] = "embeddings/position_embeddings"
	mapping[prefix+".embeddings.token_type_embeddings.weight"] = "embeddings/token_type_embeddings"
	mapping[prefix+".embeddings.LayerNorm.weight"] = "embeddings/layer_norm/gain"
	mapping[prefix+".embeddings.LayerNorm.bias"] = "embeddings/layer_norm/offset"

	// Encoder layers.
	for i := 0; i < b.config.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("%s.encoder.layer.%d", prefix, i)
		layerScope := fmt.Sprintf("encoder/layer/%d", i)

		// Self-attention.
		mapping[layerPrefix+".attention.self.query.weight"] = layerScope + "/attention/query/weights"
		mapping[layerPrefix+".attention.self.query.bias"] = layerScope + "/attention/query/biases"
		mapping[layerPrefix+".attention.self.key.weight"] = layerScope + "/attention/key/weights"
		mapping[layerPrefix+".attention.self.key.bias"] = layerScope + "/attention/key/biases"
		mapping[layerPrefix+".attention.self.value.weight"] = layerScope + "/attention/value/weights"
		mapping[layerPrefix+".attention.self.value.bias"] = layerScope + "/attention/value/biases"

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

	// Pooler (optional).
	mapping[prefix+".pooler.dense.weight"] = "pooler/weights"
	mapping[prefix+".pooler.dense.bias"] = "pooler/biases"

	return mapping
}

// BuildEmbeddings builds the embedding layer.
func (b *Builder) BuildEmbeddings(ctx *context.Context, inputIDs, tokenTypeIDs, positionIDs *Node) *Node {
	g := inputIDs.Graph()
	embCtx := ctx.In("embeddings")

	batchSize := inputIDs.Shape().Dimensions[0]
	seqLen := inputIDs.Shape().Dimensions[1]

	// Word embeddings.
	embeddings := common.Embedding(embCtx, inputIDs, b.config.VocabSize, b.config.HiddenSize)

	// Position embeddings.
	if positionIDs == nil {
		positionIDs = common.GetPositionIDs(g, batchSize, seqLen)
	}
	posEmb := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "position_embeddings").ValueGraph(g)
	// Gather position embeddings.
	posEmbGathered := Gather(posEmb, Reshape(positionIDs, batchSize, seqLen, 1))
	embeddings = Add(embeddings, posEmbGathered)

	// Token type embeddings (optional).
	if tokenTypeIDs != nil {
		typeEmb := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "token_type_embeddings").ValueGraph(g)
		typeEmbGathered := Gather(typeEmb, Reshape(tokenTypeIDs, batchSize, seqLen, 1))
		embeddings = Add(embeddings, typeEmbGathered)
	}

	// Layer normalization.
	embeddings = common.LayerNorm(embCtx.In("layer_norm"), embeddings, b.config.LayerNormEps)

	return embeddings
}

// BuildEncoderLayer builds a single transformer encoder layer.
func (b *Builder) BuildEncoderLayer(ctx *context.Context, hidden, attentionMask *Node) *Node {
	_ = hidden.Graph() // Ensure we have a graph reference
	cfg := b.config

	// Self-attention.
	attnCtx := ctx.In("attention")
	residual := hidden

	// Q, K, V projections.
	query := common.DenseWithBias(attnCtx.In("query"), hidden)
	key := common.DenseWithBias(attnCtx.In("key"), hidden)
	value := common.DenseWithBias(attnCtx.In("value"), hidden)

	// Reshape for multi-head attention: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	headDim := cfg.HeadDim()

	query = Reshape(query, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	key = Reshape(key, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	value = Reshape(value, batchSize, seqLen, cfg.NumAttentionHeads, headDim)

	query = Transpose(query, 1, 2)
	key = Transpose(key, 1, 2)
	value = Transpose(value, 1, 2)

	// Attention scores: Q @ K^T / sqrt(d_k)
	scores := Einsum("bhqd,bhkd->bhqk", query, key)
	scale := ConstAs(scores, 1.0/float64(headDim))
	scores = Mul(scores, Sqrt(scale))

	// Apply attention mask if provided.
	if attentionMask != nil {
		// Expand mask: [batch, seq] -> [batch, 1, 1, seq]
		mask := common.ExpandAttentionMask(attentionMask, scores.DType())
		scores = Add(scores, mask)
	}

	// Softmax and attention output.
	attnWeights := Softmax(scores, -1)
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, value)

	// Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, seqLen, cfg.HiddenSize)

	// Output projection and residual.
	attnOutput = common.DenseWithBias(attnCtx.In("output").In("dense"), attnOutput)
	hidden = Add(residual, attnOutput)
	hidden = common.LayerNorm(attnCtx.In("output").In("layer_norm"), hidden, cfg.LayerNormEps)

	// Feed-forward network.
	ffCtx := ctx.In("ff")
	residual = hidden
	hidden = common.DenseWithBiasAndActivation(ffCtx.In("intermediate"), hidden, activations.TypeGeluApprox)
	hidden = common.DenseWithBias(ffCtx.In("output"), hidden)
	hidden = Add(residual, hidden)
	hidden = common.LayerNorm(ffCtx.In("layer_norm"), hidden, cfg.LayerNormEps)

	return hidden
}

// BuildEncoder builds the full encoder stack.
func (b *Builder) BuildEncoder(ctx *context.Context, hidden, attentionMask *Node) *Node {
	encCtx := ctx.In("encoder")

	for i := 0; i < b.config.NumHiddenLayers; i++ {
		hidden = b.BuildEncoderLayer(encCtx.In("layer").In(itoa(i)), hidden, attentionMask)
	}

	return hidden
}

// BuildPooler applies the pooler layer (CLS token projection).
func (b *Builder) BuildPooler(ctx *context.Context, hidden *Node) *Node {
	_ = hidden.Graph() // Keep graph reference available
	poolerCtx := ctx.In("pooler")

	// Check if pooler weights exist.
	if poolerCtx.GetVariableByScopeAndName(poolerCtx.Scope(), "weights") == nil {
		return nil
	}

	// Get CLS token output: [batch, hidden]
	batchSize := hidden.Shape().Dimensions[0]
	clsOutput := Slice(hidden, AxisRange(), AxisElem(0), AxisRange())
	clsOutput = Reshape(clsOutput, batchSize, b.config.HiddenSize)

	// Apply pooler dense + tanh.
	pooled := common.DenseWithBias(poolerCtx, clsOutput)
	pooled = Tanh(pooled)

	return pooled
}

// Forward runs the forward pass.
func (b *Builder) Forward(ctx *context.Context, inputIDs, attentionMask, tokenTypeIDs, positionIDs *Node) (*Node, *Node) {
	// Embeddings.
	hidden := b.BuildEmbeddings(ctx, inputIDs, tokenTypeIDs, positionIDs)

	// Encoder.
	hidden = b.BuildEncoder(ctx, hidden, attentionMask)

	// Pooler.
	pooled := b.BuildPooler(ctx, hidden)

	return hidden, pooled
}

// CreateExecGraphFn returns a function suitable for context.NewExec.
// The function signature is: func(ctx, inputIDs, attentionMask) -> lastHiddenState
func (b *Builder) CreateExecGraphFn() func(*context.Context, *Node, *Node) *Node {
	return func(ctx *context.Context, inputIDs, attentionMask *Node) *Node {
		hidden, _ := b.Forward(ctx, inputIDs, attentionMask, nil, nil)
		return hidden
	}
}

// GetVariableShape returns the expected shape for a variable.
func (b *Builder) GetVariableShape(name string) shapes.Shape {
	cfg := b.config

	switch {
	case strings.Contains(name, "embeddings"):
		return shapes.Make(dtypes.Float32, cfg.VocabSize, cfg.HiddenSize)
	case strings.Contains(name, "position_embeddings"):
		return shapes.Make(dtypes.Float32, cfg.MaxPositionEmbeddings, cfg.HiddenSize)
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
