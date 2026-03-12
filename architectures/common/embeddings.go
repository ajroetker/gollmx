package common

import (
	"fmt"
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

// Embedding retrieves embeddings from the context.
// Uses the GoMLX layers.Embedding which expects "embeddings" variable in scope.
func Embedding(ctx *context.Context, inputIDs *graph.Node, vocabSize, hiddenSize int) *graph.Node {
	embeddings := layers.Embedding(ctx, inputIDs, dtypes.Float32, vocabSize, hiddenSize)

	// Ensure 3D output: [batch, seq, hidden].
	// layers.Embedding may return 2D when seq_len=1.
	if embeddings.Shape().Rank() == 2 {
		embeddings = graph.InsertAxes(embeddings, 1)
	}

	return embeddings
}

// EmbeddingFromVar performs an embedding lookup using an already-loaded variable.
// The result is converted to Float32 for downstream computation.
func EmbeddingFromVar(ctx *context.Context, inputIDs *graph.Node, v *context.Variable) *graph.Node {
	g := inputIDs.Graph()
	table := v.ValueGraph(g)

	// Prepare indices with trailing dim of 1 for Gather.
	indices := inputIDs
	inputShape := inputIDs.Shape()
	if inputShape.IsScalar() || inputShape.Dimensions[inputShape.Rank()-1] != 1 {
		indices = graph.InsertAxes(indices, -1)
	}
	embeddings := graph.Gather(table, indices)

	// Convert to Float32 if needed (e.g. Float16 from GGUF).
	if embeddings.DType() != dtypes.Float32 {
		embeddings = graph.ConvertDType(embeddings, dtypes.Float32)
	}

	// Ensure 3D output: [batch, seq, hidden].
	if embeddings.Shape().Rank() == 2 {
		embeddings = graph.InsertAxes(embeddings, 1)
	}

	return embeddings
}

// QuantizedEmbedding performs a quantized embedding lookup using GGML-format weights.
// Expects "embeddings" variable in scope as [vocabSize, bytesPerRow] Uint8.
// Dequantizes only the selected rows on-the-fly (like llama.cpp's ggml_get_rows).
// Returns [batch, seqLen, K] Float32 where K is the logical embedding dimension.
func QuantizedEmbedding(ctx *context.Context, inputIDs *graph.Node, ggmlType backends.GGMLQuantType) *graph.Node {
	g := inputIDs.Graph()

	embVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "embeddings")
	if embVar == nil {
		panic(fmt.Sprintf("QuantizedEmbedding: missing variable 'embeddings' in scope %q", ctx.Scope()))
	}
	table := embVar.ValueGraph(g)

	// Prepare indices: add trailing dim of 1 if needed (Gather convention).
	indices := inputIDs
	inputShape := inputIDs.Shape()
	if inputShape.IsScalar() || inputShape.Dimensions[inputShape.Rank()-1] != 1 {
		indices = graph.InsertAxes(indices, -1)
	}

	quant := &graph.Quantization{
		Scheme:   backends.QuantGGML,
		GGMLType: ggmlType,
	}
	embeddings := nn.QuantizedGather(table, indices, quant)

	// Ensure 3D output: [batch, seq, hidden].
	if embeddings.Shape().Rank() == 2 {
		embeddings = graph.InsertAxes(embeddings, 1)
	}

	return embeddings
}

// AbsolutePositionEmbedding adds absolute position embeddings.
// Expects "position_embeddings" variable in scope with shape [max_positions, hidden_size].
func AbsolutePositionEmbedding(ctx *context.Context, x *graph.Node, seqLen int) *graph.Node {
	g := x.Graph()

	posEmbVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "position_embeddings")
	if posEmbVar == nil {
		return x // No position embeddings, just return input.
	}
	posEmb := posEmbVar.ValueGraph(g)

	// Slice to sequence length: [max_pos, hidden] -> [seq_len, hidden]
	posEmb = graph.Slice(posEmb, graph.AxisRange(0, seqLen), graph.AxisRange())

	// Add position embeddings (broadcasts over batch).
	return graph.Add(x, posEmb)
}

// BuildRelativePositionEmbeddings creates a [seq_len, seq_len, hidden] tensor of relative position embeddings.
// For each (query_pos=i, key_pos=j) pair, looks up the embedding for relative position (i - j).
// This follows DeBERTa convention where relative_pos = query_pos - key_pos.
func BuildRelativePositionEmbeddings(g *graph.Graph, relEmbeddings *graph.Node, seqLen int) *graph.Node {
	const maxRelPos = 256

	// Build the indices matrix [seq_len, seq_len, 1] for Gather.
	indices := make([]int32, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			relPos := i - j // query_pos - key_pos (DeBERTa convention)
			// Clamp to valid range and add offset.
			if relPos < -maxRelPos {
				relPos = -maxRelPos
			} else if relPos >= maxRelPos {
				relPos = maxRelPos - 1
			}
			idx := relPos + maxRelPos // Shift to [0, 511]
			indices[i*seqLen+j] = int32(idx)
		}
	}

	// Create indices tensor with shape [seq_len, seq_len, 1].
	indicesNode := graph.Const(g, indices)
	indicesNode = graph.Reshape(indicesNode, seqLen, seqLen, 1)

	// Gather position embeddings: [512, hidden] indexed by [seq, seq, 1] -> [seq, seq, hidden]
	return graph.Gather(relEmbeddings, indicesNode)
}

// RoPE applies Rotary Position Embedding to query and key tensors.
// query/key shape: [batch, heads, seq, head_dim]
// positionIDs shape: [batch, seq] or nil (will use sequential positions)
// scalingFactor: for linear RoPE scaling, frequencies are divided by this factor.
// Use 1.0 or 0.0 for no scaling.
func RoPE(query, key *graph.Node, positionIDs *graph.Node, theta float64, seqLen, headDim int, scalingFactor ...float64) (*graph.Node, *graph.Node) {
	g := query.Graph()

	// Apply linear RoPE scaling: divide frequencies by the factor.
	// This stretches position encodings to support longer contexts.
	factor := 1.0
	if len(scalingFactor) > 0 && scalingFactor[0] > 1.0 {
		factor = scalingFactor[0]
	}

	// Compute rotary frequencies.
	// freq_i = theta^(-2i/d) / factor for i in [0, d/2)
	invFreq := make([]float32, headDim/2)
	for i := 0; i < headDim/2; i++ {
		invFreq[i] = float32(1.0 / (math.Pow(theta, float64(2*i)/float64(headDim)) * factor))
	}
	invFreqNode := graph.Const(g, invFreq)
	invFreqNode = graph.Reshape(invFreqNode, 1, 1, headDim/2) // [1, 1, head_dim/2]

	// Create position indices.
	var positions *graph.Node
	if positionIDs != nil {
		positions = graph.ConvertDType(positionIDs, dtypes.Float32)
		positions = graph.Reshape(positions, positions.Shape().Dimensions[0], seqLen, 1) // [batch, seq, 1]
	} else {
		posArray := make([]float32, seqLen)
		for i := 0; i < seqLen; i++ {
			posArray[i] = float32(i)
		}
		positions = graph.Const(g, posArray)
		positions = graph.Reshape(positions, 1, seqLen, 1) // [1, seq, 1]
	}

	// freqs = positions * inv_freq: [batch, seq, head_dim/2]
	freqs := graph.Mul(positions, invFreqNode)

	// Compute sin and cos.
	sinFreqs := graph.Sin(freqs)
	cosFreqs := graph.Cos(freqs)

	// Expand for heads: [batch, 1, seq, head_dim/2]
	sinFreqs = graph.InsertAxes(sinFreqs, 1)
	cosFreqs = graph.InsertAxes(cosFreqs, 1)

	// Apply rotary embedding.
	query = applyRotaryEmb(query, sinFreqs, cosFreqs, headDim)
	key = applyRotaryEmb(key, sinFreqs, cosFreqs, headDim)

	return query, key
}

// applyRotaryEmb applies rotary embedding to a tensor.
// x shape: [batch, heads, seq, head_dim]
// sin/cos shape: [batch, 1, seq, head_dim/2]
func applyRotaryEmb(x, sin, cos *graph.Node, headDim int) *graph.Node {
	// Split into two halves.
	x1 := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(0, headDim/2))
	x2 := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(headDim/2, headDim))

	// Rotate: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
	rotatedX1 := graph.Sub(graph.Mul(x1, cos), graph.Mul(x2, sin))
	rotatedX2 := graph.Add(graph.Mul(x2, cos), graph.Mul(x1, sin))

	// Concatenate back.
	return graph.Concatenate([]*graph.Node{rotatedX1, rotatedX2}, -1)
}

// CreateCausalMask creates a causal attention mask.
// Returns a mask of shape [1, 1, seq_len, seq_len] where mask[i][j] = 0 if i >= j else -inf.
func CreateCausalMask(g *graph.Graph, seqLen int, dtype dtypes.DType) *graph.Node {
	mask := make([]float32, seqLen*seqLen)
	negInf := float32(-1e9)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i {
				mask[i*seqLen+j] = negInf
			}
		}
	}

	maskNode := graph.Const(g, mask)
	maskNode = graph.Reshape(maskNode, 1, 1, seqLen, seqLen)
	return graph.ConvertDType(maskNode, dtype)
}

// ExpandAttentionMask expands attention mask for multi-head attention.
// Input mask: [batch, seq_len] (1 for valid, 0 for masked)
// Output: [batch, 1, 1, seq_len] with 0 for valid, large negative for masked.
func ExpandAttentionMask(mask *graph.Node, dtype dtypes.DType) *graph.Node {
	// Expand dimensions: [batch, seq] -> [batch, 1, 1, seq]
	mask = graph.InsertAxes(mask, 1, 1)

	// Convert to float and apply masking.
	mask = graph.ConvertDType(mask, dtype)

	// Convert 0->large_negative, 1->0
	negInf := graph.ConstAs(mask, float64(-1e9))
	one := graph.ConstAs(mask, 1.0)
	return graph.Mul(graph.Sub(one, mask), negInf)
}

// GetPositionIDs creates sequential position IDs.
func GetPositionIDs(g *graph.Graph, batchSize, seqLen int) *graph.Node {
	positions := make([]int32, seqLen)
	for i := 0; i < seqLen; i++ {
		positions[i] = int32(i)
	}
	posNode := graph.Const(g, positions)
	posNode = graph.Reshape(posNode, 1, seqLen)

	// Broadcast to batch size.
	return graph.BroadcastToDims(posNode, batchSize, seqLen)
}

// CreateSinusoidalPositionEmbedding creates sinusoidal position embeddings.
// Returns tensor of shape [maxLen, hiddenSize].
func CreateSinusoidalPositionEmbedding(g *graph.Graph, maxLen, hiddenSize int, dtype dtypes.DType) *graph.Node {
	posEmb := make([]float32, maxLen*hiddenSize)

	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < hiddenSize; i++ {
			if i%2 == 0 {
				// sin(pos / 10000^(2i/d))
				posEmb[pos*hiddenSize+i] = float32(math.Sin(float64(pos) / math.Pow(10000.0, float64(i)/float64(hiddenSize))))
			} else {
				// cos(pos / 10000^(2(i-1)/d))
				posEmb[pos*hiddenSize+i] = float32(math.Cos(float64(pos) / math.Pow(10000.0, float64(i-1)/float64(hiddenSize))))
			}
		}
	}

	embeddings := graph.Const(g, posEmb)
	embeddings = graph.Reshape(embeddings, maxLen, hiddenSize)
	return graph.ConvertDType(embeddings, dtype)
}

// Variable helper to get variable value from context or create it with a shape.
func GetOrCreateVariable(ctx *context.Context, g *graph.Graph, name string, shape shapes.Shape) *graph.Node {
	v := ctx.GetVariableByScopeAndName(ctx.Scope(), name)
	if v != nil {
		return v.ValueGraph(g)
	}
	return ctx.VariableWithShape(name, shape).ValueGraph(g)
}
