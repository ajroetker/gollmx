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

// RoPEConfig holds configuration for Rotary Position Embedding.
type RoPEConfig struct {
	Theta         float64   // Base frequency (e.g., 10000.0).
	HeadDim       int       // Full dimension of each attention head.
	RotaryDim     int       // Number of dimensions to apply RoPE to (0 = all). For partial_rotary_factor=0.75 with headDim=128, set to 96.
	ScalingFactor float64   // Uniform scaling factor (divides all frequencies). 0 or 1 = no scaling.
	LongFactors   []float32 // Per-dimension LongRoPE factors for long sequences (len = rotaryDim/2). Nil = unused.
	ShortFactors  []float32 // Per-dimension LongRoPE factors for short sequences (len = rotaryDim/2). Nil = unused.
	OrigMaxSeqLen int       // Original max sequence length for LongRoPE threshold. Sequences > this use LongFactors.
}

// RoPE applies Rotary Position Embedding to query and key tensors.
// query/key shape: [batch, heads, seq, head_dim]
// positionIDs shape: [batch, seq] or nil (will use sequential positions)
// scalingFactor: for linear RoPE scaling, frequencies are divided by this factor.
// Use 1.0 or 0.0 for no scaling.
func RoPE(query, key *graph.Node, positionIDs *graph.Node, theta float64, seqLen, headDim int, scalingFactor ...float64) (*graph.Node, *graph.Node) {
	cfg := RoPEConfig{
		Theta:   theta,
		HeadDim: headDim,
	}
	if len(scalingFactor) > 0 {
		cfg.ScalingFactor = scalingFactor[0]
	}
	return RoPEWithConfig(query, key, positionIDs, seqLen, cfg)
}

// RoPEWithConfig applies Rotary Position Embedding with full configuration control.
// Supports partial rotary (RotaryDim < HeadDim) and LongRoPE per-dimension scaling.
// query/key shape: [batch, heads, seq, head_dim]
// positionIDs shape: [batch, seq] or nil (will use sequential positions)
func RoPEWithConfig(query, key *graph.Node, positionIDs *graph.Node, seqLen int, cfg RoPEConfig) (*graph.Node, *graph.Node) {
	g := query.Graph()

	rotaryDim := cfg.HeadDim
	if cfg.RotaryDim > 0 {
		rotaryDim = cfg.RotaryDim
	}
	halfRotary := rotaryDim / 2

	// Compute base inverse frequencies: freq_i = theta^(-2i/d) for i in [0, rotaryDim/2).
	invFreq := make([]float32, halfRotary)
	for i := range halfRotary {
		invFreq[i] = float32(1.0 / math.Pow(cfg.Theta, float64(2*i)/float64(rotaryDim)))
	}

	// Apply scaling: either per-dimension (LongRoPE) or uniform.
	if cfg.LongFactors != nil && seqLen > cfg.OrigMaxSeqLen {
		for i := range halfRotary {
			invFreq[i] /= cfg.LongFactors[i]
		}
	} else if cfg.ShortFactors != nil && seqLen <= cfg.OrigMaxSeqLen {
		for i := range halfRotary {
			invFreq[i] /= cfg.ShortFactors[i]
		}
	} else if cfg.ScalingFactor > 1.0 {
		for i := range halfRotary {
			invFreq[i] /= float32(cfg.ScalingFactor)
		}
	}

	invFreqNode := graph.Const(g, invFreq)
	invFreqNode = graph.Reshape(invFreqNode, 1, 1, halfRotary) // [1, 1, halfRotary]

	// Create position indices.
	var positions *graph.Node
	if positionIDs != nil {
		positions = graph.ConvertDType(positionIDs, dtypes.Float32)
		positions = graph.Reshape(positions, positions.Shape().Dimensions[0], seqLen, 1) // [batch, seq, 1]
	} else {
		posArray := make([]float32, seqLen)
		for i := range seqLen {
			posArray[i] = float32(i)
		}
		positions = graph.Const(g, posArray)
		positions = graph.Reshape(positions, 1, seqLen, 1) // [1, seq, 1]
	}

	// freqs = positions * inv_freq: [batch, seq, halfRotary]
	freqs := graph.Mul(positions, invFreqNode)

	sinFreqs := graph.Sin(freqs)
	cosFreqs := graph.Cos(freqs)

	// Expand for heads: [batch, 1, seq, halfRotary]
	sinFreqs = graph.InsertAxes(sinFreqs, 1)
	cosFreqs = graph.InsertAxes(cosFreqs, 1)

	// Apply rotary embedding to the rotary dimensions.
	if rotaryDim < cfg.HeadDim {
		// Partial rotary: apply RoPE to first rotaryDim dims, pass through the rest.
		query = applyPartialRotaryEmb(query, sinFreqs, cosFreqs, rotaryDim, cfg.HeadDim)
		key = applyPartialRotaryEmb(key, sinFreqs, cosFreqs, rotaryDim, cfg.HeadDim)
	} else {
		query = applyRotaryEmb(query, sinFreqs, cosFreqs, rotaryDim)
		key = applyRotaryEmb(key, sinFreqs, cosFreqs, rotaryDim)
	}

	return query, key
}

// applyRotaryEmb applies rotary embedding to a tensor.
// x shape: [batch, heads, seq, head_dim]
// sin/cos shape: [batch, 1, seq, rotaryDim/2]
func applyRotaryEmb(x, sin, cos *graph.Node, rotaryDim int) *graph.Node {
	half := rotaryDim / 2
	x1 := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(0, half))
	x2 := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(half, rotaryDim))

	rotatedX1 := graph.Sub(graph.Mul(x1, cos), graph.Mul(x2, sin))
	rotatedX2 := graph.Add(graph.Mul(x2, cos), graph.Mul(x1, sin))

	return graph.Concatenate([]*graph.Node{rotatedX1, rotatedX2}, -1)
}

// applyPartialRotaryEmb applies rotary embedding to only the first rotaryDim
// dimensions and passes through the remaining dimensions unchanged.
func applyPartialRotaryEmb(x, sin, cos *graph.Node, rotaryDim, headDim int) *graph.Node {
	half := rotaryDim / 2
	xRot1 := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(0, half))
	xRot2 := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(half, rotaryDim))
	xPass := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(rotaryDim, headDim))

	rotatedX1 := graph.Sub(graph.Mul(xRot1, cos), graph.Mul(xRot2, sin))
	rotatedX2 := graph.Add(graph.Mul(xRot2, cos), graph.Mul(xRot1, sin))

	return graph.Concatenate([]*graph.Node{rotatedX1, rotatedX2, xPass}, -1)
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
