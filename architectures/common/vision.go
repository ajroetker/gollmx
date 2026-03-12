package common

import (
	"math"
	"strconv"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

// VisionConfig holds configuration for a CLIP/SigLIP vision encoder.
type VisionConfig struct {
	HiddenSize   int     // e.g., 1024 (CLIP ViT-L) or 1152 (SigLIP)
	NumLayers    int     // e.g., 24 (CLIP ViT-L) or 27 (SigLIP)
	NumHeads     int     // e.g., 16
	MLPDim       int     // e.g., 4096 (CLIP) or 4304 (SigLIP)
	ImageSize    int     // e.g., 336 (CLIP) or 896 (SigLIP)
	PatchSize    int     // e.g., 14
	NumChannels  int     // e.g., 3
	LayerNormEps float64 // e.g., 1e-6
	UseGELU      bool    // true for CLIP (approximate GELU), false uses exact GELU
}

// NumPatches returns the number of image patches from the vision encoder.
func (vc *VisionConfig) NumPatches() int {
	s := vc.ImageSize / vc.PatchSize
	return s * s
}

// PatchesPerSide returns the number of patches along one spatial dimension.
func (vc *VisionConfig) PatchesPerSide() int {
	return vc.ImageSize / vc.PatchSize
}

// HeadDim returns the per-head dimension for vision attention.
func (vc *VisionConfig) HeadDim() int {
	return vc.HiddenSize / vc.NumHeads
}

// BuildCLIPVisionEncoder runs a CLIP-style vision encoder (also works for SigLIP).
// pixelValues: [batch, channels, height, width] float32.
// Returns: [batch, numPatches, hiddenSize] float32.
func BuildCLIPVisionEncoder(ctx *context.Context, pixelValues *Node, vc *VisionConfig) *Node {
	vCtx := ctx.In("vision")
	g := pixelValues.Graph()
	batchSize := pixelValues.Shape().Dimensions[0]

	// Patch embedding: extract patches and linearly project.
	hidden := PatchEmbed(vCtx, pixelValues, batchSize, vc)

	// Add position embeddings.
	posVar := vCtx.GetVariableByScopeAndName(vCtx.Scope(), "position_embeddings")
	if posVar != nil {
		posEmb := posVar.ValueGraph(g)
		if posEmb.DType() != dtypes.Float32 {
			posEmb = ConvertDType(posEmb, dtypes.Float32)
		}
		// GGUF may store as [hiddenSize, numPatches]; transpose to [numPatches, hiddenSize].
		numPatches := vc.NumPatches()
		if posEmb.Shape().Dimensions[0] != numPatches && posEmb.Shape().Dimensions[1] == numPatches {
			posEmb = Transpose(posEmb, 0, 1)
		}
		posEmb = InsertAxes(posEmb, 0) // [1, numPatches, hiddenSize]
		hidden = Add(hidden, posEmb)
	}

	// Encoder layers.
	for i := range vc.NumLayers {
		hidden = VisionEncoderLayer(vCtx.In("layers").In(strconv.Itoa(i)), hidden, vc)
	}

	// Post-layer normalization.
	hidden = LayerNorm(vCtx.In("post_layernorm"), hidden, vc.LayerNormEps)

	return hidden
}

// VisionEncoderLayer runs a single CLIP/SigLIP encoder layer.
// LayerNorm1 → SelfAttention → residual → LayerNorm2 → MLP → residual.
func VisionEncoderLayer(ctx *context.Context, hidden *Node, vc *VisionConfig) *Node {
	normalized := LayerNorm(ctx.In("layer_norm1"), hidden, vc.LayerNormEps)
	attnOutput := VisionAttention(ctx, normalized, vc)
	hidden = Add(hidden, attnOutput)

	normalized = LayerNorm(ctx.In("layer_norm2"), hidden, vc.LayerNormEps)
	mlpOutput := VisionMLP(ctx.In("mlp"), normalized, vc)
	hidden = Add(hidden, mlpOutput)

	return hidden
}

// VisionAttention runs multi-head self-attention for the vision encoder.
// Standard attention with bias, no GQA, no RoPE.
func VisionAttention(ctx *context.Context, hidden *Node, vc *VisionConfig) *Node {
	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	headDim := vc.HeadDim()
	numHeads := vc.NumHeads

	query := DenseWithBias(ctx.In("attn_q"), hidden)
	key := DenseWithBias(ctx.In("attn_k"), hidden)
	value := DenseWithBias(ctx.In("attn_v"), hidden)

	query = Reshape(query, batchSize, seqLen, numHeads, headDim)
	query = Transpose(query, 1, 2)
	key = Reshape(key, batchSize, seqLen, numHeads, headDim)
	key = Transpose(key, 1, 2)
	value = Reshape(value, batchSize, seqLen, numHeads, headDim)
	value = Transpose(value, 1, 2)

	scale := 1.0 / math.Sqrt(float64(headDim))
	attnOutput, _ := attention.Core(nil, query, key, value, scale, nil, 0, attention.LayoutBHSD, false, false)

	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, seqLen, numHeads*headDim)

	return DenseWithBias(ctx.In("attn_output"), attnOutput)
}

// VisionMLP applies the vision encoder MLP: fc1(GELU) → fc2.
func VisionMLP(ctx *context.Context, hidden *Node, vc *VisionConfig) *Node {
	hidden = DenseWithBias(ctx.In("fc1"), hidden)
	if vc.UseGELU {
		hidden = activations.GeluApproximate(hidden)
	} else {
		hidden = activations.Gelu(hidden)
	}
	return DenseWithBias(ctx.In("fc2"), hidden)
}

// PatchEmbed converts pixel values to patch embeddings.
// pixelValues: [batch, channels, height, width]
// Returns: [batch, numPatches, hiddenSize]
func PatchEmbed(ctx *context.Context, pixelValues *Node, batchSize int, vc *VisionConfig) *Node {
	g := pixelValues.Graph()

	patchCtx := ctx.In("patch_embedding")
	weightVar := patchCtx.GetVariableByScopeAndName(patchCtx.Scope(), "weights")
	if weightVar == nil {
		panic("vision: missing patch_embedding weights")
	}
	convWeight := weightVar.ValueGraph(g)
	if convWeight.DType() != dtypes.Float32 {
		convWeight = ConvertDType(convWeight, dtypes.Float32)
	}

	biasVar := patchCtx.GetVariableByScopeAndName(patchCtx.Scope(), "biases")
	var convBias *Node
	if biasVar != nil {
		convBias = biasVar.ValueGraph(g)
		if convBias.DType() != dtypes.Float32 {
			convBias = ConvertDType(convBias, dtypes.Float32)
		}
	}

	patchSize := vc.PatchSize
	gridSize := vc.ImageSize / patchSize
	numPatches := gridSize * gridSize
	channels := vc.NumChannels
	patchDim := patchSize * patchSize * channels

	// pixelValues: [B, C, H, W]. Reshape to [B, C, gridH, patchH, gridW, patchW].
	x := Reshape(pixelValues, batchSize, channels, gridSize, patchSize, gridSize, patchSize)
	// Transpose to [B, gridH, gridW, C, patchH, patchW] — patches in [C, pH, pW] order.
	x = TransposeAllDims(x, 0, 2, 4, 1, 3, 5)
	// Reshape to [B, numPatches, C*patchH*patchW].
	x = Reshape(x, batchSize, numPatches, patchDim)

	// Conv weight: [hiddenSize, C, kH, kW] → [hiddenSize, patchDim] → transpose to [patchDim, hiddenSize].
	convWeight = Reshape(convWeight, vc.HiddenSize, patchDim)
	convWeight = Transpose(convWeight, 0, 1)

	return nn.Dense(x, convWeight, convBias)
}

// MLPProjector applies a multi-layer MLP projector with the given activation.
// Supports 2-layer (mlp2x_gelu) and 3-layer projectors.
// Expects "0/weights", "0/biases", "2/weights", "2/biases" etc. in scope.
func MLPProjector(ctx *context.Context, hidden *Node, numLayers int, activation func(*Node) *Node) *Node {
	for i := 0; i < numLayers; i++ {
		layerIdx := i * 2 // skip activation indices (0, 2, 4, ...)
		hidden = DenseWithBias(ctx.In(strconv.Itoa(layerIdx)), hidden)
		if i < numLayers-1 {
			hidden = activation(hidden)
		}
	}
	return hidden
}
