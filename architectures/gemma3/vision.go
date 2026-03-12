package gemma3

import (
	"math"
	"strconv"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/gomlx/pkg/ml/nn"

	"github.com/gomlx/gollmx/architectures/common"
)

// VisionConfig holds SigLIP vision encoder configuration.
type VisionConfig struct {
	HiddenSize    int     // e.g., 1152
	NumLayers     int     // e.g., 27
	NumHeads      int     // e.g., 16
	MLPDim        int     // e.g., 4304
	ImageSize     int     // e.g., 896
	PatchSize     int     // e.g., 14
	NumChannels   int     // e.g., 3
	LayerNormEps  float64 // e.g., 1e-6
}

// NumPatches returns the number of image patches from the vision encoder.
func (vc *VisionConfig) NumPatches() int {
	s := vc.ImageSize / vc.PatchSize
	return s * s
}

// PoolingKernel returns the average pooling kernel size used by the multimodal
// projector to reduce patches to image tokens. patches_per_side / tokens_per_side.
func (vc *VisionConfig) PoolingKernel() int {
	patchesPerSide := vc.ImageSize / vc.PatchSize
	tokensPerSide := vc.NumImageTokensPerSide()
	if tokensPerSide <= 0 {
		return 1
	}
	return patchesPerSide / tokensPerSide
}

// NumImageTokensPerSide returns the number of image tokens per spatial side
// after average pooling in the projector. Default: 16 (for 256 total tokens).
func (vc *VisionConfig) NumImageTokensPerSide() int {
	// mm_tokens_per_image = 256 for Gemma 3, so tokens_per_side = 16.
	// This is hardcoded since GGUF doesn't store mm_tokens_per_image.
	return 16
}

// NumImageTokens returns the number of image tokens after projector pooling.
// This is the number of <image> placeholder tokens to insert in the prompt.
func (vc *VisionConfig) NumImageTokens() int {
	s := vc.NumImageTokensPerSide()
	return s * s
}

// HeadDim returns the per-head dimension for vision attention.
func (vc *VisionConfig) HeadDim() int {
	return vc.HiddenSize / vc.NumHeads
}

// HasVision returns true if the model has vision encoder weights loaded.
func (b *Builder) HasVision() bool {
	return b.visionConfig != nil
}

// VisionConfig returns the vision configuration, or nil if not a multimodal model.
func (b *Builder) VisionCfg() *VisionConfig {
	return b.visionConfig
}

// BuildVisionEncoder runs the SigLIP vision encoder.
// pixelValues: [batch, channels, height, width] float32 in [0, 1].
// Returns: [batch, numPatches, hiddenSize] float32.
func (b *Builder) BuildVisionEncoder(ctx *context.Context, pixelValues *Node) *Node {
	vc := b.visionConfig
	vCtx := ctx.In("vision")
	g := pixelValues.Graph()

	batchSize := pixelValues.Shape().Dimensions[0]

	// Patch embedding: Conv2D with stride=patchSize.
	// Weight: [patchSize, patchSize, channels, hiddenSize] stored in GGUF.
	// We implement as: reshape image into patches, then linear projection.
	hidden := b.patchEmbed(vCtx, pixelValues, batchSize)

	// Add position embeddings.
	// Position embedding weight: [hiddenSize, numPatches] in GGUF (transposed).
	posVar := vCtx.GetVariableByScopeAndName(vCtx.Scope(), "position_embeddings")
	if posVar == nil {
		panic("vision: missing position_embeddings")
	}
	posEmb := posVar.ValueGraph(g)
	if posEmb.DType() != dtypes.Float32 {
		posEmb = ConvertDType(posEmb, dtypes.Float32)
	}
	// GGUF stores as [hiddenSize, numPatches]; transpose to [numPatches, hiddenSize].
	if posEmb.Shape().Dimensions[0] == vc.HiddenSize {
		posEmb = Transpose(posEmb, 0, 1)
	}
	// Expand to [1, numPatches, hiddenSize] for broadcasting with batched input.
	posEmb = InsertAxes(posEmb, 0)
	hidden = Add(hidden, posEmb)

	// Encoder layers.
	for i := range vc.NumLayers {
		hidden = b.visionEncoderLayer(vCtx.In("layers").In(strconv.Itoa(i)), hidden)
	}

	// Post-layer normalization.
	hidden = common.LayerNorm(vCtx.In("post_layernorm"), hidden, vc.LayerNormEps)

	_ = batchSize
	return hidden
}

// BuildMultiModalProjector projects vision features to the LLM embedding space.
// The projector: AvgPool2d(4×4) → RMSNorm → Linear.
// Input: [batch, numPatches, visionHidden] (4096 patches for 64×64 grid).
// Output: [batch, numImageTokens, textHidden] (256 tokens for 16×16 grid).
func (b *Builder) BuildMultiModalProjector(ctx *context.Context, visionFeatures *Node) *Node {
	vc := b.visionConfig
	mmCtx := ctx.In("mm")
	g := visionFeatures.Graph()

	batchSize := visionFeatures.Shape().Dimensions[0]
	hiddenSize := visionFeatures.Shape().Dimensions[2]
	patchesPerSide := vc.ImageSize / vc.PatchSize // 64
	kernel := vc.PoolingKernel()                   // 4
	tokensPerSide := vc.NumImageTokensPerSide()    // 16

	// Average pooling: reshape patches to 2D grid, pool with kernel×kernel stride.
	// [batch, numPatches, hidden] → [batch, hidden, patchesPerSide, patchesPerSide]
	pooled := Transpose(visionFeatures, 1, 2) // [batch, hidden, numPatches]
	pooled = Reshape(pooled, batchSize, hiddenSize, patchesPerSide, patchesPerSide)

	// AvgPool2d with kernel×kernel stride: manually reshape and mean.
	// [batch, hidden, patchesPerSide, patchesPerSide]
	// → [batch, hidden, tokensPerSide, kernel, tokensPerSide, kernel]
	pooled = Reshape(pooled, batchSize, hiddenSize, tokensPerSide, kernel, tokensPerSide, kernel)
	// → [batch, hidden, tokensPerSide, tokensPerSide, kernel, kernel]
	pooled = TransposeAllDims(pooled, 0, 1, 2, 4, 3, 5)
	// Mean over the kernel dims (last two axes).
	pooled = ReduceMean(pooled, -1)
	pooled = ReduceMean(pooled, -1)
	// [batch, hidden, tokensPerSide, tokensPerSide] → [batch, hidden, numImageTokens]
	numImageTokens := tokensPerSide * tokensPerSide
	pooled = Reshape(pooled, batchSize, hiddenSize, numImageTokens)
	// → [batch, numImageTokens, hidden]
	pooled = Transpose(pooled, 1, 2)

	// RMSNorm on pooled features.
	normalized := common.RMSNorm(mmCtx.In("soft_emb_norm"), pooled, vc.LayerNormEps)

	// Linear projection: [visionHidden] → [textHidden].
	projCtx := mmCtx.In("input_projection")
	wVar := projCtx.GetVariableByScopeAndName(projCtx.Scope(), "weights")
	if wVar == nil {
		panic("mm: missing input_projection weights")
	}
	w := wVar.ValueGraph(g)
	if w.DType() != normalized.DType() {
		w = ConvertDType(w, normalized.DType())
	}

	_ = g
	return nn.Dense(normalized, w, nil)
}

// patchEmbed converts pixel values to patch embeddings.
// pixelValues: [batch, channels, height, width]
// Returns: [batch, numPatches, hiddenSize]
func (b *Builder) patchEmbed(ctx *context.Context, pixelValues *Node, batchSize int) *Node {
	vc := b.visionConfig
	g := pixelValues.Graph()

	// Get conv weight: [patchSize, patchSize, channels, hiddenSize] from GGUF.
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

	// Manual patch extraction and linear projection.
	// Conv weight from GGUF is [hiddenSize, C, kH, kW]. We extract patches
	// in [C, pH, pW] order to match the kernel layout, then project.
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

	// Matmul: [B, numPatches, patchDim] @ [patchDim, hiddenSize] = [B, numPatches, hiddenSize]
	result := nn.Dense(x, convWeight, convBias)

	return result
}

// visionEncoderLayer runs a single SigLIP encoder layer.
// LayerNorm1 → SelfAttention → residual → LayerNorm2 → MLP → residual.
func (b *Builder) visionEncoderLayer(ctx *context.Context, hidden *Node) *Node {
	vc := b.visionConfig

	// Pre-attention LayerNorm.
	normalized := common.LayerNorm(ctx.In("layer_norm1"), hidden, vc.LayerNormEps)

	// Self-attention.
	attnOutput := b.visionAttention(ctx, normalized)
	hidden = Add(hidden, attnOutput)

	// Pre-MLP LayerNorm.
	normalized = common.LayerNorm(ctx.In("layer_norm2"), hidden, vc.LayerNormEps)

	// MLP: fc1(GELU) → fc2, both with bias.
	mlpOutput := b.visionMLP(ctx.In("mlp"), normalized)
	hidden = Add(hidden, mlpOutput)

	return hidden
}

// visionAttention runs multi-head self-attention for the vision encoder.
// Standard attention with bias, no GQA, no RoPE.
func (b *Builder) visionAttention(ctx *context.Context, hidden *Node) *Node {
	vc := b.visionConfig
	g := hidden.Graph()

	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1] // numPatches
	headDim := vc.HeadDim()
	numHeads := vc.NumHeads

	// Q, K, V projections with bias.
	query := common.DenseWithBias(ctx.In("attn_q"), hidden)
	key := common.DenseWithBias(ctx.In("attn_k"), hidden)
	value := common.DenseWithBias(ctx.In("attn_v"), hidden)

	// Reshape to [batch, numHeads, seqLen, headDim].
	query = Reshape(query, batchSize, seqLen, numHeads, headDim)
	query = Transpose(query, 1, 2)

	key = Reshape(key, batchSize, seqLen, numHeads, headDim)
	key = Transpose(key, 1, 2)

	value = Reshape(value, batchSize, seqLen, numHeads, headDim)
	value = Transpose(value, 1, 2)

	// Scaled dot-product attention using fused op (no causal mask — vision is bidirectional).
	scale := 1.0 / math.Sqrt(float64(headDim))
	attnOutput, _ := attention.Core(nil, query, key, value, scale, nil, 0, attention.LayoutBHSD, false, false)

	// Reshape back: [batch, numHeads, seqLen, headDim] → [batch, seqLen, hiddenSize]
	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, seqLen, numHeads*headDim)

	// Output projection with bias.
	_ = g
	return common.DenseWithBias(ctx.In("attn_output"), attnOutput)
}

// visionMLP applies the vision encoder MLP: fc1(GELU) → fc2.
func (b *Builder) visionMLP(ctx *context.Context, hidden *Node) *Node {
	// fc1 with GELU activation.
	hidden = common.DenseWithBias(ctx.In("fc1"), hidden)
	hidden = activations.GeluApproximate(hidden)

	// fc2.
	return common.DenseWithBias(ctx.In("fc2"), hidden)
}
