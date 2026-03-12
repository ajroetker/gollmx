package phi3

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"

	"github.com/gomlx/gollmx/architectures/common"
)

// HasVision returns true if the model has vision encoder weights loaded.
func (b *Builder) HasVision() bool {
	return b.visionConfig != nil
}

// VisionCfg returns the vision configuration, or nil if not a multimodal model.
func (b *Builder) VisionCfg() *common.VisionConfig {
	return b.visionConfig
}

// ImageTokenID returns the token ID used for image placeholders.
func (b *Builder) ImageTokenID() int32 {
	return b.imageTokenID
}

// NumImageTokens returns the number of image tokens for Phi-3-Vision.
// Phi-3-Vision uses all patches directly: (imageSize/patchSize)^2.
func (b *Builder) NumImageTokens() int {
	if b.visionConfig == nil {
		return 0
	}
	return b.visionConfig.NumPatches()
}

// BuildVisionEncoder runs the CLIP vision encoder.
// pixelValues: [batch, channels, height, width] float32.
// Returns: [batch, numPatches, visionHiddenSize] float32.
func (b *Builder) BuildVisionEncoder(ctx *context.Context, pixelValues *Node) *Node {
	return common.BuildCLIPVisionEncoder(ctx, pixelValues, b.visionConfig)
}

// BuildMultiModalProjector projects vision features to the LLM embedding space.
// Phi-3-Vision uses a 3-layer MLP projector: Linear → GELU → Linear → GELU → Linear.
// Input: [batch, numPatches, visionHidden].
// Output: [batch, numPatches, textHidden].
func (b *Builder) BuildMultiModalProjector(ctx *context.Context, visionFeatures *Node) *Node {
	return common.MLPProjector(ctx.In("mm"), visionFeatures, b.projectorLayers, activations.GeluApproximate)
}
