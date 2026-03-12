package common

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// MergeImageFeatures replaces embeddings at image token positions with projected vision features.
// hidden: [batch, seqLen, hiddenSize], imageFeatures: [batch, numPatches, hiddenSize],
// tokens: [batch, seqLen] int32, imageTokenID: the token ID used as placeholder for image tokens.
//
// Image tokens may not start at position 0 (e.g., they follow a chat template header).
// Uses CumSum on the image mask to build a proper index into imageFeatures so that
// the 1st image token gets feature 0, the 2nd gets feature 1, etc.
func MergeImageFeatures(hidden, imageFeatures, tokens *Node, imageTokenID int32) *Node {
	g := hidden.Graph()
	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	hiddenSize := hidden.Shape().Dimensions[2]

	// Create a boolean mask: true where token == imageTokenID.
	imageTokenConst := Scalar(g, dtypes.Int32, imageTokenID)
	isImage := Equal(tokens, imageTokenConst) // [batch, seqLen] bool

	// Build per-position indices into imageFeatures using CumSum.
	isImageF32 := ConvertDType(isImage, dtypes.Float32)
	cumIdx := CumSum(isImageF32, 1)
	cumIdx = ConvertDType(cumIdx, dtypes.Int32)
	featureIdx := Sub(cumIdx, Scalar(g, dtypes.Int32, int32(1))) // 0-based
	numPatches := imageFeatures.Shape().Dimensions[1]
	featureIdx = MinScalar(MaxScalar(featureIdx, 0), int32(numPatches-1)) // clamp

	// Convert imageFeatures dtype to match hidden if needed.
	if imageFeatures.DType() != hidden.DType() {
		imageFeatures = ConvertDType(imageFeatures, hidden.DType())
	}

	// Gather: use OneHot+Einsum to index into imageFeatures per batch element.
	featureIdxFlat := Reshape(featureIdx, batchSize*seqLen)
	oneHot := OneHot(featureIdxFlat, numPatches, dtypes.Float32) // [batch*seqLen, numPatches]
	oneHot = Reshape(oneHot, batchSize, seqLen, numPatches)
	gathered := Einsum("bsp,bph->bsh", oneHot, imageFeatures)

	// Select: image positions get gathered features, others keep hidden.
	isImage3D := InsertAxes(isImage, -1)
	isImage3D = BroadcastToDims(isImage3D, batchSize, seqLen, hiddenSize)
	return Where(isImage3D, gathered, hidden)
}
