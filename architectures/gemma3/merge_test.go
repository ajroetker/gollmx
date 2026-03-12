package gemma3

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"

	"github.com/gomlx/gollmx/architectures/common"
)

func TestMergeImageFeatures(t *testing.T) {
	backend := backends.MustNew()

	// Small test: batch=1, seqLen=8, hiddenSize=4, numPatches=3
	// Tokens: [BOS, startTurn, user, newline, imgTok, imgTok, imgTok, endTurn]
	// Image tokens at positions 4, 5, 6 → should get features 0, 1, 2
	tokens := []int32{2, 105, 2364, 107, 262144, 262144, 262144, 106}

	// Hidden embeddings: all 1.0 (text positions) — we expect image positions to be replaced.
	hiddenData := make([]float32, 8*4)
	for i := range hiddenData {
		hiddenData[i] = 1.0
	}

	// Image features: feature[i] = (i+1)*10.0 for all hidden dims.
	// feature 0 = [10, 10, 10, 10]
	// feature 1 = [20, 20, 20, 20]
	// feature 2 = [30, 30, 30, 30]
	imageData := make([]float32, 3*4)
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			imageData[i*4+j] = float32((i + 1) * 10)
		}
	}

	tokensTensor := tensors.FromShape(shapes.Make(dtypes.Int32, 1, 8))
	tokensTensor.MutableFlatData(func(data any) { copy(data.([]int32), tokens) })

	hiddenTensor := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 8, 4))
	hiddenTensor.MutableFlatData(func(data any) { copy(data.([]float32), hiddenData) })

	imageTensor := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 3, 4))
	imageTensor.MutableFlatData(func(data any) { copy(data.([]float32), imageData) })

	ctx := mlctx.New()
	exec, err := mlctx.NewExec(backend, ctx,
		func(ctx *mlctx.Context, hidden *Node, imageFeatures *Node, tokensNode *Node) *Node {
			return common.MergeImageFeatures(hidden, imageFeatures, tokensNode, 262144)
		})
	if err != nil {
		t.Fatalf("NewExec: %v", err)
	}

	outputs, err := exec.Exec(hiddenTensor, imageTensor, tokensTensor)
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}

	result := outputs[0]
	fmt.Printf("Result shape: %s\n", result.Shape())

	result.ConstFlatData(func(data any) {
		flat := data.([]float32)
		// Expected:
		// pos 0-3 (text): [1, 1, 1, 1]
		// pos 4 (img 0):  [10, 10, 10, 10]
		// pos 5 (img 1):  [20, 20, 20, 20]
		// pos 6 (img 2):  [30, 30, 30, 30]
		// pos 7 (text):   [1, 1, 1, 1]
		for pos := 0; pos < 8; pos++ {
			vals := flat[pos*4 : (pos+1)*4]
			fmt.Printf("  pos %d: %v\n", pos, vals)
		}

		// Check text positions kept original values.
		for _, pos := range []int{0, 1, 2, 3, 7} {
			for j := 0; j < 4; j++ {
				if flat[pos*4+j] != 1.0 {
					t.Errorf("pos %d dim %d: got %.1f, want 1.0", pos, j, flat[pos*4+j])
				}
			}
		}
		// Check image positions got correct features.
		for i, pos := range []int{4, 5, 6} {
			expected := float32((i + 1) * 10)
			for j := 0; j < 4; j++ {
				if flat[pos*4+j] != expected {
					t.Errorf("pos %d dim %d: got %.1f, want %.1f", pos, j, flat[pos*4+j], expected)
				}
			}
		}
	})
}
