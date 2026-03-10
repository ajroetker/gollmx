package main

import (
	"math"
	"os"
	"testing"

	"fmt"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"

	"github.com/gomlx/go-huggingface/models/gguf"

	models "github.com/ajroetker/gollmx"
	"github.com/ajroetker/gollmx/architectures/common"
	"github.com/ajroetker/gollmx/architectures/gemma3"
)

// TestQuantizedEmbeddingAccuracy compares the quantized embedding gather output
// against the reference dequantization from the GGUF reader.
func TestQuantizedEmbeddingAccuracy(t *testing.T) {
	ggufPath := getGGUFPath()
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF file not found, skipping")
	}

	// Get reference dequantized embedding table.
	file, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatalf("Failed to parse GGUF: %v", err)
	}
	reader, err := gguf.NewReader(file)
	if err != nil {
		t.Fatalf("Failed to open GGUF reader: %v", err)
	}
	defer reader.Close()

	refTensor, err := reader.ReadTensor("token_embd.weight")
	if err != nil {
		t.Fatalf("ReadTensor failed: %v", err)
	}
	t.Logf("Reference embedding shape: %v", refTensor.Shape())

	var refValues []float32
	refTensor.ConstFlatData(func(flat any) {
		refValues = flat.([]float32)
	})
	hiddenSize := 2560

	// Load model and weights.
	model, err := models.NewFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	backend := backends.MustNew()
	t.Logf("Backend: %s", backend.Name())

	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	builder, ok := model.Builder.(*gemma3.Builder)
	if !ok {
		t.Fatal("Not a Gemma 3 builder")
	}
	cfg := builder.Gemma3Config()
	scale := float32(math.Sqrt(float64(cfg.HiddenSize)))
	t.Logf("Embedding scale: %f (sqrt(%d))", scale, cfg.HiddenSize)

	// Test each token individually to isolate any multi-token issue.
	testTokens := []int32{2, 100, 777, 1000, 5000, 100000}

	for _, tokenID := range testTokens {
		t.Run(fmt.Sprintf("token_%d", tokenID), func(t *testing.T) {
			exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, tokens *Node) *Node {
				return builder.BuildEmbeddings(ctx, tokens)
			})
			if err != nil {
				t.Fatalf("NewExec: %v", err)
			}

			tokensTensor := tensors.FromFlatDataAndDimensions([]int32{tokenID}, 1, 1) // [1, 1]
			results := exec.MustExec(tokensTensor)
			defer results[0].FinalizeAll()

			var embOutput []float32
			results[0].ConstFlatData(func(flat any) {
				embOutput = flat.([]float32)
			})

			refStart := int(tokenID) * hiddenSize
			refRow := refValues[refStart : refStart+hiddenSize]

			var maxDiff float32
			var sumDiffSq float64
			for k := 0; k < hiddenSize; k++ {
				expected := refRow[k] * scale
				diff := embOutput[k] - expected
				if diff < 0 {
					diff = -diff
				}
				if diff > maxDiff {
					maxDiff = diff
				}
				sumDiffSq += float64(diff) * float64(diff)
			}
			rmsDiff := math.Sqrt(sumDiffSq / float64(hiddenSize))

			t.Logf("maxDiff=%.8f, rmsDiff=%.8f", maxDiff, rmsDiff)
			t.Logf("ref[0:5]=%v", []float32{refRow[0] * scale, refRow[1] * scale, refRow[2] * scale, refRow[3] * scale, refRow[4] * scale})
			t.Logf("out[0:5]=%v", embOutput[0:5])

			if maxDiff > 0.01 {
				t.Errorf("max diff %.6f exceeds tolerance 0.01", maxDiff)
			}
		})
	}

	// Also test multiple tokens in one batch.
	t.Run("multi_token", func(t *testing.T) {
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, tokens *Node) *Node {
			return builder.BuildEmbeddings(ctx, tokens)
		})
		if err != nil {
			t.Fatalf("NewExec: %v", err)
		}

		tokensTensor := tensors.FromFlatDataAndDimensions(testTokens, 1, len(testTokens))
		results := exec.MustExec(tokensTensor)
		defer results[0].FinalizeAll()

		var embOutput []float32
		results[0].ConstFlatData(func(flat any) {
			embOutput = flat.([]float32)
		})
		t.Logf("Output shape: %v", results[0].Shape())

		for i, tokenID := range testTokens {
			refStart := int(tokenID) * hiddenSize
			refRow := refValues[refStart : refStart+hiddenSize]

			outStart := i * hiddenSize
			outRow := embOutput[outStart : outStart+hiddenSize]

			var maxDiff float32
			for k := 0; k < hiddenSize; k++ {
				expected := refRow[k] * scale
				diff := outRow[k] - expected
				if diff < 0 {
					diff = -diff
				}
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			t.Logf("Token %6d: maxDiff=%.8f, ref[0:3]=%v, out[0:3]=%v",
				tokenID, maxDiff,
				[]float32{refRow[0] * scale, refRow[1] * scale, refRow[2] * scale},
				outRow[0:3])

			if maxDiff > 0.01 {
				t.Errorf("Token %d: max diff %.6f exceeds tolerance 0.01", tokenID, maxDiff)
			}
		}
	})
}

// TestSmallTableGather creates a small Q6_K table from specific rows of the GGUF
// embedding table and tests FusedQuantizedGather directly. This isolates whether
// the issue is in the dequant function or the large buffer handling.
func TestSmallTableGather(t *testing.T) {
	ggufPath := getGGUFPath()
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF file not found, skipping")
	}

	file, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatalf("Failed to parse GGUF: %v", err)
	}
	reader, err := gguf.NewReader(file)
	if err != nil {
		t.Fatalf("Failed to open GGUF reader: %v", err)
	}
	defer reader.Close()

	// Read raw bytes and reference dequant.
	rawBytes, info, err := reader.ReadTensorRaw("token_embd.weight")
	if err != nil {
		t.Fatalf("ReadTensorRaw failed: %v", err)
	}
	t.Logf("Raw bytes len: %d, type: %s", len(rawBytes), info.Type)

	refTensor, err := reader.ReadTensor("token_embd.weight")
	if err != nil {
		t.Fatalf("ReadTensor failed: %v", err)
	}
	var refValues []float32
	refTensor.ConstFlatData(func(flat any) {
		refValues = flat.([]float32)
	})

	const bytesPerRow = 2100 // Q6_K: 210 bytes/block, 256 values/block, K=2560 → 10 blocks
	const K = 2560

	// Select specific rows to test.
	sourceRows := []int{0, 2, 100, 777, 5000}
	numRows := len(sourceRows)

	// Build a small table: [numRows, bytesPerRow] Uint8.
	smallTable := make([]uint8, numRows*bytesPerRow)
	for i, srcRow := range sourceRows {
		copy(smallTable[i*bytesPerRow:(i+1)*bytesPerRow],
			rawBytes[srcRow*bytesPerRow:(srcRow+1)*bytesPerRow])
	}

	backend := backends.MustNew()

	// Run FusedQuantizedGather for each row in the small table.
	for localIdx, srcRow := range sourceRows {
		t.Run(fmt.Sprintf("row_%d", srcRow), func(t *testing.T) {
			tableTensor := tensors.FromFlatDataAndDimensions(smallTable, numRows, bytesPerRow)
			indexTensor := tensors.FromFlatDataAndDimensions([]int32{int32(localIdx)}, 1, 1)

			exec, err := context.NewExec(backend, context.New(), func(ctx *context.Context, table, indices *Node) *Node {
				quant := &backends.Quantization{
					Scheme:   backends.QuantGGML,
					GGMLType: backends.GGMLQ6_K,
				}
				return FusedQuantizedGather(table, indices, quant)
			})
			if err != nil {
				t.Fatalf("NewExec: %v", err)
			}

			results := exec.MustExec(tableTensor, indexTensor)
			defer results[0].FinalizeAll()

			var output []float32
			results[0].ConstFlatData(func(flat any) {
				output = flat.([]float32)
			})
			t.Logf("Output shape: %v, first 5: %v", results[0].Shape(), output[:5])

			// Compare against reference dequant.
			refStart := srcRow * K
			refRow := refValues[refStart : refStart+K]

			var maxDiff float32
			nonZero := 0
			for k := range K {
				if output[k] != 0 {
					nonZero++
				}
				diff := output[k] - refRow[k]
				if diff < 0 {
					diff = -diff
				}
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("Non-zero: %d/%d, maxDiff=%.8f", nonZero, K, maxDiff)
			t.Logf("ref[0:5]=%v", refRow[:5])

			if nonZero == 0 {
				t.Error("All output values are zero")
			}
			if maxDiff > 0.001 {
				t.Errorf("maxDiff %.6f exceeds tolerance 0.001", maxDiff)
			}
		})
	}
}

// TestQuantizedDenseAccuracy compares one dense layer against reference weights.
func TestQuantizedDenseAccuracy(t *testing.T) {
	ggufPath := getGGUFPath()
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF file not found, skipping")
	}

	// Get reference dequantized weight.
	file, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatalf("Failed to parse GGUF: %v", err)
	}
	reader, err := gguf.NewReader(file)
	if err != nil {
		t.Fatalf("Failed to open GGUF reader: %v", err)
	}
	defer reader.Close()

	refWeight, err := reader.ReadTensor("blk.0.attn_q.weight")
	if err != nil {
		t.Fatalf("ReadTensor failed: %v", err)
	}
	t.Logf("Reference attn_q shape: %v", refWeight.Shape())

	var refWeightValues []float32
	refWeight.ConstFlatData(func(flat any) {
		refWeightValues = flat.([]float32)
	})

	N := refWeight.Shape().Dimensions[0] // 2048
	K := refWeight.Shape().Dimensions[1] // 2560
	t.Logf("Weight: N=%d, K=%d", N, K)

	// Deterministic test input.
	testInput := make([]float32, K)
	for i := range testInput {
		testInput[i] = float32(i%7-3) * 0.01
	}

	// Compute reference output: Y = X @ W^T.
	refOutput := make([]float32, N)
	for n := range N {
		var dot float32
		for k := range K {
			dot += testInput[k] * refWeightValues[n*K+k]
		}
		refOutput[n] = dot
	}
	t.Logf("Reference output: first 10 = %v", refOutput[:10])

	// Load model.
	model, err := models.NewFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	backend := backends.MustNew()
	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	_, ok := model.Builder.(*gemma3.Builder)
	if !ok {
		t.Fatal("Not a Gemma 3 builder")
	}

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
		attnCtx := ctx.In("layers").In("0").In("attention").In("query")
		return common.QuantizedDenseWeightOnly(attnCtx, x, backends.GGMLQ4_K)
	})
	if err != nil {
		t.Fatalf("NewExec: %v", err)
	}

	inputTensor := tensors.FromFlatDataAndDimensions(testInput, 1, K)
	results := exec.MustExec(inputTensor)
	defer results[0].FinalizeAll()

	var quantOutput []float32
	results[0].ConstFlatData(func(flat any) {
		quantOutput = flat.([]float32)
	})
	t.Logf("Quantized output shape: %v", results[0].Shape())
	t.Logf("Quantized output: first 10 = %v", quantOutput[:10])

	// Compare.
	var maxDiff float32
	var sumDiffSq float64
	for i := range N {
		diff := quantOutput[i] - refOutput[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
		sumDiffSq += float64(diff) * float64(diff)
	}
	rmsDiff := math.Sqrt(sumDiffSq / float64(N))
	t.Logf("Dense comparison: maxDiff=%.6f, rmsDiff=%.6f", maxDiff, rmsDiff)

	if maxDiff > 0.5 {
		t.Errorf("Max diff %.6f exceeds tolerance 0.5", maxDiff)
	}
}
