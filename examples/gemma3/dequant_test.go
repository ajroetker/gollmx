package main

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/nn"

	models "github.com/ajroetker/gollmx"
)

func getGGUFPath() string {
	p := os.Getenv("GGUF_PATH")
	if p == "" {
		p = "/Users/ajroetker/.ollama/models/blobs/sha256-aeda25e63ebd698fab8638ffb778e68bed908b960d39d0becc650fa981609d25"
	}
	return p
}

func openGGUF(t *testing.T) (*gguf.File, *gguf.Reader) {
	t.Helper()
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

	return file, reader
}

// TestGGUFMetadata dumps config-relevant metadata from the GGUF file.
func TestGGUFMetadata(t *testing.T) {
	ggufPath := getGGUFPath()
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF file not found, skipping")
	}

	file, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatalf("Failed to parse GGUF: %v", err)
	}

	arch := file.Architecture()
	t.Logf("Architecture: %q", arch)
	t.Logf("Number of KV pairs: %d", len(file.KeyValues))
	t.Logf("Number of tensors: %d", len(file.TensorInfos))

	// Print all non-token metadata.
	for _, kv := range file.KeyValues {
		key := kv.Key
		if key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.scores" ||
			key == "tokenizer.ggml.token_type" || key == "tokenizer.ggml.merges" {
			t.Logf("  %-55s (large array, skipped)", key)
			continue
		}
		// Try different value types.
		if s := kv.String(); s != "" {
			t.Logf("  %-55s = %q", key, s)
		} else if f := kv.Float64(); f != 0 {
			t.Logf("  %-55s = %f", key, f)
		} else if i := kv.Uint64(); i != 0 {
			t.Logf("  %-55s = %d", key, i)
		} else {
			t.Logf("  %-55s = (zero or array)", key)
		}
	}

	// Print the specific config keys.
	t.Log("\n--- Config-relevant metadata ---")
	configKeys := []string{
		arch + ".block_count",
		arch + ".embedding_length",
		arch + ".attention.head_count",
		arch + ".attention.head_count_kv",
		arch + ".attention.key_length",
		arch + ".feed_forward_length",
		arch + ".context_length",
		arch + ".rope.freq_base",
		arch + ".attention.sliding_window",
		arch + ".attention.layer_norm_rms_epsilon",
	}
	for _, key := range configKeys {
		kv, ok := file.GetKeyValue(key)
		if !ok {
			t.Logf("  %-55s NOT FOUND", key)
		} else {
			// Print as both int and float for flexibility.
			t.Logf("  %-55s uint=%d float=%f", key, kv.Uint64(), kv.Float64())
		}
	}
}

// TestDequantComparison loads a quantized tensor from the GGUF file using the
// dequantized path (ReadTensor) and verifies the values are reasonable.
func TestDequantComparison(t *testing.T) {
	file, reader := openGGUF(t)
	defer reader.Close()

	tensors := []string{
		"blk.0.attn_q.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight",
		"token_embd.weight",
	}

	for _, name := range tensors {
		t.Run(name, func(t *testing.T) {
			info, ok := file.GetTensorInfo(name)
			if !ok {
				t.Skipf("Tensor %q not found", name)
			}
			t.Logf("Tensor %q: type=%s, shape=%v", name, info.Type, info.Shape)

			if !info.Type.IsQuantized() {
				t.Skipf("Tensor %q is not quantized (type=%s)", name, info.Type)
			}

			_, dims := info.GoMLXShape()
			t.Logf("  GoMLX dims: %v", dims)

			refTensor, err := reader.ReadTensor(name)
			if err != nil {
				t.Fatalf("ReadTensor failed: %v", err)
			}

			var refValues []float32
			refTensor.ConstFlatData(func(flat any) {
				refValues = flat.([]float32)
			})

			n := len(refValues)
			t.Logf("  Dequantized count: %d, first 10: %v", n, refValues[:min(10, n)])

			nonZero := 0
			for _, v := range refValues[:min(1000, n)] {
				if v != 0 {
					nonZero++
				}
			}
			t.Logf("  Non-zero in first 1000: %d", nonZero)
			if nonZero == 0 {
				t.Error("All values are zero")
			}

			var sum, sumSq float64
			var minVal, maxVal float32 = math.MaxFloat32, -math.MaxFloat32
			for _, v := range refValues {
				sum += float64(v)
				sumSq += float64(v) * float64(v)
				if v < minVal {
					minVal = v
				}
				if v > maxVal {
					maxVal = v
				}
			}
			mean := sum / float64(n)
			variance := sumSq/float64(n) - mean*mean
			t.Logf("  Stats: min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
				minVal, maxVal, mean, math.Sqrt(variance))
		})
	}
}

// TestRawBytesNonZero checks if the raw quantized tensor has non-zero data
// beyond the first few rows.
func TestRawBytesNonZero(t *testing.T) {
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

	// Read raw bytes of embedding table.
	rawBytes, info, err := reader.ReadTensorRaw("token_embd.weight")
	if err != nil {
		t.Fatalf("ReadTensorRaw failed: %v", err)
	}
	t.Logf("Raw bytes len: %d, type: %s, shape: %v", len(rawBytes), info.Type, info.Shape)

	// Q6_K: 210 bytes/block, 256 values/block. K=2560 → bytesPerRow=2100
	bytesPerRow := 2100
	N := len(rawBytes) / bytesPerRow
	t.Logf("N=%d, bytesPerRow=%d", N, bytesPerRow)

	// Check raw bytes for specific rows.
	rows := []int{0, 1, 2, 3, 100, 777, 1000, 5000, 100000}
	for _, row := range rows {
		if row >= N {
			continue
		}
		rowData := rawBytes[row*bytesPerRow : (row+1)*bytesPerRow]
		nonZero := 0
		for _, b := range rowData {
			if b != 0 {
				nonZero++
			}
		}
		t.Logf("Row %6d: non-zero bytes = %d/%d, first 16 bytes = %v",
			row, nonZero, bytesPerRow, rowData[:16])
	}

	// Now load via our weight source and check the raw tensor.
	t.Log("\n--- Checking loaded raw tensor ---")

	model, err := models.NewFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	// Get the raw embedding variable.
	embVar := ctx.In("embeddings").GetVariableByScopeAndName("/embeddings", "embeddings")
	if embVar == nil {
		t.Fatal("Embedding variable not found")
	}
	t.Logf("Embedding variable shape: %v", embVar.Shape())

	embTensor, err := embVar.Value()
	if err != nil {
		t.Fatalf("Failed to get embedding value: %v", err)
	}
	var rawData []uint8
	embTensor.ConstFlatData(func(flat any) {
		rawData = flat.([]uint8)
	})
	t.Logf("Loaded raw data len: %d", len(rawData))

	loadedBytesPerRow := embVar.Shape().Dimensions[1]
	t.Logf("Loaded bytesPerRow: %d (expected %d)", loadedBytesPerRow, bytesPerRow)

	for _, row := range rows {
		if row >= embVar.Shape().Dimensions[0] {
			continue
		}
		rowData := rawData[row*loadedBytesPerRow : (row+1)*loadedBytesPerRow]
		nonZero := 0
		for _, b := range rowData {
			if b != 0 {
				nonZero++
			}
		}
		t.Logf("Loaded row %6d: non-zero bytes = %d/%d, first 16 bytes = %v",
			row, nonZero, loadedBytesPerRow, rowData[:16])
	}
}

// TestQuantizedDenseVsReference compares the output of our QuantizedDense (GGML backend)
// with a reference matmul using the go-huggingface dequantized weights.
// If these differ significantly, the dequantization in our backend is wrong.
func TestQuantizedDenseVsReference(t *testing.T) {
	file, reader := openGGUF(t)
	defer reader.Close()

	backend := backends.MustNew()

	// Test two different quant types: Q4_K and Q6_K.
	tests := []struct {
		name     string
		quantKey string // Which GGUF tensor to test.
	}{
		{"Q4_K_attn_q", "blk.0.attn_q.weight"},
		{"Q6_K_attn_v", "blk.0.attn_v.weight"},
		{"Q6_K_ffn_down", "blk.0.ffn_down.weight"},
		{"Q4_K_ffn_gate", "blk.0.ffn_gate.weight"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			info, ok := file.GetTensorInfo(tc.quantKey)
			if !ok {
				t.Skipf("Tensor %q not found", tc.quantKey)
			}
			if !info.Type.IsQuantized() {
				t.Skipf("Tensor %q is not quantized", tc.quantKey)
			}

			ggmlType, _ := info.Type.ToGGMLQuantType()
			_, dims := info.GoMLXShape()
			N := dims[0] // output features
			K := dims[1] // input features
			t.Logf("Tensor %q: type=%s, GoMLX dims=[%d, %d] (N=%d, K=%d)", tc.quantKey, info.Type, N, K, N, K)

			// 1. Reference: dequantize using go-huggingface and compute matmul.
			refTensor, err := reader.ReadTensor(tc.quantKey)
			if err != nil {
				t.Fatalf("ReadTensor: %v", err)
			}
			var refWeights []float32
			refTensor.ConstFlatData(func(flat any) {
				refWeights = make([]float32, len(flat.([]float32)))
				copy(refWeights, flat.([]float32))
			})
			t.Logf("Reference dequant: %d values, first 5: %v", len(refWeights), refWeights[:5])

			// 2. Raw bytes for our QuantizedDense.
			rawBytes, _, err := reader.ReadTensorRaw(tc.quantKey)
			if err != nil {
				t.Fatalf("ReadTensorRaw: %v", err)
			}

			vpb := ggmlType.ValuesPerBlock()
			bpb := ggmlType.BytesPerBlock()
			bytesPerRow := (K / vpb) * bpb

			// Create a small test input: 4 rows of K features with known values.
			M := 4
			xData := make([]float32, M*K)
			// Row 0: all ones → should give sum of each weight column.
			for j := range K {
				xData[j] = 1.0
			}
			// Row 1: one-hot at position 0 → should give weight row's 0th element.
			xData[K] = 1.0
			// Row 2: one-hot at position K-1
			xData[2*K+K-1] = 1.0
			// Row 3: alternating 1,-1
			for j := range K {
				if j%2 == 0 {
					xData[3*K+j] = 1.0
				} else {
					xData[3*K+j] = -1.0
				}
			}

			// 3. Reference matmul: x @ refWeights^T.
			// refWeights is [N, K] (row-major), we compute x[M, K] @ refWeights[N, K]^T = out[M, N].
			refOutput := make([]float32, M*N)
			for m := range M {
				for n := range N {
					var dot float32
					for k := range K {
						dot += xData[m*K+k] * refWeights[n*K+k]
					}
					refOutput[m*N+n] = dot
				}
			}

			// 4. Our QuantizedDense: build and execute GoMLX graph.
			rawTensor := tensors.FromShape(shapes.Make(dtypes.Uint8, N, bytesPerRow))
			rawTensor.MutableBytes(func(data []byte) {
				copy(data, rawBytes)
			})
			xTensor := tensors.FromFlatDataAndDimensions(xData, M, K)

			ctx := context.New()
			wCtx := ctx.In("test")
			wCtx.VariableWithValue("weights", rawTensor)

			quant := &Quantization{
				Scheme:   backends.QuantGGML,
				GGMLType: ggmlType,
			}

			exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
				g := x.Graph()
				wVar := ctx.In("test").GetVariableByScopeAndName("/test", "weights")
				weights := wVar.ValueGraph(g)
				return nn.QuantizedDense(x, weights, quant, nil)
			})
			if err != nil {
				t.Fatalf("NewExec: %v", err)
			}

			results := exec.MustExec(xTensor)
			var ourOutput []float32
			results[0].ConstFlatData(func(flat any) {
				ourOutput = flat.([]float32)
			})
			results[0].FinalizeAll()

			// 5. Compare.
			if len(ourOutput) != len(refOutput) {
				t.Fatalf("Output size mismatch: ours=%d, ref=%d", len(ourOutput), len(refOutput))
			}

			var maxAbsErr, maxRelErr float64
			var worstN, worstM int
			mismatches := 0
			for m := range M {
				for n := range N {
					idx := m*N + n
					got := float64(ourOutput[idx])
					want := float64(refOutput[idx])
					absErr := math.Abs(got - want)
					var relErr float64
					if math.Abs(want) > 1e-6 {
						relErr = absErr / math.Abs(want)
					}
					if absErr > maxAbsErr {
						maxAbsErr = absErr
						worstN = n
						worstM = m
					}
					if relErr > maxRelErr {
						maxRelErr = relErr
					}
					// Flag significant differences.
					if absErr > 1e-3 {
						if mismatches < 10 {
							t.Errorf("Mismatch at [m=%d, n=%d]: got=%.6f, want=%.6f, absErr=%.6f",
								m, n, got, want, absErr)
						}
						mismatches++
					}
				}
			}

			t.Logf("Max absolute error: %.6e (at m=%d, n=%d)", maxAbsErr, worstM, worstN)
			t.Logf("Max relative error: %.6e", maxRelErr)
			t.Logf("Total mismatches (>1e-3): %d / %d", mismatches, M*N)

			// Print some sample values for manual inspection.
			for m := range M {
				t.Logf("Row %d: first 3 ours=[%.4f, %.4f, %.4f] ref=[%.4f, %.4f, %.4f]",
					m,
					ourOutput[m*N], ourOutput[m*N+1], ourOutput[m*N+2],
					refOutput[m*N], refOutput[m*N+1], refOutput[m*N+2])
			}
		})
	}
}

// TestWeightShapes prints the types and shapes of all weights.
func TestWeightShapes(t *testing.T) {
	file, reader := openGGUF(t)
	defer reader.Close()

	names := []string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.attn_q_norm.weight",
		"blk.0.attn_k_norm.weight",
		"blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
		"blk.0.post_attention_norm.weight",
		"blk.0.post_ffw_norm.weight",
	}

	for _, name := range names {
		info, ok := file.GetTensorInfo(name)
		if !ok {
			fmt.Printf("  %-40s NOT FOUND\n", name)
			continue
		}
		_, dims := info.GoMLXShape()
		fmt.Printf("  %-40s type=%-8s gomlx_dims=%v raw_shape=%v\n", name, info.Type, dims, info.Shape)
	}
}

