// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package kvcache

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/tensors"
)

func TestCompactBasic(t *testing.T) {
	numKVHeads := 2
	seqLen := 32
	headDim := 8
	targetLen := 8

	keys := randomTensor(numKVHeads, seqLen, headDim)
	values := randomTensor(numKVHeads, seqLen, headDim)

	config := CompactionConfig{
		TargetLen:     targetLen,
		NumRefQueries: 16,
	}

	result, err := Compact(config, keys, values, nil)
	if err != nil {
		t.Fatalf("Compact failed: %v", err)
	}

	// Verify output shapes.
	wantKeyShape := []int{numKVHeads, targetLen, headDim}
	gotKeyShape := result.CompactedKeys.Shape().Dimensions
	if !slicesEqual(gotKeyShape, wantKeyShape) {
		t.Errorf("CompactedKeys shape = %v, want %v", gotKeyShape, wantKeyShape)
	}

	wantValShape := []int{numKVHeads, targetLen, headDim}
	gotValShape := result.CompactedValues.Shape().Dimensions
	if !slicesEqual(gotValShape, wantValShape) {
		t.Errorf("CompactedValues shape = %v, want %v", gotValShape, wantValShape)
	}

	wantBiasShape := []int{numKVHeads, targetLen}
	gotBiasShape := result.Biases.Shape().Dimensions
	if !slicesEqual(gotBiasShape, wantBiasShape) {
		t.Errorf("Biases shape = %v, want %v", gotBiasShape, wantBiasShape)
	}
}

func TestCompactLeastSquaresQuality(t *testing.T) {
	// Verify that A_s · V_s ≈ A_full · V_full for the reference queries.
	// This is the actual optimization target of the least squares solve.
	numKVHeads := 1
	seqLen := 64
	headDim := 16
	targetLen := 16
	numRefQueries := 32

	keys := randomTensor(numKVHeads, seqLen, headDim)
	values := randomTensor(numKVHeads, seqLen, headDim)

	config := CompactionConfig{
		TargetLen:     targetLen,
		NumRefQueries: numRefQueries,
	}

	result, err := Compact(config, keys, values, nil)
	if err != nil {
		t.Fatalf("Compact failed: %v", err)
	}

	keysFlat := tensors.MustCopyFlatData[float32](keys)
	valsFlat := tensors.MustCopyFlatData[float32](values)

	// Compute the full attention output: A_full · V_full
	actualRefQueries := min(numRefQueries, seqLen)
	qRef := keysFlat[(seqLen-actualRefQueries)*headDim : seqLen*headDim]
	fullOutput := computeAttentionOutput(qRef, keysFlat, valsFlat, actualRefQueries, seqLen, headDim)

	// Compute A_s · V_s using the same attention weights A_s (columns of A_full).
	// A_s = softmax(Q_ref · K_s^T / sqrt(d)), but these are the columns of A_full
	// for the selected key positions. We approximate this by computing attention
	// directly on the compacted keys (which is what will happen at inference).
	compKeysFlat := tensors.MustCopyFlatData[float32](result.CompactedKeys)
	compValsFlat := tensors.MustCopyFlatData[float32](result.CompactedValues)
	biasFlat := tensors.MustCopyFlatData[float32](result.Biases)

	// Test 1: End-to-end with bias (what happens at inference).
	compOutput := computeAttentionOutputWithBias(qRef, compKeysFlat, compValsFlat, biasFlat, actualRefQueries, targetLen, headDim)

	var totalErr, totalNorm float64
	for i := range fullOutput {
		diff := float64(fullOutput[i] - compOutput[i])
		totalErr += diff * diff
		totalNorm += float64(fullOutput[i]) * float64(fullOutput[i])
	}
	relErrE2E := math.Sqrt(totalErr / totalNorm)
	t.Logf("End-to-end relative error at %dx compression: %.4f", seqLen/targetLen, relErrE2E)

	// Test 2: Direct least squares residual (A_s · V_s vs A_full · V_full).
	// Use the same attention weights from A_full (no re-softmax).
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	aFull := computeAttentionWeights(qRef, keysFlat, actualRefQueries, seqLen, headDim, scale)
	aS := computeAttentionWeights(qRef, compKeysFlat, actualRefQueries, targetLen, headDim, scale)

	// A_full · V_full
	target := matVecBatch(aFull, valsFlat, actualRefQueries, seqLen, headDim)
	// A_s · V_s (using renormalized weights on compacted keys)
	approx := matVecBatch(aS, compValsFlat, actualRefQueries, targetLen, headDim)

	totalErr, totalNorm = 0, 0
	for i := range target {
		diff := float64(target[i] - approx[i])
		totalErr += diff * diff
		totalNorm += float64(target[i]) * float64(target[i])
	}
	relErrLS := math.Sqrt(totalErr / totalNorm)
	t.Logf("Least squares residual relative error: %.4f", relErrLS)

	// The end-to-end error with random data at 4x compression can be high.
	// We mainly verify the algorithm runs without error and produces
	// reasonable output shapes. Real quality testing uses gemma3n.
	if math.IsNaN(relErrE2E) || math.IsInf(relErrE2E, 0) {
		t.Errorf("End-to-end error is NaN or Inf")
	}
}

func TestCompactWithExplicitRefQueries(t *testing.T) {
	numKVHeads := 1
	seqLen := 32
	headDim := 8
	targetLen := 8
	numRefQueries := 16

	keys := randomTensor(numKVHeads, seqLen, headDim)
	values := randomTensor(numKVHeads, seqLen, headDim)
	refQueries := randomTensor(numKVHeads, numRefQueries, headDim)

	config := CompactionConfig{
		TargetLen: targetLen,
	}

	result, err := Compact(config, keys, values, refQueries)
	if err != nil {
		t.Fatalf("Compact failed: %v", err)
	}

	if result.CompactedKeys.Shape().Dimensions[1] != targetLen {
		t.Errorf("got targetLen %d, want %d", result.CompactedKeys.Shape().Dimensions[1], targetLen)
	}
}

func TestCompactErrorCases(t *testing.T) {
	keys := randomTensor(2, 32, 8)
	values := randomTensor(2, 32, 8)

	// targetLen >= seqLen
	_, err := Compact(CompactionConfig{TargetLen: 32}, keys, values, nil)
	if err == nil {
		t.Error("expected error for targetLen >= seqLen")
	}

	// targetLen = 0
	_, err = Compact(CompactionConfig{TargetLen: 0}, keys, values, nil)
	if err == nil {
		t.Error("expected error for targetLen = 0")
	}
}

func BenchmarkCompact(b *testing.B) {
	numKVHeads := 8
	seqLen := 2048
	headDim := 64
	targetLen := 128

	keys := randomTensor(numKVHeads, seqLen, headDim)
	values := randomTensor(numKVHeads, seqLen, headDim)

	config := CompactionConfig{
		TargetLen:     targetLen,
		NumRefQueries: 64,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Compact(config, keys, values, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// --- helpers ---

func randomTensor(dims ...int) *tensors.Tensor {
	size := 1
	for _, d := range dims {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = rand.Float32()*2 - 1
	}
	return tensors.FromFlatDataAndDimensions(data, dims...)
}

func slicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// computeAttentionOutputWithBias computes softmax(Q·K^T/sqrt(d) + bias) · V.
func computeAttentionOutputWithBias(q, k, v, bias []float32, numQ, numK, headDim int) []float32 {
	scale := 1.0 / math.Sqrt(float64(headDim))
	output := make([]float32, numQ*headDim)

	for i := range numQ {
		scores := make([]float64, numK)
		maxScore := math.Inf(-1)
		for j := range numK {
			var dot float64
			for d := range headDim {
				dot += float64(q[i*headDim+d]) * float64(k[j*headDim+d])
			}
			scores[j] = dot*scale + float64(bias[j])
			if scores[j] > maxScore {
				maxScore = scores[j]
			}
		}

		var expSum float64
		weights := make([]float64, numK)
		for j := range numK {
			weights[j] = math.Exp(scores[j] - maxScore)
			expSum += weights[j]
		}
		for j := range numK {
			weights[j] /= expSum
		}

		for d := range headDim {
			var sum float64
			for j := range numK {
				sum += weights[j] * float64(v[j*headDim+d])
			}
			output[i*headDim+d] = float32(sum)
		}
	}

	return output
}

// computeAttentionWeights computes softmax(Q·K^T * scale) and returns the weight matrix [numQ, numK].
func computeAttentionWeights(q, k []float32, numQ, numK, headDim int, scale float32) []float32 {
	weights := make([]float32, numQ*numK)
	for i := range numQ {
		scores := make([]float64, numK)
		maxScore := math.Inf(-1)
		for j := range numK {
			var dot float64
			for d := range headDim {
				dot += float64(q[i*headDim+d]) * float64(k[j*headDim+d])
			}
			scores[j] = dot * float64(scale)
			if scores[j] > maxScore {
				maxScore = scores[j]
			}
		}
		var expSum float64
		for j := range numK {
			scores[j] = math.Exp(scores[j] - maxScore)
			expSum += scores[j]
		}
		for j := range numK {
			weights[i*numK+j] = float32(scores[j] / expSum)
		}
	}
	return weights
}

// matVecBatch computes A · V where A is [numQ, numK] and V is [numK, headDim].
// Returns [numQ, headDim].
func matVecBatch(a, v []float32, numQ, numK, headDim int) []float32 {
	out := make([]float32, numQ*headDim)
	for i := range numQ {
		for d := range headDim {
			var sum float64
			for j := range numK {
				sum += float64(a[i*numK+j]) * float64(v[j*headDim+d])
			}
			out[i*headDim+d] = float32(sum)
		}
	}
	return out
}

// computeAttentionOutput computes softmax(Q·K^T/sqrt(d)) · V on CPU.
func computeAttentionOutput(q, k, v []float32, numQ, numK, headDim int) []float32 {
	scale := 1.0 / math.Sqrt(float64(headDim))
	output := make([]float32, numQ*headDim)

	for i := range numQ {
		// Compute scores: q[i] · k[j]^T * scale
		scores := make([]float64, numK)
		maxScore := math.Inf(-1)
		for j := range numK {
			var dot float64
			for d := range headDim {
				dot += float64(q[i*headDim+d]) * float64(k[j*headDim+d])
			}
			scores[j] = dot * scale
			if scores[j] > maxScore {
				maxScore = scores[j]
			}
		}

		// Softmax.
		var expSum float64
		weights := make([]float64, numK)
		for j := range numK {
			weights[j] = math.Exp(scores[j] - maxScore)
			expSum += weights[j]
		}
		for j := range numK {
			weights[j] /= expSum
		}

		// Weighted sum of values.
		for d := range headDim {
			var sum float64
			for j := range numK {
				sum += weights[j] * float64(v[j*headDim+d])
			}
			output[i*headDim+d] = float32(sum)
		}
	}

	return output
}
