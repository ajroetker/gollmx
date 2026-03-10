// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Attention Matching KV cache compaction.
//
// Implements the Attention Matching technique for compressing KV caches:
// given reference queries, it selects the most important keys by attention
// weight, then solves a least-squares problem to compute adjusted values
// for the retained keys. A per-key scalar bias preserves attention mass.
//
// All computation runs on CPU using SIMD-accelerated go-highway primitives
// (attention weights, matmul, Cholesky solve). This is designed to run
// between graph executions (e.g., after prefill, before decode).

package kvcache

import (
	"fmt"
	"math"
	"sort"

	"github.com/ajroetker/go-highway/hwy/contrib/linalg"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// CompactionConfig configures KV cache compaction.
type CompactionConfig struct {
	// TargetLen is the desired compacted sequence length.
	TargetLen int

	// NumRefQueries is the number of reference queries to use from the
	// end of the key sequence. These proxy future attention patterns.
	// Default: 64.
	NumRefQueries int

	// MinSeqLen is the minimum sequence length to trigger compaction.
	// Sequences shorter than this are not compacted.
	MinSeqLen int

	// Regularization is the ridge regularization parameter added to the
	// diagonal of A^T A for numerical stability. Default: 1e-6.
	Regularization float32
}

// CompactionResult holds the output of a compaction operation.
type CompactionResult struct {
	// CompactedKeys contains the selected key vectors.
	// Shape: [numKVHeads, targetLen, headDim]
	CompactedKeys *tensors.Tensor

	// CompactedValues contains the least-squares-fitted value vectors.
	// Shape: [numKVHeads, targetLen, headDim]
	CompactedValues *tensors.Tensor

	// Biases contains per-key additive attention logit biases that
	// preserve the attention mass of dropped keys.
	// Shape: [numKVHeads, targetLen]
	Biases *tensors.Tensor
}

// Compact performs per-head attention-matching compaction on CPU.
//
// The algorithm per head:
//  1. Compute attention weights: A_full = softmax(Q_ref · K^T / sqrt(d))
//  2. Score importance: importance[j] = sum_i(A_full[i,j])
//  3. Select top-targetLen keys by importance
//  4. Solve least squares: A_s · V_s ≈ A_full · V_full via normal equations
//  5. Compute bias to preserve attention mass
//
// Parameters:
//   - keys: [numKVHeads, seqLen, headDim] — cached key vectors
//   - values: [numKVHeads, seqLen, headDim] — cached value vectors
//   - refQueries: [numKVHeads, numRefQueries, headDim] — reference queries
//     (if nil, the last NumRefQueries keys are used as proxy queries)
//
// For GQA models where numQueryHeads > numKVHeads, pass queries grouped
// by KV head (averaged or concatenated).
func Compact(config CompactionConfig, keys, values *tensors.Tensor, refQueries *tensors.Tensor) (*CompactionResult, error) {
	if config.NumRefQueries <= 0 {
		config.NumRefQueries = 64
	}
	if config.Regularization <= 0 {
		config.Regularization = 1e-6
	}

	keyShape := keys.Shape()
	valShape := values.Shape()
	if keyShape.Rank() != 3 || valShape.Rank() != 3 {
		return nil, fmt.Errorf("compaction: keys and values must be rank-3 [numKVHeads, seqLen, headDim], got keys=%v values=%v", keyShape, valShape)
	}

	numKVHeads := keyShape.Dimensions[0]
	seqLen := keyShape.Dimensions[1]
	headDim := keyShape.Dimensions[2]
	targetLen := config.TargetLen

	if targetLen >= seqLen {
		return nil, fmt.Errorf("compaction: targetLen (%d) must be less than seqLen (%d)", targetLen, seqLen)
	}
	if targetLen <= 0 {
		return nil, fmt.Errorf("compaction: targetLen must be positive, got %d", targetLen)
	}

	// Use last keys as reference queries if none provided.
	var numRefQueries int
	if refQueries != nil {
		refShape := refQueries.Shape()
		if refShape.Rank() != 3 {
			return nil, fmt.Errorf("compaction: refQueries must be rank-3 [numKVHeads, numRefQueries, headDim], got %v", refShape)
		}
		numRefQueries = refShape.Dimensions[1]
	} else {
		numRefQueries = min(config.NumRefQueries, seqLen)
	}

	// Extract flat data.
	keysFlat := tensors.MustCopyFlatData[float32](keys)
	valsFlat := tensors.MustCopyFlatData[float32](values)

	var refsFlat []float32
	if refQueries != nil {
		refsFlat = tensors.MustCopyFlatData[float32](refQueries)
	}

	// Allocate output buffers.
	outKeys := make([]float32, numKVHeads*targetLen*headDim)
	outVals := make([]float32, numKVHeads*targetLen*headDim)
	outBias := make([]float32, numKVHeads*targetLen)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for h := range numKVHeads {
		kHead := keysFlat[h*seqLen*headDim : (h+1)*seqLen*headDim]
		vHead := valsFlat[h*seqLen*headDim : (h+1)*seqLen*headDim]

		// Get reference queries for this head.
		var qRef []float32
		if refsFlat != nil {
			qRef = refsFlat[h*numRefQueries*headDim : (h+1)*numRefQueries*headDim]
		} else {
			// Use last numRefQueries keys as proxy queries.
			startIdx := seqLen - numRefQueries
			qRef = kHead[startIdx*headDim : seqLen*headDim]
		}

		err := compactHead(
			qRef, kHead, vHead,
			numRefQueries, seqLen, headDim, targetLen,
			scale, config.Regularization,
			outKeys[h*targetLen*headDim:(h+1)*targetLen*headDim],
			outVals[h*targetLen*headDim:(h+1)*targetLen*headDim],
			outBias[h*targetLen:(h+1)*targetLen],
		)
		if err != nil {
			return nil, fmt.Errorf("compaction head %d: %w", h, err)
		}
	}

	return &CompactionResult{
		CompactedKeys:   tensors.FromFlatDataAndDimensions(outKeys, numKVHeads, targetLen, headDim),
		CompactedValues: tensors.FromFlatDataAndDimensions(outVals, numKVHeads, targetLen, headDim),
		Biases:          tensors.FromFlatDataAndDimensions(outBias, numKVHeads, targetLen),
	}, nil
}

// compactHead performs compaction for a single attention head.
func compactHead(
	qRef, kFull, vFull []float32,
	numRefQueries, seqLen, headDim, targetLen int,
	scale, regularization float32,
	outKeys, outVals, outBias []float32,
) error {
	// Step 1: Compute attention weights A_full = softmax(Q_ref · K^T / sqrt(d))
	// Shape: [numRefQueries, seqLen]
	aFull := make([]float32, numRefQueries*seqLen)
	nn.AttentionWeights(qRef, kFull, nil, aFull, numRefQueries, seqLen, headDim, scale)

	// Step 2: Compute importance scores (column-sum of attention weights).
	importance := make([]float32, seqLen)
	for i := range numRefQueries {
		row := aFull[i*seqLen : (i+1)*seqLen]
		for j := range seqLen {
			importance[j] += row[j]
		}
	}

	// Step 3: Select top-targetLen keys by importance.
	indices := make([]int, seqLen)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(a, b int) bool {
		return importance[indices[a]] > importance[indices[b]]
	})
	selected := indices[:targetLen]
	// Sort selected indices so output order matches original position order.
	sort.Ints(selected)

	// Copy selected keys to output.
	for i, idx := range selected {
		copy(outKeys[i*headDim:(i+1)*headDim], kFull[idx*headDim:(idx+1)*headDim])
	}

	// Step 4: Gather A_s = A_full[:, selected] — [numRefQueries, targetLen]
	aS := make([]float32, numRefQueries*targetLen)
	for i := range numRefQueries {
		for j, idx := range selected {
			aS[i*targetLen+j] = aFull[i*seqLen+idx]
		}
	}

	// Step 5: Compute target = A_full · V_full — [numRefQueries, headDim]
	target := make([]float32, numRefQueries*headDim)
	matmul.MatMul(aFull, vFull, target, numRefQueries, headDim, seqLen)

	// Step 6: Solve A_s · V_s = target via normal equations.
	// A_s^T · A_s → [targetLen, targetLen]
	aST := transposeMatrix(aS, numRefQueries, targetLen)
	ata := make([]float32, targetLen*targetLen)
	matmul.MatMul(aST, aS, ata, targetLen, targetLen, numRefQueries)

	// Add regularization: ata += λI
	for i := range targetLen {
		ata[i*targetLen+i] += regularization
	}

	// A_s^T · target → [targetLen, headDim]
	atb := make([]float32, targetLen*headDim)
	matmul.MatMul(aST, target, atb, targetLen, headDim, numRefQueries)

	// Solve (A_s^T A_s) V_s = A_s^T target
	if err := linalg.CholeskySolve(ata, targetLen, atb, headDim, false); err != nil {
		// Fall back to simple value gathering if Cholesky fails.
		for i, idx := range selected {
			copy(outVals[i*headDim:(i+1)*headDim], vFull[idx*headDim:(idx+1)*headDim])
		}
		// Zero biases on fallback.
		for i := range targetLen {
			outBias[i] = 0
		}
		return nil
	}

	// atb now contains V_s [targetLen, headDim].
	copy(outVals, atb)

	// Step 7: Compute bias to preserve attention mass.
	// For each retained key j (originally at position selected[j]):
	//   bias[j] = log(col_sum_full[selected[j]] / col_sum_selected[j])
	colSumSelected := make([]float32, targetLen)
	for i := range numRefQueries {
		for j := range targetLen {
			colSumSelected[j] += aS[i*targetLen+j]
		}
	}

	for j, idx := range selected {
		if colSumSelected[j] > 0 && importance[idx] > 0 {
			outBias[j] = float32(math.Log(float64(importance[idx] / colSumSelected[j])))
		}
	}

	return nil
}

// transposeMatrix transposes a row-major [m, n] matrix to [n, m].
func transposeMatrix(src []float32, m, n int) []float32 {
	dst := make([]float32, m*n)
	for i := range m {
		for j := range n {
			dst[j*m+i] = src[i*n+j]
		}
	}
	return dst
}
