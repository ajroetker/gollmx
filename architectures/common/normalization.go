// Package common provides shared components for transformer architectures.
package common

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

// LayerNorm applies layer normalization using pre-loaded weights.
// Expects variables "gain" and "offset" in the context scope.
func LayerNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	g := x.Graph()

	gainVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "gain")
	if gainVar == nil {
		panic(fmt.Sprintf("LayerNorm: missing variable 'gain' in scope %q", ctx.Scope()))
	}
	gain := gainVar.ValueGraph(g)

	offsetVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "offset")
	if offsetVar == nil {
		panic(fmt.Sprintf("LayerNorm: missing variable 'offset' in scope %q", ctx.Scope()))
	}
	offset := offsetVar.ValueGraph(g)

	return ApplyLayerNormWithParams(x, gain, offset, epsilon)
}

// ApplyLayerNormWithParams applies layer normalization with explicit parameters.
// Uses nn.LayerNorm which handles fused dispatch internally.
func ApplyLayerNormWithParams(x, gain, offset *Node, epsilon float64) *Node {
	return nn.LayerNorm(x, []int{-1}, epsilon, gain, offset, nil)
}

// RMSNorm applies root mean square layer normalization (used by Llama).
// Expects variable "weight" in the context scope.
func RMSNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	g := x.Graph()

	weightVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weight")
	if weightVar == nil {
		panic(fmt.Sprintf("RMSNorm: missing variable 'weight' in scope %q", ctx.Scope()))
	}
	weight := weightVar.ValueGraph(g)

	return ApplyRMSNormWithParams(x, weight, epsilon)
}

// ApplyRMSNormWithParams applies RMS normalization with explicit weight parameter.
func ApplyRMSNormWithParams(x, weight *Node, epsilon float64) *Node {
	// RMS = sqrt(mean(x^2))
	variance := ReduceAndKeep(Square(x), ReduceMean, -1)
	eps := ConstAs(x, epsilon)
	rms := Sqrt(Add(variance, eps))
	normalized := Div(x, rms)

	// Convert weight dtype to match x if needed (e.g. Float16 weights with Float32 input).
	if weight.DType() != x.DType() {
		weight = ConvertDType(weight, x.DType())
	}

	// Reshape weight to broadcast with x.
	xRank := x.Shape().Rank()
	broadcastShape := make([]int, xRank)
	for i := range broadcastShape {
		broadcastShape[i] = 1
	}
	broadcastShape[xRank-1] = weight.Shape().Dimensions[0]

	weight = Reshape(weight, broadcastShape...)

	return Mul(normalized, weight)
}
