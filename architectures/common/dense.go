package common

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

// DenseWithBias applies a dense layer using pre-loaded weights.
// Expects variables "weights" and "biases" in the context scope.
// Handles 2D [batch, features] and 3D [batch, seq, features] inputs.
func DenseWithBias(ctx *context.Context, x *Node) *Node {
	g := x.Graph()

	weightsVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights")
	if weightsVar == nil {
		panic(fmt.Sprintf("DenseWithBias: missing variable 'weights' in scope %q", ctx.Scope()))
	}
	weights := weightsVar.ValueGraph(g)

	biasesVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "biases")
	if biasesVar == nil {
		panic(fmt.Sprintf("DenseWithBias: missing variable 'biases' in scope %q", ctx.Scope()))
	}
	biases := biasesVar.ValueGraph(g)

	return ApplyDenseWithBias(x, weights, biases)
}

// ApplyDenseWithBias applies a dense layer with explicit weight and bias tensors.
// weights shape: [out_features, in_features] (PyTorch convention)
func ApplyDenseWithBias(x, weights, biases *Node) *Node {
	return applyDenseWithBiasAndActivation(x, weights, biases, activations.TypeNone)
}

// applyDenseWithBiasAndActivation applies a dense layer with optional fused activation.
// weights are in PyTorch convention [out_features, in_features].
// When activation is TypeNone, this is equivalent to ApplyDenseWithBias.
func applyDenseWithBiasAndActivation(x, weights, biases *Node, activation activations.Type) *Node {
	// nn.Dense expects weight as [in_features, out_features].
	// Our weights are [out_features, in_features] (PyTorch convention), so transpose.
	wT := Transpose(weights, 0, 1)

	if activation != activations.TypeNone {
		return nn.Dense(x, wT, biases, activation)
	}
	return nn.Dense(x, wT, biases)
}

// DenseWeightOnly applies only the weight matrix (no bias).
// Expects variable "weights" in the context scope.
func DenseWeightOnly(ctx *context.Context, x *Node) *Node {
	g := x.Graph()

	weightsVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights")
	if weightsVar == nil {
		panic(fmt.Sprintf("DenseWeightOnly: missing variable 'weights' in scope %q", ctx.Scope()))
	}
	weights := weightsVar.ValueGraph(g)

	return ApplyDenseWeightOnly(x, weights)
}

// ApplyDenseWeightOnly applies a weight-only dense layer.
// weights shape: [out_features, in_features] (PyTorch convention)
func ApplyDenseWeightOnly(x, weights *Node) *Node {
	// nn.Dense expects [in_features, out_features], transpose from PyTorch convention.
	wT := Transpose(weights, 0, 1)
	return nn.Dense(x, wT, nil)
}

// DenseWithBiasAndActivation applies a dense layer with a fused activation function.
// On backends that support FusedDenseActivation, this runs as a single fused op.
// Expects variables "weights" and "biases" in the context scope.
func DenseWithBiasAndActivation(ctx *context.Context, x *Node, activation activations.Type) *Node {
	g := x.Graph()

	weightsVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights")
	if weightsVar == nil {
		panic(fmt.Sprintf("DenseWithBiasAndActivation: missing variable 'weights' in scope %q", ctx.Scope()))
	}
	weights := weightsVar.ValueGraph(g)

	biasesVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "biases")
	if biasesVar == nil {
		panic(fmt.Sprintf("DenseWithBiasAndActivation: missing variable 'biases' in scope %q", ctx.Scope()))
	}
	biases := biasesVar.ValueGraph(g)

	return applyDenseWithBiasAndActivation(x, weights, biases, activation)
}

// MLP applies a 2-layer MLP with ReLU activation.
// Expects variables in scopes "0" (first layer) and "3" (second layer),
// matching PyTorch nn.Sequential indexing with dropout at indices 1 and 2.
func MLP(ctx *context.Context, x *Node) *Node {
	// First layer with ReLU.
	x = DenseWithBiasAndActivation(ctx.In("0"), x, activations.TypeRelu)

	// Second layer.
	x = DenseWithBias(ctx.In("3"), x)

	return x
}

// MLPWithGELU applies a 2-layer MLP with GELU activation.
func MLPWithGELU(ctx *context.Context, x *Node) *Node {
	// First layer with GELU.
	x = DenseWithBiasAndActivation(ctx.In("0"), x, activations.TypeGeluApprox)

	// Second layer.
	x = DenseWithBias(ctx.In("3"), x)

	return x
}

// QuantizedDenseWeightOnly applies a quantized dense layer using GGML-format weights.
// Expects variable "weights" in the context scope as [N, bytesPerRow] Uint8.
// Unlike DenseWeightOnly, no transposition is performed — GGML weights go directly
// to the QuantizedDense executor which handles the native block layout.
func QuantizedDenseWeightOnly(ctx *context.Context, x *Node, ggmlType backends.GGMLQuantType) *Node {
	g := x.Graph()

	weightsVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights")
	if weightsVar == nil {
		panic(fmt.Sprintf("QuantizedDenseWeightOnly: missing variable 'weights' in scope %q", ctx.Scope()))
	}
	weights := weightsVar.ValueGraph(g)

	quant := &Quantization{
		Scheme:   backends.QuantGGML,
		GGMLType: ggmlType,
	}
	return nn.QuantizedDense(x, weights, quant, nil)
}
