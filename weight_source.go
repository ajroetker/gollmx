package models

import (
	"fmt"
	"strings"

	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// LoadedWeight wraps a tensor with optional quantization metadata.
// For non-quantized tensors (safetensors, native GGUF types), GGMLType is zero.
// For GGML-quantized tensors, Tensor contains raw quantized bytes as [N, bytesPerRow] Uint8,
// GGMLType indicates the quantization format, and LogicalShape gives the original [N, K] dimensions.
type LoadedWeight struct {
	Tensor       *tensors.Tensor
	GGMLType     backends.GGMLQuantType // Zero for non-quantized tensors.
	LogicalShape []int                  // Original dimensions before block packing; nil for non-quantized.
}

// IsQuantized returns true if the weight uses GGML quantization.
func (lw *LoadedWeight) IsQuantized() bool {
	return lw.GGMLType != 0
}

// WeightSource abstracts over different weight storage formats (safetensors, GGUF).
type WeightSource interface {
	// GetTensor loads a single tensor by name.
	// For quantized GGUF tensors, returns raw bytes with quantization metadata.
	GetTensor(name string) (*LoadedWeight, error)

	// ListTensorNames returns all available tensor names.
	ListTensorNames() []string
}

// SafetensorsSource adapts *safetensors.Model to the WeightSource interface.
type SafetensorsSource struct {
	Model *safetensors.Model
}

// GetTensor loads a tensor from the safetensors model.
func (s *SafetensorsSource) GetTensor(name string) (*LoadedWeight, error) {
	tn, err := s.Model.GetTensor(name)
	if err != nil {
		return nil, err
	}
	return &LoadedWeight{Tensor: tn.Tensor}, nil
}

// ListTensorNames returns all tensor names in the safetensors model.
func (s *SafetensorsSource) ListTensorNames() []string {
	return s.Model.ListTensorNames()
}

// GGUFSource adapts *gguf.Model to the WeightSource interface.
type GGUFSource struct {
	Model *gguf.Model
}

// GetTensor loads a tensor from the GGUF model.
// For supported quantized types, returns raw quantized bytes as [N, bytesPerRow] Uint8
// with GGMLType set. For native types, returns the tensor directly.
// Callers must handle quantized tensors appropriately (QuantizedDense for matmul,
// QuantizedGather for embedding lookups, etc.).
func (g *GGUFSource) GetTensor(name string) (*LoadedWeight, error) {
	info, ok := g.Model.File.GetTensorInfo(name)
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}

	// Check if this is a supported quantized type.
	ggmlType, isQuant := info.Type.ToGGMLQuantType()
	if isQuant {
		return g.getTensorQuantized(name, &info, ggmlType)
	}

	// Non-quantized: read normally.
	tn, err := g.Model.GetTensor(name)
	if err != nil {
		return nil, err
	}
	return &LoadedWeight{Tensor: tn.Tensor}, nil
}

// getTensorQuantized reads a quantized tensor as raw bytes and wraps it as [N, bytesPerRow] Uint8.
func (g *GGUFSource) getTensorQuantized(name string, info *gguf.TensorInfo, ggmlType backends.GGMLQuantType) (*LoadedWeight, error) {
	rawBytes, _, err := g.Model.GetTensorRaw(name)
	if err != nil {
		return nil, err
	}

	// GoMLXShape reverses GGUF's innermost-first dims to GoMLX's outermost-first: [N, K].
	_, dims := info.GoMLXShape()
	if len(dims) != 2 {
		return nil, fmt.Errorf("gguf: quantized tensor %q has %d dimensions, expected 2", name, len(dims))
	}
	N := dims[0]
	K := dims[1]

	vpb := ggmlType.ValuesPerBlock()
	bpb := ggmlType.BytesPerBlock()
	if K%vpb != 0 {
		return nil, fmt.Errorf("gguf: tensor %q K=%d not divisible by values-per-block=%d for %s",
			name, K, vpb, ggmlType)
	}
	bytesPerRow := (K / vpb) * bpb

	expectedSize := N * bytesPerRow
	if len(rawBytes) != expectedSize {
		return nil, fmt.Errorf("gguf: tensor %q raw size %d != expected %d (N=%d, bytesPerRow=%d)",
			name, len(rawBytes), expectedSize, N, bytesPerRow)
	}

	// Create Uint8 tensor with raw quantized bytes.
	rawTensor := tensors.FromShape(shapes.Make(dtypes.Uint8, N, bytesPerRow))
	rawTensor.MutableBytes(func(data []byte) {
		copy(data, rawBytes)
	})

	return &LoadedWeight{
		Tensor:       rawTensor,
		GGMLType:     ggmlType,
		LogicalShape: dims,
	}, nil
}

// ListTensorNames returns all tensor names in the GGUF model.
func (g *GGUFSource) ListTensorNames() []string {
	return g.Model.ListTensorNames()
}

// QuantInfo maps context scope paths to their GGML quantization type.
// Only contains entries for quantized tensors.
type QuantInfo map[string]backends.GGMLQuantType

// LoadWeightsFromMapping loads weights from a WeightSource into a GoMLX context
// using the given mapping from tensor names to context scope paths.
// Missing tensors (not found errors) are silently skipped.
// Returns quantization info for any quantized tensors that were loaded.
func LoadWeightsFromMapping(weights WeightSource, mapping map[string]string, ctx *context.Context) (QuantInfo, error) {
	quantInfo := make(QuantInfo)

	for tensorKey, scopePath := range mapping {
		lw, err := weights.GetTensor(tensorKey)
		if err != nil {
			// Skip missing weights.
			if strings.Contains(err.Error(), "not found") {
				continue
			}
			return nil, fmt.Errorf("failed to load tensor %q: %w", tensorKey, err)
		}

		// Navigate to the right scope and create variable.
		scopeParts := strings.Split(scopePath, "/")
		varCtx := ctx
		for _, part := range scopeParts[:len(scopeParts)-1] {
			varCtx = varCtx.In(part)
		}
		varName := scopeParts[len(scopeParts)-1]
		varCtx.VariableWithValue(varName, lw.Tensor)

		if lw.IsQuantized() {
			quantInfo[scopePath] = lw.GGMLType
		}
	}

	return quantInfo, nil
}
