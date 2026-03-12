package models

import (
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// Model represents a loaded Hugging Face model with its weights and architecture.
type Model struct {
	// Config is the parsed model configuration.
	Config *BaseConfig

	// Builder is the architecture-specific builder.
	Builder ArchitectureBuilder

	// Weights provides access to model tensors (safetensors or GGUF).
	Weights WeightSource
}

// New creates a Model from a Hugging Face repository.
// It downloads config.json and model.safetensors, parses the config,
// and sets up the architecture builder.
func New(repo *hub.Repo) (*Model, error) {
	// Download repository info first.
	if err := repo.DownloadInfo(false); err != nil {
		return nil, errors.Wrap(err, "failed to download repo info")
	}

	// Download and parse config.json.
	configPath, err := repo.DownloadFile("config.json")
	if err != nil {
		return nil, errors.Wrap(err, "failed to download config.json")
	}

	config, err := ParseConfigFile(configPath)
	if err != nil {
		return nil, err
	}

	// Look up architecture builder.
	builder, err := NewBuilder(config.ModelType)
	if err != nil {
		return nil, err
	}

	// Parse architecture-specific config.
	if err := builder.ParseConfig(config); err != nil {
		return nil, errors.Wrapf(err, "failed to parse %s config", config.ModelType)
	}

	// Load safetensors model (handles both single-file and sharded models).
	weights, err := safetensors.New(repo)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load safetensors weights")
	}

	return &Model{
		Config:  config,
		Builder: builder,
		Weights: &SafetensorsSource{Model: weights},
	}, nil
}

// NewFromLocal creates a Model from a local directory containing config.json and model.safetensors.
// The directory should be a cached HuggingFace model directory (e.g., from a previous download).
func NewFromLocal(dir string) (*Model, error) {
	// Parse config.json.
	configPath := filepath.Join(dir, "config.json")
	config, err := ParseConfigFile(configPath)
	if err != nil {
		return nil, err
	}

	// Look up architecture builder.
	builder, err := NewBuilder(config.ModelType)
	if err != nil {
		return nil, err
	}

	// Parse architecture-specific config.
	if err := builder.ParseConfig(config); err != nil {
		return nil, errors.Wrapf(err, "failed to parse %s config", config.ModelType)
	}

	// Create a hub repo pointing to the local directory as cache.
	// Extract model ID from directory name if possible.
	modelID := filepath.Base(dir)
	repo := hub.New(modelID).WithCacheDir(filepath.Dir(dir))

	// Load safetensors model.
	weights, err := safetensors.New(repo)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load weights from %s", dir)
	}

	return &Model{
		Config:  config,
		Builder: builder,
		Weights: &SafetensorsSource{Model: weights},
	}, nil
}

// NewFromGGUF creates a Model from a local GGUF file.
// Configuration is parsed from GGUF metadata (no config.json needed).
func NewFromGGUF(path string) (*Model, error) {
	gm, err := gguf.NewFromFile(path)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load GGUF file %s", path)
	}

	config, err := ParseConfigFromGGUF(gm.File)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse config from GGUF metadata")
	}

	builder, err := NewBuilder(config.ModelType)
	if err != nil {
		return nil, err
	}

	if err := builder.ParseConfig(config); err != nil {
		return nil, errors.Wrapf(err, "failed to parse %s config", config.ModelType)
	}

	return &Model{
		Config:  config,
		Builder: builder,
		Weights: &GGUFSource{Model: gm},
	}, nil
}

// NewFromGGUFRepo creates a Model from a HuggingFace repo containing GGUF files.
// If filenames are specified, only those files are downloaded. Otherwise, all
// .gguf files in the repo are downloaded. For repos with many quantization
// variants (e.g., second-state/Llava-v1.5-7B-GGUF), specify filenames to
// avoid downloading all variants.
func NewFromGGUFRepo(repo *hub.Repo, filenames ...string) (*Model, error) {
	if err := repo.DownloadInfo(false); err != nil {
		return nil, errors.Wrap(err, "failed to download repo info")
	}

	gm := gguf.NewEmpty(repo)
	if len(filenames) > 0 {
		if err := gm.LoadFiles(filenames...); err != nil {
			return nil, errors.Wrap(err, "failed to load GGUF files from repo")
		}
	} else {
		if err := gm.LoadAll(); err != nil {
			return nil, errors.Wrap(err, "failed to load GGUF from repo")
		}
	}

	config, err := ParseConfigFromGGUF(gm.File)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse config from GGUF metadata")
	}

	// For multimodal models (e.g., LLaVA), the vision config may be in a
	// separate mmproj GGUF file with "clip" architecture.
	for _, extra := range gm.ExtraFiles() {
		mergeVisionConfigFromGGUF(config, extra)
	}

	builder, err := NewBuilder(config.ModelType)
	if err != nil {
		return nil, err
	}

	if err := builder.ParseConfig(config); err != nil {
		return nil, errors.Wrapf(err, "failed to parse %s config", config.ModelType)
	}

	return &Model{
		Config:  config,
		Builder: builder,
		Weights: &GGUFSource{Model: gm},
	}, nil
}

// LoadWeightsIntoContext loads all model weights into the given GoMLX context.
// This should be called once before building the computation graph.
func (m *Model) LoadWeightsIntoContext(ctx *context.Context) error {
	return m.Builder.LoadWeights(ctx, m.Weights)
}

// WeightMapping returns the mapping from safetensors keys to context scope paths.
func (m *Model) WeightMapping() map[string]string {
	return m.Builder.WeightMapping()
}

// Summary returns a summary of the model configuration and weights.
func (m *Model) Summary() string {
	var sb strings.Builder
	sb.WriteString("Model Summary:\n")
	sb.WriteString("  Architecture: " + m.Builder.Name() + "\n")
	sb.WriteString("  Model Type: " + m.Config.ModelType + "\n")
	sb.WriteString("  Hidden Size: " + strconv.Itoa(m.Config.HiddenSize) + "\n")
	sb.WriteString("  Num Layers: " + strconv.Itoa(m.Config.NumHiddenLayers) + "\n")
	sb.WriteString("  Num Heads: " + strconv.Itoa(m.Config.NumAttentionHeads) + "\n")
	sb.WriteString("  Vocab Size: " + strconv.Itoa(m.Config.VocabSize) + "\n")
	if m.Weights != nil {
		sb.WriteString("  Tensors: " + strconv.Itoa(len(m.Weights.ListTensorNames())) + "\n")
	}
	return sb.String()
}

