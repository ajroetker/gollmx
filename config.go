// Package models provides support for loading and running Hugging Face models in GoMLX.
//
// It parses config.json files to understand model architectures and loads weights
// from safetensors format into GoMLX contexts.
package models

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/pkg/errors"
)

// BaseConfig contains fields common to all Hugging Face models.
// Architecture-specific fields are available in Raw for custom parsing.
type BaseConfig struct {
	// Path to the config file (not from JSON).
	ConfigFile string `json:"-"`

	// Core architecture identifier.
	ModelType     string   `json:"model_type"`
	Architectures []string `json:"architectures,omitempty"`

	// Common dimensions.
	VocabSize             int `json:"vocab_size"`
	HiddenSize            int `json:"hidden_size"`
	NumHiddenLayers       int `json:"num_hidden_layers"`
	NumAttentionHeads     int `json:"num_attention_heads"`
	IntermediateSize      int `json:"intermediate_size"`
	MaxPositionEmbeddings int `json:"max_position_embeddings"`

	// Normalization.
	LayerNormEps float64 `json:"layer_norm_eps,omitempty"`
	RMSNormEps   float64 `json:"rms_norm_eps,omitempty"`

	// Activation function.
	HiddenAct string `json:"hidden_act,omitempty"`

	// Dropout (used during training).
	HiddenDropoutProb     float64 `json:"hidden_dropout_prob,omitempty"`
	AttentionProbsDropout float64 `json:"attention_probs_dropout_prob,omitempty"`

	// Type embeddings (BERT-style).
	TypeVocabSize int `json:"type_vocab_size,omitempty"`

	// The raw JSON for architecture-specific parsing.
	Raw map[string]interface{} `json:"-"`
}

// ParseConfigFile loads and parses a config.json file.
func ParseConfigFile(filePath string) (*BaseConfig, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read config file %q", filePath)
	}

	config, err := ParseConfigContent(content)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse config file %q", filePath)
	}
	config.ConfigFile = filePath

	return config, nil
}

// ParseConfigContent parses config.json content from bytes.
func ParseConfigContent(content []byte) (*BaseConfig, error) {
	config := &BaseConfig{}

	// First unmarshal into the struct for common fields.
	if err := json.Unmarshal(content, config); err != nil {
		return nil, errors.Wrap(err, "failed to unmarshal config JSON")
	}

	// Also unmarshal into Raw for architecture-specific fields.
	if err := json.Unmarshal(content, &config.Raw); err != nil {
		return nil, errors.Wrap(err, "failed to unmarshal config JSON to raw map")
	}

	// Apply defaults.
	if config.LayerNormEps == 0 {
		config.LayerNormEps = 1e-12 // Common default
	}
	if config.HiddenAct == "" {
		config.HiddenAct = "gelu"
	}

	return config, nil
}

// GetString retrieves a string field from Raw config.
func (c *BaseConfig) GetString(key string) (string, bool) {
	if v, ok := c.Raw[key]; ok {
		if s, ok := v.(string); ok {
			return s, true
		}
	}
	return "", false
}

// GetInt retrieves an integer field from Raw config.
func (c *BaseConfig) GetInt(key string) (int, bool) {
	if v, ok := c.Raw[key]; ok {
		switch n := v.(type) {
		case float64:
			return int(n), true
		case int:
			return n, true
		}
	}
	return 0, false
}

// GetFloat retrieves a float field from Raw config.
func (c *BaseConfig) GetFloat(key string) (float64, bool) {
	if v, ok := c.Raw[key]; ok {
		if f, ok := v.(float64); ok {
			return f, true
		}
	}
	return 0, false
}

// GetBool retrieves a boolean field from Raw config.
func (c *BaseConfig) GetBool(key string) (bool, bool) {
	if v, ok := c.Raw[key]; ok {
		if b, ok := v.(bool); ok {
			return b, true
		}
	}
	return false, false
}

// GetStringSlice retrieves a string slice from Raw config.
func (c *BaseConfig) GetStringSlice(key string) ([]string, bool) {
	if v, ok := c.Raw[key]; ok {
		if arr, ok := v.([]interface{}); ok {
			result := make([]string, 0, len(arr))
			for _, item := range arr {
				if s, ok := item.(string); ok {
					result = append(result, s)
				}
			}
			return result, true
		}
	}
	return nil, false
}

// HeadDim returns the dimension of each attention head.
// If an explicit head_dim is set in Raw config, it is used (e.g., Gemma 3).
// Otherwise falls back to hidden_size / num_attention_heads.
func (c *BaseConfig) HeadDim() int {
	if v, ok := c.GetInt("head_dim"); ok && v > 0 {
		return v
	}
	if c.NumAttentionHeads == 0 {
		return 0
	}
	return c.HiddenSize / c.NumAttentionHeads
}

// ParseConfigFromGGUF creates a BaseConfig from GGUF file metadata.
// GGUF files embed configuration as metadata key-value pairs using the pattern
// "{architecture}.{field}" (e.g., "gemma3.block_count", "llama.embedding_length").
func ParseConfigFromGGUF(f *gguf.File) (*BaseConfig, error) {
	arch := f.Architecture()
	if arch == "" {
		return nil, fmt.Errorf("GGUF file missing general.architecture metadata")
	}

	config := &BaseConfig{
		ModelType: arch,
		Raw:       make(map[string]interface{}),
	}

	// Helper to read GGUF uint metadata as int.
	getInt := func(key string) (int, bool) {
		kv, ok := f.GetKeyValue(key)
		if !ok {
			return 0, false
		}
		return int(kv.Uint64()), true
	}

	// Helper to read GGUF float metadata.
	getFloat := func(key string) (float64, bool) {
		kv, ok := f.GetKeyValue(key)
		if !ok {
			return 0, false
		}
		return kv.Float64(), true
	}

	// Core dimensions.
	if v, ok := getInt(arch + ".block_count"); ok {
		config.NumHiddenLayers = v
	}
	if v, ok := getInt(arch + ".embedding_length"); ok {
		config.HiddenSize = v
	}
	if v, ok := getInt(arch + ".attention.head_count"); ok {
		config.NumAttentionHeads = v
	}
	if v, ok := getInt(arch + ".feed_forward_length"); ok {
		config.IntermediateSize = v
	}
	if v, ok := getInt(arch + ".context_length"); ok {
		config.MaxPositionEmbeddings = v
	}

	// Vocab size from tokenizer metadata.
	if kv, ok := f.GetKeyValue("tokenizer.ggml.tokens"); ok {
		config.VocabSize = len(kv.Strings())
	}

	// Normalization epsilon.
	if v, ok := getFloat(arch + ".attention.layer_norm_rms_epsilon"); ok {
		config.RMSNormEps = v
	}

	// Architecture-specific fields stored in Raw for builder parsing.
	if v, ok := getInt(arch + ".attention.head_count_kv"); ok {
		config.Raw["num_key_value_heads"] = float64(v)
	}
	if v, ok := getInt(arch + ".attention.key_length"); ok {
		config.Raw["head_dim"] = float64(v)
	}
	if v, ok := getFloat(arch + ".rope.freq_base"); ok {
		config.Raw["rope_theta"] = v
	}
	if v, ok := getFloat(arch + ".rope.scaling.factor"); ok {
		config.Raw["rope_scaling.factor"] = v
	}
	if v, ok := getInt(arch + ".attention.sliding_window"); ok {
		config.Raw["sliding_window"] = float64(v)
	}
	if v, ok := getFloat(arch + ".attention.layer_norm_rms_epsilon"); ok {
		config.Raw["rms_norm_eps"] = v
	}

	// Vision encoder metadata (e.g., SigLIP for Gemma 3 multimodal).
	if v, ok := getInt(arch + ".vision.block_count"); ok {
		config.Raw["vision.block_count"] = float64(v)
	}
	if v, ok := getInt(arch + ".vision.embedding_length"); ok {
		config.Raw["vision.embedding_length"] = float64(v)
	}
	if v, ok := getInt(arch + ".vision.attention.head_count"); ok {
		config.Raw["vision.attention.head_count"] = float64(v)
	}
	if v, ok := getInt(arch + ".vision.feed_forward_length"); ok {
		config.Raw["vision.feed_forward_length"] = float64(v)
	}
	if v, ok := getInt(arch + ".vision.image_size"); ok {
		config.Raw["vision.image_size"] = float64(v)
	}
	if v, ok := getInt(arch + ".vision.patch_size"); ok {
		config.Raw["vision.patch_size"] = float64(v)
	}
	if v, ok := getInt(arch + ".vision.num_channels"); ok {
		config.Raw["vision.num_channels"] = float64(v)
	}
	if v, ok := getFloat(arch + ".vision.attention.layer_norm_epsilon"); ok {
		config.Raw["vision.attention.layer_norm_epsilon"] = v
	}

	return config, nil
}
