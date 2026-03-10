package models_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ajroetker/gollmx"

	// Import architectures to register them for testing.
	_ "github.com/ajroetker/gollmx/architectures/bert"
	_ "github.com/ajroetker/gollmx/architectures/deberta"
	_ "github.com/ajroetker/gollmx/architectures/gemma3"
	_ "github.com/ajroetker/gollmx/architectures/llama"
)

func TestParseConfigContent_BERT(t *testing.T) {
	configJSON := `{
		"model_type": "bert",
		"vocab_size": 30522,
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"num_attention_heads": 12,
		"intermediate_size": 3072,
		"hidden_act": "gelu",
		"layer_norm_eps": 1e-12,
		"max_position_embeddings": 512
	}`

	cfg, err := models.ParseConfigContent([]byte(configJSON))
	require.NoError(t, err)

	assert.Equal(t, "bert", cfg.ModelType)
	assert.Equal(t, 30522, cfg.VocabSize)
	assert.Equal(t, 768, cfg.HiddenSize)
	assert.Equal(t, 12, cfg.NumHiddenLayers)
	assert.Equal(t, 12, cfg.NumAttentionHeads)
	assert.Equal(t, 3072, cfg.IntermediateSize)
	assert.Equal(t, "gelu", cfg.HiddenAct)
	assert.Equal(t, 1e-12, cfg.LayerNormEps)
	assert.Equal(t, 64, cfg.HeadDim())
}

func TestParseConfigContent_DeBERTa(t *testing.T) {
	configJSON := `{
		"model_type": "deberta-v2",
		"vocab_size": 128100,
		"hidden_size": 768,
		"num_hidden_layers": 6,
		"num_attention_heads": 12,
		"intermediate_size": 3072,
		"hidden_act": "gelu",
		"layer_norm_eps": 1e-7,
		"relative_attention": true,
		"pos_att_type": ["c2p", "p2c"],
		"norm_rel_ebd": ["layer_norm"],
		"share_att_key": true,
		"max_relative_positions": 256
	}`

	cfg, err := models.ParseConfigContent([]byte(configJSON))
	require.NoError(t, err)

	assert.Equal(t, "deberta-v2", cfg.ModelType)
	assert.Equal(t, 128100, cfg.VocabSize)
	assert.Equal(t, 768, cfg.HiddenSize)

	// Check architecture-specific fields in Raw.
	relAttn, ok := cfg.GetBool("relative_attention")
	assert.True(t, ok)
	assert.True(t, relAttn)

	posAttType, ok := cfg.GetStringSlice("pos_att_type")
	assert.True(t, ok)
	assert.Equal(t, []string{"c2p", "p2c"}, posAttType)

	normRelEbd, ok := cfg.GetStringSlice("norm_rel_ebd")
	assert.True(t, ok)
	assert.Equal(t, []string{"layer_norm"}, normRelEbd)

	shareAttKey, ok := cfg.GetBool("share_att_key")
	assert.True(t, ok)
	assert.True(t, shareAttKey)
}

func TestParseConfigContent_Llama(t *testing.T) {
	configJSON := `{
		"model_type": "llama",
		"vocab_size": 32000,
		"hidden_size": 4096,
		"num_hidden_layers": 32,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size": 11008,
		"hidden_act": "silu",
		"rms_norm_eps": 1e-5,
		"rope_theta": 10000.0
	}`

	cfg, err := models.ParseConfigContent([]byte(configJSON))
	require.NoError(t, err)

	assert.Equal(t, "llama", cfg.ModelType)
	assert.Equal(t, 32000, cfg.VocabSize)
	assert.Equal(t, 4096, cfg.HiddenSize)

	// Check Llama-specific fields.
	ropeTheta, ok := cfg.GetFloat("rope_theta")
	assert.True(t, ok)
	assert.Equal(t, 10000.0, ropeTheta)

	rmsNormEps, ok := cfg.GetFloat("rms_norm_eps")
	assert.True(t, ok)
	assert.Equal(t, 1e-5, rmsNormEps)

	numKVHeads, ok := cfg.GetInt("num_key_value_heads")
	assert.True(t, ok)
	assert.Equal(t, 8, numKVHeads)
}

func TestParseConfigContent_Gemma3(t *testing.T) {
	configJSON := `{
		"model_type": "gemma3",
		"vocab_size": 262144,
		"hidden_size": 2560,
		"num_hidden_layers": 34,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"intermediate_size": 10240,
		"hidden_act": "gelu",
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000.0,
		"head_dim": 256,
		"sliding_window": 1024
	}`

	cfg, err := models.ParseConfigContent([]byte(configJSON))
	require.NoError(t, err)

	assert.Equal(t, "gemma3", cfg.ModelType)
	assert.Equal(t, 262144, cfg.VocabSize)
	assert.Equal(t, 2560, cfg.HiddenSize)
	assert.Equal(t, 34, cfg.NumHiddenLayers)
	assert.Equal(t, 8, cfg.NumAttentionHeads)
	assert.Equal(t, 10240, cfg.IntermediateSize)

	// Explicit head_dim should override hidden_size/num_heads.
	assert.Equal(t, 256, cfg.HeadDim())

	// Architecture-specific fields.
	headDim, ok := cfg.GetInt("head_dim")
	assert.True(t, ok)
	assert.Equal(t, 256, headDim)

	slidingWindow, ok := cfg.GetInt("sliding_window")
	assert.True(t, ok)
	assert.Equal(t, 1024, slidingWindow)

	ropeTheta, ok := cfg.GetFloat("rope_theta")
	assert.True(t, ok)
	assert.Equal(t, 1e6, ropeTheta)
}

func TestRegisteredArchitectures(t *testing.T) {
	architectures := models.ListArchitectures()
	assert.NotEmpty(t, architectures)

	// Check that our architectures are registered.
	archSet := make(map[string]bool)
	for _, a := range architectures {
		archSet[a] = true
	}

	assert.True(t, archSet["bert"], "BERT should be registered")
	assert.True(t, archSet["deberta"], "DeBERTa should be registered")
	assert.True(t, archSet["deberta-v2"], "DeBERTa v2 should be registered")
	assert.True(t, archSet["llama"], "Llama should be registered")
	assert.True(t, archSet["gemma3"], "Gemma 3 should be registered")
}

func TestNewBuilder(t *testing.T) {
	// Try creating a builder for an unknown type.
	_, err := models.NewBuilder("unknown-model-type")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported model type")
}
