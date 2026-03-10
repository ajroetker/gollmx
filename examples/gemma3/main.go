// Command gemma3 generates text using a Gemma 3 model loaded from a GGUF file,
// using the serving engine with engine-managed KV cache.
//
// Usage:
//
//	go run ./examples/gemma3/ --gguf /path/to/gemma-3-4b-it-qat.gguf
//	go run ./examples/gemma3/ --repo google/gemma-3-4b-it-gguf
//
// The tokenizer is loaded from the HuggingFace repository (requires network on first run).
// The GGUF file provides weights and config via metadata.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/ajroetker/gollmx/kvcache"
	"github.com/ajroetker/gollmx/serving"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"

	models "github.com/ajroetker/gollmx"
	"github.com/ajroetker/gollmx/architectures/gemma3"
)

var (
	flagGGUF          = flag.String("gguf", "", "Path to local GGUF model file.")
	flagRepo          = flag.String("repo", "", "HuggingFace repo containing a GGUF model (e.g. google/gemma-3-4b-it-gguf).")
	flagTokenizerRepo = flag.String("tokenizer-repo", "google/gemma-3-4b-it", "HuggingFace repo for tokenizer.")
	flagPrompt        = flag.String("prompt", "Write a short poem about the sea.", "User message for chat prompt.")
	flagMaxTokens     = flag.Int("max-tokens", 100, "Maximum number of tokens to generate.")
	flagMaxSeqLen     = flag.Int("max-seq-len", 256, "Maximum total sequence length (prompt + generated).")
	flagBlockSize     = flag.Int("block-size", 16, "Paged KV cache block size (tokens per block).")
	flagNumBlocks     = flag.Int("num-blocks", 0, "Total paged KV cache blocks (0 = auto from max-seq-len).")
)

func main() {
	flag.Parse()
	if *flagGGUF == "" && *flagRepo == "" {
		log.Fatal("either --gguf or --repo is required")
	}

	hfToken := os.Getenv("HF_TOKEN")

	// Load GGUF model (weights + config from metadata).
	var model *models.Model
	var err error
	if *flagRepo != "" {
		fmt.Printf("Downloading GGUF model from %s...\n", *flagRepo)
		repo := hub.New(*flagRepo).WithAuth(hfToken)
		model, err = models.NewFromGGUFRepo(repo)
	} else {
		fmt.Println("Loading GGUF model...")
		model, err = models.NewFromGGUF(*flagGGUF)
	}
	if err != nil {
		log.Fatalf("Failed to load GGUF model: %v", err)
	}
	fmt.Print(model.Summary())

	// Load tokenizer from HuggingFace repo.
	fmt.Println("Loading tokenizer...")
	repo := hub.New(*flagTokenizerRepo).WithAuth(hfToken)
	tok, err := tokenizers.New(repo)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Create backend.
	backend := backends.MustNew()
	fmt.Printf("Backend: %s\n", backend.Name())

	// Load weights into context.
	fmt.Println("Loading weights into context...")
	ctx := mlctx.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	// Get the Gemma 3 builder and create ModelFn.
	builder, ok := model.Builder.(*gemma3.Builder)
	if !ok {
		log.Fatal("Model builder is not a Gemma 3 builder")
	}
	cfg := builder.Gemma3Config()
	modelFn := builder.BuildModelFn()

	// Tokenize prompt.
	chatPrompt := formatChatPrompt(*flagPrompt)
	promptTokens := tokenizePrompt(tok, chatPrompt)
	fmt.Printf("Prompt: %q\n", *flagPrompt)
	fmt.Printf("Prompt tokens: %d\n", len(promptTokens))

	if len(promptTokens) >= *flagMaxSeqLen {
		log.Fatalf("Prompt (%d tokens) exceeds max sequence length (%d)", len(promptTokens), *flagMaxSeqLen)
	}

	// Create the serving engine with paged KV cache.
	engineCfg := serving.Config{
		MaxSeqLen:    *flagMaxSeqLen,
		MaxBatchSize: 1,
	}

	eosID := int32(1) // Gemma default EOS.
	if id, err := tok.SpecialTokenID(api.TokEndOfSentence); err == nil {
		eosID = int32(id)
	}

	blockSize := *flagBlockSize
	numBlocks := *flagNumBlocks
	if numBlocks <= 0 {
		// Auto: enough blocks for maxSeqLen with some headroom.
		numBlocks = (*flagMaxSeqLen/blockSize + 1) * 2
	}

	cacheDType := dtypes.BFloat16
	if backend.Name() == "SimpleGo (go)" {
		// SimpleGo doesn't support mixed-precision DotGeneral; use Float32.
		cacheDType = dtypes.Float32
	}
	pagedCfg := kvcache.PagedKVCacheConfig{
		NumBlocks:  numBlocks,
		BlockSize:  blockSize,
		NumKVHeads: cfg.KVHeads(),
		HeadDim:    cfg.HeadDim,
		DType:      cacheDType,
	}
	fmt.Printf("Paged KV cache: %d blocks x %d tokens, %d KV heads x %d dim\n",
		numBlocks, blockSize, cfg.KVHeads(), cfg.HeadDim)

	engineTok := &servingTokenizer{tok: tok, eosID: eosID}
	engine := serving.NewPaged(
		backend, ctx, modelFn, engineTok, engineCfg, pagedCfg,
	)
	defer engine.Stop()

	// Submit the request.
	fmt.Println("\nGenerating...")
	startTime := time.Now()

	outputCh, errCh, err := engine.Submit(
		context.Background(),
		promptTokens,
		serving.RequestOptions{MaxNewTokens: *flagMaxTokens},
		nil,
	)
	if err != nil {
		log.Fatalf("Submit failed: %v", err)
	}

	// Stream output.
	numGenerated := 0
	for delta := range outputCh {
		fmt.Print(delta.Token)
		numGenerated++
		if delta.Token == "<end_of_turn>" {
			break
		}
	}

	// Check for errors.
	if err := <-errCh; err != nil {
		log.Fatalf("Generation error: %v", err)
	}

	totalDuration := time.Since(startTime)
	fmt.Println("\n\n---")
	fmt.Printf("Generated %d tokens in %.2fs (%.1f tokens/s)\n",
		numGenerated, totalDuration.Seconds(),
		float64(numGenerated)/totalDuration.Seconds())
}

// servingTokenizer wraps a HuggingFace tokenizer to implement serving.Tokenizer.
type servingTokenizer struct {
	tok   tokenizers.Tokenizer
	eosID int32
}

func (t *servingTokenizer) Decode(tokenID int32) (string, error) {
	return t.tok.Decode([]int{int(tokenID)}), nil
}

func (t *servingTokenizer) IsEOS(tokenID int32) bool {
	return tokenID == t.eosID
}

func (t *servingTokenizer) Reset() {}

// formatChatPrompt wraps a user message in the Gemma 3 chat template.
func formatChatPrompt(userMessage string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", userMessage)
}

// tokenizePrompt encodes the prompt and prepends BOS.
func tokenizePrompt(tok tokenizers.Tokenizer, prompt string) []int32 {
	bosID := 2 // Gemma default BOS.
	if id, err := tok.SpecialTokenID(api.TokBeginningOfSentence); err == nil {
		bosID = id
	}

	encoded := tok.Encode(prompt)
	tokens := make([]int32, 0, len(encoded)+1)
	tokens = append(tokens, int32(bosID))
	for _, t := range encoded {
		tokens = append(tokens, int32(t))
	}
	return tokens
}
