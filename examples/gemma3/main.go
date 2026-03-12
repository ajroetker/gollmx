// Command gemma3 generates text using a Gemma 3 model loaded from a GGUF file,
// using the serving engine with engine-managed KV cache.
//
// Usage:
//
//	go run ./examples/gemma3/ --gguf /path/to/gemma-3-4b-it-qat.gguf
//	go run ./examples/gemma3/ --gguf /path/to/gemma-3-4b-it-qat.gguf --image photo.jpg --prompt "Describe this image"
//	go run ./examples/gemma3/ --repo google/gemma-3-4b-it-gguf
//
// The tokenizer is loaded from the HuggingFace repository (requires network on first run).
// The GGUF file provides weights and config via metadata.
package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"runtime/pprof"
	"time"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gollmx/kvcache"
	"github.com/gomlx/gollmx/serving"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"

	"golang.org/x/image/draw"

	models "github.com/gomlx/gollmx"
	"github.com/gomlx/gollmx/architectures/gemma3"
)

var (
	flagGGUF          = flag.String("gguf", "", "Path to local GGUF model file.")
	flagRepo          = flag.String("repo", "", "HuggingFace repo containing a GGUF model (e.g. google/gemma-3-4b-it-gguf).")
	flagTokenizerRepo = flag.String("tokenizer-repo", "google/gemma-3-4b-it", "HuggingFace repo for tokenizer.")
	flagPrompt        = flag.String("prompt", "Write a short poem about the sea.", "User message for chat prompt.")
	flagImage         = flag.String("image", "", "Path to an image file for multimodal input.")
	flagMaxTokens     = flag.Int("max-tokens", 100, "Maximum number of tokens to generate.")
	flagMaxSeqLen     = flag.Int("max-seq-len", 256, "Maximum total sequence length (prompt + generated).")
	flagCPUProfile    = flag.String("cpuprofile", "", "Write CPU profile to file.")
	flagFlat          = flag.Bool("flat", false, "Use flat KV cache instead of paged.")
)

func main() {
	flag.Parse()

	if *flagCPUProfile != "" {
		f, err := os.Create(*flagCPUProfile)
		if err != nil {
			log.Fatalf("Failed to create CPU profile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("Failed to start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

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
	fmt.Printf("RoPE scaling factor: %.1f\n", cfg.RopeScalingFactor)
	modelFn := builder.BuildModelFn()

	hasImage := *flagImage != ""
	if hasImage && !builder.HasVision() {
		log.Fatal("--image specified but model has no vision encoder (use a multimodal GGUF)")
	}

	eosID := int32(1) // Gemma default EOS.
	if id, err := tok.SpecialTokenID(api.TokEndOfSentence); err == nil {
		eosID = int32(id)
	}

	// Most backends (SimpleGo, XLA) don't support mixed-precision DotGeneral;
	// use Float32 for the KV cache to match the compute dtype.
	cacheDType := dtypes.Float32

	// Create serving engine.
	engineCfg := serving.Config{
		MaxSeqLen:    *flagMaxSeqLen,
		MaxBatchSize: 1,
	}
	engineTok := &servingTokenizer{tok: tok, eosID: eosID}

	var engine *serving.Engine
	if *flagFlat {
		engine = serving.NewEngine(
			backend, ctx, modelFn, engineTok, engineCfg,
			cfg.KVHeads(), cfg.HeadDim, cacheDType,
		)
		fmt.Printf("Flat KV cache: %d KV heads x %d dim\n", cfg.KVHeads(), cfg.HeadDim)
	} else {
		blockSize := 16
		maxBlocks := (*flagMaxSeqLen + blockSize - 1) / blockSize
		pagedCfg := kvcache.PagedKVCacheConfig{
			NumBlocks:  maxBlocks * engineCfg.MaxBatchSize,
			BlockSize:  blockSize,
			NumKVHeads: cfg.KVHeads(),
			HeadDim:    cfg.HeadDim,
			DType:      cacheDType,
		}
		engine = serving.NewPaged(
			backend, ctx, modelFn, engineTok, engineCfg, pagedCfg,
		)
		fmt.Printf("Paged KV cache: %d blocks x %d tokens, %d KV heads x %d dim\n",
			pagedCfg.NumBlocks, blockSize, cfg.KVHeads(), cfg.HeadDim)
	}
	defer engine.Stop()

	var auxData *serving.AuxData
	if hasImage {
		imageFeatures := runVisionEncoder(backend, ctx, builder)
		auxData = &serving.AuxData{ImageFeatures: imageFeatures}
	}

	// Tokenize prompt.
	var chatPrompt string
	if hasImage {
		chatPrompt = formatImageChatPrompt(*flagPrompt)
	} else {
		chatPrompt = formatChatPrompt(*flagPrompt)
	}
	promptTokens := tokenizePrompt(tok, chatPrompt)
	fmt.Printf("Prompt: %q\n", *flagPrompt)
	fmt.Printf("Prompt tokens: %d\n", len(promptTokens))

	if hasImage {
		// Insert image placeholder tokens (256 after projector pooling, not 4096 raw patches).
		vc := builder.VisionCfg()
		numImageTokens := vc.NumImageTokens()
		promptTokens = insertImageTokens(promptTokens, numImageTokens)
		fmt.Printf("Prompt tokens (with %d image tokens): %d\n", numImageTokens, len(promptTokens))
	}

	if len(promptTokens) >= *flagMaxSeqLen {
		log.Fatalf("Prompt (%d tokens) exceeds max sequence length (%d)", len(promptTokens), *flagMaxSeqLen)
	}

	// Submit the request.
	fmt.Println("\nGenerating...")
	startTime := time.Now()

	outputCh, errCh, err := engine.Submit(
		context.Background(),
		promptTokens,
		serving.RequestOptions{MaxNewTokens: *flagMaxTokens},
		auxData,
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

// runVisionEncoder loads an image, preprocesses it, and runs the SigLIP encoder + projector.
// Returns pre-computed image features [1, numPatches, textHiddenSize].
func runVisionEncoder(backend backends.Backend, ctx *mlctx.Context, builder *gemma3.Builder) *tensors.Tensor {
	vc := builder.VisionCfg()

	// Load and preprocess image.
	fmt.Printf("Loading image: %s\n", *flagImage)
	pixelValues := loadAndPreprocessImage(*flagImage, vc.ImageSize)

	// Build and run vision encoder + projector as a GoMLX executor.
	fmt.Println("Running vision encoder...")
	visionExec, err := mlctx.NewExec(backend, ctx.Reuse(),
		func(ctx *mlctx.Context, pixels *Node) *Node {
			features := builder.BuildVisionEncoder(ctx, pixels)
			return builder.BuildMultiModalProjector(ctx, features)
		})
	if err != nil {
		log.Fatalf("Failed to create vision executor: %v", err)
	}

	outputs, err := visionExec.Exec(pixelValues)
	if err != nil {
		log.Fatalf("Vision encoder failed: %v", err)
	}

	result := outputs[0]
	fmt.Printf("Image features: %s\n", result.Shape())
	return result
}

// loadAndPreprocessImage loads an image, resizes it, and converts to [1, C, H, W] float32
// normalized for SigLIP: rescale to [0, 1] then normalize with mean=0.5, std=0.5 → [-1, 1].
func loadAndPreprocessImage(path string, targetSize int) *tensors.Tensor {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("Failed to open image: %v", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatalf("Failed to decode image: %v", err)
	}

	// Resize to targetSize x targetSize using bilinear interpolation.
	resized := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Convert to [1, 3, H, W] float32.
	// SigLIP normalization: rescale to [0,1], then (x - 0.5) / 0.5 = 2x - 1 → [-1, 1].
	pixels := make([]float32, 3*targetSize*targetSize)
	for y := range targetSize {
		for x := range targetSize {
			r, g, b, _ := resized.At(x, y).RGBA()
			// RGBA returns [0, 65535]. Rescale to [0, 1] then normalize to [-1, 1].
			idx := y*targetSize + x
			pixels[0*targetSize*targetSize+idx] = float32(r)/65535.0*2.0 - 1.0
			pixels[1*targetSize*targetSize+idx] = float32(g)/65535.0*2.0 - 1.0
			pixels[2*targetSize*targetSize+idx] = float32(b)/65535.0*2.0 - 1.0
		}
	}

	t := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 3, targetSize, targetSize))
	t.MutableFlatData(func(data any) {
		copy(data.([]float32), pixels)
	})
	return t
}

// Gemma 3 multimodal special token IDs.
const (
	imageTokenID      = int32(262144) // <image> soft token placeholder
	startOfImageID    = int32(255999) // <start_of_image>
	endOfImageID      = int32(256000) // <end_of_image>
)

// insertImageTokens inserts <start_of_image>, numPatches image tokens, and <end_of_image>
// after the "<start_of_turn>user\n" prefix in the token sequence.
// Expected input: [BOS, <start_of_turn>, user, \n, ...]
// Output: [BOS, <start_of_turn>, user, \n, <start_of_image>, <image>×N, <end_of_image>, \n, ...]
func insertImageTokens(tokens []int32, numPatches int) []int32 {
	// Find the position after "<start_of_turn>user\n" — typically index 4.
	// We insert the image block at index 4 (after BOS + turn header).
	insertPos := 4
	if insertPos > len(tokens) {
		insertPos = 1 // fallback: after BOS
	}

	result := make([]int32, 0, len(tokens)+numPatches+3) // +start_of_image, +end_of_image, +\n
	result = append(result, tokens[:insertPos]...)
	result = append(result, startOfImageID)
	for range numPatches {
		result = append(result, imageTokenID)
	}
	result = append(result, endOfImageID)
	result = append(result, tokens[insertPos:]...)
	return result
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

// formatImageChatPrompt wraps a user message with image placeholder in the Gemma 3 chat template.
func formatImageChatPrompt(userMessage string) string {
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
