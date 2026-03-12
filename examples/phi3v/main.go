// Command phi3v generates text using a Phi-3-Vision model loaded from a GGUF file,
// using the serving engine with paged KV cache.
//
// Usage:
//
//	go run ./examples/phi3v/ --gguf /path/to/phi-3-vision.gguf
//	go run ./examples/phi3v/ --gguf /path/to/phi-3-vision.gguf --image photo.jpg --prompt "Describe this image"
//	go run ./examples/phi3v/ --repo microsoft/Phi-3-vision-128k-instruct-gguf
//
// The tokenizer is loaded from the HuggingFace repository (requires network on first run).
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
	"github.com/gomlx/gollmx/architectures/phi3"
)

var (
	flagGGUF          = flag.String("gguf", "", "Path to local GGUF model file.")
	flagRepo          = flag.String("repo", "", "HuggingFace repo containing a GGUF model.")
	flagTokenizerRepo = flag.String("tokenizer-repo", "microsoft/Phi-3-vision-128k-instruct", "HuggingFace repo for tokenizer.")
	flagPrompt        = flag.String("prompt", "Write a short poem about the sea.", "User message.")
	flagImage         = flag.String("image", "", "Path to an image file for multimodal input.")
	flagMaxTokens     = flag.Int("max-tokens", 100, "Maximum number of tokens to generate.")
	flagMaxSeqLen     = flag.Int("max-seq-len", 256, "Maximum total sequence length (prompt + generated).")
	flagFlat          = flag.Bool("flat", false, "Use flat KV cache instead of paged.")
)

func main() {
	flag.Parse()

	if *flagGGUF == "" && *flagRepo == "" {
		log.Fatal("either --gguf or --repo is required")
	}

	hfToken := os.Getenv("HF_TOKEN")

	// Load GGUF model.
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

	// Load tokenizer.
	fmt.Println("Loading tokenizer...")
	repo := hub.New(*flagTokenizerRepo).WithAuth(hfToken)
	tok, err := tokenizers.New(repo)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	backend := backends.MustNew()
	fmt.Printf("Backend: %s\n", backend.Name())

	// Load weights.
	fmt.Println("Loading weights into context...")
	ctx := mlctx.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	builder, ok := model.Builder.(*phi3.Builder)
	if !ok {
		log.Fatal("Model builder is not a Phi-3 builder")
	}
	cfg := builder.Phi3Config()
	imageTokenID = builder.ImageTokenID()
	modelFn := builder.BuildModelFn()

	hasImage := *flagImage != ""
	if hasImage && !builder.HasVision() {
		log.Fatal("--image specified but model has no vision encoder (use a multimodal GGUF)")
	}

	eosID := int32(model.Config.EOSTokenID())
	if id, err := tok.SpecialTokenID(api.TokEndOfSentence); err == nil {
		eosID = int32(id)
	}

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
			cfg.KVHeads(), cfg.KVHeadDim(), cacheDType,
		)
		fmt.Printf("Flat KV cache: %d KV heads x %d dim\n", cfg.KVHeads(), cfg.KVHeadDim())
	} else {
		blockSize := 16
		maxBlocks := (*flagMaxSeqLen + blockSize - 1) / blockSize
		pagedCfg := kvcache.PagedKVCacheConfig{
			NumBlocks:  maxBlocks * engineCfg.MaxBatchSize,
			BlockSize:  blockSize,
			NumKVHeads: cfg.KVHeads(),
			HeadDim:    cfg.KVHeadDim(),
			DType:      cacheDType,
		}
		engine = serving.NewPaged(
			backend, ctx, modelFn, engineTok, engineCfg, pagedCfg,
		)
		fmt.Printf("Paged KV cache: %d blocks x %d tokens, %d KV heads x %d dim\n",
			pagedCfg.NumBlocks, blockSize, cfg.KVHeads(), cfg.KVHeadDim())
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
		numImageTokens := builder.NumImageTokens()
		promptTokens = insertImageTokens(promptTokens, numImageTokens)
		fmt.Printf("Prompt tokens (with %d image tokens): %d\n", numImageTokens, len(promptTokens))
	}

	if len(promptTokens) >= *flagMaxSeqLen {
		log.Fatalf("Prompt (%d tokens) exceeds max sequence length (%d)", len(promptTokens), *flagMaxSeqLen)
	}

	// Generate.
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

	numGenerated := 0
	for delta := range outputCh {
		fmt.Print(delta.Token)
		numGenerated++
	}

	if err := <-errCh; err != nil {
		log.Fatalf("Generation error: %v", err)
	}

	totalDuration := time.Since(startTime)
	fmt.Println("\n\n---")
	fmt.Printf("Generated %d tokens in %.2fs (%.1f tokens/s)\n",
		numGenerated, totalDuration.Seconds(),
		float64(numGenerated)/totalDuration.Seconds())
}

// runVisionEncoder loads an image, preprocesses it, and runs the CLIP encoder + projector.
func runVisionEncoder(backend backends.Backend, ctx *mlctx.Context, builder *phi3.Builder) *tensors.Tensor {
	vc := builder.VisionCfg()

	fmt.Printf("Loading image: %s\n", *flagImage)
	pixelValues := loadAndPreprocessImage(*flagImage, vc.ImageSize)

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
// normalized for CLIP: ImageNet mean=[0.48145466, 0.4578275, 0.40821073],
// std=[0.26862954, 0.26130258, 0.27577711].
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

	resized := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	mean := [3]float32{0.48145466, 0.4578275, 0.40821073}
	std := [3]float32{0.26862954, 0.26130258, 0.27577711}

	pixels := make([]float32, 3*targetSize*targetSize)
	for y := range targetSize {
		for x := range targetSize {
			r, g, b, _ := resized.At(x, y).RGBA()
			idx := y*targetSize + x
			pixels[0*targetSize*targetSize+idx] = (float32(r)/65535.0 - mean[0]) / std[0]
			pixels[1*targetSize*targetSize+idx] = (float32(g)/65535.0 - mean[1]) / std[1]
			pixels[2*targetSize*targetSize+idx] = (float32(b)/65535.0 - mean[2]) / std[2]
		}
	}

	t := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 3, targetSize, targetSize))
	t.MutableFlatData(func(data any) {
		copy(data.([]float32), pixels)
	})
	return t
}

// imageTokenID is set from the model's tokenizer during initialization.
var imageTokenID int32

// insertImageTokens finds the first occurrence of the image token and expands it
// to numPatches copies. If not found, inserts after BOS.
func insertImageTokens(tokens []int32, numPatches int) []int32 {
	for i, t := range tokens {
		if t == imageTokenID {
			result := make([]int32, 0, len(tokens)+numPatches-1)
			result = append(result, tokens[:i]...)
			for range numPatches {
				result = append(result, imageTokenID)
			}
			result = append(result, tokens[i+1:]...)
			return result
		}
	}

	// Fallback: insert after BOS.
	result := make([]int32, 0, len(tokens)+numPatches)
	result = append(result, tokens[0])
	for range numPatches {
		result = append(result, imageTokenID)
	}
	result = append(result, tokens[1:]...)
	return result
}

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

// formatChatPrompt wraps a user message in the Phi-3 chat template.
func formatChatPrompt(userMessage string) string {
	return fmt.Sprintf("<|user|>\n%s<|end|>\n<|assistant|>\n", userMessage)
}

// formatImageChatPrompt wraps a user message with image placeholder.
func formatImageChatPrompt(userMessage string) string {
	return fmt.Sprintf("<|user|>\n<|image_1|>\n%s<|end|>\n<|assistant|>\n", userMessage)
}

func tokenizePrompt(tok tokenizers.Tokenizer, prompt string) []int32 {
	bosID := 1 // Phi-3 default BOS.
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
