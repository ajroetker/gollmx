package main

import (
	gocontext "context"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/gomlx/pkg/ml/nn"

	models "github.com/ajroetker/huggingface-gomlx"
	"github.com/ajroetker/huggingface-gomlx/architectures/common"
	"github.com/ajroetker/huggingface-gomlx/architectures/gemma3"
	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/ajroetker/huggingface-gomlx/kvcache"
	"github.com/ajroetker/huggingface-gomlx/serving"
)

// loadModel loads the GGUF model and returns the builder and context.
func loadModel(t *testing.T) (*gemma3.Builder, *context.Context, backends.Backend) {
	t.Helper()
	ggufPath := getGGUFPath()
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF file not found, skipping")
	}

	model, err := models.NewFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	backend := backends.MustNew()
	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	builder, ok := model.Builder.(*gemma3.Builder)
	if !ok {
		t.Fatal("Not a Gemma 3 builder")
	}
	return builder, ctx, backend
}

// TestSingleLayerHiddenStates runs embeddings → 1 decoder layer and checks
// the hidden states are reasonable (non-zero, varying, good range).
func TestSingleLayerHiddenStates(t *testing.T) {
	builder, ctx, backend := loadModel(t)
	cfg := builder.Gemma3Config()

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, inputIDs *Node) *Node {
		g := inputIDs.Graph()
		hidden := builder.BuildEmbeddings(ctx, inputIDs)
		batchSize := inputIDs.Shape().Dimensions[0]
		seqLen := inputIDs.Shape().Dimensions[1]
		positionIDs := common.GetPositionIDs(g, batchSize, seqLen)
		hidden = builder.BuildDecoderLayer(ctx.In("layers").In("0"), hidden, positionIDs, 0)
		return hidden
	})
	if err != nil {
		t.Fatalf("NewExec: %v", err)
	}

	tokens := []int32{2, 100, 5000} // BOS + token 100 + token 5000
	tokensTensor := tensors.FromFlatDataAndDimensions(tokens, 1, 3)

	t.Log("Running embeddings + 1 decoder layer...")
	start := time.Now()
	results := exec.MustExec(tokensTensor)
	t.Logf("Took %s", time.Since(start))
	defer results[0].FinalizeAll()

	t.Logf("Output shape: %v", results[0].Shape())

	var hiddenValues []float32
	results[0].ConstFlatData(func(flat any) {
		hiddenValues = flat.([]float32)
	})

	hiddenSize := cfg.HiddenSize
	numPositions := 3

	for pos := 0; pos < numPositions; pos++ {
		row := hiddenValues[pos*hiddenSize : (pos+1)*hiddenSize]
		var minV, maxV float32 = math.MaxFloat32, -math.MaxFloat32
		var sum, sumSq float64
		nonZero := 0
		for _, v := range row {
			if v != 0 {
				nonZero++
			}
			sum += float64(v)
			sumSq += float64(v) * float64(v)
			if v < minV {
				minV = v
			}
			if v > maxV {
				maxV = v
			}
		}
		mean := sum / float64(hiddenSize)
		std := math.Sqrt(sumSq/float64(hiddenSize) - mean*mean)
		t.Logf("Position %d: range=[%.6f, %.6f], mean=%.6f, std=%.6f, nonZero=%d/%d, first5=%v",
			pos, minV, maxV, mean, std, nonZero, hiddenSize, row[:5])

		if nonZero < hiddenSize/2 {
			t.Errorf("Position %d: too many zeros (%d/%d)", pos, hiddenSize-nonZero, hiddenSize)
		}
		if maxV-minV < 0.01 {
			t.Errorf("Position %d: range too small (%.6f)", pos, maxV-minV)
		}
	}

	// Check that positions produce different hidden states.
	row0 := hiddenValues[0:hiddenSize]
	row1 := hiddenValues[hiddenSize : 2*hiddenSize]
	var diff float64
	for i := range hiddenSize {
		d := float64(row0[i] - row1[i])
		diff += d * d
	}
	rmsDiff := math.Sqrt(diff / float64(hiddenSize))
	t.Logf("RMS diff between pos 0 and pos 1: %.6f", rmsDiff)
	if rmsDiff < 0.001 {
		t.Error("Hidden states are essentially identical across positions")
	}
}

// TestTiedEmbeddingLMHead tests the LM head using tied embeddings.
// This verifies that embedding → LM head (skip decoder) produces non-trivial logits.
func TestTiedEmbeddingLMHead(t *testing.T) {
	builder, ctx, backend := loadModel(t)

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, inputIDs *Node) *Node {
		hidden := builder.BuildEmbeddings(ctx, inputIDs)

		// Manually apply tied embedding LM head.
		// This replicates applyLMHead's quantized tied embedding path.
		embCtx := ctx.In("embeddings")
		embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
		if embVar == nil {
			panic("embeddings not found")
		}
		g := inputIDs.Graph()
		embWeights := embVar.ValueGraph(g)

		quant := &Quantization{
			Scheme:   backends.QuantGGML,
			GGMLType: backends.GGMLQ6_K,
		}
		return nn.QuantizedDense(hidden, embWeights, quant, nil)
	})
	if err != nil {
		t.Fatalf("NewExec: %v", err)
	}

	tokens := []int32{2, 100} // BOS + token 100
	tokensTensor := tensors.FromFlatDataAndDimensions(tokens, 1, 2)

	t.Log("Running embedding → LM head (no decoder)...")
	start := time.Now()
	results := exec.MustExec(tokensTensor)
	t.Logf("Took %s", time.Since(start))
	defer results[0].FinalizeAll()

	t.Logf("Output shape: %v", results[0].Shape())

	var logits []float32
	results[0].ConstFlatData(func(flat any) {
		logits = flat.([]float32)
	})

	vocabSize := results[0].Shape().Dimensions[len(results[0].Shape().Dimensions)-1]
	numPositions := 2

	for pos := 0; pos < numPositions; pos++ {
		posLogits := logits[pos*vocabSize : (pos+1)*vocabSize]

		// Find top 10 tokens.
		type tokenScore struct {
			id    int
			score float32
		}
		top := make([]tokenScore, vocabSize)
		for i := range top {
			top[i] = tokenScore{i, posLogits[i]}
		}
		sort.Slice(top, func(i, j int) bool {
			return top[i].score > top[j].score
		})

		t.Logf("Position %d - Top 10 tokens:", pos)
		for i := 0; i < 10 && i < len(top); i++ {
			t.Logf("  Token %6d: %.4f", top[i].id, top[i].score)
		}

		// Check logits have reasonable range.
		var minL, maxL float32 = posLogits[0], posLogits[0]
		for _, v := range posLogits {
			if v < minL {
				minL = v
			}
			if v > maxL {
				maxL = v
			}
		}
		t.Logf("Position %d - Logit range: [%.4f, %.4f]", pos, minL, maxL)

		if maxL-minL < 1.0 {
			t.Errorf("Position %d: logits nearly flat (range=%.4f)", pos, maxL-minL)
		}
	}
}

// TestModelFnWithFlatCache tests the BuildModelFn with a flat KV cache accessor.
// This exercises the attentionWithKV path (used by the serving engine) but with
// the simpler flat cache instead of paged cache.
func TestModelFnWithFlatCache(t *testing.T) {
	builder, ctx, backend := loadModel(t)
	cfg := builder.Gemma3Config()
	modelFn := builder.BuildModelFn()

	maxSeqLen := 64

	// Prefill: process full prompt with KV cache.
	// Use Reuse().Checked(false) like the serving engine does — this allows
	// KV cache variables to be created alongside existing weight variables.
	exec, err := context.NewExec(backend, ctx.Reuse().Checked(false), func(ctx *context.Context, tokens *Node, positions *Node) *Node {
		kv := attention.NewFlatKVCacheAccessor(
			1, cfg.KVHeads(), maxSeqLen, cfg.HeadDim, dtypes.Float32, positions,
		)
		logits := modelFn(ctx, tokens, positions, kv, nil)
		// Extract last position logits.
		lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
		return Squeeze(lastLogits, 1) // [vocabSize]
	})
	if err != nil {
		t.Fatalf("NewExec: %v", err)
	}

	// Use the 18-token prompt to compare flat vs paged cache behavior.
	// Correct Gemma 3 tokenizer IDs from llama-tokenize.
	tokens := []int32{2, 105, 2430, 107, 6974, 496, 2822, 27355, 1003, 506, 5442, 236761, 106, 236743, 107, 105, 2028, 107}
	tokensTensor := tensors.FromFlatDataAndDimensions(tokens, 1, len(tokens))
	positionsTensor := tensors.FromFlatDataAndDimensions([]int32{0}, 1) // starting position = 0

	t.Logf("Running ModelFn with flat KV cache (prefill), prompt length=%d...", len(tokens))
	start := time.Now()
	results := exec.MustExec(tokensTensor, positionsTensor)
	t.Logf("Took %s", time.Since(start))
	defer results[0].FinalizeAll()

	t.Logf("Output shape: %v", results[0].Shape())

	var logits []float32
	results[0].ConstFlatData(func(flat any) {
		logits = flat.([]float32)
	})

	type tokenScore struct {
		id    int
		score float32
	}
	top := make([]tokenScore, len(logits))
	for i := range top {
		top[i] = tokenScore{i, logits[i]}
	}
	sort.Slice(top, func(i, j int) bool {
		return top[i].score > top[j].score
	})

	t.Log("Top 20 tokens (ModelFn + flat KV cache):")
	for i := 0; i < 20 && i < len(top); i++ {
		t.Logf("  Token %6d: %.4f", top[i].id, top[i].score)
	}

	var minL, maxL float32 = logits[0], logits[0]
	for _, v := range logits {
		if v < minL {
			minL = v
		}
		if v > maxL {
			maxL = v
		}
	}
	t.Logf("Logit range: [%.4f, %.4f]", minL, maxL)

	if maxL-minL < 1.0 {
		t.Error("Logits are nearly flat - attentionWithKV is not working correctly")
	}
}

// TestModelFnWithPagedCache tests the BuildModelFn with a paged KV cache accessor.
// This is the same path the serving engine uses.
func TestModelFnWithPagedCache(t *testing.T) {
	builder, ctx, backend := loadModel(t)
	cfg := builder.Gemma3Config()
	modelFn := builder.BuildModelFn()

	blockSize := 16
	numBlocks := 10 // total physical blocks
	maxBlocksPerReq := 2 // enough for 32 tokens

	pagedCfg := kvcache.PagedKVCacheConfig{
		NumBlocks:  numBlocks,
		BlockSize:  blockSize,
		NumKVHeads: cfg.KVHeads(),
		HeadDim:    cfg.HeadDim,
		DType:      dtypes.Float32,
	}

	// Allocate 1 block for the 2-token prompt.
	blockMgr := kvcache.NewBlockManager(pagedCfg)
	if err := blockMgr.EnsureBlocks(1, 2); err != nil {
		t.Fatalf("EnsureBlocks: %v", err)
	}
	pt := blockMgr.GetPageTable(1)
	t.Logf("Page table: %v", pt)

	// Build page table tensor [1, maxBlocksPerReq].
	ptRow := make([]int32, maxBlocksPerReq)
	for i, b := range pt {
		if i >= maxBlocksPerReq {
			break
		}
		ptRow[i] = int32(b)
	}
	ptTensor := tensors.FromValue([][]int32{ptRow})

	exec, err := context.NewExec(backend, ctx.Reuse().Checked(false),
		func(ctx *context.Context, tokens *Node, positions *Node, pageTables *Node) *Node {
			kv := &kvcache.PagedKVCacheAccessor{
				Config:        pagedCfg,
				PageTables:    pageTables,
				Positions:     positions,
				ReadNumBlocks: maxBlocksPerReq,
			}
			logits := modelFn(ctx, tokens, positions, kv, nil)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			return Squeeze(lastLogits, 1)
		},
	)
	if err != nil {
		t.Fatalf("NewExec: %v", err)
	}

	tokens := []int32{2, 26352} // BOS + Hello
	tokensTensor := tensors.FromFlatDataAndDimensions(tokens, 1, len(tokens))
	positionsTensor := tensors.FromFlatDataAndDimensions([]int32{0}, 1)

	t.Log("Running ModelFn with paged KV cache (prefill)...")
	start := time.Now()
	results := exec.MustExec(tokensTensor, positionsTensor, ptTensor)
	t.Logf("Took %s", time.Since(start))
	defer results[0].FinalizeAll()

	t.Logf("Output shape: %v", results[0].Shape())

	var logits []float32
	results[0].ConstFlatData(func(flat any) {
		logits = flat.([]float32)
	})

	type tokenScore struct {
		id    int
		score float32
	}
	top := make([]tokenScore, len(logits))
	for i := range top {
		top[i] = tokenScore{i, logits[i]}
	}
	sort.Slice(top, func(i, j int) bool {
		return top[i].score > top[j].score
	})

	t.Log("Top 20 tokens (ModelFn + paged KV cache):")
	for i := 0; i < 20 && i < len(top); i++ {
		t.Logf("  Token %6d: %.4f", top[i].id, top[i].score)
	}

	var minL, maxL float32 = logits[0], logits[0]
	for _, v := range logits {
		if v < minL {
			minL = v
		}
		if v > maxL {
			maxL = v
		}
	}
	t.Logf("Logit range: [%.4f, %.4f]", minL, maxL)

	if maxL-minL < 1.0 {
		t.Error("Logits are nearly flat - paged KV cache is not working correctly")
	}

	// Compare with flat cache result: top token should be 236743 (space) for BOS+Hello.
	if top[0].id != 236743 {
		t.Errorf("Expected top token 236743 (space), got %d (score %.4f)", top[0].id, top[0].score)
	}
}

// TestFullForwardPass runs the complete forward pass (all layers) on a short prompt.
// This is slow but definitive - if logits are reasonable, the architecture is correct
// and the bug is in the serving engine / KV cache integration.
func TestFullForwardPass(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping slow full forward pass in short mode")
	}

	builder, ctx, backend := loadModel(t)

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, inputIDs *Node) *Node {
		return builder.Forward(ctx, inputIDs, nil)
	})
	if err != nil {
		t.Fatalf("NewExec: %v", err)
	}

	// Short prompt: BOS + "Hello"
	tokens := []int32{2, 26352}
	tokensTensor := tensors.FromFlatDataAndDimensions(tokens, 1, len(tokens))

	t.Log("Running full forward pass (26 layers)... this may take a while")
	start := time.Now()
	results := exec.MustExec(tokensTensor)
	t.Logf("Full forward took %s", time.Since(start))
	defer results[0].FinalizeAll()

	t.Logf("Output shape: %v", results[0].Shape())

	var logits []float32
	results[0].ConstFlatData(func(flat any) {
		logits = flat.([]float32)
	})

	vocabSize := results[0].Shape().Dimensions[2]

	// Check last position logits.
	lastLogits := logits[(len(tokens)-1)*vocabSize : len(tokens)*vocabSize]

	type tokenScore struct {
		id    int
		score float32
	}
	top := make([]tokenScore, vocabSize)
	for i := range top {
		top[i] = tokenScore{i, lastLogits[i]}
	}
	sort.Slice(top, func(i, j int) bool {
		return top[i].score > top[j].score
	})

	t.Log("Top 20 tokens from last position:")
	for i := 0; i < 20 && i < len(top); i++ {
		t.Logf("  Token %6d: %.4f", top[i].id, top[i].score)
	}

	// Check logits have good range.
	var minL, maxL float32 = lastLogits[0], lastLogits[0]
	for _, v := range lastLogits {
		if v < minL {
			minL = v
		}
		if v > maxL {
			maxL = v
		}
	}
	t.Logf("Logit range: [%.4f, %.4f]", minL, maxL)

	if maxL-minL < 1.0 {
		t.Error("Logits are nearly flat - model is not working correctly")
	}
}

// TestPrefillAndDecode simulates the serving engine's prefill → decode flow.
// This is the critical test: prefill works (verified above), but end-to-end
// inference produces gibberish. The bug must be in the decode step.
func TestPrefillAndDecode(t *testing.T) {
	builder, ctx, backend := loadModel(t)
	cfg := builder.Gemma3Config()
	modelFn := builder.BuildModelFn()

	blockSize := 16
	numBlocks := 10
	maxBlocksPerReq := 4 // enough for 64 tokens

	pagedCfg := kvcache.PagedKVCacheConfig{
		NumBlocks:  numBlocks,
		BlockSize:  blockSize,
		NumKVHeads: cfg.KVHeads(),
		HeadDim:    cfg.HeadDim,
		DType:      dtypes.Float32,
	}

	// Allocate blocks for 2 prompt tokens + some generation.
	blockMgr := kvcache.NewBlockManager(pagedCfg)
	if err := blockMgr.EnsureBlocks(1, 32); err != nil {
		t.Fatalf("EnsureBlocks: %v", err)
	}
	pt := blockMgr.GetPageTable(1)
	t.Logf("Page table: %v", pt)

	ptRow := make([]int32, maxBlocksPerReq)
	for i, b := range pt {
		if i >= maxBlocksPerReq {
			break
		}
		ptRow[i] = int32(b)
	}
	ptTensor := tensors.FromValue([][]int32{ptRow})

	reuseCtx := ctx.Reuse().Checked(false)

	// ── Step 1: Prefill ──
	// Matches initPagedPromptExec: tokens [1, promptLen], positions [1], pageTables [1, maxBlocks].
	// Returns logits [1, vocabSize] (last position extracted).
	prefillExec, err := context.NewExec(backend, reuseCtx,
		func(ctx *context.Context, tokens *Node, positions *Node, pageTables *Node) *Node {
			kv := &kvcache.PagedKVCacheAccessor{
				Config:        pagedCfg,
				PageTables:    pageTables,
				Positions:     positions,
				ReadNumBlocks: maxBlocksPerReq,
			}
			logits := modelFn(ctx, tokens, positions, kv, nil)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			return Squeeze(lastLogits, 1) // [1, vocabSize]
		},
	)
	if err != nil {
		t.Fatalf("NewExec (prefill): %v", err)
	}

	promptTokens := []int32{2, 26352} // BOS + Hello
	tokensTensor := tensors.FromFlatDataAndDimensions(promptTokens, 1, len(promptTokens))
	positionsTensor := tensors.FromFlatDataAndDimensions([]int32{0}, 1)

	t.Log("Running prefill...")
	start := time.Now()
	prefillResults := prefillExec.MustExec(tokensTensor, positionsTensor, ptTensor)
	t.Logf("Prefill took %s", time.Since(start))
	defer prefillResults[0].FinalizeAll()

	// CPU greedy sample (same as serving engine).
	var prefillLogits []float32
	prefillResults[0].ConstFlatData(func(flat any) {
		prefillLogits = flat.([]float32)
	})

	prefillToken := int32(0)
	bestScore := float32(-math.MaxFloat32)
	for i, v := range prefillLogits {
		if v > bestScore {
			bestScore = v
			prefillToken = int32(i)
		}
	}
	t.Logf("Prefill top token: %d (score %.4f)", prefillToken, bestScore)

	if prefillToken != 236743 {
		t.Errorf("Expected prefill token 236743 (space), got %d", prefillToken)
	}

	// ── Step 2: Decode ──
	// Matches getPagedDecodeExec: tokens [1, 1], positions [1], cacheWritePositions [1], pageTables [1, maxBlocks].
	// Returns logits [1, vocabSize] (NOT sampled — we want to inspect logits).
	decodeExec, err := context.NewExec(backend, reuseCtx,
		func(ctx *context.Context, tokens *Node, positions *Node, cacheWritePositions *Node, pageTables *Node) *Node {
			kv := &kvcache.PagedKVCacheAccessor{
				Config:        pagedCfg,
				PageTables:    pageTables,
				Positions:     cacheWritePositions,
				ReadNumBlocks: maxBlocksPerReq,
			}
			aux := &decode.AuxInputs{CacheWritePositions: cacheWritePositions}
			logits := modelFn(ctx, tokens, positions, kv, aux)
			return Squeeze(logits, 1) // [1, vocabSize]
		},
	)
	if err != nil {
		t.Fatalf("NewExec (decode): %v", err)
	}

	// Simulate 5 decode steps.
	currentToken := prefillToken
	absPosition := int32(len(promptTokens)) // 2
	generatedTokens := []int32{prefillToken}

	for step := 0; step < 5; step++ {
		decodeTokens := tensors.FromFlatDataAndDimensions([]int32{currentToken}, 1, 1)
		decodePositions := tensors.FromFlatDataAndDimensions([]int32{absPosition}, 1)
		decodeCacheWritePos := tensors.FromFlatDataAndDimensions([]int32{absPosition}, 1)

		t.Logf("Decode step %d: token=%d, position=%d", step, currentToken, absPosition)
		start := time.Now()
		decodeResults := decodeExec.MustExec(decodeTokens, decodePositions, decodeCacheWritePos, ptTensor)
		t.Logf("  Decode took %s", time.Since(start))

		var decodeLogits []float32
		decodeResults[0].ConstFlatData(func(flat any) {
			decodeLogits = flat.([]float32)
		})
		decodeResults[0].FinalizeAll()

		// Find top token.
		nextToken := int32(0)
		bestScore = float32(-math.MaxFloat32)
		for i, v := range decodeLogits {
			if v > bestScore {
				bestScore = v
				nextToken = int32(i)
			}
		}

		// Logit stats.
		var minL, maxL float32 = decodeLogits[0], decodeLogits[0]
		for _, v := range decodeLogits {
			if v < minL {
				minL = v
			}
			if v > maxL {
				maxL = v
			}
		}
		t.Logf("  Top token: %d (score %.4f), logit range: [%.4f, %.4f]", nextToken, bestScore, minL, maxL)

		if maxL-minL < 1.0 {
			t.Errorf("Step %d: logits nearly flat (range=%.4f)", step, maxL-minL)
		}
		if nextToken == 2 {
			t.Errorf("Step %d: predicted BOS token (2) — likely degenerate", step)
		}

		currentToken = nextToken
		absPosition++
		generatedTokens = append(generatedTokens, nextToken)
	}

	t.Logf("Generated sequence: %v", generatedTokens)

	// Check for repetitive output.
	allSame := true
	for _, tok := range generatedTokens[1:] {
		if tok != generatedTokens[0] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("All generated tokens are identical — model is stuck in a loop")
	}
}

// TestFlatVsPagedDecode compares flat KV cache and paged KV cache decode output.
// If they match, the paged cache is correct and any degenerate output is model behavior.
// If they differ, there's a paged cache bug.
func TestFlatVsPagedDecode(t *testing.T) {
	builder, flatCtx, backend := loadModel(t)
	cfg := builder.Gemma3Config()
	modelFn := builder.BuildModelFn()

	// 17-token prompt that crosses block boundary.
	prompt := []int32{2, 106, 1645, 108, 5765, 476, 3309, 19575, 1105, 573, 5671, 235265, 107, 108, 106, 2516, 108}
	numDecodeSteps := 5
	maxSeqLen := 64

	// ── Flat KV cache path ──
	flatReuseCtx := flatCtx.Reuse().Checked(false)

	flatPrefillExec, err := context.NewExec(backend, flatReuseCtx,
		func(ctx *context.Context, tokens *Node, positions *Node) *Node {
			kv := attention.NewFlatKVCacheAccessor(
				1, cfg.KVHeads(), maxSeqLen, cfg.HeadDim, dtypes.Float32, positions,
			)
			logits := modelFn(ctx, tokens, positions, kv, nil)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			return Squeeze(lastLogits, 1)
		},
	)
	if err != nil {
		t.Fatalf("NewExec (flat prefill): %v", err)
	}

	flatDecodeExec, err := context.NewExec(backend, flatReuseCtx,
		func(ctx *context.Context, tokens *Node, positions *Node, cacheWritePositions *Node) *Node {
			kv := attention.NewFlatKVCacheAccessor(
				1, cfg.KVHeads(), maxSeqLen, cfg.HeadDim, dtypes.Float32, cacheWritePositions,
			)
			aux := &decode.AuxInputs{CacheWritePositions: cacheWritePositions}
			logits := modelFn(ctx, tokens, positions, kv, aux)
			return Squeeze(logits, 1)
		},
	)
	if err != nil {
		t.Fatalf("NewExec (flat decode): %v", err)
	}

	// Flat prefill.
	tokensTensor := tensors.FromFlatDataAndDimensions(prompt, 1, len(prompt))
	posTensor := tensors.FromFlatDataAndDimensions([]int32{0}, 1)
	results := flatPrefillExec.MustExec(tokensTensor, posTensor)

	var logits []float32
	results[0].ConstFlatData(func(flat any) { logits = flat.([]float32) })
	results[0].FinalizeAll()

	flatTokens := []int32{greedySampleSlice(logits)}
	t.Logf("Flat prefill → token %d", flatTokens[0])

	pos := int32(len(prompt))
	for step := 0; step < numDecodeSteps; step++ {
		dt := tensors.FromFlatDataAndDimensions([]int32{flatTokens[len(flatTokens)-1]}, 1, 1)
		dp := tensors.FromFlatDataAndDimensions([]int32{pos}, 1)
		dcp := tensors.FromFlatDataAndDimensions([]int32{pos}, 1)
		results := flatDecodeExec.MustExec(dt, dp, dcp)
		results[0].ConstFlatData(func(flat any) { logits = flat.([]float32) })
		results[0].FinalizeAll()
		flatTokens = append(flatTokens, greedySampleSlice(logits))
		pos++
	}
	t.Logf("Flat generated: %v", flatTokens)

	// ── Paged KV cache path ──
	// Need a fresh context since KV cache vars are different between flat and paged.
	_, pagedCtx, _ := loadModel(t)
	pagedReuseCtx := pagedCtx.Reuse().Checked(false)

	blockSize := 16
	numBlocks := 10
	maxBlocksPerReq := (maxSeqLen + blockSize - 1) / blockSize

	pagedCfg := kvcache.PagedKVCacheConfig{
		NumBlocks:  numBlocks,
		BlockSize:  blockSize,
		NumKVHeads: cfg.KVHeads(),
		HeadDim:    cfg.HeadDim,
		DType:      dtypes.Float32,
	}

	blockMgr := kvcache.NewBlockManager(pagedCfg)
	if err := blockMgr.EnsureBlocks(1, maxSeqLen); err != nil {
		t.Fatalf("EnsureBlocks: %v", err)
	}
	pt := blockMgr.GetPageTable(1)
	ptRow := make([]int32, maxBlocksPerReq)
	for i, b := range pt {
		if i >= maxBlocksPerReq {
			break
		}
		ptRow[i] = int32(b)
	}
	ptTensor := tensors.FromValue([][]int32{ptRow})

	pagedPrefillExec, err := context.NewExec(backend, pagedReuseCtx,
		func(ctx *context.Context, tokens *Node, positions *Node, pageTables *Node) *Node {
			kv := &kvcache.PagedKVCacheAccessor{
				Config:        pagedCfg,
				PageTables:    pageTables,
				Positions:     positions,
				ReadNumBlocks: maxBlocksPerReq,
			}
			logits := modelFn(ctx, tokens, positions, kv, nil)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			return Squeeze(lastLogits, 1)
		},
	)
	if err != nil {
		t.Fatalf("NewExec (paged prefill): %v", err)
	}

	pagedDecodeExec, err := context.NewExec(backend, pagedReuseCtx,
		func(ctx *context.Context, tokens *Node, positions *Node, cacheWritePositions *Node, pageTables *Node) *Node {
			kv := &kvcache.PagedKVCacheAccessor{
				Config:        pagedCfg,
				PageTables:    pageTables,
				Positions:     cacheWritePositions,
				ReadNumBlocks: maxBlocksPerReq,
			}
			aux := &decode.AuxInputs{CacheWritePositions: cacheWritePositions}
			logits := modelFn(ctx, tokens, positions, kv, aux)
			return Squeeze(logits, 1)
		},
	)
	if err != nil {
		t.Fatalf("NewExec (paged decode): %v", err)
	}

	// Paged prefill.
	tokensTensor2 := tensors.FromFlatDataAndDimensions(prompt, 1, len(prompt))
	posTensor2 := tensors.FromFlatDataAndDimensions([]int32{0}, 1)
	results2 := pagedPrefillExec.MustExec(tokensTensor2, posTensor2, ptTensor)

	results2[0].ConstFlatData(func(flat any) { logits = flat.([]float32) })
	results2[0].FinalizeAll()

	pagedTokens := []int32{greedySampleSlice(logits)}
	t.Logf("Paged prefill → token %d", pagedTokens[0])

	pos = int32(len(prompt))
	for step := 0; step < numDecodeSteps; step++ {
		dt := tensors.FromFlatDataAndDimensions([]int32{pagedTokens[len(pagedTokens)-1]}, 1, 1)
		dp := tensors.FromFlatDataAndDimensions([]int32{pos}, 1)
		dcp := tensors.FromFlatDataAndDimensions([]int32{pos}, 1)
		results := pagedDecodeExec.MustExec(dt, dp, dcp, ptTensor)
		results[0].ConstFlatData(func(flat any) { logits = flat.([]float32) })
		results[0].FinalizeAll()
		pagedTokens = append(pagedTokens, greedySampleSlice(logits))
		pos++
	}
	t.Logf("Paged generated: %v", pagedTokens)

	// Compare.
	if len(flatTokens) != len(pagedTokens) {
		t.Fatalf("Length mismatch: flat=%d, paged=%d", len(flatTokens), len(pagedTokens))
	}
	mismatch := false
	for i := range flatTokens {
		if flatTokens[i] != pagedTokens[i] {
			t.Errorf("Token %d mismatch: flat=%d, paged=%d", i, flatTokens[i], pagedTokens[i])
			mismatch = true
		}
	}
	if !mismatch {
		t.Log("Flat and paged KV cache produce IDENTICAL decode sequences")
	}
}

// TestPromptLengthSweep tests the full forward pass at different prompt lengths
// to find where the output starts to degrade.
func TestPromptLengthSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping slow sweep in short mode")
	}

	builder, ctx, backend := loadModel(t)
	cfg := builder.Gemma3Config()
	t.Logf("Config: layers=%d, hidden=%d, heads=%d, kvHeads=%d, headDim=%d, intermediate=%d",
		cfg.NumHiddenLayers, cfg.HiddenSize, cfg.NumAttentionHeads, cfg.KVHeads(), cfg.HeadDim, cfg.IntermediateSize)

	// Full chat prompt tokens (from llama-tokenize with this GGUF file).
	// <bos><start_of_turn> user\nWrite a short poem about the sea.<end_of_turn> \n<start_of_turn> model\n
	fullPrompt := []int32{2, 105, 2430, 107, 6974, 496, 2822, 27355, 1003, 506, 5442, 236761, 106, 236743, 107, 105, 2028, 107}

	// Test at different prefix lengths.
	lengths := []int{2, 3, 5, 8, 10, 13, 15, 18}
	for _, length := range lengths {
		if length > len(fullPrompt) {
			continue
		}
		prompt := fullPrompt[:length]

		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, inputIDs *Node) *Node {
			return builder.Forward(ctx, inputIDs, nil)
		})
		if err != nil {
			t.Fatalf("NewExec: %v", err)
		}

		tokensTensor := tensors.FromFlatDataAndDimensions(prompt, 1, len(prompt))
		results := exec.MustExec(tokensTensor)

		var logits []float32
		results[0].ConstFlatData(func(flat any) {
			logits = flat.([]float32)
		})

		vocabSize := results[0].Shape().Dimensions[2]
		lastLogits := logits[(length-1)*vocabSize : length*vocabSize]

		// Top 5 tokens.
		type ts struct {
			id    int
			score float32
		}
		top := make([]ts, vocabSize)
		for i := range top {
			top[i] = ts{i, lastLogits[i]}
		}
		sort.Slice(top, func(i, j int) bool {
			return top[i].score > top[j].score
		})

		t.Logf("Prompt len %2d: top5=[%d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f)]",
			length,
			top[0].id, top[0].score, top[1].id, top[1].score,
			top[2].id, top[2].score, top[3].id, top[3].score,
			top[4].id, top[4].score)

		results[0].FinalizeAll()
	}
}

// TestCheckOutputWeight verifies whether the GGUF file has a separate output.weight
// or uses tied embeddings for the LM head.
func TestCheckOutputWeight(t *testing.T) {
	ggufPath := getGGUFPath()
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF file not found, skipping")
	}

	model, err := models.NewFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	t.Log(model.Summary())

	names := model.Weights.ListTensorNames()
	t.Logf("Total tensors: %d", len(names))

	// Check for output.weight and token_embd.weight.
	hasOutput := false
	hasEmbd := false
	for _, name := range names {
		if name == "output.weight" {
			hasOutput = true
			t.Log("Found output.weight (separate LM head)")
		}
		if name == "token_embd.weight" {
			hasEmbd = true
			t.Log("Found token_embd.weight (embedding)")
		}
	}
	if !hasOutput {
		t.Log("NO output.weight found — using tied embeddings for LM head")
	}
	if !hasEmbd {
		t.Error("Missing token_embd.weight!")
	}

	// Print first 20 and last 20 tensor names for orientation.
	t.Log("First 20 tensors:")
	for i := 0; i < 20 && i < len(names); i++ {
		t.Logf("  %s", names[i])
	}
	t.Log("Last 20 tensors:")
	for i := max(0, len(names)-20); i < len(names); i++ {
		t.Logf("  %s", names[i])
	}
}

// TestQuantizedVsFloat32Forward compares the quantized model output against
// a fully dequantized float32 model to isolate quantized-pipeline bugs.
func TestQuantizedVsFloat32Forward(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping slow comparison in short mode")
	}

	ggufPath := getGGUFPath()
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF file not found, skipping")
	}

	backend := backends.MustNew()

	// --- Build 1: normal quantized model ---
	model, err := models.NewFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("NewFromGGUF: %v", err)
	}
	quantCtx := context.New()
	if err := model.LoadWeightsIntoContext(quantCtx); err != nil {
		t.Fatalf("LoadWeights (quantized): %v", err)
	}
	quantBuilder := model.Builder.(*gemma3.Builder)

	// --- Build 2: fully dequantized float32 model ---
	// Re-open the GGUF file to read dequantized tensors.
	model2, err := models.NewFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("NewFromGGUF (2): %v", err)
	}
	floatCtx := context.New()

	// Get the weight mapping.
	floatBuilder := model2.Builder.(*gemma3.Builder)
	mapping := floatBuilder.WeightMapping()

	// Load all weights as dequantized float32 using Model.GetTensor (always dequantizes).
	ggufSource := model2.Weights.(*models.GGUFSource)

	loaded := 0
	skipped := 0
	for tensorKey, scopePath := range mapping {
		// GetTensor from the gguf.Model always returns dequantized float32.
		tn, err := ggufSource.Model.GetTensor(tensorKey)
		if err != nil {
			skipped++
			continue
		}
		tensor := tn.Tensor

		// Create variable in floatCtx at the right scope.
		parts := make([]string, 0)
		for _, p := range splitScopePath(scopePath) {
			parts = append(parts, p)
		}
		varCtx := floatCtx
		for _, part := range parts[:len(parts)-1] {
			varCtx = varCtx.In(part)
		}
		varCtx.VariableWithValue(parts[len(parts)-1], tensor)
		loaded++
	}
	t.Logf("Loaded %d float32 weights, skipped %d", loaded, skipped)

	// Tell the context to reuse existing variables (we pre-loaded them above).
	floatCtx = floatCtx.Reuse()

	// Input: short prompt that should produce a known output.
	prompt := []int32{2, 105, 2430, 107} // <bos><start_of_turn> user\n
	tokensTensor := tensors.FromFlatDataAndDimensions(prompt, 1, len(prompt))

	// Run quantized forward.
	quantExec, err := context.NewExec(backend, quantCtx, func(ctx *context.Context, inputIDs *Node) *Node {
		return quantBuilder.Forward(ctx, inputIDs, nil)
	})
	if err != nil {
		t.Fatalf("NewExec (quantized): %v", err)
	}
	quantResults := quantExec.MustExec(tokensTensor)
	quantShape := quantResults[0].Shape()
	t.Logf("Quantized output shape: %v", quantShape)
	var quantLogits []float32
	quantResults[0].ConstFlatData(func(flat any) {
		quantLogits = make([]float32, len(flat.([]float32)))
		copy(quantLogits, flat.([]float32))
	})
	quantResults[0].FinalizeAll()

	// Run float32 forward.
	floatExec, err := context.NewExec(backend, floatCtx, func(ctx *context.Context, inputIDs *Node) *Node {
		return floatBuilder.Forward(ctx, inputIDs, nil)
	})
	if err != nil {
		t.Fatalf("NewExec (float32): %v", err)
	}
	floatResults := floatExec.MustExec(tokensTensor)
	floatShape := floatResults[0].Shape()
	t.Logf("Float32 output shape: %v", floatShape)
	var floatLogits []float32
	floatResults[0].ConstFlatData(func(flat any) {
		floatLogits = make([]float32, len(flat.([]float32)))
		copy(floatLogits, flat.([]float32))
	})
	floatResults[0].FinalizeAll()

	// Compare logits at last position.
	vocabSize := quantShape.Dimensions[2]
	seqLen := len(prompt)
	qLast := quantLogits[(seqLen-1)*vocabSize : seqLen*vocabSize]
	fLast := floatLogits[(seqLen-1)*vocabSize : seqLen*vocabSize]

	// Find top tokens for each.
	type ts struct {
		id    int
		score float32
	}
	qTop := make([]ts, vocabSize)
	fTop := make([]ts, vocabSize)
	for i := range vocabSize {
		qTop[i] = ts{i, qLast[i]}
		fTop[i] = ts{i, fLast[i]}
	}
	sort.Slice(qTop, func(i, j int) bool { return qTop[i].score > qTop[j].score })
	sort.Slice(fTop, func(i, j int) bool { return fTop[i].score > fTop[j].score })

	t.Logf("Quantized  top5: [%d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f)]",
		qTop[0].id, qTop[0].score, qTop[1].id, qTop[1].score,
		qTop[2].id, qTop[2].score, qTop[3].id, qTop[3].score,
		qTop[4].id, qTop[4].score)
	t.Logf("Float32    top5: [%d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f)]",
		fTop[0].id, fTop[0].score, fTop[1].id, fTop[1].score,
		fTop[2].id, fTop[2].score, fTop[3].id, fTop[3].score,
		fTop[4].id, fTop[4].score)

	// Compute error stats.
	var maxAbsErr float64
	var sumAbsErr float64
	for i := range vocabSize {
		absErr := math.Abs(float64(qLast[i]) - float64(fLast[i]))
		sumAbsErr += absErr
		if absErr > maxAbsErr {
			maxAbsErr = absErr
		}
	}
	t.Logf("Logit diff: maxAbsErr=%.6f, meanAbsErr=%.6f", maxAbsErr, sumAbsErr/float64(vocabSize))

	if qTop[0].id != fTop[0].id {
		t.Errorf("Top-1 token mismatch: quantized=%d, float32=%d", qTop[0].id, fTop[0].id)
	} else {
		t.Logf("Top-1 token MATCHES: %d", qTop[0].id)
	}
}

// splitScopePath splits "a/b/c" into ["a", "b", "c"].
func splitScopePath(path string) []string {
	parts := make([]string, 0)
	for _, p := range strings.Split(path, "/") {
		if p != "" {
			parts = append(parts, p)
		}
	}
	return parts
}

func greedySampleSlice(logits []float32) int32 {
	best := int32(0)
	bestScore := float32(-math.MaxFloat32)
	for i, v := range logits {
		if v > bestScore {
			bestScore = v
			best = int32(i)
		}
	}
	return best
}

// TestEmbeddingNorms checks for anomalous embedding norms that could explain
// why certain token IDs (e.g., 236xxx) consistently score high in the LM head.
func TestEmbeddingNorms(t *testing.T) {
	ggufPath := getGGUFPath()
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF file not found, skipping")
	}

	file, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatalf("Failed to parse GGUF: %v", err)
	}
	reader, err := gguf.NewReader(file)
	if err != nil {
		t.Fatalf("Failed to create reader: %v", err)
	}

	tn, err := reader.ReadTensor("token_embd.weight")
	if err != nil {
		t.Fatalf("ReadTensor: %v", err)
	}
	// ReadTensor returns dequantized float32.
	var embData []float32
	tn.ConstFlatData(func(flat any) {
		src := flat.([]float32)
		embData = make([]float32, len(src))
		copy(embData, src)
	})
	vocabSize := tn.Shape().Dimensions[0]
	hiddenSize := tn.Shape().Dimensions[1]
	t.Logf("Embedding shape: [%d, %d]", vocabSize, hiddenSize)

	// Compute L2 norms for all tokens.
	norms := make([]float64, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sumSq float64
		for d := 0; d < hiddenSize; d++ {
			val := float64(embData[v*hiddenSize+d])
			sumSq += val * val
		}
		norms[v] = math.Sqrt(sumSq)
	}

	// Report norms for suspicious tokens.
	suspectTokens := []int{236743, 236770, 236777, 236800, 236829}
	normalTokens := []int{2, 106, 818, 1018, 1645, 2717, 4521, 108}
	t.Log("Suspect token norms (236xxx):")
	for _, id := range suspectTokens {
		t.Logf("  Token %6d: norm=%.6f", id, norms[id])
	}
	t.Log("Normal token norms:")
	for _, id := range normalTokens {
		t.Logf("  Token %6d: norm=%.6f", id, norms[id])
	}

	// Compute stats over ranges.
	var sumLow, sumHigh, sumMid float64
	var countLow, countHigh, countMid int
	for v := 0; v < vocabSize; v++ {
		if v < 256000 {
			if v < 128000 {
				sumLow += norms[v]
				countLow++
			} else {
				sumMid += norms[v]
				countMid++
			}
		} else {
			sumHigh += norms[v]
			countHigh++
		}
	}
	t.Logf("Mean norm [0, 128K):    %.6f (count=%d)", sumLow/float64(countLow), countLow)
	t.Logf("Mean norm [128K, 256K): %.6f (count=%d)", sumMid/float64(countMid), countMid)
	t.Logf("Mean norm [256K, 262K): %.6f (count=%d)", sumHigh/float64(countHigh), countHigh)

	// Check 236K-237K range specifically.
	var sum236 float64
	for v := 236000; v < 237000; v++ {
		sum236 += norms[v]
	}
	t.Logf("Mean norm [236K, 237K): %.6f", sum236/1000.0)

	// Find top-10 highest norm tokens.
	type tokenNorm struct {
		id   int
		norm float64
	}
	sorted := make([]tokenNorm, vocabSize)
	for i := range sorted {
		sorted[i] = tokenNorm{i, norms[i]}
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].norm > sorted[j].norm })
	t.Log("Top-10 highest embedding norms:")
	for i := 0; i < 10; i++ {
		t.Logf("  Token %6d: norm=%.6f", sorted[i].id, sorted[i].norm)
	}
}

// TestLayerByLayerDiagnostic runs the forward pass with increasing numbers of
// transformer layers to find where the predictions go wrong.
func TestLayerByLayerDiagnostic(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping slow diagnostic in short mode")
	}

	builder, ctx, backend := loadModel(t)
	cfg := builder.Gemma3Config()

	// Correct Gemma 3 tokenizer IDs from llama-tokenize.
	prompt := []int32{2, 105, 2430, 107, 6974, 496, 2822, 27355, 1003, 506, 5442, 236761, 106, 236743, 107, 105, 2028, 107}
	tokensTensor := tensors.FromFlatDataAndDimensions(prompt, 1, len(prompt))

	// Test with different layer counts.
	for _, numLayers := range []int{0, 1, 2, 5, 10, 17, 34} {
		if numLayers > cfg.NumHiddenLayers {
			continue
		}
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, inputIDs *Node) *Node {
			g := inputIDs.Graph()
			hidden := builder.BuildEmbeddings(ctx, inputIDs)

			batchSize := inputIDs.Shape().Dimensions[0]
			seqLen := inputIDs.Shape().Dimensions[1]
			positionIDs := common.GetPositionIDs(g, batchSize, seqLen)

			for i := 0; i < numLayers; i++ {
				hidden = builder.BuildDecoderLayer(ctx.In("layers").In(fmt.Sprintf("%d", i)), hidden, positionIDs, i)
			}

			hidden = common.RMSNorm(ctx.In("norm"), hidden, cfg.RMSNormEps)

			return builder.ApplyLMHead(ctx, hidden, g)
		})
		if err != nil {
			t.Fatalf("NewExec (layers=%d): %v", numLayers, err)
		}

		results := exec.MustExec(tokensTensor)
		shape := results[0].Shape()
		vocabSize := shape.Dimensions[2]

		var logits []float32
		results[0].ConstFlatData(func(flat any) {
			src := flat.([]float32)
			logits = make([]float32, len(src))
			copy(logits, src)
		})
		results[0].FinalizeAll()

		// Check last position.
		lastLogits := logits[(len(prompt)-1)*vocabSize : len(prompt)*vocabSize]
		type ts struct {
			id    int
			score float32
		}
		top := make([]ts, vocabSize)
		for i := range vocabSize {
			top[i] = ts{i, lastLogits[i]}
		}
		sort.Slice(top, func(i, j int) bool { return top[i].score > top[j].score })

		t.Logf("Layers=%2d: top5=[%d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f)]",
			numLayers,
			top[0].id, top[0].score, top[1].id, top[1].score,
			top[2].id, top[2].score, top[3].id, top[3].score,
			top[4].id, top[4].score)
	}
}

// mockTokenizer implements serving.Tokenizer for testing.
type mockTokenizer struct {
	eosID int32
}

func (t *mockTokenizer) Decode(tokenID int32) (string, error) {
	return fmt.Sprintf("[%d]", tokenID), nil
}

func (t *mockTokenizer) IsEOS(tokenID int32) bool {
	return tokenID == t.eosID
}

func (t *mockTokenizer) Reset() {}

// TestServingEngine uses the actual serving engine (serving.NewPaged) end-to-end.
// This reproduces the "HelloHelloHello..." repetitive output from main.go.
func TestServingEngine(t *testing.T) {
	builder, ctx, backend := loadModel(t)
	cfg := builder.Gemma3Config()
	modelFn := builder.BuildModelFn()

	maxSeqLen := 64
	blockSize := 16
	numBlocks := (maxSeqLen/blockSize + 1) * 2 // same formula as main.go

	cacheDType := dtypes.Float32
	pagedCfg := kvcache.PagedKVCacheConfig{
		NumBlocks:  numBlocks,
		BlockSize:  blockSize,
		NumKVHeads: cfg.KVHeads(),
		HeadDim:    cfg.HeadDim,
		DType:      cacheDType,
	}

	engineCfg := serving.Config{
		MaxSeqLen:    maxSeqLen,
		MaxBatchSize: 1,
	}

	tok := &mockTokenizer{eosID: 1}
	engine := serving.NewPaged(backend, ctx, modelFn, tok, engineCfg, pagedCfg)
	defer engine.Stop()

	// Use the same 2-token prompt as TestPrefillAndDecode.
	promptTokens := []int32{2, 26352} // BOS + Hello
	maxNewTokens := 10

	outputCh, errCh, err := engine.Submit(
		gocontext.Background(),
		promptTokens,
		serving.RequestOptions{MaxNewTokens: maxNewTokens},
		nil,
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var generated []int32
	for delta := range outputCh {
		generated = append(generated, delta.TokenID)
		t.Logf("Token %d: id=%d text=%q", len(generated), delta.TokenID, delta.Token)
	}

	if err := <-errCh; err != nil {
		t.Fatalf("Generation error: %v", err)
	}

	t.Logf("Generated %d tokens: %v", len(generated), generated)

	// Check first token matches expected (236743=space for BOS+Hello prompt).
	if len(generated) > 0 && generated[0] != 236743 {
		t.Errorf("Expected first token 236743 (space), got %d", generated[0])
	}

	// Check for repetitive output.
	if len(generated) >= 3 {
		allSame := true
		for _, tok := range generated[1:] {
			if tok != generated[0] {
				allSame = false
				break
			}
		}
		if allSame {
			t.Error("All generated tokens are identical — model is stuck in a loop")
		}
	}
}
