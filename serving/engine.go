package serving

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/decode/sample"
	"github.com/gomlx/gollmx/kvcache"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

var (
	// ErrEngineStopped is returned by Submit when the engine has been stopped.
	ErrEngineStopped = errors.New("engine is stopped")

	// ErrPromptEmpty is returned by Submit when the input tokens slice is empty.
	ErrPromptEmpty = errors.New("prompt is empty")
)

// EmbedFn converts token IDs to embeddings eagerly, outside the compiled graph.
// The engine calls this before every forward pass (prefill and decode).
//
// This enables models whose embedding step contains operations incompatible
// with static graph compilation (e.g., ONNX NonZero for data-dependent shapes).
// The returned AuxData must have InputsEmbeds populated; PerLayerInputs is
// optional and model-specific.
//
// Parameters:
//   - tokens: [batchSize, seqLen] int32 token IDs to embed.
//   - auxData: request-level data (e.g., image features for multimodal prefill).
//     nil for decode steps and text-only prefill.
//
// The EmbedFn implementation is responsible for its own executor caching and
// efficiency. For decode (seqLen=1), implementations should use a cached
// native embedding lookup rather than recompiling per token.
type EmbedFn func(tokens *tensors.Tensor, auxData *AuxData) (*AuxData, error)

// Config holds engine configuration.
type Config struct {
	// MaxSeqLen is the maximum total sequence length (prompt + generated).
	// This determines the KV cache size per request.
	MaxSeqLen int

	// MaxBatchSize is the maximum number of concurrent requests in a single
	// forward pass. Only used in batched mode (NewEngine/NewPaged).
	// Default: 8.
	MaxBatchSize int

	// SubmitQueueSize is the capacity of the internal submit channel.
	// Default: 256.
	SubmitQueueSize int

	// Preemption configures request preemption under memory pressure.
	// nil means preemption is disabled.
	Preemption *PreemptionPolicy

	// Speculative configures speculative decoding.
	// nil means speculative decoding is disabled.
	Speculative *SpeculativeConfig

	// Compaction configures KV cache compaction after prefill.
	// When set, the engine compresses the KV cache using attention matching
	// after prefill completes, reducing memory usage for long prompts.
	// nil means compaction is disabled.
	Compaction *kvcache.CompactionConfig
}

// DefaultConfig returns a default engine configuration.
func DefaultConfig() Config {
	return Config{
		MaxSeqLen:       2048,
		MaxBatchSize:    8,
		SubmitQueueSize: 256,
	}
}

// applyDefaults fills in zero-valued fields with sensible defaults.
func (c *Config) applyDefaults() {
	if c.SubmitQueueSize <= 0 {
		c.SubmitQueueSize = 256
	}
	if c.MaxSeqLen <= 0 {
		c.MaxSeqLen = 2048
	}
	if c.MaxBatchSize <= 0 {
		c.MaxBatchSize = 8
	}
}

// Engine provides concurrent inference with token streaming.
//
// Multiple goroutines may call Submit concurrently. The engine processes
// requests via a single background loop, streaming generated tokens back
// through channels.
//
// Two modes are supported:
//   - Sequential (New): processes one request at a time using IncrementalModelFn.
//   - Batched (NewEngine/NewPaged): continuous batching using ModelFn with
//     per-element positions, slot management, and power-of-2 padded batch sizes.
type Engine struct {
	backend   backends.Backend
	modelCtx  *mlctx.Context
	modelFn   decode.IncrementalModelFn // Phase 1 sequential model
	unifiedFn decode.ModelFn            // Unified model with engine-controlled KV cache
	tokenizer Tokenizer
	config    Config

	// KV cache config for unified mode (numKVHeads, headDim, maxSeqLen, dtype).
	unifiedKVConfig *decode.KVCacheConfig

	// Batched mode infrastructure.
	batchedMode bool
	slotMgr     *slotManager
	sched       *scheduler

	// Paged KV cache (Phase 3).
	pagedMode bool
	blockMgr  *kvcache.BlockManager
	pagedCfg  kvcache.PagedKVCacheConfig

	// Speculative decoding (Phase 4).
	specConfig     *SpeculativeConfig
	draftExecCache map[int]*mlctx.Exec
	verifyExec     *mlctx.Exec // cached verify executor (returns all-position logits)

	// Preemption (Phase 4).
	preemptMgr *preemptionManager

	// Prefix cache (Phase 4, paged mode).
	prefixCache *kvcache.PrefixCache

	// Request tracking.
	mu       sync.Mutex
	requests map[uint64]*engineRequest
	nextID   uint64

	// Concurrency control.
	submitCh chan *engineRequest
	stopCh   chan struct{}
	stopped  atomic.Bool
	stopOnce sync.Once
	submitMu sync.RWMutex // guards stopped check + submitCh send atomicity
	wg       sync.WaitGroup

	// Cached executors — sequential mode (keyed by position).
	promptExec   *mlctx.Exec
	genExecCache map[int]*mlctx.Exec

	// Cached executors — batched mode (keyed by paddedBatchSize).
	batchedPromptExec          *mlctx.Exec
	batchedMultimodalExec      *mlctx.Exec            // prefill with aux inputs (images, etc.)
	batchedDecodeExecCache     map[int]*mlctx.Exec     // paddedBatchSize → exec

	// Cached executors — paged mode (separate from flat because they take page tables).
	pagedPromptExec            *mlctx.Exec
	pagedMultimodalExec        *mlctx.Exec             // paged prefill with aux inputs (images, etc.)
	pagedDecodeExecCache       map[int]*mlctx.Exec     // paddedBatchSize → exec

	// Eager embedding support.
	embedFn                    EmbedFn                 // optional eager embedding function
	embedPromptExec            *mlctx.Exec             // prefill with pre-computed embeddings
	embedPerLayerPromptExec    *mlctx.Exec             // prefill with embeddings + per-layer inputs
	embedDecodeExecCache       map[int]*mlctx.Exec     // decode with pre-computed embeddings
	embedPerLayerDecodeCache   map[int]*mlctx.Exec     // decode with embeddings + per-layer inputs
}

// New creates and starts a new Engine. The background step loop begins
// immediately. Call Stop to shut down.
//
// Parameters:
//   - backend: Backend for computation (e.g., SimpleGo or XLA).
//   - ctx: Model context containing weights. The engine calls ctx.Reuse()
//     internally for executor creation.
//   - modelFn: An IncrementalModelFn for KV-cached generation.
//   - tokenizer: Tokenizer for output decoding and EOS detection.
//   - config: Engine configuration.
func New(
	backend backends.Backend,
	ctx *mlctx.Context,
	modelFn decode.IncrementalModelFn,
	tokenizer Tokenizer,
	config Config,
) *Engine {
	config.applyDefaults()

	e := &Engine{
		backend:      backend,
		modelCtx:     ctx,
		modelFn:      modelFn,
		tokenizer:    tokenizer,
		config:       config,
		requests:     make(map[uint64]*engineRequest),
		genExecCache: make(map[int]*mlctx.Exec),
		submitCh:     make(chan *engineRequest, config.SubmitQueueSize),
		stopCh:       make(chan struct{}),
	}

	e.wg.Add(1)
	go e.runStepLoop()

	return e
}

// NewPaged creates a batched Engine with paged KV cache for memory-efficient
// inference. Instead of pre-allocating maxSeqLen per batch slot, KV entries
// are stored in fixed-size blocks allocated on demand.
//
// Parameters:
//   - backend: Backend for computation.
//   - ctx: Model context containing weights.
//   - modelFn: A ModelFn that accepts positions as tensors and a KVCacheAccessor.
//   - tokenizer: Tokenizer for output decoding and EOS detection.
//   - config: Engine configuration.
//   - pagedCfg: Paged KV cache configuration (block size, num blocks, etc.).
func NewPaged(
	backend backends.Backend,
	ctx *mlctx.Context,
	modelFn decode.ModelFn,
	tokenizer Tokenizer,
	config Config,
	pagedCfg kvcache.PagedKVCacheConfig,
) *Engine {
	config.applyDefaults()

	e := &Engine{
		backend:                backend,
		modelCtx:               ctx,
		unifiedFn:              modelFn,
		tokenizer:              tokenizer,
		config:                 config,
		batchedMode:            true,
		slotMgr:                newSlotManager(config.MaxBatchSize),
		sched:                  newScheduler(config.MaxBatchSize),
		requests:               make(map[uint64]*engineRequest),
		batchedDecodeExecCache: make(map[int]*mlctx.Exec),
		pagedDecodeExecCache:   make(map[int]*mlctx.Exec),
		submitCh:               make(chan *engineRequest, config.SubmitQueueSize),
		stopCh:                 make(chan struct{}),
		pagedMode:              true,
		blockMgr:               kvcache.NewBlockManager(pagedCfg),
		pagedCfg:               pagedCfg,
		prefixCache:            kvcache.NewPrefixCache(0),
		unifiedKVConfig: &decode.KVCacheConfig{
			NumKVHeads: pagedCfg.NumKVHeads,
			HeadDim:    pagedCfg.HeadDim,
			MaxSeqLen:  config.MaxSeqLen,
			DType:      pagedCfg.DType,
		},
	}

	if config.Preemption != nil {
		e.preemptMgr = newPreemptionManager(*config.Preemption)
	}
	if config.Speculative != nil {
		e.specConfig = config.Speculative
	}

	e.wg.Add(1)
	go e.runStepLoop()
	return e
}

// NewEngine creates a batched Engine using a unified ModelFn with
// engine-controlled KV cache. This enables O(1) compiled executors
// (positions are tensor parameters) and transparent KV cache management.
//
// Parameters:
//   - backend: Backend for computation.
//   - ctx: Model context containing weights.
//   - modelFn: A ModelFn that accepts positions as tensors and a KVCacheAccessor.
//   - tokenizer: Tokenizer for output decoding and EOS detection.
//   - config: Engine configuration.
//   - numKVHeads: Number of key/value attention heads.
//   - headDim: Dimension of each attention head.
//   - cacheDType: Data type for KV cache entries.
func NewEngine(
	backend backends.Backend,
	ctx *mlctx.Context,
	modelFn decode.ModelFn,
	tokenizer Tokenizer,
	config Config,
	numKVHeads, headDim int,
	cacheDType dtypes.DType,
) *Engine {
	config.applyDefaults()

	e := &Engine{
		backend:                backend,
		modelCtx:               ctx,
		unifiedFn:              modelFn,
		tokenizer:              tokenizer,
		config:                 config,
		batchedMode:            true,
		slotMgr:                newSlotManager(config.MaxBatchSize),
		sched:                  newScheduler(config.MaxBatchSize),
		requests:               make(map[uint64]*engineRequest),
		batchedDecodeExecCache: make(map[int]*mlctx.Exec),
		submitCh:               make(chan *engineRequest, config.SubmitQueueSize),
		stopCh:                 make(chan struct{}),
		unifiedKVConfig: &decode.KVCacheConfig{
			NumKVHeads: numKVHeads,
			HeadDim:    headDim,
			MaxSeqLen:  config.MaxSeqLen,
			DType:      cacheDType,
		},
	}

	if config.Preemption != nil {
		e.preemptMgr = newPreemptionManager(*config.Preemption)
	}
	if config.Speculative != nil {
		e.specConfig = config.Speculative
	}

	e.wg.Add(1)
	go e.runStepLoop()
	return e
}

// SetEmbedFn sets the eager embedding function. Must be called before
// submitting any requests. When set, the engine calls embedFn before every
// forward pass to convert token IDs to embeddings outside the compiled graph.
func (e *Engine) SetEmbedFn(fn EmbedFn) {
	e.embedFn = fn
	e.embedDecodeExecCache = make(map[int]*mlctx.Exec)
	e.embedPerLayerDecodeCache = make(map[int]*mlctx.Exec)
}


// Submit submits a generation request with pre-tokenized input.
//
// Returns channels for streaming output and errors. The caller should
// range over the output channel until it is closed, then read from the
// error channel (which will also be closed; nil error means success).
//
// The context controls cancellation: if ctx is cancelled, the engine will
// stop generating for this request, send the context error on errChan,
// and close both channels.
//
// auxData carries optional multimodal inputs (e.g., pre-computed image
// features from a vision encoder). Pass nil for text-only requests.
func (e *Engine) Submit(
	ctx context.Context,
	inputTokens []int32,
	opts RequestOptions,
	auxData *AuxData,
) (<-chan SequenceDelta, <-chan error, error) {
	if len(inputTokens) == 0 {
		return nil, nil, ErrPromptEmpty
	}

	if opts.MaxNewTokens <= 0 {
		opts.MaxNewTokens = 100
	}

	outputChan := make(chan SequenceDelta, 256)
	errChan := make(chan error, 1)

	req := &engineRequest{
		inputTokens: make([]int32, len(inputTokens)),
		opts:        opts,
		auxData:     auxData,
		outputChan:  outputChan,
		errChan:     errChan,
		ctx:         ctx,
		startTime:   time.Now(),
	}
	copy(req.inputTokens, inputTokens)

	// Atomic check-and-send: RLock prevents Stop from closing submitCh
	// while we're checking stopped + sending.
	e.submitMu.RLock()
	if e.stopped.Load() {
		e.submitMu.RUnlock()
		return nil, nil, ErrEngineStopped
	}
	select {
	case e.submitCh <- req:
	case <-ctx.Done():
		e.submitMu.RUnlock()
		return nil, nil, ctx.Err()
	}
	e.submitMu.RUnlock()

	return outputChan, errChan, nil
}

// Stop signals the engine to stop and waits for the background loop to drain.
// After Stop returns, no further requests will be processed. Submit calls
// after Stop returns will return ErrEngineStopped.
func (e *Engine) Stop() {
	e.stopOnce.Do(func() {
		e.stopped.Store(true)
		close(e.stopCh)
	})
	e.wg.Wait()

	// Drain any requests that were queued after the loop exited.
	// These requests were never added to e.requests, so finishAllRequests
	// did not touch them. Use failRequest which safely closes channels.
	e.submitMu.Lock()
	defer e.submitMu.Unlock()
	for {
		select {
		case req := <-e.submitCh:
			e.failRequest(req, ErrEngineStopped)
		default:
			return
		}
	}
}

// SetSpeculativeConfig enables speculative decoding with the given config.
// Prefer setting Config.Speculative before construction.
// If called after construction, must be called before submitting requests.
func (e *Engine) SetSpeculativeConfig(config SpeculativeConfig) {
	if config.NumSpecTokens <= 0 {
		config.NumSpecTokens = 4
	}
	e.mu.Lock()
	e.specConfig = &config
	// Invalidate cached executors compiled for the old speculative config.
	e.verifyExec = nil
	e.draftExecCache = nil
	e.mu.Unlock()
}

// EnablePreemption enables request preemption with the given policy.
// Prefer setting Config.Preemption before construction.
// If called after construction, must be called before submitting requests.
func (e *Engine) EnablePreemption(policy PreemptionPolicy) {
	e.mu.Lock()
	e.preemptMgr = newPreemptionManager(policy)
	e.mu.Unlock()
}

// initPromptExec lazily initializes the prompt executor.
func (e *Engine) initPromptExec() error {
	if e.promptExec != nil {
		return nil
	}

	var err error
	e.promptExec, err = mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, tokens *Node) *Node {
			logits := e.modelFn(ctx, tokens, 0)
			// Extract last token logits: [batch, seqLen, vocab] -> [batch, vocab]
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// getGenExec returns a cached executor for the given position, creating one if needed.
func (e *Engine) getGenExec(position int) (*mlctx.Exec, error) {
	if exec, ok := e.genExecCache[position]; ok {
		return exec, nil
	}

	exec, err := mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, token *Node) *Node {
			tokenReshaped := ExpandDims(token, -1) // [batch] -> [batch, 1]
			logits := e.modelFn(ctx, tokenReshaped, position)
			lastLogits := Squeeze(logits, 1) // [batch, 1, vocab] -> [batch, vocab]
			return lastLogits
		},
	)
	if err != nil {
		return nil, err
	}
	e.genExecCache[position] = exec
	return exec, nil
}

// greedySample returns the index of the maximum value in logits (argmax).
// This is CPU-side greedy sampling used by sequential mode and speculative decoding.
func greedySample(logits []float32) int32 {
	maxIdx := int32(0)
	maxVal := logits[0]
	for i := int32(1); i < int32(len(logits)); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// initBatchedPromptExec lazily initializes the batched prompt executor.
// The prompt executor takes [1, promptLen] tokens + [1] positions and returns
// [1, vocabSize] logits. Prefills are done one-at-a-time because prompt
// lengths vary.
func (e *Engine) initBatchedPromptExec() error {
	if e.batchedPromptExec != nil {
		return nil
	}

	cfg := e.unifiedKVConfig
	var err error
	// Note: flat KV cache creates variables during graph construction, so we
	// use Checked(false) to allow both creating new kv_cache variables and
	// reusing existing weight variables.
	e.batchedPromptExec, err = mlctx.NewExec(e.backend, e.modelCtx.Checked(false),
		func(ctx *mlctx.Context, tokens *Node, positions *Node) *Node {
			bs := tokens.Shape().Dimensions[0]
			kv := attention.NewFlatKVCacheAccessor(bs, cfg.NumKVHeads, cfg.MaxSeqLen, cfg.HeadDim, cfg.DType, positions)
			logits := e.unifiedFn(ctx, tokens, positions, kv, nil)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// initMultimodalPromptExec lazily initializes the multimodal prompt executor.
// This is a separate compiled executor from the text-only prompt executor because
// it takes an additional imageFeatures tensor input. Only used with unifiedFn.
func (e *Engine) initMultimodalPromptExec() error {
	if e.batchedMultimodalExec != nil {
		return nil
	}
	cfg := e.unifiedKVConfig
	var err error
	// Note: flat KV cache creates variables during graph construction, so we
	// use Checked(false) to allow both creating new kv_cache variables and
	// reusing existing weight variables.
	e.batchedMultimodalExec, err = mlctx.NewExec(e.backend, e.modelCtx.Checked(false),
		func(ctx *mlctx.Context, tokens *Node, positions *Node, imageFeatures *Node) *Node {
			bs := tokens.Shape().Dimensions[0]
			kv := attention.NewFlatKVCacheAccessor(bs, cfg.NumKVHeads, cfg.MaxSeqLen, cfg.HeadDim, cfg.DType, positions)
			aux := &decode.AuxInputs{ImageFeatures: imageFeatures}
			logits := e.unifiedFn(ctx, tokens, positions, kv, aux)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// initEmbedPromptExec lazily initializes the prompt executor for embed mode
// (without per-layer inputs). Takes (inputsEmbeds, positions) as inputs.
func (e *Engine) initEmbedPromptExec() error {
	if e.embedPromptExec != nil {
		return nil
	}
	cfg := e.unifiedKVConfig
	var err error
	e.embedPromptExec, err = mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, inputsEmbeds *Node, positions *Node) *Node {
			bs := inputsEmbeds.Shape().Dimensions[0]
			kv := attention.NewFlatKVCacheAccessor(bs, cfg.NumKVHeads, cfg.MaxSeqLen, cfg.HeadDim, cfg.DType, positions)
			aux := &decode.AuxInputs{InputsEmbeds: inputsEmbeds}
			logits := e.unifiedFn(ctx, inputsEmbeds, positions, kv, aux)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// initEmbedPerLayerPromptExec lazily initializes the prompt executor for embed
// mode with per-layer inputs. Takes (inputsEmbeds, perLayerInputs, positions).
func (e *Engine) initEmbedPerLayerPromptExec() error {
	if e.embedPerLayerPromptExec != nil {
		return nil
	}
	cfg := e.unifiedKVConfig
	var err error
	e.embedPerLayerPromptExec, err = mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, inputsEmbeds *Node, perLayerInputs *Node, positions *Node) *Node {
			bs := inputsEmbeds.Shape().Dimensions[0]
			kv := attention.NewFlatKVCacheAccessor(bs, cfg.NumKVHeads, cfg.MaxSeqLen, cfg.HeadDim, cfg.DType, positions)
			aux := &decode.AuxInputs{InputsEmbeds: inputsEmbeds, PerLayerInputs: perLayerInputs}
			logits := e.unifiedFn(ctx, inputsEmbeds, positions, kv, aux)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// getEmbedDecodeExec returns a cached decode executor for embed mode (no per-layer).
func (e *Engine) getEmbedDecodeExec(paddedBatch int) (*mlctx.Exec, error) {
	if exec, ok := e.embedDecodeExecCache[paddedBatch]; ok {
		return exec, nil
	}

	cfg := e.unifiedKVConfig
	exec, err := mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, inputsEmbeds *Node, positions *Node, cacheWritePositions *Node) *Node {
			bs := inputsEmbeds.Shape().Dimensions[0]
			kv := attention.NewFlatKVCacheAccessor(bs, cfg.NumKVHeads, cfg.MaxSeqLen, cfg.HeadDim, cfg.DType, cacheWritePositions)
			aux := &decode.AuxInputs{InputsEmbeds: inputsEmbeds, CacheWritePositions: cacheWritePositions}
			logits := e.unifiedFn(ctx, inputsEmbeds, positions, kv, aux)
			lastLogits := Squeeze(logits, 1)
			sampled := sample.Greedy(lastLogits)
			return sampled
		},
	)
	if err != nil {
		return nil, err
	}
	e.embedDecodeExecCache[paddedBatch] = exec
	return exec, nil
}

// getEmbedPerLayerDecodeExec returns a cached decode executor for embed mode with per-layer inputs.
func (e *Engine) getEmbedPerLayerDecodeExec(paddedBatch int) (*mlctx.Exec, error) {
	if exec, ok := e.embedPerLayerDecodeCache[paddedBatch]; ok {
		return exec, nil
	}

	cfg := e.unifiedKVConfig
	exec, err := mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, inputsEmbeds *Node, perLayerInputs *Node, positions *Node, cacheWritePositions *Node) *Node {
			bs := inputsEmbeds.Shape().Dimensions[0]
			kv := attention.NewFlatKVCacheAccessor(bs, cfg.NumKVHeads, cfg.MaxSeqLen, cfg.HeadDim, cfg.DType, cacheWritePositions)
			aux := &decode.AuxInputs{InputsEmbeds: inputsEmbeds, PerLayerInputs: perLayerInputs, CacheWritePositions: cacheWritePositions}
			logits := e.unifiedFn(ctx, inputsEmbeds, positions, kv, aux)
			lastLogits := Squeeze(logits, 1)
			sampled := sample.Greedy(lastLogits)
			return sampled
		},
	)
	if err != nil {
		return nil, err
	}
	e.embedPerLayerDecodeCache[paddedBatch] = exec
	return exec, nil
}

// getBatchedDecodeExec returns a cached decode executor for the given padded
// batch size, creating one if needed. The executor takes:
//   - tokens: [paddedBatch, 1] int32
//   - positions: [paddedBatch] int32
//
// and returns logits [paddedBatch, vocabSize] with greedy sampling applied.
func (e *Engine) getBatchedDecodeExec(paddedBatch int) (*mlctx.Exec, error) {
	if exec, ok := e.batchedDecodeExecCache[paddedBatch]; ok {
		return exec, nil
	}

	cfg := e.unifiedKVConfig
	exec, err := mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, tokens *Node, positions *Node, cacheWritePositions *Node) *Node {
			// The KV cache accessor uses cacheWritePositions for mask and cache writes.
			// positions carries the absolute sequence position (for RoPE/position_ids).
			bs := tokens.Shape().Dimensions[0]
			kv := attention.NewFlatKVCacheAccessor(bs, cfg.NumKVHeads, cfg.MaxSeqLen, cfg.HeadDim, cfg.DType, cacheWritePositions)
			aux := &decode.AuxInputs{CacheWritePositions: cacheWritePositions}
			logits := e.unifiedFn(ctx, tokens, positions, kv, aux)
			lastLogits := Squeeze(logits, 1)
			sampled := sample.Greedy(lastLogits)
			return sampled
		},
	)
	if err != nil {
		return nil, err
	}
	e.batchedDecodeExecCache[paddedBatch] = exec
	return exec, nil
}

// buildPaddedTokens creates a [paddedBatch, 1] int32 token tensor,
// [paddedBatch] int32 positions tensor, and [paddedBatch] int32 cache write
// positions tensor from the batch. Padding elements use token 0 and position 0
// (their outputs are discarded).
//
// cacheWritePositions carries the KV cache write position for each element.
// After compaction, this differs from positions (which carries the absolute
// sequence position for RoPE). When no compaction has occurred, cacheWritePositions
// equals positions.
func buildPaddedTokens(b *batch, paddedSize int) (tokens [][]int32, positions []int32, cacheWritePositions []int32) {
	tokens = make([][]int32, paddedSize)
	positions = make([]int32, paddedSize)
	cacheWritePositions = make([]int32, paddedSize)
	for i, req := range b.requests {
		prevToken := req.generatedTokens[len(req.generatedTokens)-1]
		tokens[i] = []int32{prevToken}
		positions[i] = b.positions[i]
		if b.cacheWritePositions != nil {
			cacheWritePositions[i] = b.cacheWritePositions[i]
		} else {
			cacheWritePositions[i] = b.positions[i]
		}
	}
	// Fill padding slots with zeros.
	for i := len(b.requests); i < paddedSize; i++ {
		tokens[i] = []int32{0}
		positions[i] = 0
		cacheWritePositions[i] = 0
	}
	return
}

// addRequest assigns an ID (and in batched mode, a KV cache slot) to the request.
// Block allocation and preemption happen without holding e.mu to avoid deadlocks.
func (e *Engine) addRequest(req *engineRequest) {
	// Assign ID under lock, then release for allocations.
	e.mu.Lock()
	req.id = e.nextID
	e.nextID++
	e.mu.Unlock()

	if e.batchedMode {
		slot, err := e.slotMgr.Allocate(req.id)
		if err != nil {
			e.failRequest(req, err)
			return
		}
		req.slot = slot
	}

	// In paged mode, pre-allocate blocks for the full sequence
	// (prompt + all generated tokens). This avoids needing to grow blocks
	// during decode, which could fail when free blocks are exhausted.
	if e.pagedMode {
		promptLen := len(req.inputTokens)
		blocksNeeded := promptLen + req.opts.MaxNewTokens

		// Check prefix cache for reusable KV blocks.
		hash := kvcache.HashTokens(req.inputTokens)
		req.prefixHash = hash
		if cachedBlocks, cachedTokens, ok := e.prefixCache.LookupAndRef(hash); ok {
			// Cache hit -- reuse prefix blocks and only allocate for the rest.
			req.prefixBlocks = cachedBlocks
			req.prefixLen = cachedTokens
			req.hasPrefixHit = true
			blocksNeeded = max(promptLen-cachedTokens+req.opts.MaxNewTokens, 1)
		}

		err := e.blockMgr.EnsureBlocks(req.id, blocksNeeded)

		// If allocation failed, try evicting prefix cache entries first
		// (cheaper than preemption — no request needs to re-prefill).
		if err != nil && e.prefixCache != nil {
			for err != nil {
				freed := e.prefixCache.EvictLRU()
				if len(freed) == 0 {
					break
				}
				e.blockMgr.RecycleBlocks(freed)
				err = e.blockMgr.EnsureBlocks(req.id, blocksNeeded)
			}
		}

		// If still not enough, try preemption (may need multiple victims).
		if err != nil && e.preemptMgr != nil {
			for err != nil {
				victimID := e.preemptLowestPriority()
				if victimID == 0 {
					break
				}
				err = e.blockMgr.EnsureBlocks(req.id, blocksNeeded)
			}
		}

		if err != nil {
			if e.batchedMode {
				e.slotMgr.Free(req.slot)
			}
			// Unref prefix blocks if we had a cache hit.
			if req.hasPrefixHit {
				freed := e.prefixCache.Unref(req.prefixBlocks)
				if len(freed) > 0 {
					e.blockMgr.RecycleBlocks(freed)
				}
			}
			e.failRequest(req, err)
			return
		}
	}

	e.mu.Lock()
	e.requests[req.id] = req
	e.mu.Unlock()
}

// failRequest sends an error on the request's channels and closes them.
// Used when a request cannot be admitted (no slots, no blocks, etc.).
func (e *Engine) failRequest(req *engineRequest, err error) {
	select {
	case req.errChan <- err:
	default:
	}
	close(req.outputChan)
	close(req.errChan)
}

// removeRequest removes a request from the active map.
func (e *Engine) removeRequest(id uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.requests, id)
}

// hasActiveRequests returns true if there are any in-flight requests.
func (e *Engine) hasActiveRequests() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.requests) > 0
}

// resetKVCache clears KV cache variables between requests.
func (e *Engine) resetKVCache() {
	attention.KVCacheReset(e.modelCtx)
}

// maxBlocksPerRequest returns the maximum number of paged KV cache blocks
// a single request can occupy. This is a compile-time constant for executors.
func (e *Engine) maxBlocksPerRequest() int {
	return (e.config.MaxSeqLen + e.pagedCfg.BlockSize - 1) / e.pagedCfg.BlockSize
}

// initPagedPromptExec lazily initializes the paged prompt executor.
// Takes [1, promptLen] tokens, [1] positions, and [1, maxBlocksPerReq] page tables.
// Returns [1, vocabSize] logits (last-token extracted).
func (e *Engine) initPagedPromptExec() error {
	if e.pagedPromptExec != nil {
		return nil
	}
	cfg := e.pagedCfg
	maxBlocks := e.maxBlocksPerRequest()
	var err error
	// Use Reuse().Checked(false) instead of just Reuse() to allow KV cache variables
	// to be created on first compilation while still reusing existing weight variables.
	// Plain Reuse() panics when KV cache variables don't exist yet; Checked(false)
	// permits dynamic creation alongside reuse.
	e.pagedPromptExec, err = mlctx.NewExec(e.backend, e.modelCtx.Reuse().Checked(false),
		func(ctx *mlctx.Context, tokens *Node, positions *Node, pageTables *Node) *Node {
			kv := &kvcache.PagedKVCacheAccessor{
				Config:        cfg,
				PageTables:    pageTables,
				Positions:     positions,
				ReadNumBlocks: maxBlocks,
			}
			logits := e.unifiedFn(ctx, tokens, positions, kv, nil)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// initPagedMultimodalPromptExec lazily initializes the paged multimodal prompt executor.
// Takes [1, promptLen] tokens, [1] positions, [1, maxBlocksPerReq] page tables,
// and [1, numPatches, hiddenDim] image features.
// Returns [1, vocabSize] logits (last-token extracted).
func (e *Engine) initPagedMultimodalPromptExec() error {
	if e.pagedMultimodalExec != nil {
		return nil
	}
	cfg := e.pagedCfg
	maxBlocks := e.maxBlocksPerRequest()
	var err error
	e.pagedMultimodalExec, err = mlctx.NewExec(e.backend, e.modelCtx.Reuse().Checked(false),
		func(ctx *mlctx.Context, tokens *Node, positions *Node, pageTables *Node, imageFeatures *Node) *Node {
			kv := &kvcache.PagedKVCacheAccessor{
				Config:        cfg,
				PageTables:    pageTables,
				Positions:     positions,
				ReadNumBlocks: maxBlocks,
			}
			aux := &decode.AuxInputs{ImageFeatures: imageFeatures}
			logits := e.unifiedFn(ctx, tokens, positions, kv, aux)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// getPagedDecodeExec returns a cached paged decode executor for the given
// padded batch size. The executor takes:
//   - tokens: [paddedBatch, 1] int32
//   - positions: [paddedBatch] int32 (absolute positions for RoPE)
//   - cacheWritePositions: [paddedBatch] int32 (logical positions for cache write/mask)
//   - pageTables: [paddedBatch, maxBlocksPerReq] int32
//
// Returns sampled token IDs [paddedBatch] int32.
func (e *Engine) getPagedDecodeExec(paddedBatch int) (*mlctx.Exec, error) {
	if exec, ok := e.pagedDecodeExecCache[paddedBatch]; ok {
		return exec, nil
	}

	cfg := e.pagedCfg
	maxBlocks := e.maxBlocksPerRequest()
	// See comment in initPagedPromptExec for why Checked(false) is needed.
	exec, err := mlctx.NewExec(e.backend, e.modelCtx.Reuse().Checked(false),
		func(ctx *mlctx.Context, tokens *Node, positions *Node, cacheWritePositions *Node, pageTables *Node) *Node {
			kv := &kvcache.PagedKVCacheAccessor{
				Config:        cfg,
				PageTables:    pageTables,
				Positions:     cacheWritePositions,
				ReadNumBlocks: maxBlocks,
			}
			aux := &decode.AuxInputs{CacheWritePositions: cacheWritePositions}
			logits := e.unifiedFn(ctx, tokens, positions, kv, aux)
			lastLogits := Squeeze(logits, 1)
			sampled := sample.Greedy(lastLogits)
			return sampled
		},
	)
	if err != nil {
		return nil, err
	}
	e.pagedDecodeExecCache[paddedBatch] = exec
	return exec, nil
}

// buildPrefillPageTableTensor creates a [1, maxBlocksPerReq] int32 tensor
// from a request's page table for paged prefill.
func (e *Engine) buildPrefillPageTableTensor(req *engineRequest) *tensors.Tensor {
	maxBlocks := e.maxBlocksPerRequest()
	pt := e.blockMgr.GetPageTable(req.id)
	row := make([]int32, maxBlocks)
	for i, block := range pt {
		if i >= maxBlocks {
			break
		}
		row[i] = int32(block)
	}
	return tensors.FromValue([][]int32{row})
}

// buildBatchPageTableTensor creates a [paddedBatch, maxBlocksPerReq] int32 tensor
// from the page tables of all requests in a decode batch.
func (e *Engine) buildBatchPageTableTensor(b *batch, paddedSize int) *tensors.Tensor {
	maxBlocks := e.maxBlocksPerRequest()
	result := make([][]int32, paddedSize)
	for i := range paddedSize {
		result[i] = make([]int32, maxBlocks)
		if i < len(b.requests) {
			pt := e.blockMgr.GetPageTable(b.requests[i].id)
			for j, block := range pt {
				if j >= maxBlocks {
					break
				}
				result[i][j] = int32(block)
			}
		}
	}
	return tensors.FromValue(result)
}
