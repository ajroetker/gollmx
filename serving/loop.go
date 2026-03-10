package serving

import (
	"fmt"
	"runtime"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"k8s.io/klog/v2"
)

// runStepLoop is the background goroutine that processes all requests.
// It follows the same pattern as ortgenai's Engine: drain submissions,
// step one request, stream tokens, check cancellations, repeat.
func (e *Engine) runStepLoop() {
	defer e.wg.Done()

	for {
		// 0. Try to restore preempted requests if blocks are available.
		e.tryRestorePreempted()

		// 1. Drain all pending submissions (non-blocking).
		e.drainSubmissions()

		// 2. Check for stop signal.
		select {
		case <-e.stopCh:
			e.finishAllRequests(ErrEngineStopped)
			return
		default:
		}

		// 3. If no active requests, block until a submission or stop.
		if !e.hasActiveRequests() {
			select {
			case req := <-e.submitCh:
				e.addRequest(req)
			case <-e.stopCh:
				e.finishAllRequests(ErrEngineStopped)
				return
			}
			continue
		}

		// 4. Step: process tokens.
		var stepErr error
		if e.batchedMode {
			stepErr = e.stepBatched()
		} else {
			stepErr = e.stepSequential()
		}
		if stepErr != nil {
			// Fatal engine error — notify all active requests and exit.
			e.finishAllRequests(stepErr)
			return
		}

		// 5. Check for cancelled contexts.
		e.checkCancellations()
	}
}

// stepSequential processes one generation step for a single request.
// Phase 1: only one request is active at a time.
func (e *Engine) stepSequential() error {
	// Pick the first active request.
	e.mu.Lock()
	var req *engineRequest
	for _, r := range e.requests {
		if !r.eosReached {
			req = r
			break
		}
	}
	e.mu.Unlock()

	if req == nil {
		runtime.Gosched()
		return nil
	}

	if req.position == 0 {
		return e.prefillRequest(req)
	}
	return e.decodeOneToken(req)
}

// prefillRequest processes the initial prompt through the model,
// samples the first token, and streams it back.
func (e *Engine) prefillRequest(req *engineRequest) error {
	// Reset KV cache for this new request.
	e.resetKVCache()

	// Initialize prompt executor if needed.
	if err := e.initPromptExec(); err != nil {
		e.finishRequest(req, fmt.Errorf("init prompt exec: %w", err))
		return nil // not a fatal engine error
	}

	// Build prompt tensor [1, promptLen].
	promptLen := len(req.inputTokens)
	prompt := tensors.FromValue([][]int32{req.inputTokens})

	// Execute prompt — returns logits [1, vocabSize].
	outputs, err := e.promptExec.Exec(prompt)
	if err != nil {
		e.finishRequest(req, fmt.Errorf("prompt execution: %w", err))
		return nil
	}

	// Sample from logits.
	logits := outputs[0]
	logitsFlat := logits.Value().([][]float32)[0]
	nextToken := greedySample(logitsFlat)

	// Update request state.
	req.position = promptLen
	req.absPosition = promptLen
	req.generatedTokens = append(req.generatedTokens, nextToken)

	// Stream token back and check completion.
	e.streamToken(req, nextToken)
	e.checkAndFinish(req)

	return nil
}

// decodeOneToken generates one new token for the given request.
func (e *Engine) decodeOneToken(req *engineRequest) error {
	position := req.absPosition

	// Get or create cached executor for this position.
	exec, err := e.getGenExec(position)
	if err != nil {
		e.finishRequest(req, fmt.Errorf("create gen exec at position %d: %w", position, err))
		return nil
	}

	// Previous token as [1] tensor (batch=1).
	prevToken := req.generatedTokens[len(req.generatedTokens)-1]
	prevTokenTensor := tensors.FromValue([]int32{prevToken})

	// Execute — returns logits [1, vocabSize].
	outputs, err := exec.Exec(prevTokenTensor)
	if err != nil {
		e.finishRequest(req, fmt.Errorf("gen step at position %d: %w", position, err))
		return nil
	}

	// Sample from logits.
	logits := outputs[0]
	logitsFlat := logits.Value().([][]float32)[0]
	nextToken := greedySample(logitsFlat)

	// Update state.
	req.position++
	req.absPosition++
	req.generatedTokens = append(req.generatedTokens, nextToken)

	// Stream token back and check completion.
	e.streamToken(req, nextToken)
	e.checkAndFinish(req)

	return nil
}

// drainSubmissions processes all pending submissions without blocking.
func (e *Engine) drainSubmissions() {
	for {
		select {
		case req := <-e.submitCh:
			e.addRequest(req)
		default:
			return
		}
	}
}

// checkAndFinish finishes the request if it has reached EOS or its token limit.
func (e *Engine) checkAndFinish(req *engineRequest) {
	if req.eosReached || len(req.generatedTokens) >= req.opts.MaxNewTokens {
		e.finishRequest(req, nil)
	}
}

// streamToken decodes a token and sends it to the request's output channel.
// It also selects on the engine's stop channel so sends never block shutdown.
func (e *Engine) streamToken(req *engineRequest, tokenID int32) {
	if req.eosReached {
		return
	}

	// Check EOS.
	if e.tokenizer.IsEOS(tokenID) {
		req.eosReached = true
		select {
		case req.outputChan <- SequenceDelta{TokenID: tokenID, EOSReached: true}:
		case <-req.ctx.Done():
		case <-e.stopCh:
		}
		return
	}

	// Decode token text.
	text, err := e.tokenizer.Decode(tokenID)
	if err != nil {
		// Non-fatal: send the token ID without text.
		text = ""
	}

	select {
	case req.outputChan <- SequenceDelta{Token: text, TokenID: tokenID}:
	case <-req.ctx.Done():
	case <-e.stopCh:
	}
}

// finishRequest closes channels for a single request and removes it from tracking.
func (e *Engine) finishRequest(req *engineRequest, err error) {
	if err != nil {
		select {
		case req.errChan <- err:
		default:
		}
	}
	close(req.outputChan)
	close(req.errChan)

	if e.batchedMode {
		// Free KV cache slot in batched mode.
		e.slotMgr.Free(req.slot)
	} else {
		// Reset incremental tokenizer state only in sequential mode.
		// In batched mode, multiple requests share the tokenizer concurrently,
		// so resetting here would corrupt other in-flight requests' multi-byte state.
		e.tokenizer.Reset()
	}

	// Free paged KV cache blocks. Detach prefix-cached blocks first so
	// ReleaseRequest only frees the non-prefix (generation) blocks.
	if e.pagedMode {
		if len(req.prefixBlocks) > 0 {
			e.blockMgr.DetachBlocks(req.id, req.prefixBlocks)
		}
		e.blockMgr.ReleaseRequest(req.id)
	}

	// Unref prefix cache blocks. If refcount drops to 0, return to free pool.
	if len(req.prefixBlocks) > 0 && e.prefixCache != nil {
		freed := e.prefixCache.Unref(req.prefixBlocks)
		if len(freed) > 0 {
			e.blockMgr.RecycleBlocks(freed)
		}
	}

	e.removeRequest(req.id)
}

// finishAllRequests sends an error to all active requests and closes their channels.
func (e *Engine) finishAllRequests(err error) {
	e.mu.Lock()
	reqs := make([]*engineRequest, 0, len(e.requests))
	for _, r := range e.requests {
		reqs = append(reqs, r)
	}
	e.mu.Unlock()

	for _, req := range reqs {
		e.finishRequest(req, err)
	}
}

// checkCancellations checks all active requests for cancelled contexts
// and removes them without blocking the step loop.
func (e *Engine) checkCancellations() {
	e.mu.Lock()
	var cancelled []*engineRequest
	for _, req := range e.requests {
		select {
		case <-req.ctx.Done():
			cancelled = append(cancelled, req)
		default:
		}
	}
	e.mu.Unlock()

	for _, req := range cancelled {
		e.finishRequest(req, req.ctx.Err())
	}
}

// tryRestorePreempted attempts to restore preempted requests when free blocks
// become available. In recompute mode, restored requests re-prefill their full
// token history (original prompt + generated tokens). In swap mode, block data
// would be copied back from CPU (not yet implemented for tensor data).
func (e *Engine) tryRestorePreempted() {
	pm := e.preemptMgr
	if pm == nil || pm.NumPreempted() == 0 {
		return
	}

	for _, reqID := range pm.PreemptedIDs() {
		entry := pm.Restore(reqID)
		if entry == nil || entry.req == nil {
			continue
		}

		req := entry.req

		// Try to allocate a new slot.
		if e.batchedMode {
			slot, err := e.slotMgr.Allocate(reqID)
			if err != nil {
				// No free slots -- re-preempt and try later.
				pm.Preempt(reqID, req, entry.pageTable)
				return
			}
			req.slot = slot
		}

		// Try to allocate blocks for re-prefill.
		// In recompute mode, we rebuild the full sequence from scratch.
		fullSeqLen := len(req.inputTokens) + len(entry.generatedTokens)
		if e.pagedMode {
			if err := e.blockMgr.EnsureBlocks(reqID, fullSeqLen+1); err != nil {
				// Not enough blocks yet -- re-preempt and try later.
				if e.batchedMode {
					e.slotMgr.Free(req.slot)
				}
				pm.Preempt(reqID, req, entry.pageTable)
				return
			}
		}

		// Restore the request: rebuild input tokens to include everything
		// generated so far, reset position to 0 for re-prefill.
		// (Both recompute and swap modes fall back to recompute for now —
		// actual KV tensor swap from CPU is not yet implemented.)
		rebuiltInput := make([]int32, 0, fullSeqLen)
		rebuiltInput = append(rebuiltInput, req.inputTokens...)
		rebuiltInput = append(rebuiltInput, entry.generatedTokens...)
		req.inputTokens = rebuiltInput
		req.generatedTokens = nil
		req.position = 0
		req.absPosition = 0

		// Re-add to active requests.
		e.mu.Lock()
		e.requests[reqID] = req
		e.mu.Unlock()
	}
}

// stepBatched processes one generation step using batched execution.
// It handles prefill and decode phases separately:
//   - Prefill: one request at a time (variable prompt lengths)
//   - Decode: batch multiple requests with per-element positions
func (e *Engine) stepBatched() error {
	// Collect active requests and snapshot config under lock.
	e.mu.Lock()
	active := make([]*engineRequest, 0, len(e.requests))
	for _, r := range e.requests {
		if !r.eosReached {
			active = append(active, r)
		}
	}
	specConfig := e.specConfig
	e.mu.Unlock()

	if len(active) == 0 {
		runtime.Gosched()
		return nil
	}

	// Priority 1: prefill any request that hasn't been processed yet.
	if prefillReq := e.sched.NextPrefillRequest(active); prefillReq != nil {
		return e.prefillRequestBatched(prefillReq)
	}

	// Priority 2: decode active requests.
	// If speculative decoding is configured, use it per-request.
	if specConfig != nil {
		for _, req := range active {
			if req.eosReached || req.position == 0 {
				continue
			}
			if err := e.speculativeDecode(req); err != nil {
				return err
			}
		}
		return nil
	}

	// Normal batched decode.
	b := e.sched.FormDecodeBatch(active)
	if b == nil {
		runtime.Gosched()
		return nil
	}
	return e.decodeBatched(b)
}

// prefillRequestBatched processes a single request's prompt through the
// batched model. Uses batch size 1 since prompt lengths vary.
//
// If the request has a prefix cache hit, only the tokens after the cached
// prefix are processed (the KV for prefix tokens is already in the cache).
// On a cache miss, the full prompt is processed and the resulting blocks
// are stored in the prefix cache for future reuse.
func (e *Engine) prefillRequestBatched(req *engineRequest) error {
	promptTokens := req.inputTokens
	startPos := int32(0)

	// If prefix cache hit, skip the cached prefix tokens.
	if req.hasPrefixHit && req.prefixLen > 0 {
		if req.prefixLen < len(req.inputTokens) {
			promptTokens = req.inputTokens[req.prefixLen:]
			startPos = int32(req.prefixLen)
		} else {
			promptTokens = req.inputTokens[len(req.inputTokens)-1:]
			startPos = int32(len(req.inputTokens) - 1)
		}
	}

	// Build prompt tensor [1, promptLen] and positions tensor [1].
	prompt := tensors.FromValue([][]int32{promptTokens})
	positions := tensors.FromValue([]int32{startPos})

	var outputs []*tensors.Tensor
	var err error

	if e.embedFn != nil {
		// Eager embedding mode: call EmbedFn to convert tokens → embeddings.
		auxData, embedErr := e.embedFn(prompt, req.auxData)
		if embedErr != nil {
			e.finishRequest(req, fmt.Errorf("embed fn: %w", embedErr))
			return nil
		}

		hasPerLayer := auxData.PerLayerInputs != nil
		if hasPerLayer {
			if err := e.initEmbedPerLayerPromptExec(); err != nil {
				e.finishRequest(req, fmt.Errorf("init embed+perLayer prompt exec: %w", err))
				return nil
			}
			outputs, err = e.embedPerLayerPromptExec.Exec(auxData.InputsEmbeds, auxData.PerLayerInputs, positions)
		} else {
			if err := e.initEmbedPromptExec(); err != nil {
				e.finishRequest(req, fmt.Errorf("init embed prompt exec: %w", err))
				return nil
			}
			outputs, err = e.embedPromptExec.Exec(auxData.InputsEmbeds, positions)
		}
	} else {
		// Standard mode: pass tokens directly to the compiled graph.
		hasAux := req.auxData != nil && req.auxData.ImageFeatures != nil
		if hasAux {
			if e.pagedMode {
				e.finishRequest(req, fmt.Errorf("multimodal inputs not yet supported with paged KV cache"))
				return nil
			}
			if err := e.initMultimodalPromptExec(); err != nil {
				e.finishRequest(req, fmt.Errorf("init multimodal prompt exec: %w", err))
				return nil
			}
			outputs, err = e.batchedMultimodalExec.Exec(prompt, positions, req.auxData.ImageFeatures)
		} else if e.pagedMode {
			if err := e.initPagedPromptExec(); err != nil {
				e.finishRequest(req, fmt.Errorf("init paged prompt exec: %w", err))
				return nil
			}
			pt := e.buildPrefillPageTableTensor(req)
			outputs, err = e.pagedPromptExec.Exec(prompt, positions, pt)
		} else {
			if err := e.initBatchedPromptExec(); err != nil {
				e.finishRequest(req, fmt.Errorf("init batched prompt exec: %w", err))
				return nil
			}
			outputs, err = e.batchedPromptExec.Exec(prompt, positions)
		}
	}

	if err != nil {
		e.finishRequest(req, fmt.Errorf("batched prompt execution: %w", err))
		return nil
	}

	// CPU-side greedy sampling from logits [1, vocabSize].
	logits := outputs[0]
	logitsFlat := logits.Value().([][]float32)[0]
	nextToken := greedySample(logitsFlat)

	// Update request state.
	req.position = len(req.inputTokens)
	req.absPosition = len(req.inputTokens)
	req.generatedTokens = append(req.generatedTokens, nextToken)
	req.auxData = nil // release aux data after prefill — not needed during decode

	// Compact KV cache if configured and sequence is long enough.
	if e.config.Compaction != nil {
		if err := e.compactKVCache(req); err != nil {
			klog.Warningf("KV cache compaction failed for request %d: %v", req.id, err)
		}
	}

	// Stream token back.
	e.streamToken(req, nextToken)

	// On a cache miss, register the prompt's blocks in the prefix cache so
	// future requests with the same prompt can reuse the KV data. Only cache
	// the blocks that contain prompt tokens (not generation blocks).
	//
	// The blocks are NOT detached from the request's page table here — the
	// request still needs them during decode. DetachBlocks happens later in
	// finishRequest, just before ReleaseRequest, so that ReleaseRequest
	// frees only the non-prefix (generation) blocks.
	if !req.hasPrefixHit && e.pagedMode {
		allBlocks := e.blockMgr.GetPageTable(req.id)
		promptBlockCount := (len(req.inputTokens) + e.pagedCfg.BlockSize - 1) / e.pagedCfg.BlockSize
		if promptBlockCount > len(allBlocks) {
			promptBlockCount = len(allBlocks)
		}
		promptBlocks := allBlocks[:promptBlockCount]
		if len(promptBlocks) > 0 {
			evicted := e.prefixCache.Store(req.prefixHash, promptBlocks, len(req.inputTokens))
			if len(evicted) > 0 {
				e.blockMgr.RecycleBlocks(evicted)
			}
			e.prefixCache.Ref(promptBlocks)
			req.prefixBlocks = make([]int, len(promptBlocks))
			copy(req.prefixBlocks, promptBlocks)
		}
	}

	e.checkAndFinish(req)
	return nil
}

// decodeBatched processes one decode step for a batch of requests.
// Positions are per-element tensors; batch size is padded to power of 2.
func (e *Engine) decodeBatched(b *batch) error {
	padded := paddedBatchSize(len(b.requests))

	// Build padded input tensors.
	tokens, positions, cacheWritePos := buildPaddedTokens(b, padded)
	tokensTensor := tensors.FromValue(tokens)
	positionsTensor := tensors.FromValue(positions)
	cacheWritePosTensor := tensors.FromValue(cacheWritePos)

	var outputs []*tensors.Tensor
	var err error

	if e.embedFn != nil {
		// Eager embedding mode: call EmbedFn then use embed-mode executor.
		auxData, embedErr := e.embedFn(tokensTensor, nil)
		if embedErr != nil {
			for _, req := range b.requests {
				e.finishRequest(req, fmt.Errorf("decode embed fn: %w", embedErr))
			}
			return nil
		}

		hasPerLayer := auxData.PerLayerInputs != nil
		if hasPerLayer {
			exec, execErr := e.getEmbedPerLayerDecodeExec(padded)
			if execErr != nil {
				return fmt.Errorf("create embed+perLayer decode exec (padded=%d): %w", padded, execErr)
			}
			outputs, err = exec.Exec(auxData.InputsEmbeds, auxData.PerLayerInputs, positionsTensor, cacheWritePosTensor)
		} else {
			exec, execErr := e.getEmbedDecodeExec(padded)
			if execErr != nil {
				return fmt.Errorf("create embed decode exec (padded=%d): %w", padded, execErr)
			}
			outputs, err = exec.Exec(auxData.InputsEmbeds, positionsTensor, cacheWritePosTensor)
		}
	} else if e.pagedMode {
		// Paged mode: pass page tables to use paged KV cache.
		exec, execErr := e.getPagedDecodeExec(padded)
		if execErr != nil {
			return fmt.Errorf("create paged decode exec (padded=%d): %w", padded, execErr)
		}
		pt := e.buildBatchPageTableTensor(b, padded)
		outputs, err = exec.Exec(tokensTensor, positionsTensor, cacheWritePosTensor, pt)
	} else {
		// Standard flat mode: pass tokens directly.
		exec, execErr := e.getBatchedDecodeExec(padded)
		if execErr != nil {
			return fmt.Errorf("create batched decode exec (padded=%d): %w", padded, execErr)
		}
		outputs, err = exec.Exec(tokensTensor, positionsTensor, cacheWritePosTensor)
	}

	if err != nil {
		for _, req := range b.requests {
			e.finishRequest(req, fmt.Errorf("batched decode step: %w", err))
		}
		return nil
	}

	// outputs[0] is sampled token IDs [paddedBatch] int32.
	sampledTensor := outputs[0]
	sampledTokens := sampledTensor.Value().([]int32)

	// Distribute results to individual requests.
	for i, req := range b.requests {
		nextToken := sampledTokens[i]
		req.position++
		req.absPosition++
		req.generatedTokens = append(req.generatedTokens, nextToken)
		e.streamToken(req, nextToken)
		e.checkAndFinish(req)
	}

	return nil
}
