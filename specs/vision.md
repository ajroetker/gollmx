# Vision Model Support for huggingface-gomlx

## Summary

Add support for vision encoder architectures and encoder-decoder models to enable Vision2Seq tasks (OCR, document understanding, image captioning).

## Motivation

Vision2Seq models like TrOCR, Donut, and Florence-2 are widely used for:
- Optical Character Recognition (OCR)
- Document understanding and parsing
- Image captioning
- Visual question answering

Currently these models require ONNX Runtime. Adding native GoMLX support would enable:
- Fine-tuning vision models with GoMLX's autodiff
- Unified inference pipeline without ONNX dependency
- Better integration with existing huggingface-gomlx models

## Target Models

| Model | Encoder | Decoder | Priority |
|-------|---------|---------|----------|
| TrOCR | ViT | RoBERTa/BART | High |
| Donut | Swin | BART | Medium |
| Florence-2 | DaViT | BART | Low |
| CLIP (vision) | ViT | - | High |

## Architecture Overview

```
Vision2Seq Model
├── Vision Encoder (ViT, Swin, etc.)
│   ├── Patch Embedding
│   ├── Position Embedding
│   └── Transformer Encoder Layers
│       ├── Self-Attention
│       ├── FFN (with GELU)
│       └── LayerNorm
│
├── Encoder-Decoder Projection (optional)
│
└── Text Decoder (with cross-attention)
    ├── Token Embedding
    ├── Position Embedding
    └── Transformer Decoder Layers
        ├── Masked Self-Attention (with KV-cache)
        ├── Cross-Attention to encoder
        ├── FFN
        └── LayerNorm
```

## Implementation Plan

### Phase 1: Common Components

#### 1.1 GELU Activation
- **Status**: ✅ Added to onnx-gomlx (PR #49)
- **Location**: Already in GoMLX `activations.Gelu()`

#### 1.2 Cross-Attention Layer
- **File**: `architectures/common/cross_attention.go`
- **Interface**:
```go
// CrossAttention computes attention where queries come from one sequence
// and keys/values come from another (encoder output).
func CrossAttention(
    ctx *context.Context,
    query *Node,           // [batch, seq, hidden] from decoder
    encoderHidden *Node,   // [batch, enc_seq, enc_hidden] from encoder
    encoderMask *Node,     // [batch, enc_seq] attention mask
    numHeads int,
    headDim int,
) *Node
```

#### 1.3 KV-Cache Support
- **File**: `architectures/common/kv_cache.go`
- **Interface**:
```go
// KVCache stores past key/value states for efficient autoregressive decoding.
type KVCache struct {
    Keys   []*Node  // Per-layer key cache: [batch, heads, seq, head_dim]
    Values []*Node  // Per-layer value cache
    SeqLen int      // Current sequence length
}

// AttentionWithCache performs attention using and updating the KV cache.
func AttentionWithCache(
    ctx *context.Context,
    query, key, value *Node,
    cache *KVCache,
    layerIdx int,
    causal bool,
) (*Node, *KVCache)
```

### Phase 2: Vision Transformer (ViT)

#### 2.1 Patch Embedding
- **File**: `architectures/vit/patch_embed.go`
- Convert image to sequence of patch embeddings
- Two variants:
  - Conv2D with kernel_size=patch_size, stride=patch_size
  - Linear projection of flattened patches

```go
// PatchEmbedding converts an image to a sequence of patch embeddings.
// Input: [batch, channels, height, width]
// Output: [batch, num_patches, embed_dim]
func PatchEmbedding(
    ctx *context.Context,
    image *Node,
    patchSize int,
    embedDim int,
    useConv bool,  // Conv2D vs linear projection
) *Node
```

#### 2.2 ViT Architecture
- **File**: `architectures/vit/vit.go`
- **Config**:
```go
type Config struct {
    *models.BaseConfig
    ImageSize      int     // e.g., 224, 384
    PatchSize      int     // e.g., 16, 32
    NumChannels    int     // e.g., 3
    HiddenSize     int     // e.g., 768
    NumLayers      int     // e.g., 12
    NumHeads       int     // e.g., 12
    IntermediateSize int   // e.g., 3072
    HiddenDropout  float64
    AttentionDropout float64
    LayerNormEps   float64
    UseCLS         bool    // Whether to use CLS token
}
```

- **Weight Mapping** (HuggingFace ViT):
```
vit.embeddings.patch_embeddings.projection.weight -> patch_embed/conv/weights
vit.embeddings.patch_embeddings.projection.bias   -> patch_embed/conv/bias
vit.embeddings.cls_token                          -> cls_token
vit.embeddings.position_embeddings                -> position_embeddings
vit.encoder.layer.{i}.attention.attention.query   -> layers/{i}/attention/query
vit.encoder.layer.{i}.attention.attention.key     -> layers/{i}/attention/key
vit.encoder.layer.{i}.attention.attention.value   -> layers/{i}/attention/value
vit.encoder.layer.{i}.attention.output.dense      -> layers/{i}/attention/output
vit.encoder.layer.{i}.layernorm_before            -> layers/{i}/norm1
vit.encoder.layer.{i}.intermediate.dense          -> layers/{i}/mlp/fc1
vit.encoder.layer.{i}.output.dense                -> layers/{i}/mlp/fc2
vit.encoder.layer.{i}.layernorm_after             -> layers/{i}/norm2
vit.layernorm                                     -> norm
```

#### 2.3 Model Types to Register
```go
models.RegisterArchitecture("vit", ...)
models.RegisterArchitecture("deit", ...)  // Same as ViT
```

### Phase 3: Encoder-Decoder Models

#### 3.1 Vision Encoder-Decoder Wrapper
- **File**: `architectures/vision_encoder_decoder/ved.go`
- Combines any vision encoder with any text decoder
- Handles dimension projection between encoder and decoder

```go
type VisionEncoderDecoder struct {
    Encoder   ArchitectureBuilder  // ViT, Swin, etc.
    Decoder   ArchitectureBuilder  // BART, RoBERTa, etc.
    EncDecProj *Node               // Optional projection layer
}

func (v *VisionEncoderDecoder) Forward(
    ctx *context.Context,
    pixelValues *Node,      // [batch, channels, height, width]
    decoderInputIDs *Node,  // [batch, seq]
    decoderAttentionMask *Node,
    kvCache *KVCache,
) (*Node, *KVCache)
```

#### 3.2 TrOCR Model
- **File**: `architectures/trocr/trocr.go`
- Encoder: ViT (microsoft/trocr-* models)
- Decoder: RoBERTa-style with cross-attention

#### 3.3 Model Types to Register
```go
models.RegisterArchitecture("trocr", ...)
models.RegisterArchitecture("vision-encoder-decoder", ...)
```

### Phase 4: Swin Transformer (Optional)

#### 4.1 Windowed Attention
- **File**: `architectures/swin/window_attention.go`
- Compute attention within local windows
- Shifted window mechanism for cross-window connections

#### 4.2 Swin Architecture
- **File**: `architectures/swin/swin.go`
- Hierarchical structure with patch merging
- More complex than ViT

### Phase 5: Generation Support

#### 5.1 Autoregressive Generation
- **File**: `generation/generate.go`
- Greedy decoding
- Beam search
- Sampling (temperature, top-k, top-p)

```go
type GenerationConfig struct {
    MaxNewTokens      int
    MinLength         int
    DoSample          bool
    Temperature       float32
    TopK              int
    TopP              float32
    RepetitionPenalty float32
    EOSTokenID        int32
    BOSTokenID        int32
    PadTokenID        int32
}

func Generate(
    ctx *context.Context,
    model EncoderDecoderModel,
    encoderOutput *Node,
    config *GenerationConfig,
) []int32
```

## File Structure

```
architectures/
├── common/
│   ├── dense.go           # Existing
│   ├── embeddings.go      # Existing
│   ├── normalization.go   # Existing
│   ├── cross_attention.go # NEW
│   └── kv_cache.go        # NEW
├── bert/
│   └── bert.go            # Existing
├── deberta/
│   └── deberta.go         # Existing
├── llama/
│   └── llama.go           # Existing
├── vit/                   # NEW
│   ├── vit.go
│   └── patch_embed.go
├── swin/                  # NEW (optional)
│   ├── swin.go
│   └── window_attention.go
├── trocr/                 # NEW
│   └── trocr.go
└── vision_encoder_decoder/ # NEW
    └── ved.go

generation/                # NEW
└── generate.go
```

## Dependencies

- GoMLX activations (GELU, SiLU, etc.) - ✅ Available
- onnx-gomlx GELU support - ✅ PR #49
- go-huggingface tokenizers - ✅ Available
- go-huggingface safetensors - ✅ Available

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Weight loading verification against PyTorch reference

### Integration Tests
- End-to-end inference on sample images
- Output comparison with HuggingFace transformers

### Test Models
- `microsoft/trocr-base-handwritten` - TrOCR for handwriting
- `microsoft/trocr-base-printed` - TrOCR for printed text
- `google/vit-base-patch16-224` - ViT encoder only
- `openai/clip-vit-base-patch32` - CLIP vision encoder

## Complexity Estimates

| Component | Complexity | Dependencies |
|-----------|------------|--------------|
| Cross-attention | Low | None |
| KV-cache | Medium | Attention refactor |
| ViT encoder | Medium | Patch embedding |
| Vision-Encoder-Decoder | Medium | ViT, cross-attention |
| TrOCR | Medium | ViT, VED |
| Swin transformer | High | Window attention |
| Generation loop | Medium | KV-cache |

## Alternative: ONNX Runtime

For immediate Vision2Seq needs, ONNX Runtime remains the pragmatic choice:
- All operators supported out of the box
- Optimized attention kernels
- Native KV-cache support
- Pre-exported models available

This spec describes the path to native GoMLX support, which enables fine-tuning and removes the ONNX dependency.

## References

- [ViT Paper](https://arxiv.org/abs/2010.11929) - An Image is Worth 16x16 Words
- [TrOCR Paper](https://arxiv.org/abs/2109.10282) - Transformer-based OCR
- [Donut Paper](https://arxiv.org/abs/2111.15664) - Document Understanding Transformer
- [Swin Paper](https://arxiv.org/abs/2103.14030) - Swin Transformer
- [HuggingFace ViT](https://huggingface.co/docs/transformers/model_doc/vit)
- [HuggingFace TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)
