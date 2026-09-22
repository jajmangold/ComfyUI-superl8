# ComfyUI-SuperL8

![CI](https://github.com/jajmangold/ComfyUI-superl8/actions/workflows/ci.yml/badge.svg)

**INT8 quantized diffusion DiTs with SQNR accuracy gating.**

ComfyUI-SuperL8 brings [SuperL8](https://github.com/jajmangold/superl8)'s INT8 DP4A kernels to ComfyUI for quantized DiT inference. The key innovation is **per-layer SQNR gating** — each attention and linear layer is quantized, verified against a signal-to-quantization-noise ratio threshold, and automatically falls back to fp16 when quality drops. You get aggressive quantization where it's free and accurate inference where it matters.

## Why quantize DiTs?

Diffusion transformers are memory-bound. A 22B-parameter DiT like LTX-2.3 needs ~44 GB in fp16 — doesn't fit on a single 24 GB card. INT8 cuts that in half. But naive INT8 quantization destroys image quality on some layers while being invisible on others. SuperL8's SQNR gating quantizes every layer, measures the actual quality loss, and only keeps the quantization where it's safe.

## What's implemented

### DiT inference

| Component | Details |
|---|---|
| INT8 W8A8 DiT forward | Full transformer block (attention + MLP) in INT8 via dp4a kernels. |
| INT8 W4A8 DiT forward | 4-bit weight variant for memory-constrained cards. |
| SQNR accuracy gating | Per-layer signal-to-quantization-noise ratio check. Automatic fp16 fallback when quality drops below threshold. |
| FP16 fallback path | Graceful degradation — quantize what's safe, keep the rest in fp16. |

### Attention

| Component | Details |
|---|---|
| Self-attention INT8 | SuperL8's `attn_int8_fwd` for DiT self-attention. |
| Cross-attention INT8 | Quantized cross-attention for text-encoder conditioning. |
| rotary positional embedding | Fused RoPE in INT8 attention path. |
| Attention masking | STA (spatio-temporal attention) masking for video DiTs. |
| DiT block fusion | Fused attention + MLP block for reduced launch overhead. |

### Model support

| Architecture | INT8 DiT | GGUF DiT | Status |
|---|---|---|---|
| Z-Image Turbo | Yes | Q4_K_M | Validated |
| FLUX.1-dev | Yes | — | Validated |
| Qwen-Image | Yes | — | Validated |
| Qwen-Image-Edit | Yes | — | Validated |
| LTX-2.3 | Yes | Q4_K_M | Validated |
| Wan2.2 | Yes | — | Experimental |
| SD3.5 | Yes | — | Experimental |
| Sana | Yes | — | Experimental |
| Ideogram | Yes | — | Experimental |
| AuraFlow | Yes | — | Experimental |

### Weight loading

| Component | Details |
|---|---|
| SuperL8 format | mmap zero-copy load of `.superl8` quantized DiTs. |
| GGUF native | Load Q2_K–Q6_K k-quant DiTs directly via dp4a kernels. |
| Compat layer | Fallback to ComfyUI-GGUF ops if native path unavailable. |
| LoRA support | Apply LoRA adapters to quantized base models. |
| Component loading | Text encoder + VAE loaded via `SuperL8ComponentLoader`. |

### Tiling and memory

| Component | Details |
|---|---|
| Tiled DiT | Split DiT inference across tiles for large images. Configurable tile size. |
| Tiled VAE | Memory-budget-controlled VAE decode. Configurable max VRAM. |
| Memory tracking | Live VRAM usage monitoring for tile scheduling. |

### Multi-GPU

| Component | Details |
|---|---|
| Pipeline parallelism | Split DiT layers across GPUs. |
| CFG parallel | Classifier-free guidance split across two GPUs. |
| GPU election | Fail-closed GPU selection based on HBM availability and process exclusion. |
| Peer routes | Direct GPU-to-GPU transfer for pipeline stages. |

### Acceleration

| Component | Details |
|---|---|
| TeaCache | Skip redundant DiT blocks based on activation similarity. |
| Step controller | Dynamic step scheduling with profiling hooks. |
| AYS guidance distill | Adversarial You Look Only Once guidance distillation support. |

### VLM / editing

| Component | Details |
|---|---|
| Qwen-Image-Edit | Image editing via Qwen-Image-Edit models. |
| SAM3 / SAM3D | Segment Anything 3D encoder integration. |
| Z-Image resident service | HTTP daemon for persistent Z-Image inference. |

### Testing

| Component | Details |
|---|---|
| 57 unit tests | Architecture registry, attention gating, GGUF ops, LoRA, tiling, quantization. |
| 18 e2e tests | Real GPU + real weights smoke tests for each validated architecture. |
| SQNR validation | Per-layer quality metrics comparing INT8 vs fp16 outputs. |
| Quality metrics | PSNR, SSIM, cosine similarity for image output validation. |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jajmangold/ComfyUI-superl8
cd ComfyUI-superl8
pip install -e .
```

Or install the wheel directly:

```bash
pip install https://github.com/jajmangold/ComfyUI-superl8/releases/download/v0.1.0/comfyui_superl8-0.1.0-py3-none-any.whl
```

Requires Python 3.10+, PyTorch 2.0+, ComfyUI 0.3.0+, and a Volta (sm_70) GPU for int8 dp4a acceleration.

## Nodes

| Node | Purpose |
|---|---|
| **UnetLoaderSuperL8** | Load a SuperL8-quantized DiT |
| **UnetLoaderSuperL8GGUF** | Load a GGUF DiT with native int8 dp4a kernels |
| **SuperL8ComponentLoader** | Load matching text encoder / VAE |
| **SuperL8Multigpu** | Move the diffusion model to a second GPU |
| **SuperL8PipelineParallel** | Split an oversized DiT across GPUs |
| **TiledVAEDecode** | Tiled VAE decode with memory budget control |
| **ApplyTiledDiT** | Tiled DiT inference for large images |
| **SuperL8StepController** | Step scheduling and profiling |
| **SuperL8TeaCache** | TeaCache acceleration |

## Environment variables

| Variable | Description |
|---|---|
| `SUPERL8_MODEL_ROOT` | Root directory for model weights (required) |
| `FNI8_WEIGHTS_DIR` | Weights directory for e2e tests and benchmarks |
| `FNI8_GPU` | GPU index for e2e tests (default: 1) |

## Roadmap

Performance improvements planned for upcoming releases:

- **Persistent DiT resident** — keep the DiT loaded in VRAM across workflows to eliminate reload overhead.
- **Tiled attention** — attention computed over tiles to support images larger than VRAM without quality loss from tiling artifacts.
- **Multi-GPU expert parallelism** — for MoE DiTs (like Hunyuan), route experts to specific GPUs.
- **INT4 GEMM for DiT MLPs** — 4-bit weight GEMM for the MLP layers which are typically the memory bottleneck.
- **Compile-time SQNR calibration** — auto-calibrate per-layer SQNR thresholds from a validation set instead of using fixed defaults.
- **Triton kernel fallback** — provide Triton implementations as fallback for non-Volta GPUs.
- **Video DiT streaming** — frame-by-frame streaming for long video generation without pre-allocating full temporal context.
- **ControlNet INT8** — quantize ControlNet conditioning to match the DiT quantization level.

## Related repos

- [**SuperL8**](https://github.com/jajmangold/superl8) — the CUDA kernels powering this node pack. INT8 DP4A FlashAttention-2 and GEMM for Volta GPUs.
- [**SuperL8 Serve**](https://github.com/jajmangold/superl8-serve) — OpenAI-compatible LLM inference server built on the same kernels.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
