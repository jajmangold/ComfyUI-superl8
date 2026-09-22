# ComfyUI-SuperL8

**INT8 diffusion that knows when to stop.**

ComfyUI-SuperL8 brings [SuperL8](https://github.com/jajmangold/superl8)'s INT8 DP4A kernels to ComfyUI for quantized DiT inference. The key innovation is **per-layer SQNR gating** — each attention and linear layer is quantized, verified against a signal-to-quantization-noise ratio threshold, and automatically falls back to fp16 when quality drops. You get aggressive quantization where it's free and accurate inference where it matters.

## Why quantize DiTs?

Diffusion transformers are memory-bound. A 22B-parameter DiT like LTX-2.3 needs ~44 GB in fp16 — doesn't fit on a single 24 GB card. INT8 cuts that in half. But naive INT8 quantization destroys image quality on some layers while being invisible on others. SuperL8's SQNR gating quantizes every layer, measures the actual quality loss, and only keeps the quantization where it's safe.

## Features

- **INT8 W8A8 and W4A8 DiT inference** via SuperL8's dp4a kernels
- **Per-layer SQNR accuracy gating** — quantize aggressively, verify automatically, fall back to fp16 where it matters
- **17 registered DiT architectures** — Z-Image, FLUX.1-dev, Qwen-Image, LTX-2.3, Wan2.2, SD3.5, Sana, and more
- **GGUF native k-quant loading** — Q2_K through Q6_K, loaded in-kernel with dp4a
- **Multi-GPU pipeline parallelism** — split oversized DiTs across GPUs
- **Tiled DiT/VAE processing** — memory-budget-controlled inference for large images
- **LoRA composition** — apply LoRA adapters to quantized models
- **TeaCache acceleration** — skip redundant DiT blocks
- **SAM3 / SAM3D encoder integration** — segmentation alongside generation

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

## Supported architectures

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

## Related repos

- [**SuperL8**](https://github.com/jajmangold/superl8) — the CUDA kernels powering this node pack. INT8 DP4A FlashAttention-2 and GEMM for Volta GPUs.
- [**SuperL8 Serve**](https://github.com/jajmangold/superl8-serve) — OpenAI-compatible LLM inference server built on the same kernels.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
