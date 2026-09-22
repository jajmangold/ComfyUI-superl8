# ComfyUI-SuperL8

INT8 quantized diffusion model nodes for ComfyUI.

## Features

- INT8 W8A8 and W4A8 quantized DiT inference via dp4a kernels
- Per-layer SQNR accuracy gating with automatic fp fallback
- Support for image DiTs (Z-Image, FLUX.1-dev, Qwen-Image, Qwen-Image-Edit)
- Support for video DiTs (LTX-2.3, Wan2.2)
- Multi-GPU pipeline parallelism for oversized models
- Tiled VAE decode with configurable memory budgets
- GGUF native k-quant DiT loading (Q2_K through Q6_K)
- LoRA composition with quantized models
- TeaCache acceleration support
- SAM3 / SAM3D encoder integration

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

### Requirements

- Python 3.10+
- PyTorch 2.0+
- ComfyUI 0.3.0+
- Volta (sm_70) GPU for int8 dp4a acceleration

## Supported Architectures

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

- **UnetLoaderSuperL8** — loads a SuperL8-quantized DiT
- **UnetLoaderSuperL8GGUF** — loads a GGUF DiT with native int8 dp4a kernels
- **SuperL8ComponentLoader** — loads matching text encoder / VAE
- **SuperL8Multigpu** — moves the diffusion model to a second GPU
- **SuperL8PipelineParallel** — splits an oversized DiT across GPUs
- **TiledVAEDecode** — tiled VAE decode with memory budget control
- **ApplyTiledDiT** — tiled DiT inference for large images
- **SuperL8StepController** — step scheduling and profiling
- **SuperL8TeaCache** — TeaCache acceleration

## Environment Variables

| Variable | Description |
|---|---|
| `SUPERL8_MODEL_ROOT` | Root directory for model weights (required) |
| `FNI8_WEIGHTS_DIR` | Weights directory for e2e tests and benchmarks |
| `FNI8_GPU` | GPU index for e2e tests (default: 1) |

## License

BSD-3-Clause. See [LICENSE](LICENSE).
