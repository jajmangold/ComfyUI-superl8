# ComfyUI-superl8 test suite

## Test lanes

| Lane | Location | GPU needed | Weights needed |
|---|---|---|---|
| Fast (unit) | `tests/test_*.py` | No (some use `cuda_only` marker) | No |
| E2E | `tests/e2e/test_*.py` | Yes | Yes |

## Running tests

```bash
# Fast lane — unit tests only (no GPU, no weights)
pytest tests/ -m "not comfy_e2e" -ra

# E2E lane — real GPU + real weights
pytest tests/ -m comfy_e2e -ra

# Everything
pytest tests/ -ra

# Single test file
pytest tests/test_core.py -ra
```

## Unit tests (fast lane)

Unit tests in `tests/test_*.py` test pure logic without ComfyUI or GPU:

- Architecture registry lookups and validation
- INT8 linear (dp4a) correctness — uses `@cuda_only` marker, skipped when CUDA unavailable
- SQNR gating logic
- DiT quantize/round-trip (`.fni8` format)
- Multi-GPU placement and CFG parallel policy
- LoRA, tiling, TeaCache, attention masking
- GGUF loader node registration

Tests that need CUDA use a `cuda_only = pytest.mark.skipif(not CUDA, reason="dp4a needs CUDA")` pattern — they run on GPU CI but skip on CPU.

## E2E tests (e2e lane)

E2E tests live in `tests/e2e/` and are all marked with `pytest.mark.comfy_e2e`. They require:

- A Volta/CMP (sm_70) GPU
- Real model weights accessible via `FNI8_WEIGHTS_DIR`
- ComfyUI installed with `folder_paths` available

Each e2e test loads a real checkpoint, runs one or more denoise steps, and asserts the output is finite and non-constant. This catches fp16 overflow and black-image regressions that unit tests with random tensors cannot see.

### E2E conftest

`tests/e2e/conftest.py` registers `FNI8_WEIGHTS_DIR` as a ComfyUI `diffusion_models` search path via `folder_paths.add_model_folder_path()`.

## Environment variables

| Variable | Purpose |
|---|---|
| `FNI8_WEIGHTS_DIR` | Path to model weight files for e2e tests. Required for the e2e lane. |
| `FNI8_GPU` | GPU index for e2e tests (default: 1). |

## What `comfy_e2e` marker does

The `comfy_e2e` marker tags tests that need the full ComfyUI runtime + real weights. Run them selectively with `-m comfy_e2e` or skip them entirely with `-m "not comfy_e2e"`.
