# SPDX-License-Identifier: Apache-2.0
"""Shared one-denoise-step smoke helper for the per-arch e2e matrix (issue #12).

Each `test_smoke_*.py` module loads its own arch via `UnetLoaderFNI8` and builds its
own noise/conditioning shapes (they differ per arch: Flux/Z-Image are 4-D image
latents, Qwen-Image is 5-D `[B, C, T, H, W]`), then calls
`assert_finite_nonconstant_denoise_step` to run the shared one-step
`comfy.sample.sample` call and the finite/non-constant assertions."""
from __future__ import annotations

import comfy.sample
import pytest
import torch


def load_dit_or_skip(load_fn):
    """Run a DiT load and SKIP (not fail) if it OOMs — the only reliable "does it
    fit" test, since ComfyUI offloads the fp16 tensors so resident VRAM is well
    below the `.fni8` file size (FLUX's 17GB file fits a 16GB card; Qwen-Image's
    20GB doesn't). File size is not a usable proxy; the OOM is. Big DiTs skip on a
    16GB CMP card and run on a larger/idle GPU. See issue #12."""
    try:
        return load_fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("DiT does not fit this card's VRAM — run on a larger/idle GPU")


def assert_finite_nonconstant_denoise_step(
    model, latent_image, positive, negative,
    sampler_name="euler", scheduler="simple", cfg=1.0, steps=1,
):
    """Run *steps* denoise steps and assert the output is the black-image/fp16-overflow
    guard this harness exists for: finite (no int8/fp16 overflow -> NaN) and
    non-constant (no collapsed-to-black latent). Defaults to 1 step for fast
    smoke testing; ``steps > 1`` is used by the AYS schedule matrix (testing
    training-free step-count reduction)."""
    noise = comfy.sample.prepare_noise(latent_image, seed=0)
    samples = comfy.sample.sample(
        model, noise, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative, latent_image=latent_image, denoise=1.0,
    )
    assert torch.isfinite(samples).all(), \
        "denoise step produced non-finite values (fp16/int8 overflow?)"
    assert samples.std().item() > 1e-6, \
        "denoise step collapsed to a constant (black-image) latent"
    return samples
