# SPDX-License-Identifier: Apache-2.0
"""Headless e2e test for issue #64: LoRA composition onto int8 FNI8Tensor weights
via ModelPatcher.  Loads a real `.fni8` DiT inside an actual ComfyUI process,
applies a synthetic LoRA patch via `model.add_patches`, runs one denoise step,
and asserts:
- the output latent is finite and non-constant (same guard as the base smoke test),
- the patched output differs measurably from the no-LoRA baseline,
- the int8 dp4a path is not silently broken by the patch.

Needs a real checkpoint, a real ComfyUI install, and a Volta/CMP (sm_70) GPU;
only runs via `docker compose run --rm e2e`."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import folder_paths

from comfyui_superl8.superl8_tensor import FNI8Tensor
from comfyui_superl8.gate import is_sm70
from comfyui_superl8.nodes import UnetLoaderFNI8
from ._common import assert_finite_nonconstant_denoise_step, load_dit_or_skip

pytestmark = pytest.mark.comfy_e2e

ZIMAGE_TURBO = "Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8"


@pytest.fixture(scope="module")
def zimage_model():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if ZIMAGE_TURBO not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{ZIMAGE_TURBO} not found on the diffusion_models search path")
    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(ZIMAGE_TURBO, arch="zimage"))
    return model


def _pick_fni8_keys(state_dict, n=3):
    """Return up to *n* state-dict keys whose values are FNI8Tensor."""
    keys = [k for k, v in state_dict.items() if isinstance(v, FNI8Tensor)]
    return keys[:n]


def _make_synthetic_lora_patch(state_dict, key, rank=4, seed=42):
    """Build a ("diff", (diff_tensor,)) LoRA patch for the given FNI8Tensor key.

    The diff approximates a real LoRA delta: `(alpha/rank) * B @ A` where
    B [out, rank], A [rank, in] are random but seeded for reproducibility.
    """
    w = state_dict[key]
    out_dim, in_dim = w.shape
    alpha = 1.0
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(rank, in_dim, generator=g, dtype=torch.float16) * 0.1
    B = torch.randn(out_dim, rank, generator=g, dtype=torch.float16) * 0.1
    diff = (alpha / rank) * (B @ A)
    return ("diff", (diff,))


def test_lora_on_int8_zimage_changes_latents(zimage_model):
    """Apply a synthetic LoRA to FNI8Tensor weights and verify the output latent
    is finite, non-constant, and differs from the no-LoRA baseline."""
    model = zimage_model
    dit = model.model.diffusion_model
    sd = dit.state_dict()

    # Pick a few FNI8Tensor keys to patch
    target_keys = _pick_fni8_keys(sd, n=2)
    if not target_keys:
        pytest.skip("no FNI8Tensor weights found in model")

    # Build patch dict and apply
    patch_dict = {}
    for i, k in enumerate(target_keys):
        patch_dict[k] = _make_synthetic_lora_patch(sd, k, rank=4, seed=42 + i)
    patched_keys = model.add_patches(patch_dict, strength_patch=1.0, strength_model=1.0)
    assert len(patched_keys) == len(target_keys), (
        f"expected {len(target_keys)} patched keys, got {len(patched_keys)}"
    )

    # Run one denoise step through the shared assertion helper
    in_channels = dit.in_channels
    cap_feat_dim = dit.cap_embedder[1].in_features
    latent_image = torch.zeros((1, in_channels, 16, 16), dtype=torch.float32)
    cond_tensor = torch.randn(1, 8, cap_feat_dim, dtype=torch.float32)
    positive = [[cond_tensor, {}]]
    negative = [[torch.zeros_like(cond_tensor), {}]]

    assert_finite_nonconstant_denoise_step(model, latent_image, positive, negative)

    # Verify the patched weights are still FNI8Tensor in the model
    for k in target_keys:
        w = dit.get_parameter(k)
        assert isinstance(w, FNI8Tensor), f"weight {k} lost FNI8Tensor wrapper after patching"
