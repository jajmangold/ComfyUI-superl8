# SPDX-License-Identifier: Apache-2.0
"""E2e smoke test (issue #28): verify that UnetLoaderFNI8 engages int8 FlashAttention,
not fp SDPA. Load a real `.fni8` DiT with the attention patch wired in by the loader,
spy on superl8.attn_int8_fwd to confirm the int8 path is taken, then assert the usual
finite/non-constant output contract.

Without the attention patch wired into UnetLoaderFNI8 the denoise step still produces
finite output (FNI8Ops handles the linears), but the self-attention inside each block
runs through fp SDPA instead of superl8.attn_int8_fwd — this test distinguishes those
two cases by the spy."""
from __future__ import annotations

import pytest

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import folder_paths
import superl8
import torch

from comfyui_superl8.gate import is_sm70
from comfyui_superl8.nodes import UnetLoaderFNI8

from ._common import assert_finite_nonconstant_denoise_step, load_dit_or_skip

pytestmark = pytest.mark.comfy_e2e

ZIMAGE_TURBO = "Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8"


@pytest.fixture(scope="module")
def zimage_with_attn_spy():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if ZIMAGE_TURBO not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{ZIMAGE_TURBO} not found on the diffusion_models search path")

    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(ZIMAGE_TURBO, arch="zimage"))

    return model


def test_unet_loader_engages_int8_flash_attention(zimage_with_attn_spy):
    """Load a DiT via UnetLoaderFNI8 (which now auto-applies FNI8AttentionPatch) and
    prove superl8.attn_int8_fwd was actually called during the denoise step (not fp
    SDPA fallback).  Also verifies the finite / non-constant output contract."""
    model = zimage_with_attn_spy
    dit = model.model.diffusion_model

    in_channels = dit.in_channels
    cap_feat_dim = dit.cap_embedder[1].in_features

    latent_image = torch.zeros((1, in_channels, 16, 16), dtype=torch.float32)
    cond_tensor = torch.randn(1, 8, cap_feat_dim, dtype=torch.float32)
    positive = [[cond_tensor, {}]]
    negative = [[torch.zeros_like(cond_tensor), {}]]

    # Spy on superl8.attn_int8_fwd to confirm the int8 FA path is taken.
    attn_int8_called = False
    original_fwd = superl8.attn_int8_fwd

    def _spy(*args, **kwargs):
        nonlocal attn_int8_called
        attn_int8_called = True
        return original_fwd(*args, **kwargs)

    superl8.attn_int8_fwd = _spy
    try:
        assert_finite_nonconstant_denoise_step(model, latent_image, positive, negative)
        assert attn_int8_called, (
            "superl8.attn_int8_fwd was never called — self-attention ran through fp SDPA "
            "instead of int8 FA.  Either the attention patch was not applied, or every "
            "call fell through to the fp fallback (unsupported head dim, masked attn, "
            "cross-attention, …)."
        )
    finally:
        superl8.attn_int8_fwd = original_fwd
