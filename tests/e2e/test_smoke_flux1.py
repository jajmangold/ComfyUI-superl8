# SPDX-License-Identifier: Apache-2.0
"""Per-arch e2e matrix (issue #12): FLUX.1-dev, the first `object_patch`-attn-seam
arch validated end-to-end (Z-Image, issue #9, uses `object_patch` too but a different
DiT family). Same load-via-UnetLoaderFNI8 + one-denoise-step + finite/non-constant
contract as `test_smoke_zimage.py`; see `docker compose run --rm e2e`."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import folder_paths

from comfyui_superl8.gate import is_sm70
from comfyui_superl8.nodes import UnetLoaderFNI8

from ._common import assert_finite_nonconstant_denoise_step, load_dit_or_skip

pytestmark = pytest.mark.comfy_e2e

FLUX1_DEV = "black-forest-labs__FLUX.1-dev.dit.b8.fni8"


@pytest.fixture(scope="module")
def flux1_model():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if FLUX1_DEV not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{FLUX1_DEV} not found on the diffusion_models search path")
    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(FLUX1_DEV, arch="flux1"))
    return model


def test_flux1_dev_denoise_step_is_finite_and_non_constant(flux1_model):
    model = flux1_model
    dit = model.model.diffusion_model

    # Shapes come off the *loaded* model (itself detected from the real checkpoint's
    # tensor shapes by comfy.sd's model_detection) — never hand-guessed. Flux's own
    # `in_channels`/`img_in` are the *patchified* dim (raw_channels * patch_size**2),
    # so the raw (VAE-facing) latent channel count is derived by dividing it back out
    # — the same relation `comfy.ldm.flux.model.Flux.__init__` uses to build img_in.
    raw_in_channels = dit.img_in.in_features // (dit.patch_size ** 2)
    context_dim = dit.txt_in.in_features

    latent_image = torch.zeros((1, raw_in_channels, 16, 16), dtype=torch.float32)
    # Random conditioning is fine here — the point is catching a numerical failure
    # (fp16/int8 overflow -> collapsed latent), not checking image quality. cfg=1
    # (no negative branch) matches Flux-dev's guidance-distilled usage.
    cond_tensor = torch.randn(1, 8, context_dim, dtype=torch.float32)
    positive = [[cond_tensor, {}]]
    negative = [[torch.zeros_like(cond_tensor), {}]]

    assert_finite_nonconstant_denoise_step(model, latent_image, positive, negative)
