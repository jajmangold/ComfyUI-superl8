# SPDX-License-Identifier: Apache-2.0
"""Headless e2e smoke test (issue #9): load a real `.fni8` DiT through
`UnetLoaderFNI8` inside an actual ComfyUI process, run one denoise step, and assert
the output latent is finite and non-constant.

This is the black-image/fp16-overflow guard that the random-tensor unit tests in
tests/test_core.py can't see — it needs a real checkpoint, a real ComfyUI install,
and a Volta/CMP (sm_70) GPU, so it only runs via `docker compose run --rm e2e`
(see docker/Dockerfile, docker-compose.yml)."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import folder_paths

from comfyui_superl8.gate import is_sm70
from comfyui_superl8.nodes import UnetLoaderFNI8
from ._common import load_dit_or_skip

from ._common import assert_finite_nonconstant_denoise_step

pytestmark = pytest.mark.comfy_e2e

# Z-Image Turbo is the pack's first validation target (README: "FIRST target") —
# 6B, 8-step, single joint-attention stream, so cfg=1 (no negative branch) fits it.
ZIMAGE_TURBO = "Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8"


@pytest.fixture(scope="module")
def zimage_model():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if ZIMAGE_TURBO not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{ZIMAGE_TURBO} not found on the diffusion_models search path")
    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(ZIMAGE_TURBO, arch="zimage"))
    return model


def test_zimage_turbo_denoise_step_is_finite_and_non_constant(zimage_model):
    model = zimage_model
    dit = model.model.diffusion_model

    # Shapes come off the *loaded* model (itself detected from the real checkpoint's
    # tensor shapes by comfy.sd's model_detection) — never hand-guessed.
    in_channels = dit.in_channels
    cap_feat_dim = dit.cap_embedder[1].in_features

    latent_image = torch.zeros((1, in_channels, 16, 16), dtype=torch.float32)
    # Random conditioning is fine here — the point is catching a numerical failure
    # (fp16/int8 overflow -> collapsed latent), not checking image quality.
    cond_tensor = torch.randn(1, 8, cap_feat_dim, dtype=torch.float32)
    positive = [[cond_tensor, {}]]
    negative = [[torch.zeros_like(cond_tensor), {}]]

    assert_finite_nonconstant_denoise_step(model, latent_image, positive, negative)
