# SPDX-License-Identifier: Apache-2.0
"""Per-arch e2e (issue #30): Wan 2.2 video DiT loads from its diffusers-layout `.fni8`.

The `.fni8`s the pack publishes for Wan were quantized straight from the HuggingFace
**diffusers** `Wan2.2-*-Diffusers` checkpoints, whose state-dict key layout ComfyUI's
`comfy.model_detection.detect_unet_config` does not recognize — so before this the
DiT could not be built into a `MODEL` at all (see the companion
`test_smoke_video_diffusers_gap.py` repro). `loader.remap_diffusers_to_native` rewrites
the keys to ComfyUI's native `comfy.ldm.wan.model.WanModel` layout, and
`UnetLoaderFNI8.load` re-asserts the int8 weights onto the built model so the dp4a path
engages (ComfyUI's stock loader would otherwise copy the int8 codes into fp params).

Same load-via-UnetLoaderFNI8 + one-denoise-step + finite/non-constant contract as
`test_smoke_flux1.py`, plus an explicit assertion that int8 `FNI8Tensor` weights are
resident on the built model (the point of the pack). `docker compose run --rm e2e`."""
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

# The smallest published Wan (5B) — fits a 16GB CMP card where the 14B does not.
WAN22_TI2V_5B = "Wan-AI__Wan2.2-TI2V-5B-Diffusers.dit.b8.fni8"


@pytest.fixture(scope="module")
def wan_model():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if WAN22_TI2V_5B not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{WAN22_TI2V_5B} not found on the diffusion_models search path")
    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(WAN22_TI2V_5B, arch="wan22"))
    return model


def test_wan_builds_as_native_wan_model_with_int8_weights(wan_model):
    """The diffusers `.fni8` was detected + built into ComfyUI's native WanModel, and
    the int8 dp4a weights survived loading (this is the whole point of the pack)."""
    dit = wan_model.model.diffusion_model
    assert type(dit).__name__ == "WanModel", (
        f"expected native WanModel, got {type(dit).__name__} — the diffusers->native "
        "remap or detection regressed"
    )
    n_int8 = sum(
        1 for m in dit.modules()
        if getattr(m, "weight", None) is not None and isinstance(m.weight, FNI8Tensor)
    )
    assert n_int8 > 0, (
        "no int8 FNI8Tensor weights resident on the built WanModel — the dp4a path is "
        "not engaged (ComfyUI's stock loader copied the int8 codes into fp params)"
    )


def test_wan_denoise_step_is_finite_and_non_constant(wan_model):
    model = wan_model
    dit = model.model.diffusion_model

    # Shapes come off the *loaded* model (detected from the real checkpoint's tensor
    # shapes), never hand-guessed. Wan is a 3D DiT: latents are [B, C, T, H, W] and the
    # patch_embedding is a Conv3d whose in_channels is the raw (VAE-facing) latent dim.
    raw_in_channels = dit.patch_embedding.in_channels
    context_dim = dit.text_embedding[0].in_features

    latent_image = torch.zeros((1, raw_in_channels, 1, 16, 16), dtype=torch.float32)
    cond_tensor = torch.randn(1, 8, context_dim, dtype=torch.float32)
    positive = [[cond_tensor, {}]]
    negative = [[torch.zeros_like(cond_tensor), {}]]

    assert_finite_nonconstant_denoise_step(model, latent_image, positive, negative)
