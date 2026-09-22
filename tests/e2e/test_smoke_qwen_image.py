# SPDX-License-Identifier: Apache-2.0
"""Per-arch e2e matrix (issue #12): Qwen-Image + Qwen-Image-Edit (2509/2511). Same
underlying `QwenImageTransformer2DModel` and `.fni8` layout for all three checkpoints
(confirmed via `comfy.sd`'s model_detection: identical `unet_config` for base and both
edit revisions) — Edit's dual-path (VL semantic + VAE latent) conditioning is optional
at the model level (`ref_latents=None` is a valid forward), so a plain one-step
denoise exercises the same int8 dp4a DiT path for all three.

Unlike Flux/Z-Image's 4-D `[B, C, H, W]` image latent, `QwenImageTransformer2DModel`
patchifies a 5-D `[B, C, T, H, W]` latent (`comfy_extras/nodes_qwen.py`'s
`EmptyQwenImageLayeredLatentImage` builds the same shape with T=1 for a flat image)."""
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

QWEN_CHECKPOINTS = [
    pytest.param(("Qwen__Qwen-Image.dit.b8.fni8", "qwen_image"), id="qwen_image"),
    pytest.param(("Qwen__Qwen-Image-Edit-2509.dit.b8.fni8", "qwen_image_edit"), id="qwen_image_edit_2509"),
    pytest.param(("Qwen__Qwen-Image-Edit-2511.dit.b8.fni8", "qwen_image_edit"), id="qwen_image_edit_2511"),
]


@pytest.fixture(params=QWEN_CHECKPOINTS)
def qwen_model(request):
    unet_name, arch = request.param
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if unet_name not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{unet_name} not found on the diffusion_models search path")
    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(unet_name, arch=arch))
    return model


def test_qwen_image_denoise_step_is_finite_and_non_constant(qwen_model):
    model = qwen_model
    dit = model.model.diffusion_model

    # Shapes come off the *loaded* model — never hand-guessed. `dit.out_channels`
    # (16, the class default since the real checkpoint doesn't override it) is the
    # raw VAE-facing channel count; `dit.in_channels` (64) is already patchified
    # (16 * patch_size**2) and is NOT the latent shape to build here.
    raw_in_channels = dit.out_channels
    context_dim = dit.txt_in.in_features

    # 5-D: [B, C, T, H, W] — QwenImageTransformer2DModel.process_img unpacks this
    # itself (see comfy/ldm/qwen_image/model.py); T=1 for a flat (non-layered) image.
    latent_image = torch.zeros((1, raw_in_channels, 1, 16, 16), dtype=torch.float32)
    # Random conditioning is fine here — the point is catching a numerical failure
    # (fp16/int8 overflow -> collapsed latent), not checking image quality.
    cond_tensor = torch.randn(1, 8, context_dim, dtype=torch.float32)
    positive = [[cond_tensor, {}]]
    negative = [[torch.zeros_like(cond_tensor), {}]]

    assert_finite_nonconstant_denoise_step(model, latent_image, positive, negative)
