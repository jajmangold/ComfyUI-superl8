# SPDX-License-Identifier: Apache-2.0
"""Sampler/scheduler/cfg matrix (issue #31): exercise euler+dpmpp / simple+karras
+ ays / cfg>1 and guidance-distilled (cfg=1) on the FLUX.1-dev DiT to ensure
finite/non-constant output across the matrix.

AYS (Align Your Steps, 2404.14507) schedules are tested at reduced step counts
(2-5 steps) vs the baseline (1 step) to verify the training-free step-count
reduction produces finite, non-constant latents.

The model is loaded once (module-scoped fixture) and reused for all parametrized
combinations — the int8 dp4a DiT path (the costly part) is the same; only the
ComfyUI sampling call differs."""
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

SAMPLER_SCHEDULER_CFG = [
    # Baseline: standard schedulers at 1 step (existing smoke test coverage).
    pytest.param("euler", "simple", 1.0, 1, id="euler_simple_cfg1_baseline"),
    pytest.param("euler", "simple", 2.0, 1, id="euler_simple_cfg2_baseline"),
    pytest.param("euler", "karras", 1.0, 1, id="euler_karras_cfg1_baseline"),
    pytest.param("euler", "karras", 2.0, 1, id="euler_karras_cfg2_baseline"),
    pytest.param("dpmpp_2m", "simple", 1.0, 1, id="dpmpp_simple_cfg1_baseline"),
    pytest.param("dpmpp_2m", "simple", 2.0, 1, id="dpmpp_simple_cfg2_baseline"),
    pytest.param("dpmpp_2m", "karras", 1.0, 1, id="dpmpp_karras_cfg1_baseline"),
    pytest.param("dpmpp_2m", "karras", 2.0, 1, id="dpmpp_karras_cfg2_baseline"),
    # AYS: training-free step-count reduction at 2-5 steps on FLUX (flow-matching).
    # FLUX.1-dev is guidance-distilled (cfg=1 is the correct usage).
    pytest.param("euler", "ays", 1.0, 2, id="euler_ays_cfg1_2step"),
    pytest.param("euler", "ays", 1.0, 3, id="euler_ays_cfg1_3step"),
    pytest.param("euler", "ays", 1.0, 4, id="euler_ays_cfg1_4step"),
    pytest.param("euler", "ays", 1.0, 5, id="euler_ays_cfg1_5step"),
    pytest.param("euler", "ays_flow", 1.0, 2, id="euler_ays_flow_cfg1_2step"),
    pytest.param("euler", "ays_flow", 1.0, 3, id="euler_ays_flow_cfg1_3step"),
    pytest.param("euler", "ays_flow", 1.0, 4, id="euler_ays_flow_cfg1_4step"),
    pytest.param("euler", "ays_flow", 1.0, 5, id="euler_ays_flow_cfg1_5step"),
    # DPM++ with AYS schedules.
    pytest.param("dpmpp_2m", "ays", 1.0, 3, id="dpmpp_ays_cfg1_3step"),
    pytest.param("dpmpp_2m", "ays", 1.0, 5, id="dpmpp_ays_cfg1_5step"),
    pytest.param("dpmpp_2m", "ays_flow", 1.0, 3, id="dpmpp_ays_flow_cfg1_3step"),
    pytest.param("dpmpp_2m", "ays_flow", 1.0, 5, id="dpmpp_ays_flow_cfg1_5step"),
]


@pytest.fixture(scope="module")
def flux1_model():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if FLUX1_DEV not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{FLUX1_DEV} not found on the diffusion_models search path")
    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(FLUX1_DEV, arch="flux1"))
    return model


@pytest.mark.parametrize(
    "sampler_name,scheduler,cfg,steps",
    SAMPLER_SCHEDULER_CFG,
)
def test_flux1_denoise_step_matrix(flux1_model, sampler_name, scheduler, cfg, steps):
    model = flux1_model
    dit = model.model.diffusion_model

    raw_in_channels = dit.img_in.in_features // (dit.patch_size ** 2)
    context_dim = dit.txt_in.in_features

    latent_image = torch.zeros((1, raw_in_channels, 16, 16), dtype=torch.float32)
    cond_tensor = torch.randn(1, 8, context_dim, dtype=torch.float32)
    positive = [[cond_tensor, {}]]
    negative = [[torch.zeros_like(cond_tensor), {}]]

    assert_finite_nonconstant_denoise_step(
        model, latent_image, positive, negative,
        sampler_name=sampler_name, scheduler=scheduler, cfg=cfg, steps=steps,
    )
