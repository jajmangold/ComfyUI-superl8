# SPDX-License-Identifier: Apache-2.0
"""Quantitative int8-vs-fp16 quality comparison: loads a DiT ONCE with FNI8Ops,
runs one denoise step with int8 (dp4a) weights, then replaces weights with
dequantized fp16 and re-runs on the same model to isolate the matmul-path
difference. Asserts cosine-similarity (pre-VAE latent) and optionally PSNR/SSIM
(decoded image) against minimum floors.

This addresses the metric gap noted in docs/e2e-coverage.md:
    "Wiring an actual fp16 baseline run + comparison metric per arch is follow-up work"

The same FNI8Ops class is used for both runs: FNI8Ops.Linear.forward_comfy_cast_weights
checks isinstance(self.weight, FNI8Tensor) — when the weight is a regular fp tensor it
falls back to manual_cast's standard torch matmul. This keeps the architecture, the
op-patching, and every non-linear kernel identical; only the matmul path changes:
  - int8:  W8A8 dp4a GEMM via superl8.linear + per-row activation quant
  - fp:    standard torch matmul (no quantisation)
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

# Make the repo root importable so `bench.quality` resolves.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import comfy.sample
import comfy.utils
import folder_paths
import comfy.sd

from comfyui_superl8.superl8_tensor import FNI8Tensor
from comfyui_superl8.gate import is_sm70
from comfyui_superl8.loader import fni8_dit_native_dtype, load_fni8_dit
from comfyui_superl8.nodes import globals_arch_prefix
from comfyui_superl8.ops import FNI8Ops

from bench.quality import cosine_similarity, format_report, psnr, ssim

pytestmark = pytest.mark.comfy_e2e

# ---- floors ----------------------------------------------------------------

COS_FLOOR = 0.985
PSNR_FLOOR = 28.0
SSIM_FLOOR = 0.90

# ---- per-arch shape resolvers -----------------------------------------------

def _zimage_shapes(dit):
    return dict(in_channels=dit.in_channels, cond_dim=dit.cap_embedder[1].in_features)

def _flux_shapes(dit):
    return dict(
        in_channels=dit.img_in.in_features // (dit.patch_size ** 2),
        cond_dim=dit.txt_in.in_features,
    )

ARCH_CASES = [
    pytest.param(
        "Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8",
        "zimage",
        _zimage_shapes,
        None,
        id="zimage",
    ),
]


def _dequantize_model_state(sd: dict) -> dict:
    sd_fp = {}
    for name, w in sd.items():
        if isinstance(w, FNI8Tensor):
            sd_fp[name] = (w.int8_data().float() * w.q_scale.unsqueeze(1)).to(
                torch.float16)
        else:
            sd_fp[name] = w
    return sd_fp


def _one_step_denoise(model, latent_image, cond_tensor):
    noise = comfy.sample.prepare_noise(latent_image, seed=0)
    positive = [[cond_tensor, {}]]
    negative = [[torch.zeros_like(cond_tensor), {}]]
    return comfy.sample.sample(
        model, noise, steps=1, cfg=1.0, sampler_name="euler", scheduler="simple",
        positive=positive, negative=negative, latent_image=latent_image, denoise=1.0,
    )


@pytest.mark.parametrize("unet_name,arch,shape_fn,vae_override", ARCH_CASES)
def test_quality_int8_vs_fp16(unet_name, arch, shape_fn, vae_override):
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if unet_name not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{unet_name} not found on the diffusion_models search path")

    path = folder_paths.get_full_path("diffusion_models", unet_name)
    native_dtype = fni8_dit_native_dtype(path)
    dtype = torch.bfloat16 if native_dtype == "bfloat16" else torch.float16
    prefix = "diffusion_model." if globals_arch_prefix(arch) else ""

    # Load state dict once — dequantized copy saved for the fp pass.
    sd = load_fni8_dit(path, device="cpu", strip_prefix=prefix)
    sd_fp = _dequantize_model_state(dict(sd))

    # ---- load model ONCE with FNI8Ops ---------------------------------------
    try:
        model = comfy.sd.load_diffusion_model_state_dict(
            dict(sd),
            model_options={"custom_operations": FNI8Ops, "dtype": dtype},
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("model OOM — run on a larger/idle GPU")

    dit = model.model.diffusion_model
    sh = shape_fn(dit)
    latent_hw = sh.get("latent_hw", 8)
    latent_shape = (1, sh["in_channels"], latent_hw, latent_hw)
    cond_tensor = torch.randn(1, 8, sh["cond_dim"], dtype=torch.float32)
    latent_image = torch.zeros(latent_shape, dtype=torch.float32)

    # ---- int8 pass ----------------------------------------------------------
    try:
        samples_int8 = _one_step_denoise(model, latent_image, cond_tensor)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("int8 denoise step OOM")
    samples_int8 = samples_int8.detach().clone()
    assert torch.isfinite(samples_int8).all(), "int8 latent has non-finite values"

    # ---- replace weights with dequantized fp, then re-run -------------------
    # load_state_dict(strict=False) replaces only matching keys; unmatched
    # parameters (norms, buffers, etc.) keep their current values, so the
    # architecture is bit-identical across both passes.
    model.model.load_state_dict(sd_fp, strict=False)

    try:
        samples_fp = _one_step_denoise(model, latent_image, cond_tensor)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("fp denoise step OOM")
    samples_fp = samples_fp.detach().clone()
    del model
    torch.cuda.empty_cache()

    assert torch.isfinite(samples_fp).all(), "fp reference latent has non-finite values"

    # ---- compare latents ----------------------------------------------------
    cos = cosine_similarity(samples_int8, samples_fp)

    # ---- optional VAE decode for PSNR/SSIM ---------------------------------
    psnr_val, ssim_val = None, None
    vae_fn = vae_override
    if vae_fn is not None and vae_fn in folder_paths.get_filename_list("vae"):
        vae_path = folder_paths.get_full_path("vae", vae_fn)
        try:
            vae_sd = comfy.utils.load_torch_file(vae_path, safe_load=True)
            vae = comfy.sd.VAE(sd=vae_sd)
            img_int8 = vae.decode(samples_int8.to(device="cuda"))
            img_fp = vae.decode(samples_fp.to(device="cuda"))
            psnr_val = psnr(img_int8, img_fp)
            ssim_val = ssim(img_int8, img_fp)
        except (torch.cuda.OutOfMemoryError, RuntimeError, FileNotFoundError):
            torch.cuda.empty_cache()

    print(format_report(arch, cos, psnr_val, ssim_val))

    assert cos >= COS_FLOOR, (
        f"latent cosine similarity {cos:.6f} < floor {COS_FLOOR}"
    )
    if psnr_val is not None:
        assert psnr_val >= PSNR_FLOOR, (
            f"PSNR {psnr_val:.2f} dB < floor {PSNR_FLOOR}"
        )
    if ssim_val is not None:
        assert ssim_val >= SSIM_FLOOR, (
            f"SSIM {ssim_val:.6f} < floor {SSIM_FLOOR}"
        )
