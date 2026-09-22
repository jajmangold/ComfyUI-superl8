# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU e2e tests for ``FNI8Multigpu``.

Honest status (measured on real int8 Z-Image, 2× CMP GPUs — see
docs/multigpu-cfg-parallel-finding.md):

  * ``component_parallel`` is a validated device move (no batch split, no numeric
    change).
  * ``cfg_parallel`` is NOT quality-validated: splitting the batched cond+uncond
    forward into two batch-1 forwards yields cos ≈ 0.88 vs the single-GPU batched
    reference, below the cos ≥ 0.99 bar (the int8 DiT forward is batch-size
    dependent). The node therefore SAFELY FALLS BACK to single-GPU. The
    ``test_cfg_batch_split_diverges`` test below MEASURES that divergence directly
    so the fallback is backed by a real number, not an assumption. The old
    "cfg_parallel matches single-GPU" test used cfg=1.0 (which runs no uncond pass
    at all, so it never exercised the split) — removed as misleading.

Requires 2+ CUDA GPUs and a real checkpoint. Marked ``comfy_e2e``; runs only via
``docker compose run --rm e2e`` (see docker/Dockerfile)."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import comfy.sample
import folder_paths

from comfyui_superl8.gate import is_sm70
from comfyui_superl8.multigpu import available_devices
from comfyui_superl8.nodes import FNI8Multigpu, UnetLoaderFNI8
from ._common import assert_finite_nonconstant_denoise_step, load_dit_or_skip

pytestmark = pytest.mark.comfy_e2e

ZIMAGE_TURBO = "Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8"
QUALITY_BAR = 0.99


@pytest.fixture(scope="module")
def zimage_model():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if len(available_devices()) < 1:
        pytest.skip("needs a CUDA GPU")
    if ZIMAGE_TURBO not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{ZIMAGE_TURBO} not found on the diffusion_models search path")
    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(ZIMAGE_TURBO, arch="zimage"))
    return model


def test_cfg_parallel_node_falls_back_to_single_gpu(zimage_model):
    """FNI8Multigpu(cfg_parallel) must produce output IDENTICAL to the single-GPU
    reference — because it safely falls back (it does not batch-split, which was
    measured to diverge). This is the honest contract: the node never silently
    ships a divergent image."""
    base = zimage_model
    dit = base.model.diffusion_model
    latent_image = torch.zeros((1, dit.in_channels, 32, 32), dtype=torch.float32)
    cond = torch.randn(1, 8, dit.cap_embedder[1].in_features, dtype=torch.float32)
    positive = [[cond, {}]]
    negative = [[torch.zeros_like(cond), {}]]

    ref = assert_finite_nonconstant_denoise_step(base, latent_image, positive, negative, cfg=2.0)
    (cfg_model,) = FNI8Multigpu().apply(base, "cfg_parallel")
    test = assert_finite_nonconstant_denoise_step(cfg_model, latent_image, positive, negative, cfg=2.0)

    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().float(), test.flatten().float(), dim=0).item()
    assert cos >= 0.999, (
        f"cfg_parallel fallback must match single-GPU exactly (cos={cos:.6f}); "
        "if it diverges, the node is no longer safely falling back")


def test_cfg_batch_split_diverges(zimage_model):
    """MEASURE the reason cfg_parallel falls back: manually splitting the batched
    cfg=2 cond+uncond forward into two batch-1 forwards does NOT reproduce the
    batched output on the int8 DiT path (documented cos ≈ 0.88 < 0.99 bar).

    Marked xfail(strict=False): it asserts the batch-split MISSES the quality bar.
    If a future batch-invariant int8 forward makes the split match (xpass), this
    test flips green and cfg_parallel can be genuinely enabled."""
    base = zimage_model
    captured = {}

    def _slice_c(c, idx, batch):
        out = {}
        for k, v in c.items():
            if k == "transformer_options" and isinstance(v, dict):
                to = dict(v)
                cou = to.get("cond_or_uncond")
                if isinstance(cou, list):
                    to["cond_or_uncond"] = [cou[j] for j in idx]
                out[k] = to
            elif torch.is_tensor(v) and v.shape[:1] == torch.Size([batch]):
                out[k] = v[idx]
            else:
                out[k] = v
        return out

    def wrapper(func, kw):
        cou = kw["cond_or_uncond"]
        if len(set(cou)) > 1 and "cos" not in captured:
            x, t, c = kw["input"], kw["timestep"], kw["c"]
            b = x.shape[0]
            ref = func(x, t, **c)
            ci = [i for i, v in enumerate(cou) if v == 0]
            ui = [i for i, v in enumerate(cou) if v == 1]
            out = torch.empty_like(ref)
            out[ci] = func(x[ci], t[ci], **_slice_c(c, ci, b))
            out[ui] = func(x[ui], t[ui], **_slice_c(c, ui, b))
            captured["cos"] = torch.nn.functional.cosine_similarity(
                ref.flatten().float(), out.flatten().float(), dim=0).item()
            return ref
        return func(kw["input"], kw["timestep"], **kw["c"])

    m = base.clone()
    m.model_options = dict(base.model_options)
    m.model_options["model_function_wrapper"] = wrapper
    dit = base.model.diffusion_model
    latent = torch.zeros((1, dit.in_channels, 128, 128), dtype=torch.float32)
    cond = torch.randn(1, 8, dit.cap_embedder[1].in_features, dtype=torch.float32)
    noise = comfy.sample.prepare_noise(latent, seed=0)
    comfy.sample.sample(
        m, noise, steps=1, cfg=2.0, sampler_name="euler", scheduler="simple",
        positive=[[cond, {}]], negative=[[torch.zeros_like(cond), {}]],
        latent_image=latent, denoise=1.0)

    cos = captured.get("cos")
    assert cos is not None, "cfg=2 batched wrapper never fired"
    print(f"\nMEASURED cfg batch-split cos vs single-GPU batched = {cos:.6f} (bar {QUALITY_BAR})")
    if cos >= QUALITY_BAR:
        pytest.xfail(f"batch-split now MATCHES (cos={cos:.4f}) — cfg_parallel can be enabled")
    assert cos < QUALITY_BAR, f"unexpectedly high split cos {cos:.4f}"


def test_cfg_parallel_fallback_no_crash_fake_model():
    """Pure-logic: FNI8Multigpu(cfg_parallel) never crashes and returns the model
    unchanged when it can't parallelise (fake model, no real GPUs to split onto)."""
    if len(available_devices()) >= 2:
        pytest.skip("this fallback path is only guaranteed with <2 GPUs")
    import torch.nn as nn

    class FakeModel:
        model_options: dict = {}
        model = type("M", (), {"diffusion_model": nn.Linear(8, 8)})()

        def clone(self):
            return self

    model = FakeModel()
    (out,) = FNI8Multigpu().apply(model, "cfg_parallel")
    assert out is model, "single-GPU fallback must return the model unchanged"
