# SPDX-License-Identifier: Apache-2.0
"""FULL end-to-end Z-Image-Turbo pipeline e2e test (not just the DiT denoise smoke).

Exercises the whole generation path the way a ComfyUI graph does:

    Qwen3-4B text encoder  ->  int8 dp4a Z-Image DiT (multi-step Turbo denoise)
                           ->  tiled VAE decode (#97)  ->  a decoded RGB image

and a matching fp reference (same conditioning, same VAE, DiT weights dequantized to
bf16). This is the pass ``docs/int8-dit-validation.md`` said "wouldn't fit on the 16 GB
card" before the tiled VAE (#97) landed. Two things are checked separately:

  * ``test_..._runs_and_fits`` — the pipeline runs end to end and fits 16 GB, the tiled
    VAE decodes to a finite structured RGB image, and the **fp** reference is coherent.
    This is the #97 memory + wiring validation and is expected to PASS.
  * ``test_..._int8_matches_fp`` — the int8-vs-fp *decoded-image* match. The full
    multi-step + VAE-decode pass (affordable via #97) first revealed the int8 DiT
    collapsing to an incoherent patch-grid (PSNR ≈ 7.6 dB); the root cause was a loader
    bug (comfy fuses Z-Image `to_q`/`to_k`/`to_v` into `qkv`, corrupting int8), fixed by
    loading the attention projections fp (`DiTArch.fp_dequant`). Now a real gate: the
    int8 image reproduces the fp image (PSNR ≈ 31 dB @ 8-step/512px). No bar is weakened.
    See ``docs/zimage-full-pipeline-findings.md``.

Gated, like the rest of the e2e matrix, on the real weights being present:
  * the int8 DiT ``Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8`` on the diffusion_models path,
  * the Qwen3-4B TE shards under ``<weights>/zimage_te``,
  * the Z-Image VAE under ``<weights>/zimage_vae``.
Any missing -> skip (never fail); OOM -> skip (run on a larger/idle GPU).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import folder_paths

# Make the repo root importable so `bench.full_pipeline_zimage` resolves.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from comfyui_superl8.gate import is_sm70

import bench.full_pipeline_zimage as fp
from bench.quality import cosine_similarity, psnr, ssim

pytestmark = pytest.mark.comfy_e2e

# Keep the e2e run cheap: a 4-step denoise at 256 px still exercises multi-step +
# tiled VAE + the full TE->DiT->VAE wiring, without the bench's 8-step/512 px cost.
STEPS = 4
SIZE = 32           # 256 px at 8x VAE stride
CARD_GIB = 16.0     # the deployment card
PSNR_MATCH = 20.0   # a real "same image" floor for the int8-vs-fp match (xfail today)


def _have_aux() -> bool:
    te = os.path.join(fp.TE_DIR, "model-00001-of-00003.safetensors")
    return os.path.isfile(te) and os.path.isfile(fp.VAE_FILE)


def _skip_on_oom(fn):
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("full pipeline does not fit this card — run on a larger/idle GPU")


@pytest.fixture(scope="module")
def pipeline_outputs():
    """Run the full pipeline once (int8 + fp), return latents, images, and peak HBM."""
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    folder_paths.add_model_folder_path("diffusion_models", fp.WEIGHTS_DIR)
    if fp.ZIMAGE_DIT not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{fp.ZIMAGE_DIT} not on the diffusion_models search path")
    if not _have_aux():
        pytest.skip("Z-Image Qwen3-4B TE / VAE aux weights not present "
                    f"(expected under {fp.TE_DIR} and {fp.VAE_FILE})")

    import comfy.sd
    import comfy.utils

    path = folder_paths.get_full_path("diffusion_models", fp.ZIMAGE_DIT)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(fp.VAE_FILE, safe_load=True))
    cond, empty = fp.encode_prompt(fp.DEFAULT_PROMPT)

    torch.cuda.reset_peak_memory_stats()

    attn_stats: dict = {}

    def _int8():
        m = fp.load_int8_model()
        in_ch = m.model.diffusion_model.in_channels
        attn_stats.update(_int8_attn_stats(m.model.diffusion_model))
        lat = fp.denoise(m, cond, empty, in_ch, SIZE, STEPS, seed=0)
        del m
        torch.cuda.empty_cache()
        return in_ch, lat

    in_ch, lat_int8 = _skip_on_oom(_int8)
    img_int8 = _skip_on_oom(
        lambda: fp.vae_decode_tiled(vae, lat_int8, tile_size=512, overlap=64))

    def _fp():
        m = fp.load_fp_model(path)
        lat = fp.denoise(m, cond, empty, in_ch, SIZE, STEPS, seed=0)
        del m
        torch.cuda.empty_cache()
        return lat

    lat_fp = _skip_on_oom(_fp)
    img_fp = _skip_on_oom(
        lambda: fp.vae_decode_tiled(vae, lat_fp, tile_size=512, overlap=64))

    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return dict(lat_int8=lat_int8, img_int8=img_int8, lat_fp=lat_fp, img_fp=img_fp,
                peak_gib=peak_gib, attn_stats=attn_stats)


def _int8_attn_stats(diffusion_model) -> dict:
    """Count how many attention projection params are int8 FNI8Tensor after loading —
    the check that the Z-Image qkv re-fusion actually ENGAGED int8 attention (not the
    fp fallback #103 shipped). `attention.qkv.weight` is the fused per-row-int8 [3*dim,dim]
    weight; `attention.out.weight` the output projection."""
    from comfyui_superl8.superl8_tensor import FNI8Tensor

    qkv = out = 0
    for n, p in diffusion_model.named_parameters():
        if n.endswith("attention.qkv.weight"):
            qkv += int(isinstance(p, FNI8Tensor))
        elif n.endswith("attention.out.weight"):
            out += int(isinstance(p, FNI8Tensor))
    return {"qkv_int8": qkv, "out_int8": out}


def test_zimage_full_pipeline_runs_and_fits(pipeline_outputs):
    """The full Qwen3-4B -> int8 DiT -> tiled VAE pipeline runs end to end, fits the
    16 GB card, and the fp reference decodes to a coherent (finite, structured) image.
    This is the #97 memory + wiring validation."""
    o = pipeline_outputs

    # int8 latent survived the multi-step denoise without overflow-to-NaN / black.
    assert torch.isfinite(o["lat_int8"]).all(), "int8 latent non-finite (overflow)"
    assert o["lat_int8"].std().item() > 1e-6, "int8 latent collapsed to constant"

    # Tiled VAE (#97) produced a real RGB image tensor from the latent.
    for tag in ("img_int8", "img_fp"):
        img = o[tag]
        assert img.shape[1] == 3, f"{tag}: expected RGB, got {tuple(img.shape)}"
        assert torch.isfinite(img).all(), f"{tag} non-finite"

    # The fp reference is a coherent, structured image (not black/constant).
    assert o["img_fp"].std().item() > 1e-3, "fp image collapsed to constant"

    # The whole pipeline fit the 16 GB budget.
    assert o["peak_gib"] < CARD_GIB, \
        f"pipeline peak HBM {o['peak_gib']:.2f} GiB exceeds the {CARD_GIB} GiB card"


def test_zimage_int8_attention_is_engaged(pipeline_outputs):
    """The Z-Image attention PROJECTIONS run int8 dp4a — not the fp fallback #103 shipped.

    #103 kept `to_q`/`to_k`/`to_v`/`to_out` fp because ComfyUI's Z-Image model fuses
    q/k/v into one `qkv` and a naive int8-code concat (three independent per-row scales)
    is corrupt. The loader now re-fuses them LOSSLESSLY — `per_row_i8` carries one scale
    per output row, so concatenating q/k/v is just concatenating code rows AND scale
    vectors — yielding a single int8 `attention.qkv.weight` that runs dp4a. This asserts
    every attention block's fused qkv + out projection is an int8 FNI8Tensor after load,
    so 'full W8A8' (attention projections + FFN on dp4a) is real, not claimed."""
    stats = pipeline_outputs["attn_stats"]
    assert stats.get("qkv_int8", 0) > 0, (
        "no int8 attention.qkv params — the qkv re-fusion did not engage int8 attention "
        f"(stats={stats}); attention fell back to fp")
    assert stats["out_int8"] == stats["qkv_int8"], (
        f"int8 out projections ({stats['out_int8']}) != int8 qkv ({stats['qkv_int8']}) — "
        "some attention block is not fully int8")


def test_zimage_int8_multistep_image_matches_fp(pipeline_outputs):
    """int8-vs-fp on the DECODED image: the int8 8-step generation reproduces the fp
    image (a coherent 'same image', not the earlier patch-grid collapse).

    History: the full multi-step + VAE-decode pass first revealed the int8 image
    collapsing to an incoherent patch-grid (PSNR ~7.6 dB). Root cause was NOT precision
    but a loader bug — ComfyUI's Z-Image model FUSES the diffusers `to_q`/`to_k`/`to_v`
    projections into one `qkv`, and comfy's concat corrupts int8 (three independent
    per-row scales), leaving attention weights garbage. #103 first fixed it by loading the
    attention projections fp; this build re-fuses them losslessly into a single int8 `qkv`
    so attention runs dp4a too (FFN was always int8). The int8 image must still reproduce
    the fp image (#103 measured ~31 dB @ 8-step/512px). See
    docs/zimage-full-pipeline-findings.md. The bar is a genuine 'same image' floor,
    never weakened to pass."""
    o = pipeline_outputs
    cos = cosine_similarity(o["lat_int8"], o["lat_fp"])
    ps = psnr(o["img_int8"], o["img_fp"])
    ss = ssim(o["img_int8"], o["img_fp"])
    print(f"\n[zimage full pipeline] latent cos={cos:.6f}  image PSNR={ps:.2f}dB  "
          f"SSIM={ss:.6f}  peak={o['peak_gib']:.2f}GiB")
    assert ps >= PSNR_MATCH, f"int8-vs-fp image PSNR {ps:.2f} dB < {PSNR_MATCH}"
