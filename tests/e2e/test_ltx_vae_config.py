# SPDX-License-Identifier: Apache-2.0
"""End-to-end: the LTX-2.3 video VAE builds from the converter-PERSISTED
`metadata['config']['vae']` and decodes a real latent → frames (no channel mismatch).

The bug (see docs/ltx23-single-card-findings.md, blocker #3): the `.fni8` carries the raw
`vae.` weights but the VAE *config* was dropped, so `comfy.sd.VAE` falls back to sizing the
LTX video VAE from a single conv shape + a built-in `version` preset (0/1/2). LTX-2.3's VAE
matches none of them — auto-detect mis-sizes the decoder and dies at `decoder.up_blocks.4`
(512-vs-256 channel mismatch), so no latent→frames decode is possible.

The fix (this PR): the converter persists the VAE config (from the checkpoint's safetensors
`__metadata__['config']`, since LTX-2.3 ships no config.json) and the loader builds the VAE
from it (`ltx2_vae_metadata` → `comfy.sd.VAE(metadata=...)` →
`VideoVAE(config=config['vae'])`), yielding the EXACT LTX-2.3 `CausalVideoAutoencoder`.

Because the already-shipped `.fni8` predates the converter persisting the config, this test
builds a compact vae-only `.fni8` from the mounted bundle's raw `vae.` sub-tree PLUS the real
VAE config (fixture `ltx23_vae_config.json` = what the fixed converter writes), then proves
both the reproduced bug and the fix on the real weights. Skips cleanly without weights/GPU.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

# The repo root (this node pack) must be importable so `bench.full_pipeline_ltx` resolves
# when pytest runs from /opt/ComfyUI (same pattern as the other e2e tests).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import comfy.sd
import folder_paths

from comfyui_superl8.gate import is_sm70
from comfyui_superl8.loader import (
    load_fni8_dit,
    ltx2_canonical_vae_config,
    ltx2_persisted_vae_config,
    ltx2_vae_metadata,
)

pytestmark = pytest.mark.comfy_e2e

LTX23 = "Lightricks__LTX-2.3.dit.b8.fni8"
_FIXTURE = os.path.join(os.path.dirname(__file__), "ltx23_vae_config.json")


@pytest.fixture(scope="module")
def vae_only_fni8(tmp_path_factory):
    """A compact vae-only `.fni8` (bundle's raw `vae.*` sub-tree) carrying the persisted VAE
    config in `meta['config']['vae']`, exactly as the fixed converter writes it."""
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    names = folder_paths.get_filename_list("diffusion_models")
    if LTX23 not in names:
        pytest.skip(f"{LTX23} not found on the diffusion_models search path")
    from superl8 import FQReader, save_fni8

    bundle = folder_paths.get_full_path("diffusion_models", LTX23)
    vae_config = json.load(open(_FIXTURE))
    with FQReader(bundle) as r:
        vae_names = [n for n in r.names if n.startswith("vae.")]
        if not vae_names:
            pytest.skip("bundle has no `vae.` sub-tree")
        qsd = {n: r.get_qtensor(n, device="cpu") for n in vae_names}
    out = str(tmp_path_factory.mktemp("ltxvae") / "ltx23_vae_only.fni8")
    save_fni8(out, qsd, meta={"kind": "dit", "repo": "Lightricks/LTX-2.3", "bits": 8,
                              "native_dtype": "bfloat16", "config": {"vae": vae_config}})
    return out


def test_persisted_vae_config_roundtrips(vae_only_fni8):
    """The loader reads back the persisted VAE config + emits the `metadata=` blob comfy
    consumes. This is the durable source that supersedes comfy's version-preset auto-detect."""
    cfg = ltx2_persisted_vae_config(vae_only_fni8)
    assert cfg is not None and cfg["_class_name"] == "CausalVideoAutoencoder"
    assert cfg["latent_channels"] == 128
    # LTX-2.3's real schedule is 9 decoder blocks — comfy's presets are 7 (the mismatch).
    assert len(cfg["decoder_blocks"]) == 9
    md = ltx2_vae_metadata(vae_only_fni8)
    assert md is not None and json.loads(md["config"])["vae"]["latent_channels"] == 128


def test_autodetect_missizes_without_config(vae_only_fni8):
    """Without the persisted config, comfy's shape/version auto-detect builds the WRONG LTX
    VAE and fails loading the real weights (the 512-vs-256 channel mismatch) — the bug."""
    sd = load_fni8_dit(vae_only_fni8, device="cpu", strip_prefix="vae.",
                       keep_only_prefixed=True, dequant_fp=())
    with pytest.raises(Exception):
        comfy.sd.VAE(sd=dict(sd))               # NO metadata -> version-preset auto-detect


def test_vae_builds_and_decodes_from_config(vae_only_fni8):
    """The fix: built from the persisted config the VAE loads all weights (0 missing) and
    decodes a real latent → frames (correct 8×/32× upscale, finite, structured)."""
    from bench.full_pipeline_ltx import load_bundle_video_vae

    sd = load_fni8_dit(vae_only_fni8, device="cpu", strip_prefix="vae.",
                       keep_only_prefixed=True, dequant_fp=())
    vae = load_bundle_video_vae(vae_only_fni8)
    assert type(vae.first_stage_model).__name__ == "VideoVAE"
    missing = set(vae.first_stage_model.state_dict().keys()) - set(sd.keys())
    assert not missing, f"config built the wrong VAE — missing weights: {sorted(missing)[:8]}"

    g = torch.Generator(device="cpu").manual_seed(0)
    lat = torch.randn(1, 128, 3, 8, 8, generator=g, dtype=torch.float32)  # [B,128,T,H,W]
    with torch.no_grad():
        frames = vae.decode(lat.to("cuda:0"))
    # VAE upscale (8,32,32): 3 latent frames -> 8*3-7 = 17 video frames; 8px -> 256px.
    assert frames.shape[1] == 17 and frames.shape[-2] == 256
    assert torch.isfinite(frames).all()
    assert float(frames.float().std()) > 1e-3, "decoded frames are degenerate/constant"

    # int8-magnitude latent error (the DiT's error source; the VAE stays fp) survives the
    # decode with cos ~ 1 — the decode does not amplify int8 latent noise.
    amax = lat.abs().amax(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-8)
    lat_i8 = (torch.round(lat / (amax / 127.0)).clamp(-127, 127) * (amax / 127.0))
    with torch.no_grad():
        frames_q = vae.decode(lat_i8.to("cuda:0"))
    cos = float(torch.nn.functional.cosine_similarity(
        frames.float().flatten(), frames_q.float().flatten(), dim=0))
    assert cos > 0.99, f"int8-vs-fp latent-decode cosine {cos:.6f} below bar"


def test_canonical_config_matches_packaged_fixture():
    """The packaged canonical config (loader data) is the SAME LTX-2.3 VAE config as the test
    fixture — one durable source of truth, no drift between the loader fallback and the test."""
    canon = ltx2_canonical_vae_config()
    assert canon is not None and canon["_class_name"] == "CausalVideoAutoencoder"
    assert canon == json.load(open(_FIXTURE))


def test_pre_persist_real_bundle_falls_back_and_decodes():
    """THE DELIVERABLE PATH: the already-shipped LTX-2.3 `.fni8` on the fleet carries NO
    persisted VAE config (its meta is only kind/repo/bits/native_dtype). `ltx2_vae_metadata`
    must fall back to the packaged canonical config so `comfy.sd.VAE` builds the EXACT
    LTX-2.3 VAE — loading the real bundle's `vae.*` weights (0 missing) and decoding a real
    latent → finite frames — instead of the auto-detect 512-vs-256 crash. This is what makes
    the default pipeline (no re-conversion) produce a clip."""
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    names = folder_paths.get_filename_list("diffusion_models")
    if LTX23 not in names:
        pytest.skip(f"{LTX23} not found on the diffusion_models search path")
    bundle = folder_paths.get_full_path("diffusion_models", LTX23)

    # The real bundle has NO persisted config, but the fallback supplies one.
    assert ltx2_persisted_vae_config(bundle) is None, "expected a pre-persist bundle"
    md = ltx2_vae_metadata(bundle)
    assert md is not None, "canonical fallback did not fire for an LTX-2.3 bundle"
    assert json.loads(md["config"])["vae"]["latent_channels"] == 128

    from bench.full_pipeline_ltx import load_bundle_video_vae

    sd = load_fni8_dit(bundle, device="cpu", strip_prefix="vae.",
                       keep_only_prefixed=True, dequant_fp=())
    vae = load_bundle_video_vae(bundle)
    assert type(vae.first_stage_model).__name__ == "VideoVAE"
    missing = set(vae.first_stage_model.state_dict().keys()) - set(sd.keys())
    assert not missing, f"fallback built the wrong VAE — missing weights: {sorted(missing)[:8]}"

    g = torch.Generator(device="cpu").manual_seed(0)
    lat = torch.randn(1, 128, 3, 8, 8, generator=g, dtype=torch.float32)
    with torch.no_grad():
        frames = vae.decode(lat.to("cuda:0"))
    assert frames.shape[1] == 17 and frames.shape[-2] == 256
    assert torch.isfinite(frames).all()
    assert float(frames.float().std()) > 1e-3, "decoded frames are degenerate/constant"
