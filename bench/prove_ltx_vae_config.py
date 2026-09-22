# SPDX-License-Identifier: Apache-2.0
"""Proof: the LTX-2.3 video VAE decodes correctly when built from the converter-PERSISTED
`metadata['config']['vae']`, and does NOT (channel mismatch) under comfy's auto-detect.

Runs on the fleet (Volta/CMP sm_70) against the mounted LTX-2.3 bundle. Because the
already-shipped `.fni8` predates the converter persisting the config, this builds a compact
vae-only `.fni8` from the bundle's raw `vae.` sub-tree PLUS the real VAE config (persisted in
meta exactly as the fixed converter does), then exercises the loader's config-driven build.

    FNI8_GPU=13 docker compose run --rm --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI:custom_nodes/ComfyUI-superl8 python3 \
       custom_nodes/ComfyUI-superl8/bench/prove_ltx_vae_config.py'
"""
from __future__ import annotations

import json
import os
import sys

import torch

BUNDLE = os.path.join(os.environ.get("FNI8_WEIGHTS_DIR", ""), "Lightricks__LTX-2.3.dit.b8.fni8")
FIXTURE = os.path.join(os.path.dirname(__file__), "..", "tests", "e2e", "ltx23_vae_config.json")
SMALL = "/tmp/ltx23_vae_only.fni8"


def build_vae_only_fni8(bundle: str, out: str, vae_config: dict) -> int:
    """Copy the bundle's raw `vae.*` tensors into a small `.fni8`, persisting the VAE config
    in meta['config']['vae'] exactly as the fixed converter (tools/forge.py #89) does."""
    from superl8 import FQReader, save_fni8

    with FQReader(bundle) as r:
        names = [n for n in r.names if n.startswith("vae.")]
        qsd = {n: r.get_qtensor(n, device="cpu") for n in names}
    meta = {"kind": "dit", "repo": "Lightricks/LTX-2.3", "bits": 8,
            "native_dtype": "bfloat16", "config": {"vae": vae_config}}
    save_fni8(out, qsd, meta=meta)
    return len(qsd)


def main() -> int:
    from comfyui_superl8.gate import is_sm70
    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return 0
    if not os.path.exists(BUNDLE):
        print(f"SKIP: bundle not mounted at {BUNDLE}")
        return 0

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import comfy.sd
    from comfyui_superl8.loader import (
        load_fni8_dit,
        ltx2_persisted_vae_config,
        ltx2_vae_metadata,
    )
    from bench.full_pipeline_ltx import load_bundle_video_vae

    def decode(v, latent):
        with torch.no_grad():
            return v.decode(latent.to("cuda:0"))

    vae_config = json.load(open(os.path.abspath(FIXTURE)))
    print(f"[fixture] VAE config: {vae_config['_class_name']} "
          f"latent_channels={vae_config['latent_channels']} "
          f"decoder_blocks={len(vae_config['decoder_blocks'])}")

    n = build_vae_only_fni8(BUNDLE, SMALL, vae_config)
    print(f"[build] vae-only .fni8 written ({n} raw vae.* tensors) -> {SMALL}")

    # ---- 1. loader reads the persisted config back ----------------------------------
    got = ltx2_persisted_vae_config(SMALL)
    assert got is not None and got["_class_name"] == "CausalVideoAutoencoder", got
    md = ltx2_vae_metadata(SMALL)
    assert md and json.loads(md["config"])["vae"]["latent_channels"] == 128
    print("[1] ltx2_persisted_vae_config / ltx2_vae_metadata: OK "
          f"(decoder_blocks={len(got['decoder_blocks'])})")

    # ---- 2. reproduce the BUG: comfy auto-detect (no metadata) mis-sizes the VAE -----
    vae_sd = load_fni8_dit(SMALL, device="cpu", strip_prefix="vae.",
                           keep_only_prefixed=True, dequant_fp=())
    try:
        comfy.sd.VAE(sd=dict(vae_sd))            # NO metadata -> version-preset auto-detect
        print("[2] auto-detect UNEXPECTEDLY built a VAE (bug not reproduced?)")
        autodetect_failed = False
    except Exception as e:
        msg = str(e).splitlines()[0][:160]
        print(f"[2] auto-detect (no config) FAILS as expected: {type(e).__name__}: {msg}")
        autodetect_failed = True

    # ---- 3. config-driven build succeeds (the fix) -----------------------------------
    vae = load_bundle_video_vae(SMALL)
    missing, unexpected = [], []
    model_keys = set(vae.first_stage_model.state_dict().keys())
    got_keys = set(vae_sd.keys())
    missing = sorted(model_keys - got_keys)
    unexpected = sorted(got_keys - model_keys)
    print(f"[3] config-built VAE: {type(vae.first_stage_model).__name__}; "
          f"state_dict {len(model_keys)} keys; missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing, f"missing keys (wrong VAE built): {missing[:8]}"

    # ---- 4. decode a real latent -> frames (no channel mismatch) ---------------------
    dev = "cuda:0"
    g = torch.Generator(device="cpu").manual_seed(0)
    # LTX latent: [B, 128, T, H, W]; VAE upscale (8,32,32). Small dims keep HBM modest.
    lat = torch.randn(1, 128, 3, 8, 8, generator=g, dtype=torch.float32)
    frames = decode(vae, lat)
    finite = bool(torch.isfinite(frames).all())
    structured = float(frames.float().std()) > 1e-3
    print(f"[4] decoded latent {tuple(lat.shape)} -> frames {tuple(frames.shape)} "
          f"finite={finite} structured={structured} std={float(frames.float().std()):.4f}")
    assert finite and structured, "decoded frames non-finite or degenerate"

    # ---- 5. int8-vs-fp latent-decode cosine ------------------------------------------
    # The VAE is fp (never quantized); the int8 error enters in the DiT's LATENT. Model that
    # here by round-tripping the sample latent through per-channel int8 (the same per-row-i8
    # recipe the DiT uses) and decoding both — measuring how much int8-magnitude latent error
    # survives the decode. (The full DiT int8-vs-fp denoise cosine is gated on the fit blocker
    # — 2-GPU split / shrunk DiT — and is unchanged by this VAE fix.)
    amax = lat.abs().amax(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    lat_int8 = (torch.round(lat / scale).clamp(-127, 127) * scale).to(lat.dtype)
    frames_q = decode(vae, lat_int8)
    a, b = frames.float().flatten(), frames_q.float().flatten()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print(f"[5] int8-vs-fp latent-decode cosine (sample latent): {cos:.6f}")

    ok = autodetect_failed and finite and structured and not missing
    print("=" * 72)
    print(f"RESULT: LTX-2.3 video VAE decodes correctly from persisted config = {ok}")
    print(f"  auto-detect mis-sizes (bug reproduced): {autodetect_failed}")
    print(f"  config-built decode finite+structured : {finite and structured}")
    print(f"  int8-vs-fp latent-decode cosine       : {cos:.6f}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
