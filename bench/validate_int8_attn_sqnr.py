# SPDX-License-Identifier: Apache-2.0
"""Measure the REAL per-call-site int8 self-attention SQNR on a live DiT denoise
(issue #82). Loads a `.fni8` DiT through the production `UnetLoaderFNI8` path (which
now wires the `Int8AttnGate`), spies on `attn_sqnr` to capture the SQNR of every int8
attention call vs its fp SDPA reference during a multi-step denoise, and reports the
distribution + how many call-sites the gate demotes at the configured floor.

This grounds the `ATTN_SQNR_FLOOR_DB` choice in measured data and proves both that
int8 attention engages AND that the gate makes a sane accept/reject decision on real
weights.

    python3 bench/validate_int8_attn_sqnr.py [--unet <name>] [--arch zimage]
                                             [--steps 8] [--hw 32]
"""
from __future__ import annotations

import argparse
import statistics

import folder_paths
import torch

import comfyui_superl8.attention as A
from comfyui_superl8.attention import ATTN_SQNR_FLOOR_DB
from comfyui_superl8.gate import is_sm70
from comfyui_superl8.nodes import UnetLoaderFNI8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8")
    ap.add_argument("--arch", default="zimage")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--hw", type=int, default=32)
    args = ap.parse_args()

    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return
    # Register the mounted weights archive (mirrors tests/e2e/conftest.py) so the DiT
    # is discoverable when this script runs standalone (outside pytest).
    import os
    weights_dir = os.environ.get("FNI8_WEIGHTS_DIR", "")
    if os.path.isdir(weights_dir):
        folder_paths.add_model_folder_path("diffusion_models", weights_dir)
    if args.unet not in folder_paths.get_filename_list("diffusion_models"):
        print(f"SKIP: {args.unet} not on the diffusion_models search path")
        return

    import comfy.sample

    (model,) = UnetLoaderFNI8().load(args.unet, arch=args.arch)
    dit = model.model.diffusion_model
    in_channels = dit.in_channels
    cap_feat_dim = dit.cap_embedder[1].in_features

    # Capture every SQNR the gate measures + count int8-kernel invocations.
    measured: list[float] = []
    orig_sqnr = A.attn_sqnr

    def spy_sqnr(y_int8, y_fp):
        v = orig_sqnr(y_int8, y_fp)
        measured.append(v)
        return v

    int8_calls = {"n": 0}
    import superl8
    orig_fwd = superl8.attn_int8_fwd

    def spy_fwd(*a, **k):
        int8_calls["n"] += 1
        return orig_fwd(*a, **k)

    A.attn_sqnr = spy_sqnr
    superl8.attn_int8_fwd = spy_fwd
    try:
        latent = torch.zeros((1, in_channels, args.hw, args.hw), dtype=torch.float32)
        cond = torch.randn(1, 8, cap_feat_dim, dtype=torch.float32)
        noise = comfy.sample.prepare_noise(latent, seed=0)
        samples = comfy.sample.sample(
            model, noise, steps=args.steps, cfg=1.0, sampler_name="euler",
            scheduler="simple", positive=[[cond, {}]],
            negative=[[torch.zeros_like(cond), {}]], latent_image=latent, denoise=1.0,
        )
    finally:
        A.attn_sqnr = orig_sqnr
        superl8.attn_int8_fwd = orig_fwd

    finite = bool(torch.isfinite(samples).all())
    nonconst = samples.std().item() > 1e-6

    print("=" * 68)
    print(f"int8 DiT self-attention SQNR validation — {args.unet}")
    print("=" * 68)
    print(f"int8 attn_int8_fwd invocations : {int8_calls['n']}  "
          f"({'ENGAGED' if int8_calls['n'] > 0 else 'NOT ENGAGED — ran fp SDPA'})")
    print(f"denoise output finite          : {finite}")
    print(f"denoise output non-constant    : {nonconst}")
    print(f"SQNR gate floor                : {ATTN_SQNR_FLOOR_DB:.1f} dB")
    print(f"distinct call-sites gated       : {len(measured)}")
    if measured:
        finite_vals = [v for v in measured if v not in (float("inf"), float("-inf"))]
        # A site is demoted whenever it does NOT clear the floor — including the
        # non-finite -inf case (untrustworthy SQNR -> fp fallback).
        below = [v for v in measured if not (v >= ATTN_SQNR_FLOOR_DB)]
        print(f"SQNR min / median / max (dB)   : "
              f"{min(measured):.2f} / {statistics.median(measured):.2f} / {max(measured):.2f}")
        if finite_vals:
            print(f"SQNR mean (finite, dB)         : {statistics.mean(finite_vals):.2f}")
        print(f"call-sites BELOW floor (demoted): {len(below)} / {len(measured)}")
        print("per-call-site SQNR (dB): " +
              ", ".join(f"{v:.1f}" for v in measured))
    print("=" * 68)


if __name__ == "__main__":
    main()
