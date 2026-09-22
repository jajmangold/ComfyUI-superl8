#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure STA quality vs dense attention on a live video DiT denoise.

Loads a video DiT through the production ``UnetLoaderFNI8`` path and hooks
every self-attention call to compare:
  1. Dense attention (the existing int8 kernel + fp SDPA reference)
  2. STA (Sliding-Tile Attention via masked SDPA)

Reports per-call-site SQNR, cosine similarity, theoretical FLOP savings, and
whether the STA gate would keep or demote each site at the configured floor.

Tile sizes are configurable; the benchmark tries multiple configurations to
map the quality-cost frontier::

    python3 bench/validate_sta_quality.py
        --unet Wan2.2-TI2V-5B-Diffusers.dit.b8.fni8
        --arch wan22
        --tile-f 9 --tile-h 8 --tile-w 8

Negative results (low SQNR for aggressive tiles) are reported honestly —
this is a draft-PR measurement tool, not a quality gate.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from comfyui_superl8.sta import (
    StaGate,
    flops_report,
    make_sta_mask_3d,
    sparsity_ratio,
    sta_sqnr,
)


def main():
    ap = argparse.ArgumentParser(description="STA quality vs dense attention benchmark (video DiT)")
    ap.add_argument("--unet", default="Wan-AI__Wan2.2-TI2V-5B-Diffusers.dit.b8.fni8")
    ap.add_argument("--arch", default="wan22")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--f", type=int, default=8, help="Token grid frames (after patch embed)")
    ap.add_argument("--h", type=int, default=12, help="Token grid height (after patch embed)")
    ap.add_argument("--w", type=int, default=8, help="Token grid width (after patch embed)")
    ap.add_argument("--tile-f", type=int, default=9, help="STA tile size in frames")
    ap.add_argument("--tile-h", type=int, default=8, help="STA tile size in height")
    ap.add_argument("--tile-w", type=int, default=8, help="STA tile size in width")
    ap.add_argument(
        "--sqnr-floor", type=float, default=20.0, help="SQNR floor in dB for the STA gate"
    )
    ap.add_argument(
        "--multi-tile",
        type=str,
        default=None,
        help="Comma-separated tile configs to scan, e.g. '5x4x4,9x8x8,13x12x12'",
    )
    args = ap.parse_args()

    try:
        from comfyui_superl8.gate import is_sm70

        if not is_sm70():
            print("SKIP: needs a Volta/CMP (sm_70) GPU")
            return
    except Exception:
        print("SKIP: could not check GPU capability")
        return

    import os
    import folder_paths

    weights_dir = os.environ.get("FNI8_WEIGHTS_DIR", "")
    if os.path.isdir(weights_dir):
        folder_paths.add_model_folder_path("diffusion_models", weights_dir)
    if args.unet not in folder_paths.get_filename_list("diffusion_models"):
        print(f"SKIP: {args.unet} not on the diffusion_models search path")
        return

    from comfyui_superl8.nodes import UnetLoaderFNI8

    (model,) = UnetLoaderFNI8().load(args.unet, arch=args.arch)
    dit = model.model.diffusion_model

    F, H, W = args.f, args.h, args.w
    N = F * H * W
    tile_f, tile_h, tile_w = args.tile_f, args.tile_h, args.tile_w

    fr = flops_report(N, tile_f * tile_h * tile_w)
    print(f"STA config: F={F} H={H} W={W}  N={N}")
    print(f"  tile: {tile_f}x{tile_h}x{tile_w}  (volume={fr['tile_volume']})")
    print(f"  dense flops (rel): {fr['dense_flops']}")
    print(f"  sta flops (rel):   {fr['sta_flops']}")
    print(f"  theoretical reduction: {fr['reduction'] * 100:.1f}%")

    mask = make_sta_mask_3d(F, H, W, tile_f, tile_h, tile_w)
    print(f"  actual mask sparsity: {sparsity_ratio(mask) * 100:.1f}%")

    raw_in_channels = dit.patch_embedding.in_channels
    context_dim = None
    te = getattr(dit, "text_embedding", None)
    if te is not None:
        context_dim = te[0].in_features

    import comfy.sample

    latent_image = torch.zeros((1, raw_in_channels, F, H, W), dtype=torch.float32)
    cond_tensor = torch.randn(1, 8, context_dim or 4096, dtype=torch.float32)

    from comfyui_superl8 import attention as A

    per_site: list[dict] = []

    orig_int8 = A._int8_dp4a

    def spy_int8(q, k, v):
        dense_out = orig_int8(q, k, v)
        sta_out = _sta_via_sdpa(q, k, v, F, H, W, tile_f, tile_h, tile_w)

        sqnr = sta_sqnr(sta_out, dense_out)
        cos = torch.nn.functional.cosine_similarity(
            sta_out.float().flatten(), dense_out.float().flatten(), dim=0
        ).item()

        record = dict(
            sqnr_db=sqnr,
            cosine=cos,
            B=q.shape[0],
            heads=q.shape[1],
            S=q.shape[2],
            D=q.shape[3],
            tile=f"{tile_f}x{tile_h}x{tile_w}",
        )
        per_site.append(record)
        return dense_out

    A._int8_dp4a = spy_int8
    try:
        noise = comfy.sample.prepare_noise(latent_image, seed=0)
        samples = comfy.sample.sample(
            model,
            noise,
            steps=args.steps,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=[[cond_tensor, {}]],
            negative=[[torch.zeros_like(cond_tensor), {}]],
            latent_image=latent_image,
            denoise=1.0,
        )
    finally:
        A._int8_dp4a = orig_int8

    finite = bool(torch.isfinite(samples).all())
    nonconst = samples.std().item() > 1e-6

    print("\n" + "=" * 72)
    print(f"STA quality vs dense attention — {args.unet}")
    print(f"  arch={args.arch}  tile={tile_f}x{tile_h}x{tile_w}  layout={F}x{H}x{W}")
    print(f"  denoise steps={args.steps}  output finite={finite}  non-constant={nonconst}")
    print(f"  SQNR floor: {args.sqnr_floor:.1f} dB")
    print(f"  attention call-sites measured: {len(per_site)}")
    print(f"  theoretical FLOP reduction: {fr['reduction'] * 100:.1f}%")
    print("-" * 72)

    if not per_site:
        print("  NO attention calls captured — int8 kernel did not engage?")
        return

    sqnrs = [s["sqnr_db"] for s in per_site]
    cosines = [s["cosine"] for s in per_site]
    finite_sqnrs = [v for v in sqnrs if v not in (float("inf"), float("-inf"))]
    below_floor = [v for v in sqnrs if not (v >= args.sqnr_floor)]

    print(
        f"  SQNR (dB):        min={min(sqnrs):6.2f}  "
        f"median={statistics.median(sqnrs):6.2f}  "
        f"max={max(sqnrs):6.2f}"
    )
    if finite_sqnrs:
        print(f"  SQNR mean (finite): {statistics.mean(finite_sqnrs):6.2f}")
    print(
        f"  Cosine similarity: min={min(cosines):.6f}  "
        f"mean={statistics.mean(cosines):.6f}  "
        f"max={max(cosines):.6f}"
    )
    print(
        f"  Sites below {args.sqnr_floor} dB floor: "
        f"{len(below_floor)}/{len(per_site)} "
        f"({len(below_floor) / len(per_site) * 100:.1f}%)"
    )
    print()
    print("  Per-site SQNR (dB):")
    for i, s in enumerate(per_site):
        flag = " *** DEMOTE ***" if s["sqnr_db"] < args.sqnr_floor else ""
        print(
            f"    [{i}] B={s['B']} H={s['heads']} S={s['S']} D={s['D']}  "
            f"sqnr={s['sqnr_db']:7.2f}  cos={s['cosine']:.6f}{flag}"
        )
    print("=" * 72)


def _sta_via_sdpa(q, k, v, F, H, W, tile_f, tile_h, tile_w):
    mask = make_sta_mask_3d(F, H, W, tile_f, tile_h, tile_w, device=q.device)
    attn_mask = mask[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)
    attn_mask = torch.where(attn_mask, 0.0, float("-inf"))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


if __name__ == "__main__":
    main()
