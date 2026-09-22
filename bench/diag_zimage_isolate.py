# SPDX-License-Identifier: Apache-2.0
"""Isolate the int8 Z-Image collapse. The .fni8 meta records native_dtype='float16',
so UnetLoaderFNI8 runs the whole DiT (norms/modulation/residual stream + int8 linears'
fp I/O) in fp16 — but Z-Image is bf16-native (AGENTS.md: fp16 overflows -> garbage), and
the fp reference is hardcoded bf16. This tests whether that dtype seam is the culprit by
running the int8 linears in bf16 vs fp16, both vs the bf16 fp reference.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bench.full_pipeline_zimage as F


def _load_int8(path, dtype, keep_fp=(), attn_patch=False, arch="zimage"):
    """Build the int8 DiT with an explicit compute dtype (override the meta) and an
    optional extra keep-fp substring set. attn_patch toggles int8 attention (proven
    irrelevant: fp/int8 attention were bit-identical, so default off)."""
    import comfy.sd

    from comfyui_superl8.superl8_tensor import FNI8Tensor
    from comfyui_superl8.loader import (
        _HEAD_FP_DEQUANT, assign_int8_weights, load_fni8_dit,
        remap_diffusers_to_native,
    )
    from comfyui_superl8.ops import FNI8Ops

    dequant_fp = tuple(_HEAD_FP_DEQUANT) + tuple(keep_fp)
    sd = load_fni8_dit(path, device="cpu", strip_prefix="", dequant_fp=dequant_fp)
    sd = remap_diffusers_to_native(sd, None)
    int8_weights = {k: v for k, v in sd.items() if isinstance(v, FNI8Tensor)}
    model = comfy.sd.load_diffusion_model_state_dict(
        dict(sd), model_options={"custom_operations": FNI8Ops, "dtype": dtype})
    n = assign_int8_weights(model.model.diffusion_model, int8_weights) if int8_weights else 0
    print(f"  [build dtype={dtype} keep_fp={keep_fp}] int8 linears={n}", flush=True)
    if attn_patch:
        from comfyui_superl8.nodes import FNI8AttentionPatch
        (model,) = FNI8AttentionPatch().patch(model, arch=arch)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.environ.get("FNI8_PIPE_OUT", ".pipe_out"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import folder_paths
    folder_paths.add_model_folder_path("diffusion_models", F.WEIGHTS_DIR)

    import comfy.sd
    import comfy.utils
    from comfyui_superl8.gate import is_sm70
    from bench.quality import cosine_similarity, psnr

    if not is_sm70():
        print("SKIP: needs sm_70"); return
    path = folder_paths.get_full_path("diffusion_models", F.ZIMAGE_DIT)

    cond, empty = F.encode_prompt(F.DEFAULT_PROMPT)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(F.VAE_FILE, safe_load=True))

    def run(model, tag):
        in_ch = model.model.diffusion_model.in_channels
        lat = F.denoise(model, cond, empty, in_ch, args.size, args.steps, args.seed)
        img = F.vae_decode_tiled(vae, lat, 512, 64)
        F.save_png(img, os.path.join(args.outdir, f"zimage_{tag}.png"))
        print(f"  [{tag}] latent std={lat.std().item():.4f} img std={img.std().item():.4f}",
              flush=True)
        return lat, img

    print("[A] fp reference (bf16, fp SDPA)", flush=True)
    m = F.load_fp_model(path)
    lat_fp, img_fp = run(m, "A_fp")
    del m; torch.cuda.empty_cache()

    results = {}
    configs = [
        ("F_attnfp_ffnint8", torch.bfloat16, ("attention.to_",)),
        ("G_ffnfp_attnint8", torch.bfloat16, ("feed_forward",)),
    ]
    for tag, dt, keep in configs:
        print(f"[{tag}] int8 linears, dtype={dt}, keep_fp={keep}", flush=True)
        m = _load_int8(path, dt, keep_fp=keep)
        lat, img = run(m, tag)
        results[tag] = (lat, img)
        del m; torch.cuda.empty_cache()

    print("=" * 64, flush=True)
    print(f"  fp ref latent std={lat_fp.std().item():.4f} img std={img_fp.std().item():.4f}")
    for tag, (lat, img) in results.items():
        print(f"  {tag:16s} vs fp: latentcos={cosine_similarity(lat, lat_fp):.4f}  "
              f"PSNR={psnr(img, img_fp):.2f}dB", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
