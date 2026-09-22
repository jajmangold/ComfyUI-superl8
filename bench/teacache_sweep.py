# SPDX-License-Identifier: Apache-2.0
"""TeaCache threshold sweep on the int8 dp4a DiT path — measures the stacked speedup
(int8-only vs int8+TeaCache) and the end-to-end latent quality at each threshold, so the
quality/speed knee is chosen by measurement (not ideology).

For each ``rel_l1_thresh`` in the sweep it runs the SAME int8 DiT denoise with the
`FNI8TeaCache` wrapper installed, and reports vs the no-cache int8 baseline:
  * skipped / total steps (the FLOP reduction),
  * denoise wall-time and speedup,
  * FINAL-latent cosine vs the no-cache baseline (the quality gate; caching error
    accumulates over steps so we gate the end result, per issue requirements).

The latent-cosine gate is >= 0.985. The reported knee is the largest passing threshold
(max speedup at acceptable quality). Honest: failing thresholds are printed too.

Two e2e models are wired:
  * ``--model zimage``  Z-Image-Turbo, cfg=1 (cond-only), single-stream S3-DiT.
  * ``--model qwen_image_edit``  Qwen-Image-Edit, cfg>1 (cond+uncond => exercises the
    controller's per-stream isolation), double-stream MMDiT, driven via KSampler.

Run inside the e2e image on a single card (GPU 8 = deployment-representative CMP; NOT a
pristine real-V100 step-latency)::

    FNI8_GPU=8 docker compose run --rm --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
       custom_nodes/ComfyUI-superl8/bench/teacache_sweep.py --model zimage --steps 8 --size 64'
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import torch

WEIGHTS_DIR = os.environ.get("FNI8_WEIGHTS_DIR", "")
QWEN_TE_FILE = os.environ.get(
    "QWEN_TE_FILE",
    os.environ.get("TEXT_ENCODER_DIR", "")
    "qwen_2.5_vl_7b_fp8_scaled.safetensors",
)
QWEN_VAE_FILE = os.environ.get(
    "QWEN_VAE_FILE",
    os.path.join(os.environ.get("VAE_DIR", ""), "qwen_image_vae.safetensors"),
)

MODELS = {
    "zimage": {"dit": "Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8", "arch": "zimage",
               "steps": 8, "cfg": 1.0},
    "qwen_image_edit": {"dit": "Qwen__Qwen-Image-Edit-2509.dit.b4.fni8",
                        "arch": "qwen_image_edit", "steps": 8, "cfg": 2.5},
}

ZIMAGE_PROMPT = ("a photograph of a red fox sitting in a field of autumn leaves, "
                 "warm golden hour light, sharp focus, highly detailed")
QWEN_PROMPT = "Change the background to a snowy mountain landscape at sunset."
QWEN_LLAMA_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
    "size, texture, objects, background), then explain how the user's text instruction "
    "should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

DEFAULT_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
LATENT_COS_GATE = 0.985


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cosine(a, b):
    from bench.quality import cosine_similarity
    return cosine_similarity(a, b)


# --------------------------------------------------------------------------- Z-Image
def zimage_encode(prompt):
    import comfy.sd
    import comfy.utils

    te_dir = os.path.join(WEIGHTS_DIR, "zimage_te")
    shards = sorted(glob.glob(os.path.join(te_dir, "model-0000*-of-00003.safetensors")))
    merged: dict = {}
    for s in shards:
        merged.update(comfy.utils.load_torch_file(s, safe_load=True))
    clip = comfy.sd.load_text_encoder_state_dicts([merged], clip_type=None)
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
    except Exception:
        pass
    del clip, merged
    torch.cuda.empty_cache()
    return {"positive": cond, "negative": empty}


def zimage_denoise_fn(model, ctx, size, steps, cfg, seed):
    import comfy.sample

    in_ch = model.model.diffusion_model.in_channels
    latent = torch.zeros((1, in_ch, size, size), dtype=torch.float32)
    noise = comfy.sample.prepare_noise(latent, seed=seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    samples = comfy.sample.sample(
        model, noise, steps=steps, cfg=cfg, sampler_name="euler", scheduler="simple",
        positive=ctx["positive"], negative=ctx["negative"], latent_image=latent,
        denoise=1.0)
    torch.cuda.synchronize()
    return samples.detach(), time.perf_counter() - t0


# ------------------------------------------------------------------- Qwen-Image-Edit
def qwen_encode(prompt, ref_area, seed):
    """Qwen-Image-Edit conditioning (mirrors comfy_extras nodes_qwen + the sibling e2e
    bench). Uses a SYNTHETIC input image (deterministic gradient) so the sweep is
    self-contained — we measure the cache's speed/quality tradeoff, not edit aesthetics."""
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils
    import node_helpers

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(QWEN_VAE_FILE, safe_load=True))

    # deterministic synthetic RGB input image [1,H,W,C] in [0,1].
    H = W = 512
    yy = torch.linspace(0, 1, H).view(H, 1).expand(H, W)
    xx = torch.linspace(0, 1, W).view(1, W).expand(H, W)
    r = 0.5 * yy + 0.5 * xx
    g = yy
    b = 1.0 - xx  # a little chroma structure
    img = torch.stack([r, g, b], dim=-1)  # [H,W,3]
    image_bhwc = img.unsqueeze(0)         # [1,H,W,3]

    te_sd = comfy.utils.load_torch_file(QWEN_TE_FILE, safe_load=True)
    clip = comfy.sd.load_text_encoder_state_dicts([te_sd], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
    del te_sd

    samples = image_bhwc.movedim(-1, 1)
    total_vl = int(384 * 384)
    sb = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
    w_vl, h_vl = round(samples.shape[3] * sb), round(samples.shape[2] * sb)
    s_vl = comfy.utils.common_upscale(samples, w_vl, h_vl, "area", "disabled")
    images_vl = [s_vl.movedim(1, -1)]

    total_ref = int(ref_area * ref_area)
    sb = math.sqrt(total_ref / (samples.shape[3] * samples.shape[2]))
    w_ref = round(samples.shape[3] * sb / 8.0) * 8
    h_ref = round(samples.shape[2] * sb / 8.0) * 8
    s_ref = comfy.utils.common_upscale(samples, w_ref, h_ref, "area", "disabled")
    ref_latent = vae.encode(s_ref.movedim(1, -1)[:, :, :, :3])

    image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    tokens = clip.tokenize(image_prompt + prompt, images=images_vl,
                           llama_template=QWEN_LLAMA_TEMPLATE)
    cond = clip.encode_from_tokens_scheduled(tokens)
    cond = node_helpers.conditioning_set_values(
        cond, {"reference_latents": [ref_latent]}, append=True)
    ntokens = clip.tokenize("", llama_template=QWEN_LLAMA_TEMPLATE)
    ncond = clip.encode_from_tokens_scheduled(ntokens)
    try:
        mm.unload_all_models()
    except Exception:
        pass
    del clip, vae
    torch.cuda.empty_cache()
    return {"positive": cond, "negative": ncond, "ref_latent": ref_latent}


def qwen_denoise_fn(model, ctx, size, steps, cfg, seed):
    import comfy.sample
    import comfy.samplers

    ref = ctx["ref_latent"]
    b, c, h, w = ref.shape
    latent_image = torch.zeros((b, c, 1, h, w), dtype=torch.float32)
    noise = comfy.sample.prepare_noise(latent_image, seed=seed)
    sampler = comfy.samplers.KSampler(
        model, steps=steps, device=model.load_device, sampler="euler",
        scheduler="simple", denoise=1.0, model_options=model.model_options)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    samples = sampler.sample(noise, ctx["positive"], ctx["negative"], cfg=cfg,
                             latent_image=latent_image, disable_pbar=True, seed=seed)
    torch.cuda.synchronize()
    return samples.detach(), time.perf_counter() - t0


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zimage", choices=list(MODELS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--size", type=int, default=64, help="zimage latent H=W")
    ap.add_argument("--ref-area", type=int, default=512, help="qwen ref-latent px area^.5")
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--max-skips", type=int, default=3)
    ap.add_argument("--strategy", default="tea", choices=["tea", "taylor"],
                    help="tea = raw rel-L1 (interpretable thresholds); taylor = "
                         "sensitivity-scaled (the node default)")
    ap.add_argument("--mode", default="auto", choices=["auto", "residual", "output"],
                    help="residual = cache block delta, re-run epilogue (higher quality, "
                         "single-stream only); output = reuse whole prediction")
    ap.add_argument("--cond-cache", default=None,
                    help="path to save/reuse the encoded conditioning (skips the slow TE "
                         "reload on re-runs)")
    args = ap.parse_args()

    sys.path.insert(0, _repo_root())
    import folder_paths
    folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)

    from comfyui_superl8.gate import is_sm70
    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return

    spec = MODELS[args.model]
    steps = args.steps or spec["steps"]
    cfg = args.cfg if args.cfg is not None else spec["cfg"]
    dit_path = folder_paths.get_full_path("diffusion_models", spec["dit"])
    if dit_path is None:
        print(f"SKIP: {spec['dit']} not found under {WEIGHTS_DIR}")
        return

    from comfyui_superl8.nodes import FNI8TeaCache, UnetLoaderFNI8

    print(f"[1/3] encode conditioning ({args.model})")
    if args.cond_cache and os.path.exists(args.cond_cache):
        ctx = torch.load(args.cond_cache, weights_only=False)
        print(f"      reused cached conditioning ({args.cond_cache})")
    elif args.model == "zimage":
        ctx = zimage_encode(args.prompt or ZIMAGE_PROMPT)
    else:
        ctx = qwen_encode(args.prompt or QWEN_PROMPT, args.ref_area, args.seed)
    if args.cond_cache and not os.path.exists(args.cond_cache):
        torch.save(ctx, args.cond_cache)
        print(f"      cached conditioning -> {args.cond_cache}")

    _step_fn = zimage_denoise_fn if args.model == "zimage" else qwen_denoise_fn

    def denoise_fn(m):
        return _step_fn(m, ctx, args.size, steps, cfg, args.seed)

    print(f"[2/3] load int8 {args.model} DiT ({os.path.basename(dit_path)})")
    (base_model,) = UnetLoaderFNI8().load(spec["dit"], arch=spec["arch"])

    # Warm up CUDA (kernel autotune / cudnn / first-touch alloc) so the timed baseline is
    # not inflated by one-time cold-start cost — otherwise the first denoise looks ~4x
    # slower than the rest and every "speedup" is contaminated.
    print("      warmup denoise (untimed) ...")
    (mw,) = FNI8TeaCache().patch(base_model, rel_l1_thresh=0.0)
    _wlat, _wdt = denoise_fn(mw)
    del mw
    torch.cuda.empty_cache()

    thresholds = sorted(set([0.0] + list(args.thresholds)))
    print(f"[3/3] sweep {thresholds} @ {steps} steps, cfg={cfg}, strategy={args.strategy}")
    baseline_latent = None
    baseline_dt = None
    rows = []
    for thr in thresholds:
        (m,) = FNI8TeaCache().patch(base_model, rel_l1_thresh=thr,
                                    warmup_steps=args.warmup_steps,
                                    max_consecutive_skips=args.max_skips,
                                    strategy=args.strategy, mode=args.mode)
        lat, dt = denoise_fn(m)
        assert torch.isfinite(lat).all(), f"non-finite latent at thresh={thr}"
        ctrl = m.model_options.get("fni8_teacache_controller")
        stats = ctrl.stats() if ctrl is not None else {
            "skipped": 0, "total": steps}
        if thr == 0.0:
            baseline_latent, baseline_dt, cos = lat, dt, 1.0
        else:
            cos = cosine(lat, baseline_latent)
        speedup = (baseline_dt / dt) if dt > 0 else float("nan")
        rows.append((thr, stats["skipped"], stats["total"], dt, speedup, cos))
        print(f"    thresh={thr:.2f}  skipped={stats['skipped']}/{stats['total']}  "
              f"wall={dt:.2f}s  speedup={speedup:.2f}x  latent_cos={cos:.5f}")
        del m
        torch.cuda.empty_cache()

    print("=" * 80)
    print(f"TeaCache sweep — {args.model} int8 dp4a DiT ({steps} steps, cfg={cfg}, "
          f"seed {args.seed})")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  (deployment-representative CMP; not a "
          "pristine real-V100 latency)")
    print("-" * 80)
    print(f"  {'thresh':>7} {'skipped':>9} {'wall_s':>8} {'speedup':>8} "
          f"{'latent_cos':>11} {'gate':>6}")
    for thr, skipped, total, dt, speedup, cos in rows:
        tag = "base" if thr == 0.0 else ("PASS" if cos >= LATENT_COS_GATE else "fail")
        print(f"  {thr:7.2f} {skipped:>5}/{total:<3} {dt:8.2f} {speedup:8.2f}x "
              f"{cos:11.5f} {tag:>6}")
    print("-" * 80)
    passing = [r for r in rows if r[0] > 0.0 and r[5] >= LATENT_COS_GATE]
    if passing:
        knee = max(passing, key=lambda r: r[0])
        print(f"  KNEE: thresh={knee[0]:.2f} -> {knee[4]:.2f}x on top of int8 "
              f"(skipped {knee[1]}/{knee[2]} steps), latent cos {knee[5]:.5f} "
              f"(>= {LATENT_COS_GATE})")
    else:
        print(f"  NO threshold cleared the latent cosine gate ({LATENT_COS_GATE}); this "
              "model tolerates caching poorly at these settings — reported honestly.")
    print("=" * 80)


if __name__ == "__main__":
    main()
