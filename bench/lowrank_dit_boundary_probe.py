# SPDX-License-Identifier: Apache-2.0
"""DiT analogue of the LLM low-rank PP-boundary probe (fni8-serve #184).

The LLM finding (`fni8-serve/docs/lowrank-codec-findings.md`): the production
low-rank wire codec looks great per-boundary (cos 0.99 at ~10x) but COLLAPSES
end-to-end next-token fidelity, because an LM's next-token readout lives in the
low-variance directions a variance-ordered projection discards first. Hypothesis:
a DiT's image latent may NOT have that pathology, so low-rank could be a real
5-10x transport win exactly where the PCIe-1.0-x1 PP link is bottlenecked.

This runs the DiT analogue of that sweep on a REAL Z-Image-Turbo `.fni8` DiT:

  * base model = the int8 dp4a Z-Image DiT (the real deployed PP path);
  * boundary   = the output of `layers[k]` (k = n_layers//2), a 2-way PP split;
  * codec      = the PRODUCTION `fni8serve.dist.lowrank.LowRankCodec`, its SVD
                 basis fit on calibration boundary activations (a held-out calib
                 prompt), swept over rank x raw_fraction;
  * baselines  = int8 / int4 / int4-had via `superl8.transport.compress_activation`
                 at the same boundary;
  * metric     = cosine + PSNR of the FINAL denoised latent vs the no-codec
                 reference latent (the image analogue of LLM top-1 agreement),
                 plus decoded-image PSNR/SSIM for a few anchor configs.

Reference and codec runs use the SAME int8 DiT, same conditioning, same seed —
so the int8-DiT quantization noise is common-mode and the measured delta is the
codec's marginal damage alone.

Run inside the comfyui-fni8 e2e image, with fni8-serve mounted so the real
production codec module is importable::

    FNI8_GPU=0 docker compose run --rm \
      -v <fni8-serve-dir>:/opt/fni8-serve:ro \
      -e FNI8_SERVE_ROOT=/opt/fni8-serve \
      --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
       custom_nodes/ComfyUI-superl8/bench/lowrank_dit_boundary_probe.py --steps 8 --size 64'
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import torch

WEIGHTS_DIR = os.environ.get("FNI8_WEIGHTS_DIR", "")
ZIMAGE_DIT = "Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8"
TE_DIR = os.path.join(WEIGHTS_DIR, "zimage_te")
VAE_FILE = os.path.join(WEIGHTS_DIR, "zimage_vae", "diffusion_pytorch_model.safetensors")

# Two DIFFERENT prompts: calib (basis fit) vs test (measured). No train/test leakage.
CALIB_PROMPT = (
    "a bustling medieval marketplace at dawn, cobblestone streets, wooden stalls "
    "with fruit and cloth, soft morning fog, cinematic wide shot, highly detailed"
)
TEST_PROMPT = (
    "a photograph of a red fox sitting in a field of autumn leaves, "
    "warm golden hour light, sharp focus, highly detailed"
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_production_codec():
    """Import the EXACT production low-rank codec file (fni8serve/dist/lowrank.py)
    by path, bypassing the heavy fni8serve package __init__ (which pulls the whole
    serving stack). The file itself only needs torch."""
    root = os.environ.get("FNI8_SERVE_ROOT", "/opt/fni8-serve")
    path = os.path.join(root, "fni8serve", "dist", "lowrank.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"production codec not found at {path}; mount fni8-serve and set "
            f"FNI8_SERVE_ROOT (see this file's module docstring)."
        )
    spec = importlib.util.spec_from_file_location("_fni8serve_lowrank", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection (which looks the module up in
    # sys.modules via cls.__module__) works under `from __future__ import annotations`.
    sys.modules["_fni8serve_lowrank"] = mod
    spec.loader.exec_module(mod)
    return mod, path


def encode_prompts(prompts: list[str]):
    """Load the Qwen3-4B TE, encode each prompt to a ComfyUI conditioning, free the
    TE before returning (so the DiT gets the VRAM)."""
    import comfy.sd
    import comfy.utils

    shards = sorted(glob.glob(os.path.join(TE_DIR, "model-0000*-of-00003.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no Qwen3-4B TE shards under {TE_DIR}")
    merged: dict = {}
    for s in shards:
        merged.update(comfy.utils.load_torch_file(s, safe_load=True))
    clip = comfy.sd.load_text_encoder_state_dicts([merged], clip_type=None)
    conds = [clip.encode_from_tokens_scheduled(clip.tokenize(p)) for p in prompts]
    empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
    except Exception:
        pass
    del clip, merged
    torch.cuda.empty_cache()
    return conds, empty


def load_int8_model():
    from comfyui_superl8.nodes import UnetLoaderFNI8

    (model,) = UnetLoaderFNI8().load(ZIMAGE_DIT, arch="zimage")
    return model


def find_block_list(diffusion_model):
    """Return the main transformer-block ModuleList ('layers' for NextDiT)."""
    import torch.nn as nn

    best = None
    for name, mod in diffusion_model.named_children():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not locate the transformer-block ModuleList")
    return best


def denoise(model, cond, empty, in_channels, size, steps, seed):
    import comfy.sample

    latent = torch.zeros((1, in_channels, size, size), dtype=torch.float32)
    noise = comfy.sample.prepare_noise(latent, seed=seed)
    samples = comfy.sample.sample(
        model, noise, steps=steps, cfg=1.0, sampler_name="euler", scheduler="simple",
        positive=cond, negative=empty, latent_image=latent, denoise=1.0,
    )
    return samples.detach()


def _out_tensor(o):
    return o[0] if isinstance(o, tuple) else o


def _rewrap(o, new):
    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=64, help="latent H=W (px = 8x)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="/tmp/lowrank_dit")
    ap.add_argument("--decode-anchors", action="store_true",
                    help="also VAE-decode a few anchor latents for image PSNR/SSIM")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()

    sys.path.insert(0, _repo_root())
    os.makedirs(args.outdir, exist_ok=True)

    import folder_paths
    folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)
    from comfyui_superl8.gate import is_sm70
    from bench.quality import cosine_similarity, psnr

    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return

    lowrank, codec_path = load_production_codec()
    print(f"[codec] production LowRankCodec from {codec_path}")

    from superl8.transport import compress_activation, decompress_activation

    dev = "cuda"

    # --- conditioning (calib + test), TE freed after ------------------------
    print("[1/6] Qwen3-4B TE: encoding calib + test prompts")
    (calib_cond, test_cond), empty = encode_prompts([CALIB_PROMPT, TEST_PROMPT])

    # --- int8 DiT (the real deployed path) ----------------------------------
    print("[2/6] loading int8 dp4a Z-Image DiT")
    model = load_int8_model()
    dm = model.model.diffusion_model
    in_ch = dm.in_channels
    blk_name, blocks = find_block_list(dm)
    k = len(blocks) // 2
    print(f"      block list '{blk_name}' len={len(blocks)}, boundary after layer {k}")

    # --- calibration: capture boundary activations on the CALIB prompt ------
    print(f"[3/6] calibration denoise ({args.steps} steps), capturing layers[{k}] output")
    cap: list[torch.Tensor] = []

    def _grab(_m, _i, o):
        t = _out_tensor(o)
        cap.append(t.detach().float().reshape(-1, t.shape[-1]).cpu())

    h = blocks[k].register_forward_hook(_grab)
    _ = denoise(model, calib_cond, empty, in_ch, args.size, args.steps, args.seed)
    h.remove()
    calib_acts = torch.cat(cap, 0)
    d = calib_acts.shape[-1]
    print(f"      calibration activations: {tuple(calib_acts.shape)}  (d={d})")

    # Precompute SVD once (bit-identical to fit_basis, sliced per rank) ------
    x = calib_acts.float()
    mean = x.mean(0)
    with torch.no_grad():
        _, _, Vt = torch.linalg.svd(x - mean, full_matrices=False)
    channel_mag = x.abs().mean(0)

    def make_basis(r, raw_fraction):
        r = min(r, Vt.shape[0], d)
        U = Vt[:r, :].T.contiguous()
        num_raw = max(1, int(d * raw_fraction))
        _, raw_idx = torch.topk(channel_mag, k=num_raw)
        return lowrank.LowRankBasis.from_state(U, mean.clone(), raw_idx.tolist())

    def lowrank_ratio(r, num_raw):
        # fp16 baseline (2 B/elem) vs wire = int8 latent (1 B) + fp16 raw (2 B) + scale.
        wire = r * 1 + num_raw * 2 + 4
        return (d * 2) / wire

    # --- reference denoise on the TEST prompt (no codec) --------------------
    print("[4/6] reference denoise (no codec) on test prompt")
    lat_ref = denoise(model, test_cond, empty, in_ch, args.size, args.steps, args.seed)
    assert torch.isfinite(lat_ref).all()

    results = []
    saved_latents = {"reference": lat_ref}

    def run_with_hook(hook_fn, label, ratio, per_boundary=None):
        handle = blocks[k].register_forward_hook(hook_fn)
        try:
            lat = denoise(model, test_cond, empty, in_ch, args.size, args.steps, args.seed)
        finally:
            handle.remove()
        finite = bool(torch.isfinite(lat).all())
        cos = cosine_similarity(lat, lat_ref) if finite else float("nan")
        ps = psnr(lat.clamp(-10, 10), lat_ref.clamp(-10, 10)) if finite else float("nan")
        row = {"config": label, "ratio": round(ratio, 2), "latent_cos": round(cos, 6),
               "latent_psnr_db": round(ps, 2), "finite": finite}
        if per_boundary is not None:
            row["boundary_sqnr_db"] = round(per_boundary[0], 2)
            row["boundary_cos"] = round(per_boundary[1], 6)
        results.append(row)
        print(f"      {label:28s} ratio={ratio:5.2f}x  latent_cos={cos:.6f}  "
              f"psnr={ps:5.2f}dB  finite={finite}")
        return lat

    def boundary_quality(codec_encode_decode, sample):
        xr = codec_encode_decode(sample)
        xf = sample.float()
        xrf = xr.float()
        noise = (xf - xrf).pow(2).sum().item()
        sqnr = 10.0 * torch.log10(xf.pow(2).sum() / noise).item() if noise > 0 else float("inf")
        cos = torch.nn.functional.cosine_similarity(
            xf.flatten(), xrf.flatten(), dim=0, eps=1e-12).item()
        return sqnr, cos

    # --- low-rank sweep: rank x raw_fraction --------------------------------
    print("[5/6] low-rank sweep (production codec)")
    rank_fracs = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
    raw_fracs = [0.005, 0.05, 0.20]
    # a probe sample to report per-boundary fidelity (from the reference run)
    probe_cap: list[torch.Tensor] = []

    def _grab_probe(_m, _i, o):
        probe_cap.append(_out_tensor(o).detach())

    hp = blocks[k].register_forward_hook(_grab_probe)
    _ = denoise(model, test_cond, empty, in_ch, args.size, args.steps, args.seed)
    hp.remove()
    probe = probe_cap[-1]  # last-step boundary activation on the test prompt

    for rf in raw_fracs:
        for rfrac in rank_fracs:
            r = int(d * rfrac)
            basis = make_basis(r, rf)
            num_raw = len(basis.raw_indices)
            codec = lowrank.LowRankCodec(basis)
            ratio = lowrank_ratio(basis.r, num_raw)
            label = f"lowrank r={basis.r} raw={rf*100:.1f}%"

            def _enc_dec(x, codec=codec):
                return codec.decode(codec.encode(x))

            pb = boundary_quality(_enc_dec, probe)

            def _hook(_m, _i, o, codec=codec):
                t = _out_tensor(o)
                y = codec.decode(codec.encode(t)).to(t.dtype)
                return _rewrap(o, y)

            lat = run_with_hook(_hook, label, ratio, pb)
            saved_latents[label] = lat

    # --- int8 / int4 / int4-had baselines at the same boundary --------------
    print("[6/6] quant-codec baselines at the same boundary")
    for scheme, gs in [("int8", None), ("int4", 128), ("int4-had", 128), ("nf4", 128)]:
        def _enc_dec(x, scheme=scheme, gs=gs):
            c = compress_activation(x.to(torch.float16), scheme=scheme, group_size=gs)
            return decompress_activation(c)

        c0 = compress_activation(probe.to(torch.float16), scheme=scheme, group_size=gs)
        ratio = (probe.numel() * 2) / c0.on_wire_bytes
        pb = boundary_quality(_enc_dec, probe)

        def _hook(_m, _i, o, scheme=scheme, gs=gs):
            t = _out_tensor(o)
            c = compress_activation(t.to(torch.float16), scheme=scheme, group_size=gs)
            y = decompress_activation(c).to(t.dtype)
            return _rewrap(o, y)

        lat = run_with_hook(_hook, f"{scheme}", ratio, pb)
        saved_latents[scheme] = lat

    # --- summary ------------------------------------------------------------
    results.sort(key=lambda r: (-r["latent_cos"]))
    out = {
        "d": d, "n_layers": len(blocks), "boundary_layer": k,
        "calib_prompt": CALIB_PROMPT, "test_prompt": TEST_PROMPT,
        "steps": args.steps, "size": args.size, "results": results,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print(f"Z-Image-Turbo low-rank DiT-boundary sweep  (d={d}, {len(blocks)} layers, "
          f"boundary after layer {k})")
    print(f"{'config':30s} {'ratio':>7s} {'lat_cos':>9s} {'lat_psnr':>9s} "
          f"{'bnd_sqnr':>9s} {'bnd_cos':>9s}")
    for r in sorted(results, key=lambda r: -r["latent_cos"]):
        print(f"{r['config']:30s} {r['ratio']:6.2f}x {r['latent_cos']:9.6f} "
              f"{r['latent_psnr_db']:8.2f}dB {r.get('boundary_sqnr_db', float('nan')):8.2f}dB "
              f"{r.get('boundary_cos', float('nan')):9.6f}")
    print("=" * 78)

    # best low-rank compression at cos >= 0.99 and >= 0.999
    for floor in (0.99, 0.999):
        lr = [r for r in results if r["config"].startswith("lowrank")
              and r["latent_cos"] >= floor]
        best = max(lr, key=lambda r: r["ratio"]) if lr else None
        if best:
            print(f"best low-rank @ latent_cos>={floor}: {best['config']} "
                  f"-> {best['ratio']}x")
        else:
            print(f"best low-rank @ latent_cos>={floor}: NONE reach it")

    # --- optional: decode anchor latents to images --------------------------
    if args.decode_anchors:
        print("[decode] VAE-decoding anchor latents")
        import comfy.sd
        import comfy.utils
        from comfyui_superl8.vae_tiled import tiled_vae_decode
        from bench.quality import ssim

        del model
        torch.cuda.empty_cache()
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE_FILE, safe_load=True))

        def decode(lat):
            def decode_fn(x):
                with torch.no_grad():
                    img = vae.decode(x.to("cuda"))
                if img.dim() == 4 and img.shape[-1] in (1, 3):
                    img = img.movedim(-1, 1)
                return img.detach().to(x.device)
            with torch.no_grad():
                return tiled_vae_decode(decode_fn, lat.to("cuda"),
                                        tile_size=args.tile_size, overlap=args.overlap)

        img_ref = decode(lat_ref)
        anchors = ["int8", "int4-had"]
        lr099 = [r for r in results if r["config"].startswith("lowrank")
                 and r["latent_cos"] >= 0.99]
        if lr099:
            anchors.append(max(lr099, key=lambda r: r["ratio"])["config"])
        anchors.append(min((r for r in results if r["config"].startswith("lowrank")),
                           key=lambda r: r["ratio"])["config"])  # lowest-cos lowrank
        img_rows = []
        for a in anchors:
            if a not in saved_latents:
                continue
            img = decode(saved_latents[a])
            ip = psnr(img, img_ref)
            iss = ssim(img, img_ref)
            img_rows.append({"config": a, "image_psnr_db": round(ip, 2),
                             "image_ssim": round(iss, 4)})
            print(f"      {a:28s} image_psnr={ip:5.2f}dB ssim={iss:.4f}")
        out["image_anchors"] = img_rows
        with open(os.path.join(args.outdir, "results.json"), "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nresults.json written to {args.outdir}")


if __name__ == "__main__":
    main()
