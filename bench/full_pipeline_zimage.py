# SPDX-License-Identifier: Apache-2.0
"""Full end-to-end Z-Image-Turbo generation on the int8 dp4a path, and its fp
reference, with an int8-vs-fp *decoded-image* quality comparison.

This closes the gap called out in ``docs/int8-dit-validation.md``: the earlier
Z-Image validation could only compare a *latent* (cosine 1.000000) because a full
multi-step + VAE-decode pass "wouldn't fit on the 16 GB card". With the tiled VAE
decode (#97) that pass now fits, so this runs the real pipeline:

    Qwen3-4B text encoder  ->  int8 dp4a Z-Image DiT (8-step Turbo denoise)
                           ->  tiled VAE decode (#97)  ->  a decoded RGB image

then repeats the DiT+VAE with the same weights *dequantized to fp* (identical
architecture, identical conditioning, identical VAE — only the DiT matmul/attention
path changes) and reports PSNR / SSIM / cosine (and LPIPS when the lib is present) of
the int8 image vs the fp image, plus peak HBM so the 16 GB fit is measured, not
assumed.

Run inside the e2e image (needs the mounted weights + a Volta/CMP sm_70 GPU)::

    FNI8_GPU=5 docker compose run --rm --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
       custom_nodes/ComfyUI-superl8/bench/full_pipeline_zimage.py --steps 8 --size 64'
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Canonical production profile. The archive mount remains available to older
# e2e fixtures, but production Z-Image resolves only from the NVMe model root.
# ---------------------------------------------------------------------------
from comfyui_superl8 import zimage_profile

WEIGHTS_DIR = zimage_profile.CANONICAL_MODEL_ROOT
ZIMAGE_DIT = os.path.basename(zimage_profile.DIT_RELATIVE_PATH)
TE_DIR = os.path.dirname(
    os.path.join(WEIGHTS_DIR, zimage_profile.TEXT_ENCODER_RELATIVE_PATH)
)
VAE_FILE = os.path.join(WEIGHTS_DIR, zimage_profile.VAE_RELATIVE_PATH)

DEFAULT_PROMPT = (
    "a photograph of a red fox sitting in a field of autumn leaves, "
    "warm golden hour light, sharp focus, highly detailed"
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def encode_prompt(prompt: str, te_path: str | None = None):
    """Load the real Qwen3-4B text encoder (Z-Image's TE) and encode ``prompt`` into
    a ComfyUI conditioning. Frees the TE before returning so the DiT has the VRAM."""
    import comfy.sd
    import comfy.utils

    te_path = te_path or os.path.join(
        WEIGHTS_DIR, zimage_profile.TEXT_ENCODER_RELATIVE_PATH
    )
    if not os.path.isfile(te_path):
        raise FileNotFoundError(f"Qwen3-4B TE not found: {te_path}")
    merged = comfy.utils.load_torch_file(te_path, safe_load=True)
    # clip_type=None -> ComfyUI detects TEModel.QWEN3_4B and builds the Z-Image TE
    # (Qwen3-4B, llama_template "<|im_start|>user\n{}<|im_end|>...", layer_idx=-2).
    clip = comfy.sd.load_text_encoder_state_dicts([merged], clip_type=None)
    tokens = clip.tokenize(prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)
    empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    # Drop the TE from VRAM before the DiT loads.
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
    except Exception:
        pass
    del clip, merged
    torch.cuda.empty_cache()
    return cond, empty


def _dequantize_state(sd: dict) -> dict:
    from comfyui_superl8.superl8_tensor import FNI8Tensor

    out = {}
    for name, w in sd.items():
        if isinstance(w, FNI8Tensor):
            out[name] = (w.int8_data().float() * w.q_scale.unsqueeze(1)).to(torch.bfloat16)
        else:
            out[name] = w
    return out


def _int8_attn_engaged(diffusion_model) -> dict:
    """Count int8 (FNI8Tensor) attention projection params — proof that the qkv re-fusion
    actually put attention on the dp4a path, not the fp fallback."""
    from comfyui_superl8.superl8_tensor import FNI8Tensor

    qkv = out = 0
    for n, p in diffusion_model.named_parameters():
        if n.endswith("attention.qkv.weight"):
            qkv += int(isinstance(p, FNI8Tensor))
        elif n.endswith("attention.out.weight"):
            out += int(isinstance(p, FNI8Tensor))
    return {"qkv_int8": qkv, "out_int8": out}


def int8_weight_inventory(diffusion_model) -> dict[str, int]:
    """Structural proof for the exact checksummed production artifact."""
    return zimage_profile.int8_weight_inventory(diffusion_model)


def assert_int8_inventory(diffusion_model, *, require_cuda: bool) -> dict[str, int]:
    return zimage_profile.assert_int8_inventory(
        diffusion_model, require_cuda=require_cuda
    )


def runtime_int8_inventory(diffusion_model) -> dict[str, int]:
    """Report actual SQNR-gated linear routing after a real denoise."""
    return zimage_profile.runtime_int8_inventory(diffusion_model)


def assert_model_sampling_shift(model, expected: float = zimage_profile.FLOW_SHIFT) -> float:
    return zimage_profile.assert_model_sampling_shift(model, expected)


def load_int8_model(model_root: str | None = None):
    """The real int8 pipeline: int8 dp4a linears + bidirectional int8 FlashAttention
    (SQNR-gated), exactly as a ComfyUI graph loads it."""
    return zimage_profile.load_int8_model(model_root)


def load_fp_model(path: str):
    """The fp reference: same architecture + conditioning + VAE, DiT weights
    dequantized to bf16 (Z-Image is bf16-native — fp16 overflows to black), standard
    torch matmul + fp SDPA. Only the DiT compute path differs from the int8 run."""
    import comfy.sd

    from comfyui_superl8.loader import load_fni8_dit

    sd = load_fni8_dit(path, device="cpu", strip_prefix="")
    sd_fp = _dequantize_state(sd)
    model = comfy.sd.load_diffusion_model_state_dict(
        sd_fp, model_options={"dtype": torch.bfloat16}
    )
    return model


def denoise(
    model,
    cond,
    empty,
    in_channels: int,
    size: int | None = None,
    steps: int = 8,
    seed: int = 0,
    latent_h: int | None = None,
    latent_w: int | None = None,
    abort_check=None,
):
    """Official-contract denoise: CFG1, res_multistep/simple, shift3, denoise=1,
    explicit seed, empty negative -- exact eight-NFE Turbo semantics (#181).

    ``size`` gives a square latent (``H=W=size``, the original harness shape).
    ``latent_h``/``latent_w`` override it independently for rectangular shapes
    (issue #181's 896x1152 acceptance case) -- this is the SAME harness, not a
    separate reimplementation, so int8-vs-fp quality/runtime evidence gathered
    here applies to both square and rectangular requests.

    ``abort_check``, if given, is called after every sampling step (via
    ``comfy.sample.sample``'s per-step ``callback``); raising from it propagates
    out of the sampler immediately, bounding a mid-denoise cancellation to at
    most one step's duration instead of only being interruptible between whole
    pipeline stages.
    """
    if latent_h is None:
        latent_h = size
    if latent_w is None:
        latent_w = size
    if latent_h is None or latent_w is None:
        raise ValueError("must pass either `size` or both `latent_h` and `latent_w`")

    return zimage_profile.denoise(
        model,
        cond,
        empty,
        in_channels,
        latent_h=latent_h,
        latent_w=latent_w,
        steps=steps,
        seed=seed,
        abort_check=abort_check,
    )


def vae_decode_tiled(vae, latent, tile_size: int, overlap: int, abort_check=None):
    """Decode via the pack's tiled VAE primitive (#97). comfy VAE.decode returns
    [B,H,W,C]; the tiler works in [B,C,H,W], so transpose on the way out/in."""
    from comfyui_superl8.vae_tiled import tiled_vae_decode

    def decode_fn(lat):
        with torch.no_grad():
            img = vae.decode(lat.to("cuda"))  # comfy returns [B,H,W,C] on CPU device
        if img.dim() == 4 and img.shape[-1] in (1, 3):
            img = img.movedim(-1, 1)          # -> [B,C,H,W]
        return img.detach().to(lat.device)    # keep tiler accumulation on one device

    with torch.no_grad():
        img_bchw = tiled_vae_decode(decode_fn, latent.to("cuda"), tile_size=tile_size,
                                    overlap=overlap, verbose=True, abort_check=abort_check)
    return img_bchw


def save_png(img_bchw: torch.Tensor, path: str):
    from PIL import Image

    x = img_bchw[0].detach().clamp(0, 1).float().cpu().movedim(0, -1).numpy()
    Image.fromarray((x * 255).round().astype("uint8")).save(path)


def try_lpips(a_bchw, b_bchw):
    try:
        import lpips
    except Exception:
        return None
    net = lpips.LPIPS(net="alex").to(a_bchw.device)
    with torch.no_grad():
        d = net(a_bchw.float() * 2 - 1, b_bchw.float() * 2 - 1)
    return float(d.flatten()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8, help="Turbo is an 8-step model")
    ap.add_argument("--size", type=int, default=64, help="latent H=W (px = 8x); ignored if --width/--height given")
    ap.add_argument("--width", type=int, default=None, help="pixel width (rectangular; #181 896x1152 case)")
    ap.add_argument("--height", type=int, default=None, help="pixel height (rectangular; #181 896x1152 case)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--outdir", default="/tmp/zimage_fullpipe")
    args = ap.parse_args()
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be given together")
    if args.width is not None:
        latent_h, latent_w = zimage_profile.validate_geometry(args.width, args.height)
    else:
        latent_h = latent_w = None
    shape_px = f"{args.width}x{args.height}" if args.width is not None else f"{args.size * 8}x{args.size * 8}"

    sys.path.insert(0, _repo_root())
    os.makedirs(args.outdir, exist_ok=True)

    import folder_paths
    files = zimage_profile.resolve_files()
    folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)

    import comfy.sd
    import comfy.utils
    from comfyui_superl8.gate import is_sm70
    from comfyui_superl8.memory import format_hbm_report, peak_hbm_monitor
    from bench.quality import cosine_similarity, psnr, ssim

    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return

    path = files.dit

    torch.cuda.reset_peak_memory_stats()

    print(f"[1/5] Qwen3-4B text encoder: encoding prompt: {args.prompt!r}")
    cond, empty = encode_prompt(args.prompt, files.text_encoder)
    print(f"      conditioning: {cond[0][0].shape}")

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(files.vae, safe_load=True))

    # ---- int8 pipeline --------------------------------------------------------
    print(f"[2/5] int8 dp4a Z-Image DiT: {args.steps}-step denoise @ {shape_px}px")
    with peak_hbm_monitor() as info:
        m_int8 = load_int8_model()
        in_ch = m_int8.model.diffusion_model.in_channels
        attn_int8 = _int8_attn_engaged(m_int8.model.diffusion_model)
        print(f"      int8 attention projections engaged: qkv={attn_int8['qkv_int8']} "
              f"out={attn_int8['out_int8']} blocks (FNI8Tensor/dp4a)")
        lat_int8 = denoise(m_int8, cond, empty, in_ch, args.size, args.steps, args.seed,
                            latent_h=latent_h, latent_w=latent_w)
    print(format_hbm_report(info, "int8 DiT denoise"))
    assert torch.isfinite(lat_int8).all(), "int8 latent non-finite"
    del m_int8
    torch.cuda.empty_cache()

    print("[3/5] tiled VAE decode (#97) of the int8 latent")
    with peak_hbm_monitor() as info_v:
        img_int8 = vae_decode_tiled(vae, lat_int8, args.tile_size, args.overlap)
    print(format_hbm_report(info_v, "tiled VAE decode"))
    save_png(img_int8, os.path.join(args.outdir, "zimage_int8.png"))

    # ---- fp reference ---------------------------------------------------------
    print(f"[4/5] fp reference DiT: {args.steps}-step denoise (bf16 weights)")
    m_fp = load_fp_model(path)
    lat_fp = denoise(m_fp, cond, empty, in_ch, args.size, args.steps, args.seed,
                      latent_h=latent_h, latent_w=latent_w)
    assert torch.isfinite(lat_fp).all(), "fp latent non-finite"
    del m_fp
    torch.cuda.empty_cache()
    img_fp = vae_decode_tiled(vae, lat_fp, args.tile_size, args.overlap)
    save_png(img_fp, os.path.join(args.outdir, "zimage_fp.png"))

    # ---- metrics --------------------------------------------------------------
    print("[5/5] int8-vs-fp quality")
    cos_lat = cosine_similarity(lat_int8, lat_fp)
    ps = psnr(img_int8, img_fp)
    ss = ssim(img_int8, img_fp)
    lp = try_lpips(img_int8, img_fp)

    def structured(img):
        return img.std().item() > 1e-3 and torch.isfinite(img).all().item()

    peak_all = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("=" * 64)
    print("Z-Image-Turbo FULL int8 pipeline  (Qwen3-4B -> int8 DiT -> tiled VAE)")
    print(f"  int8 attention projections : qkv={attn_int8['qkv_int8']} "
          f"out={attn_int8['out_int8']} blocks on dp4a (0 = fp fallback)")
    print(f"  image size                 : {img_int8.shape[-2]}x{img_int8.shape[-1]}")
    print(f"  int8 image finite+structured: {structured(img_int8)}")
    print(f"  fp   image finite+structured: {structured(img_fp)}")
    print(f"  latent cosine (int8 vs fp) : {cos_lat:.6f}")
    print(f"  PSNR  (decoded image)      : {ps:.2f} dB")
    print(f"  SSIM  (decoded image)      : {ss:.6f}")
    print(f"  LPIPS (decoded image)      : {lp if lp is not None else 'n/a (lpips not installed)'}")
    print(f"  peak HBM (whole pipeline)  : {peak_all:.2f} GiB  (card = 16 GiB)")
    print(f"  images saved to            : {args.outdir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
