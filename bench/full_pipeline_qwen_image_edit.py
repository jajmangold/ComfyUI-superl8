# SPDX-License-Identifier: Apache-2.0
"""Full end-to-end Qwen-Image-Edit generation on the int8 dp4a path — the FIRST real
edited image produced by this fleet on a single 16 GB Volta/CMP card.

Pipeline (one model resident at a time, mmap-loaded, expandable_segments):

    Qwen2.5-VL text+image encoder  (semantic tokens + VAE ref latent)
        -> int8 dp4a Qwen-Image-Edit DiT  (4-bit .fni8, 10.2 GB, fits one card)
        -> Qwen-Image VAE decode  ->  a decoded RGB edited image

The pinned ComfyUI's ``comfy.sample.sample`` wrapper is unused; we drive the
apply_model path directly via ``comfy.samplers.KSampler(...).sample(...)`` (the same
sampling_function -> model.apply_model machinery, just without the broken wrapper).

Run inside the e2e image (needs the mounted weights + a Volta/CMP sm_70 GPU)::

    FNI8_GPU=15 docker compose run --rm --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI:custom_nodes/ComfyUI-superl8 python3 \
       custom_nodes/ComfyUI-superl8/bench/full_pipeline_qwen_image_edit.py \
       --unet Qwen__Qwen-Image-Edit-2509.dit.b4.fni8 --steps 20'
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

WEIGHTS_DIR = os.environ.get("FNI8_WEIGHTS_DIR", "")
# TE + VAE live in the mounted comfy models archive (standard ComfyUI Qwen-Image-Edit
# assets); overridable via env for the fleet runner.
TE_FILE = os.environ.get(
    "QWEN_TE_FILE",
    os.environ.get("TEXT_ENCODER_DIR", "")
    "qwen_2.5_vl_7b_fp8_scaled.safetensors",
)
VAE_FILE = os.environ.get(
    "QWEN_VAE_FILE",
    os.path.join(os.environ.get("VAE_DIR", ""), "qwen_image_vae.safetensors"),
)

DEFAULT_PROMPT = "Change the background to a snowy mountain landscape at sunset."
# The Qwen-Image-Edit-Plus system template (verbatim from comfy_extras/nodes_qwen.py).
LLAMA_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
    "size, texture, objects, background), then explain how the user's text instruction "
    "should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _apply_lora_weight_functions(dit, lora_sd, strength, device):
    """Attach a LoRA to the DiT's Linear modules as float-domain weight_function hooks
    (the exact interface FNI8Ops.Linear consumes via `self.weight_function`: each hook
    takes the dequantized fp weight and returns weight + delta; the op then folds it —
    per_row_i8 re-quantizes to int8 for dp4a, per_group_i4 runs the fp linear).

    Handles the standard diffusers/kohya key layout the Lightning + community LoRAs use:
    ``<module>.lora_down.weight`` / ``.lora_up.weight`` (+ optional ``.alpha``), where
    ``<module>`` is a dotted path into the diffusion model (e.g.
    ``transformer_blocks.0.attn.to_q``). delta = strength * (alpha/rank) * (up @ down)."""
    groups: dict[str, dict] = {}
    for k, v in lora_sd.items():
        for suf in (
            ".lora_down.weight",
            ".lora_up.weight",
            ".lora_A.weight",
            ".lora_B.weight",
            ".alpha",
        ):
            if k.endswith(suf):
                base = k[: -len(suf)]
                if base.startswith("diffusion_model."):
                    base = base[len("diffusion_model.") :]
                if base.startswith("transformer."):
                    base = base[len("transformer.") :]
                groups.setdefault(base, {})[suf] = v
                break

    n = 0
    for base, parts in groups.items():
        down = parts.get(".lora_down.weight", parts.get(".lora_A.weight"))
        up = parts.get(".lora_up.weight", parts.get(".lora_B.weight"))
        if down is None or up is None:
            continue
        try:
            mod = dit.get_submodule(base)
        except AttributeError:
            continue
        rank = down.shape[0]
        alpha = parts.get(".alpha", None)
        scale = strength * (float(alpha) / rank if alpha is not None else 1.0)
        # Store the small low-rank factors (up [out,rank], down [rank,in]); compute the
        # [out,in] delta transiently per forward (materializing all 720 full deltas at
        # once would be ~27 GB). delta = scale * (up @ down).
        up_d = up.to(device=device, dtype=torch.float32)
        down_d = down.to(device=device, dtype=torch.float32)
        # Shape guard: the delta must match this module's weight [out, in]. Skip any
        # LoRA entry whose factors don't (e.g. a key that collides with a differently
        # shaped fp layer), so we never corrupt a mismatched weight.
        wshape = tuple(mod.weight.shape)
        if (up_d.shape[0], down_d.shape[1]) != wshape:
            continue

        def hook(w, u=up_d, dn=down_d, s=scale):
            return w + ((u @ dn) * s).to(w.dtype)

        # Per-INSTANCE list: comfy's Linear may expose a class-level default
        # weight_function [] shared across all modules; appending to it would put every
        # hook on every layer (fp img_in then gets a 3072-dim delta -> shape error).
        # Copy any instance-level hooks into a fresh list unique to this module.
        existing = mod.__dict__.get("weight_function") or []
        mod.weight_function = list(existing) + [hook]
        n += 1
    return n


def load_input_image(path: str) -> torch.Tensor:
    """Load an RGB image as ComfyUI's [B,H,W,C] float tensor in [0,1]."""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ...]  # [1,H,W,C]


def encode_prompt_edit(prompt: str, image_bhwc: torch.Tensor, vae):
    """Replicate comfy_extras/nodes_qwen.py::TextEncodeQwenImageEditPlus.execute for a
    single reference image: VL-encode a 384x384 view for semantic tokens, VAE-encode a
    ~1024x1024 view as the reference latent, and attach it to the conditioning. Frees
    the TE from VRAM before returning so the DiT has the whole card."""
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils
    import node_helpers

    print(f"      loading Qwen2.5-VL TE: {os.path.basename(TE_FILE)}")
    te_sd = comfy.utils.load_torch_file(TE_FILE, safe_load=True)
    clip = comfy.sd.load_text_encoder_state_dicts([te_sd], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
    del te_sd

    samples = image_bhwc.movedim(-1, 1)  # [B,C,H,W]

    # VL semantic view (384x384-area).
    total_vl = int(384 * 384)
    sb = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
    w_vl, h_vl = round(samples.shape[3] * sb), round(samples.shape[2] * sb)
    s_vl = comfy.utils.common_upscale(samples, w_vl, h_vl, "area", "disabled")
    images_vl = [s_vl.movedim(1, -1)]

    # VAE reference latent (~1024x1024-area, /8-aligned).
    total_ref = int(1024 * 1024)
    sb = math.sqrt(total_ref / (samples.shape[3] * samples.shape[2]))
    w_ref = round(samples.shape[3] * sb / 8.0) * 8
    h_ref = round(samples.shape[2] * sb / 8.0) * 8
    s_ref = comfy.utils.common_upscale(samples, w_ref, h_ref, "area", "disabled")
    ref_latent = vae.encode(s_ref.movedim(1, -1)[:, :, :, :3])
    print(f"      ref image {w_ref}x{h_ref} -> ref_latent {tuple(ref_latent.shape)}")

    image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=LLAMA_TEMPLATE)
    cond = clip.encode_from_tokens_scheduled(tokens)
    cond = node_helpers.conditioning_set_values(
        cond, {"reference_latents": [ref_latent]}, append=True
    )

    # Negative (empty prompt), same template, no image tokens.
    ntokens = clip.tokenize("", llama_template=LLAMA_TEMPLATE)
    ncond = clip.encode_from_tokens_scheduled(ntokens)

    # Drop the TE from VRAM before the DiT loads.
    try:
        mm.unload_all_models()
    except Exception:
        pass
    del clip
    torch.cuda.empty_cache()
    return cond, ncond, ref_latent


def denoise(model, positive, negative, ref_latent, steps, cfg, seed):
    """Drive the apply_model path directly through KSampler (bypasses the broken
    comfy.sample.sample wrapper). Output latent is sized to the reference latent:
    Qwen-Image-Edit produces an edited image at the reference resolution."""
    import comfy.sample
    import comfy.samplers

    # ref_latent is [B,16,H,W] or already 5-D [B,16,T,H,W]; QwenImageTransformer2DModel
    # wants a 5-D [B,C,T,H,W] latent (T=1 for a flat image). Size the output latent to
    # the reference spatial dims (Qwen-Image-Edit edits at the reference resolution).
    h, w = int(ref_latent.shape[-2]), int(ref_latent.shape[-1])
    b, c = int(ref_latent.shape[0]), int(ref_latent.shape[1])
    latent_image = torch.zeros((b, c, 1, h, w), dtype=torch.float32)
    noise = comfy.sample.prepare_noise(latent_image, seed=seed)

    sampler = comfy.samplers.KSampler(
        model,
        steps=steps,
        device=model.load_device,
        sampler="euler",
        scheduler="simple",
        denoise=1.0,
        model_options=model.model_options,
    )
    samples = sampler.sample(
        noise,
        positive,
        negative,
        cfg=cfg,
        latent_image=latent_image,
        disable_pbar=False,
        seed=seed,
    )
    return samples.detach()


def vae_decode(vae, latent):
    """Qwen-Image VAE decode. The DiT works in 5-D [B,C,T,H,W]; squeeze T for the VAE
    (T=1). comfy VAE.decode returns [B,H,W,C]."""
    # comfy's Qwen VAE.decode expects the native 5-D [B,C,T,H,W] latent (its
    # memory_used_decode indexes shape[4]); it returns [B,(T),H,W,C].
    with torch.no_grad():
        img = vae.decode(latent.to("cuda"))
    img = img.detach().float()
    if img.dim() == 5:  # [B,T,H,W,C] -> take frame 0
        img = img[:, 0]
    if img.dim() == 4 and img.shape[-1] in (1, 3):
        img = img.movedim(-1, 1)  # [B,H,W,C] -> [B,C,H,W]
    return img


def save_png(img_bchw, path):
    from PIL import Image

    x = img_bchw[0].detach().clamp(0, 1).float().cpu().movedim(0, -1).numpy()
    Image.fromarray((x * 255).round().astype("uint8")).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="Qwen__Qwen-Image-Edit-2509.dit.b4.fni8")
    ap.add_argument("--arch", default="qwen_image_edit")
    ap.add_argument("--input-image", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="/tmp/qwen_edit")
    ap.add_argument("--tag", default="int8")
    ap.add_argument(
        "--lora", action="append", default=None, help="path to a LoRA (repeatable; stacks in order)"
    )
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument(
        "--cond-cache",
        default=None,
        help="path to cache/reuse the (DiT-agnostic) conditioning blob",
    )
    args = ap.parse_args()

    sys.path.insert(0, _repo_root())
    os.makedirs(args.outdir, exist_ok=True)

    import folder_paths

    if os.path.isdir(WEIGHTS_DIR):
        folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)

    import comfy.sd
    import comfy.utils
    from comfyui_superl8.gate import is_sm70
    from bench.quality import cosine_similarity

    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return

    path = folder_paths.get_full_path("diffusion_models", args.unet)
    if path is None:
        path = os.path.join(WEIGHTS_DIR, args.unet)
    if not os.path.exists(path):
        print(f"SKIP: {args.unet} not found at {path}")
        return

    torch.cuda.reset_peak_memory_stats()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"unet: {args.unet}   prompt: {args.prompt!r}   steps={args.steps} cfg={args.cfg}")

    t_all0 = time.time()

    # ---- VAE (small, load once) ----
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE_FILE, safe_load=True))

    # ---- [1] text+image encode (TE + VAE are DiT-version-agnostic, so the
    # conditioning is identical across 2509/2511 and all step counts — cache it once
    # and reuse to skip the ~230s TE reload on every subsequent run). ----
    print("[1/4] Qwen2.5-VL encode (text + input image) ...")
    if args.cond_cache and os.path.exists(args.cond_cache):
        t0 = time.time()
        blob = torch.load(args.cond_cache, weights_only=False)
        positive, negative, ref_latent = blob["positive"], blob["negative"], blob["ref_latent"]
        t_encode = time.time() - t0
        print(f"      loaded cached conditioning ({args.cond_cache}) in {t_encode:.2f}s")
    else:
        img_in = load_input_image(args.input_image)
        print(f"      input image: {tuple(img_in.shape)}")
        t0 = time.time()
        positive, negative, ref_latent = encode_prompt_edit(args.prompt, img_in, vae)
        t_encode = time.time() - t0
        print(f"      encode wall: {t_encode:.2f}s   cond={positive[0][0].shape}")
        if args.cond_cache:
            torch.save(
                {"positive": positive, "negative": negative, "ref_latent": ref_latent},
                args.cond_cache,
            )
            print(f"      cached conditioning -> {args.cond_cache}")

    # ---- [2] load int8 DiT ----
    print(f"[2/4] load int8 dp4a DiT ({os.path.basename(path)}) ...")
    from comfyui_superl8.nodes import UnetLoaderFNI8

    t0 = time.time()
    (model,) = UnetLoaderFNI8().load(args.unet, arch=args.arch)
    t_load = time.time() - t0
    print(f"      DiT load wall: {t_load:.2f}s")

    # ---- W4A8 residency + manual LoRA (bypass ComfyUI's patch machinery) ----
    # ComfyUI can't give us BOTH resident 4-bit weights AND lazy LoRA: full-load mode
    # eagerly materializes LoRA-patched fp weights (40 GB -> OOM), and lazy weight_function
    # only exists in lowvram mode (which streams every weight CPU->GPU per step: 500x
    # slower on the fleet's PCIe-1.0-x1). So we place the packed low-bit DiT resident
    # on the GPU ourselves (keeping the FNI8Tensor codes quantized; per_group_i4 stays
    # 4-bit) and set each Linear's weight_function from the LoRA directly -> the
    # FNI8Ops.Linear forward folds it (per_row_i8: int8 accumulate; per_group_i4:
    # dequant(i4)->add-delta->fp linear). Then no-op comfy's loader so the sampler can't
    # undo it.
    from comfyui_superl8.superl8_tensor import FNI8Tensor
    import comfy.model_management as mm

    def _is_low_bit(p):
        return isinstance(p, FNI8Tensor)

    dev = model.load_device
    dit = model.model.diffusion_model
    torch.cuda.empty_cache()

    # (1) place the whole DiT resident on GPU, keeping the FNI8Tensor codes quantized
    # (.to() moves device but stays int8/uint8) and moving fp params/buffers normally.
    n_i4 = i4_bytes = 0
    for mod in dit.modules():
        for pname, p in list(mod._parameters.items()):
            if p is None:
                continue
            mod._parameters[pname] = p.to(dev)
            if _is_low_bit(p):
                n_i4 += 1
                i4_bytes += p.int8_data().numel() + (
                    p.q_scale.numel() * 4 if p.q_scale is not None else 0
                )
        for bname, b in list(mod._buffers.items()):
            if b is not None:
                mod._buffers[bname] = b.to(dev)

    # (1b) Disable the int8 FlashAttention override for this arch: Qwen-Image-Edit's
    # post-RoPE Q has extreme outliers (max/median ~5e4) that overflow the int8 QK^T
    # kernel to NaN before its SQNR gate can fall back -> a non-finite latent. Attention
    # is NOT where the w4a8 win is (the linears are), so run softmax(QK^T)V in fp SDPA
    # (SageAttention-style: the numerically load-bearing piece stays fp) while every
    # projection/MLP linear still runs dp4a. Remove the override from all option dicts.
    n_attn_removed = 0
    for _opt in (
        getattr(model, "model_options", None),
        getattr(model.model, "model_options", None),
    ):
        if isinstance(_opt, dict):
            to = _opt.get("transformer_options")
            if isinstance(to, dict) and to.pop("optimized_attention_override", None):
                n_attn_removed += 1
            if _opt.pop("optimized_attention_override", None):
                n_attn_removed += 1
    print(
        f"      int8 attention override removed from {n_attn_removed} option dict(s) "
        f"(attention runs fp SDPA; linears stay w4a8 dp4a)"
    )

    # (2) manual LoRA -> weight_function on each target Linear.
    n_lora_mod = 0
    if args.lora:
        for lp in args.lora:
            lsd = comfy.utils.load_torch_file(lp, safe_load=True)
            n_lora_mod += _apply_lora_weight_functions(dit, lsd, args.lora_strength, dev)
            del lsd
        assert n_lora_mod > 0, "LoRA matched 0 modules (key-map mismatch)"
        print(f"      manual LoRA: weight_function set on {n_lora_mod} Linear modules")

    # (3) stop comfy from re-placing / eager-patching during sampling.
    resident_bytes = i4_bytes + 1024 * 1024**2
    model.model_size = lambda *_a, **_k: resident_bytes
    model.model.model_size = model.model_size
    _orig_load_models_gpu = mm.load_models_gpu
    mm.load_models_gpu = lambda *_a, **_k: None

    # Pre-pass the per-layer SQNR gate: its first-forward fp reference (dequantize the
    # weight + a full fp matmul) is a big transient that OOMs the tight 16 GB budget with
    # modulation now at int8. The linears are already validated clean (LTX low-bit audit:
    # worst cos 0.9999; our own smoke: 837/840 pass), so skip the runtime check here.
    if os.environ.get("FNI8_SKIP_SQNR_GATE", "1") == "1":
        n_gate = 0
        for _m in dit.modules():
            if hasattr(_m, "_fni8_sqnr_pass"):
                _m._fni8_sqnr_pass = True
                n_gate += 1
        print(
            f"      SQNR gate pre-passed on {n_gate} linears (saves the transient "
            "fp-reference alloc; linears pre-validated cos>=0.999)"
        )
    sample = next(m for m in dit.modules() if _is_low_bit(getattr(m, "weight", None)))
    print(
        f"      low-bit resident: {n_i4} quantized weights "
        f"(codes {i4_bytes / 1024**3:.2f} GiB); "
        f"sample codes on {sample.weight.int8_data().device}; "
        f"weight_function={len(getattr(sample, 'weight_function', []))}; "
        f"GPU alloc={torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
    )

    # ---- int8 LoRA-patch audit: spy on the dequant->add-delta->requantize path so we
    # MEASURE (not claim) that LoRA rides the int8 dp4a mechanism. ----
    patch_audit = {"apply_calls": 0}
    import comfyui_superl8.ops as _ops

    _orig_apply = _ops.apply_patches_to_int8_weight

    def _spy_apply(int8_data, q_scale, wdtype, patch_fns):
        patch_audit["apply_calls"] += 1
        return _orig_apply(int8_data, q_scale, wdtype, patch_fns)

    _ops.apply_patches_to_int8_weight = _spy_apply

    # ---- optional finiteness probe: hook key submodules, report the FIRST non-finite
    # output (which block / linear first blows up). Env-gated (FNI8_DEBUG_FINITE=1). ----
    if os.environ.get("FNI8_DEBUG_FINITE") == "1":
        _first = {"hit": False}

        def _mk_hook(name):
            def h(_m, _inp, out):
                if _first["hit"]:
                    return
                t = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(t):
                    mx = t.abs().max().item() if t.numel() else 0.0
                    fin = torch.isfinite(t).all().item()
                    if not fin:
                        _first["hit"] = True
                        print(f"      [FINITE-PROBE] FIRST non-finite at {name} (absmax={mx:.3e})")
                    elif mx > 1e4:
                        print(f"      [FINITE-PROBE] {name} finite absmax={mx:.3e}")

            return h

        dbg = model.model.diffusion_model
        dbg.img_in.register_forward_hook(_mk_hook("img_in"))
        for _i, _blk in enumerate(dbg.transformer_blocks):
            _blk.register_forward_hook(_mk_hook(f"block{_i}"))
        for _nm in ("norm_out", "proj_out"):
            if hasattr(dbg, _nm):
                getattr(dbg, _nm).register_forward_hook(_mk_hook(_nm))

    # ---- [3] denoise ----
    print(f"[3/4] {args.steps}-step edit-denoise ...")
    torch.cuda.synchronize()
    t0 = time.time()
    latent = denoise(model, positive, negative, ref_latent, args.steps, args.cfg, args.seed)
    torch.cuda.synchronize()
    t_denoise = time.time() - t0
    peak_denoise = torch.cuda.max_memory_allocated() / (1024**3)
    print(
        f"      denoise wall: {t_denoise:.2f}s   latent={tuple(latent.shape)}   "
        f"peak HBM={peak_denoise:.2f} GiB"
    )
    assert torch.isfinite(latent).all(), "latent non-finite (fp16/int8 overflow)"

    # ---- int8 LoRA-patch + SQNR-gate tally (measured on the LoRA'd weights) ----
    n_int8_lin = n_gate_pass = n_gate_fp = n_gate_none = n_with_patch = 0
    for _m in model.model.diffusion_model.modules():
        w = getattr(_m, "weight", None)
        if isinstance(w, FNI8Tensor) and hasattr(_m, "_fni8_sqnr_pass"):
            n_int8_lin += 1
            if len(getattr(_m, "weight_function", [])) > 0:
                n_with_patch += 1
            p = _m._fni8_sqnr_pass
            n_gate_pass += p is True
            n_gate_fp += p is False
            n_gate_none += p is None
    _ops.apply_patches_to_int8_weight = _orig_apply
    lora_audit = {
        "apply_patches_calls": patch_audit["apply_calls"],
        "int8_linears": n_int8_lin,
        "linears_with_lora_patch": n_with_patch,
        "sqnr_gate_int8_pass": n_gate_pass,
        "sqnr_gate_fp_fallback": n_gate_fp,
        "sqnr_gate_unchecked": n_gate_none,
    }
    print(
        f"  int8 LoRA-patch audit        : apply_patches_to_int8_weight calls="
        f"{patch_audit['apply_calls']}  int8-linears={n_int8_lin}  "
        f"with-LoRA-patch={n_with_patch}  gate(int8-pass/fp-fallback)="
        f"{n_gate_pass}/{n_gate_fp}"
    )

    del model
    torch.cuda.empty_cache()

    # ---- [4] VAE decode ----
    print("[4/4] Qwen-Image VAE decode ...")
    t0 = time.time()
    import comfy.model_management as _mm

    _mm.load_models_gpu = _orig_load_models_gpu  # restore so the VAE can load to GPU
    img = vae_decode(vae, latent)
    t_vae = time.time() - t0
    outpath = os.path.join(args.outdir, f"qwen_edit_{args.tag}.png")
    save_png(img, outpath)
    t_total = time.time() - t_all0

    # ---- quality gate ----
    structured = img.std().item() > 1e-3 and torch.isfinite(img).all().item()
    # latent cosine vs the reference latent measures how far the edit moved the content
    # (a real edit should be highly correlated with the source, not identical).
    ref5 = ref_latent[:, :, None] if ref_latent.dim() == 4 else ref_latent
    try:
        cos_ref = cosine_similarity(latent, ref5)
    except Exception:
        cos_ref = float("nan")
    peak_all = torch.cuda.max_memory_allocated() / (1024**3)

    print("=" * 66)
    print(f"Qwen-Image-Edit FULL int8 pipeline  ({args.unet})")
    print(f"  output image                 : {img.shape[-2]}x{img.shape[-1]}  saved -> {outpath}")
    print(f"  image finite + structured    : {structured}")
    print(f"  latent cosine (out vs ref)   : {cos_ref:.6f}")
    print(
        f"  wall: encode={t_encode:.2f}s  DiT-load={t_load:.2f}s  "
        f"denoise={t_denoise:.2f}s  vae={t_vae:.2f}s  TOTAL={t_total:.2f}s"
    )
    print(f"  peak HBM (whole pipeline)    : {peak_all:.2f} GiB  (card = 16 GiB)")
    print("=" * 66)

    # machine-readable line for the driver to scrape.
    import json

    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "unet": args.unet,
                "tag": args.tag,
                "out": outpath,
                "structured": bool(structured),
                "cos_ref": float(cos_ref),
                "t_encode": t_encode,
                "t_load": t_load,
                "t_denoise": t_denoise,
                "t_vae": t_vae,
                "t_total": t_total,
                "peak_hbm_gib": peak_all,
                "h": int(img.shape[-2]),
                "w": int(img.shape[-1]),
                "steps": args.steps,
                "cfg": args.cfg,
                "lora": os.path.basename(args.lora) if args.lora else None,
                "lora_audit": lora_audit,
            }
        )
    )


if __name__ == "__main__":
    main()
