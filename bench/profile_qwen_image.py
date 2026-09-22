# SPDX-License-Identifier: Apache-2.0
"""Profile ONE Qwen-Image / Qwen-Image-Edit transformer block on the int8 dp4a path.

Why a single block, not the whole model: the b8 Qwen-Image DiT is 20.5 GB int8 —
it does not fit a 16 GB Volta/CMP card, so a whole-model single-card forward thrashes
CPU-offload PCIe traffic and profiles the offloader, not the kernels. But all 60
blocks are *identical* `QwenImageTransformerBlock`s and dominate the FLOPs (prologue /
embeddings / output head are <1%), so per-block latency x n_layers is an honest,
in-budget DiT-step estimate — and it lets us break the block down op-by-op with REAL
int8 weights (loaded from the published `.fni8`, block 0) at a realistic sequence length.

What it measures, per block:
  * wall time of each sub-region (modulation / attention-total / img_mlp / txt_mlp /
    layernorm+modulate glue), via CUDA events over N iters
  * inside attention: the int8-attn compute (softmax QK^T V) vs the RoPE + q/k RMSNorm
    fp glue vs the q/k/v/out int8 projections
  * an int8-engagement AUDIT: how many Linear call-sites took the dp4a path vs an fp
    fallback (per-layer SQNR gate), and whether int8 attention engaged + its SQNR

Run in the e2e image on a real V100 (GPU 9):

    FNI8_GPU=9 docker compose run --rm --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
       custom_nodes/ComfyUI-superl8/bench/profile_qwen_image.py --img-tokens 4096 --iters 30'
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

WEIGHTS_DIR = os.environ.get("FNI8_WEIGHTS_DIR", "")

# Real Qwen-Image / Qwen-Image-Edit QwenImageTransformer2DModel config (comfy detects
# the same unet_config for base + both Edit revisions).
QWEN_CFG = dict(
    patch_size=2, in_channels=64, out_channels=16, num_layers=1,
    attention_head_dim=128, num_attention_heads=24, joint_attention_dim=3584,
    axes_dims_rope=(16, 56, 56),
)
INNER = QWEN_CFG["num_attention_heads"] * QWEN_CFG["attention_head_dim"]  # 3072
N_LAYERS_FULL = 60


def _load_block0_weights(path, needed_prefixes):
    """Read only the block-0 + prologue/head tensors from the .fni8 (not all 20 GB)."""
    from superl8 import FQReader

    from comfyui_superl8.superl8_tensor import FNI8Tensor
    from comfyui_superl8.loader import _HEAD_FP_DEQUANT, _dequant_qtensor

    out = {}
    with FQReader(path) as r:
        for name in r.names:
            key = name[len("diffusion_model."):] if name.startswith("diffusion_model.") else name
            if not any(key.startswith(p) for p in needed_prefixes):
                continue
            qt = r.get_qtensor(name, device="cpu")
            if getattr(qt, "scheme", None) == "raw":
                out[key] = qt.data
            elif any(s in key.lower() for s in _HEAD_FP_DEQUANT):
                out[key] = _dequant_qtensor(qt)
            else:
                out[key] = FNI8Tensor(qt.data, qt.scale)
    return out


class CudaTimer:
    def __init__(self):
        self.acc = {}

    def region(self, name):
        return _Region(self, name)


class _Region:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
        self.s.record()
        return self

    def __exit__(self, *a):
        self.e.record()
        torch.cuda.synchronize()
        self.timer.acc.setdefault(self.name, []).append(self.s.elapsed_time(self.e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="Qwen__Qwen-Image-Edit-2509.dit.b8.fni8")
    ap.add_argument("--arch", default="qwen_image_edit")
    ap.add_argument("--img-tokens", type=int, default=4096,
                    help="image sequence length (4096 ~ 1024px, 1024 ~ 512px)")
    ap.add_argument("--txt-tokens", type=int, default=256)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--mask", action="store_true", help="pass a text padding mask")
    args = ap.parse_args()

    import folder_paths
    if os.path.isdir(WEIGHTS_DIR):
        folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)

    from comfyui_superl8.gate import is_sm70
    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return

    path = folder_paths.get_full_path("diffusion_models", args.unet)
    if path is None:
        path = os.path.join(WEIGHTS_DIR, args.unet)
    if not os.path.exists(path):
        print(f"SKIP: {args.unet} not found")
        return

    import comfy.ldm.qwen_image.model as qm
    from comfyui_superl8 import ops as fni8_ops
    from comfyui_superl8.attention import Int8AttnGate, make_fni8_attn_override
    from comfyui_superl8.loader import assign_int8_weights

    dev = "cuda:0"
    dtype = torch.bfloat16

    # ---- int8-engagement audit: spy on the linear path + attention gate ----
    audit = {"int8_linear": 0, "fp_linear": 0, "int8_attn": 0, "attn_sqnr": []}
    import comfyui_superl8.attention as A
    import superl8
    orig_attn_fwd = superl8.attn_int8_fwd

    def spy_attn(*a, **k):
        audit["int8_attn"] += 1
        return orig_attn_fwd(*a, **k)
    superl8.attn_int8_fwd = spy_attn
    orig_sqnr = A.attn_sqnr

    def spy_sqnr(yi, yf):
        v = orig_sqnr(yi, yf)
        audit["attn_sqnr"].append(v)
        return v
    A.attn_sqnr = spy_sqnr

    # ---- build a 1-layer Qwen DiT with the int8 ops + real block-0 weights ----
    needed = ("img_in.", "txt_in.", "txt_norm.", "time_text_embed.",
              "transformer_blocks.0.", "norm_out.", "proj_out.")
    print(f"loading block-0 weights from {os.path.basename(path)} ...")
    sd = _load_block0_weights(path, needed)

    model = qm.QwenImageTransformer2DModel(**QWEN_CFG, dtype=dtype, device="cpu",
                                           operations=fni8_ops.FNI8Ops)
    # load fp (raw) params, then re-attach int8 FNI8Tensors so dp4a engages
    from comfyui_superl8.superl8_tensor import FNI8Tensor
    fp_sd = {k: v for k, v in sd.items() if not isinstance(v, FNI8Tensor)}
    missing, unexpected = model.load_state_dict(fp_sd, strict=False, assign=True)
    n_int8 = assign_int8_weights(model, {k: v for k, v in sd.items()
                                         if isinstance(v, FNI8Tensor)})
    print(f"re-attached {n_int8} int8 FNI8Tensor weights; "
          f"{len(fp_sd)} fp params; missing(non-int8)={len(missing)}")

    model = model.to(dev).eval()
    block = model.transformer_blocks[0]

    # count int8 vs fp linears in the block by inspecting the resident weights
    for name, mod in block.named_modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, FNI8Tensor):
            audit["int8_linear"] += 1
        elif isinstance(mod, fni8_ops.FNI8Ops.Linear) or (w is not None and w.dim() == 2 and "Linear" in type(mod).__name__):
            audit["fp_linear"] += 1

    # ---- realistic inputs ----
    S_img, S_txt = args.img_tokens, args.txt_tokens
    torch.manual_seed(0)
    hidden = torch.randn(1, S_img, INNER, device=dev, dtype=dtype)
    enc = torch.randn(1, S_txt, INNER, device=dev, dtype=dtype)
    temb = torch.randn(1, INNER, device=dev, dtype=dtype)
    # rotary emb for txt+img tokens: shape [1, S, head_dim/2, 2, 2] per apply_rope1
    S = S_txt + S_img
    hd = QWEN_CFG["attention_head_dim"]
    rope = torch.randn(1, S, hd // 2, 2, 2, device=dev, dtype=dtype)

    gate = Int8AttnGate()
    tos = {"optimized_attention_override": make_fni8_attn_override(gate)}
    enc_mask = torch.ones(1, S_txt, device=dev, dtype=dtype) if args.mask else None

    def run_block():
        return block(hidden_states=hidden, encoder_hidden_states=enc,
                     encoder_hidden_states_mask=enc_mask, temb=temb,
                     image_rotary_emb=rope, transformer_options=tos)

    # ---- warmup (also triggers the one-time SQNR gates) ----
    for _ in range(3):
        run_block()
    torch.cuda.synchronize()

    # ---- region timing: wrap submodule forwards to accumulate CUDA time ----
    timer = CudaTimer()

    def wrap(mod, name):
        orig = mod.forward

        def timed(*a, **k):
            with timer.region(name):
                return orig(*a, **k)
        mod.forward = timed
        return orig

    o1 = wrap(block.attn, "attention_total")
    o2 = wrap(block.img_mlp, "img_mlp(ffn)")
    o3 = wrap(block.txt_mlp, "txt_mlp(ffn)")
    o4 = wrap(block.img_mod, "modulation")
    o5 = wrap(block.txt_mod, "modulation")
    # inside attention: time the int8/sdpa compute via the override, and rope
    orig_override = tos["optimized_attention_override"]

    def timed_override(func, *a, **k):
        with timer.region("attn_compute(int8/sdpa)"):
            return orig_override(func, *a, **k)
    tos["optimized_attention_override"] = timed_override
    orig_rope = qm.apply_rope1

    def timed_rope(*a, **k):
        with timer.region("rope(fp)"):
            return orig_rope(*a, **k)
    qm.apply_rope1 = timed_rope

    # ---- timed block iters ----
    torch.cuda.synchronize()
    whole = []
    for _ in range(args.iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_block()
        e.record()
        torch.cuda.synchronize()
        whole.append(s.elapsed_time(e))

    block_ms = statistics.median(whole)
    print("\n" + "=" * 70)
    print(f"Qwen-Image-Edit block profile — {os.path.basename(path)}")
    print(f"seq: txt={S_txt} img={S_img} (total {S})  dim={INNER}  heads=24 hd={hd}"
          f"  mask={'yes' if args.mask else 'no'}  bf16  GPU={torch.cuda.get_device_name(0)}")
    print("=" * 70)
    print("NOTE: per-region rows each cuda.synchronize(), which INFLATES them (~+10-15%);"
          "\n      trust BLOCK TOTAL + the torch.profiler aten-op table for exact splits.")
    print(f"{'region':32s} {'ms/block (median)':>18s} {'% block':>9s}")

    def med(name):
        v = timer.acc.get(name, [])
        return statistics.median(v) if v else 0.0
    regions = ["modulation", "attention_total", "  attn_compute(int8/sdpa)",
               "  rope(fp)", "img_mlp(ffn)", "txt_mlp(ffn)"]
    for r in regions:
        key = r.strip()
        m = med(key)
        pct = 100 * m / block_ms if block_ms else 0
        print(f"{r:32s} {m:18.3f} {pct:8.1f}%")
    # glue = block - (modulation + attention_total + both ffns)
    top = med("modulation") + med("attention_total") + med("img_mlp(ffn)") + med("txt_mlp(ffn)")
    glue = max(block_ms - top, 0.0)
    print(f"{'norm+modulate+gate glue (fp)':32s} {glue:18.3f} {100*glue/block_ms:8.1f}%")
    print("-" * 70)
    print(f"{'BLOCK TOTAL':32s} {block_ms:18.3f} {100.0:8.1f}%")
    print(f"\nEstimated DiT step ({N_LAYERS_FULL} blocks): "
          f"{block_ms * N_LAYERS_FULL / 1000:.3f} s "
          f"(blocks dominate; prologue/head <1%)")

    print("\n--- int8-engagement audit ---")
    print(f"block Linear call-sites int8 (dp4a): {audit['int8_linear']}")
    print(f"block Linear call-sites fp fallback: {audit['fp_linear']}")
    # NOTE: a +inf SQNR here means superl8.attn_int8_fwd's OWN SageAttention gate rejected
    # int8 QK (outlier-heavy Q) and returned plain fp SDPA (bit-identical to the fp ref).
    # So "invocations>0" counts wrapper calls, not int8 compute — check the SQNR: finite ==
    # real int8 (e.g. Z-Image ~47 dB); +inf == kernel self-fell-back to fp (e.g. Qwen-Image,
    # whose post-RoPE Q has max/median ~5e4). The updated Int8AttnGate pins the latter to fp.
    selffb = all(v == float("inf") for v in audit["attn_sqnr"]) if audit["attn_sqnr"] else False
    print(f"int8 attn_int8_fwd invocations     : {audit['int8_attn']} "
          f"({'wrapper-called' if audit['int8_attn'] else 'not called -> fp SDPA (masked)'})")
    if audit["attn_sqnr"]:
        print(f"int8 attn SQNR (dB) min/med/max     : "
              f"{min(audit['attn_sqnr']):.2f} / "
              f"{statistics.median(audit['attn_sqnr']):.2f} / {max(audit['attn_sqnr']):.2f}"
              f"{'  (+inf == kernel self-fell-back to fp SDPA; int8 NOT engaged)' if selffb else ''}")

    superl8.attn_int8_fwd = orig_attn_fwd
    A.attn_sqnr = orig_sqnr


if __name__ == "__main__":
    main()
