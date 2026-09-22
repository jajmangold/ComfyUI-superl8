# SPDX-License-Identifier: Apache-2.0
"""Profile ONE LTX-2.3 (`LTXAVModel` / `BasicAVTransformerBlock`) transformer block
forward on a single GPU, at the REAL block config, to rank what is / isn't on the int8
dp4a fast path.

Why a single block, built from config (not the loaded 26 GB `.fni8`): the full int8
LTX-2.3 DiT (26 GB) does not fit one 16 GB card, only one GPU is free here, AND the
shared weights archive is heavily disk-contended (~3 MB/s → a 26 GB load stalls for
hours). But a `BasicAVTransformerBlock` (~0.5 GB) built at the exact LTX-2.3 config
(read off the shipped `.fni8`: v_dim 4096 / 32 heads / 128 head-dim, a_dim 2048 /
64 head-dim, FFN 4096→16384, gated attn, cross-attn adaLN, bf16) runs the identical
compute. The 48 identical blocks × N steps dominate a video-DiT denoise, so this block
is the whole optimization surface. Weights are random then per-row int8-quantized — for
LATENCY this is bit-for-bit the same kernel work as real weights; the int8-vs-fp SQNR it
reports is indicative (random weights), the real-weight quality gate is a separate step.

    python3 bench/profile_ltx23_block.py [--vlen 4096] [--ctx 256] [--iters 20]
                                         [--masked-cross]

Video-only path (empty audio tensor disables audio + AV-cross). RoPE is skipped
(pe=None): a cheap elementwise op, identical on both paths, not the target.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

# Real LTX-2.3 (22b) block config, read off Lightricks__LTX-2.3.dit.b8.fni8.
CFG = dict(
    v_dim=4096,
    a_dim=2048,
    v_heads=32,
    a_heads=32,
    vd_head=128,
    ad_head=64,
    v_context_dim=4096,
    a_context_dim=2048,
    apply_gated_attention=True,
    cross_attention_adaln=True,
)


def _sync():
    torch.cuda.synchronize()


def _time_ms(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def _build_block(operations, dtype, dev):
    from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock

    blk = BasicAVTransformerBlock(
        v_dim=CFG["v_dim"],
        a_dim=CFG["a_dim"],
        v_heads=CFG["v_heads"],
        a_heads=CFG["a_heads"],
        vd_head=CFG["vd_head"],
        ad_head=CFG["ad_head"],
        v_context_dim=CFG["v_context_dim"],
        a_context_dim=CFG["a_context_dim"],
        apply_gated_attention=CFG["apply_gated_attention"],
        cross_attention_adaln=CFG["cross_attention_adaln"],
        dtype=dtype,
        device=dev,
        operations=operations,
    )
    return blk


def _init_and_quantize(fp_block, int8_block):
    """Fill fp_block linears with random bf16 weights; quantize the SAME weights per-row
    to int8 FNI8Tensor and assign onto int8_block. Result: int8_block's effective weight
    is dequant(int8) of fp_block's true weight, so the int8-vs-fp diff is PURE quant error
    (the honest quality question). Non-Linear params (norms, adaLN tables) are copied fp.
    """
    import torch.nn as nn

    from comfyui_superl8.superl8_tensor import FNI8Tensor
    from comfyui_superl8.int8_linear import quantize_linear_weight

    g = torch.Generator(device="cpu").manual_seed(1234)

    def randn_like(p):
        return (torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.02).to(p.dtype)

    fp_lin = {n: m for n, m in fp_block.named_modules() if isinstance(m, nn.Linear)}
    i8_mods = dict(int8_block.named_modules())
    # init + copy all params fp first (norms/tables), then override Linear weights int8.
    fp_params = dict(fp_block.named_parameters())
    i8_params = dict(int8_block.named_parameters())
    with torch.no_grad():
        for name, p in fp_params.items():
            newv = randn_like(p)
            p.copy_(newv.to(p.device))
            if name in i8_params:
                i8_params[name].copy_(newv.to(i8_params[name].device))
        for name, lin in fp_lin.items():
            w = lin.weight.data
            qt = quantize_linear_weight(w.float())  # QTensor(per_row_i8)
            fni = FNI8Tensor(
                qt.data.to(int8_block.attn1.to_q.weight.device),
                qt.scale.to(int8_block.attn1.to_q.weight.device),
            )
            sub = i8_mods[name]
            sub._parameters["weight"] = fni
            if lin.bias is not None:
                sub._parameters["bias"].data.copy_(lin.bias.data.to(sub.bias.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlen", type=int, default=4096)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--masked-cross", action="store_true")
    args = ap.parse_args()

    from comfyui_superl8.attention import Int8AttnGate, make_fni8_attn_override
    from comfyui_superl8.gate import is_sm70
    from comfyui_superl8.ops import FNI8Ops

    if not is_sm70():
        print("SKIP: needs a Volta/CMP sm_70 GPU")
        return

    import comfy.ops

    dev = torch.device("cuda:0")
    dt = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False

    fp_block = _build_block(comfy.ops.manual_cast, dt, dev).eval()
    i8_block = _build_block(FNI8Ops, dt, dev).eval()
    _init_and_quantize(fp_block, i8_block)
    print(f"config: {CFG}", flush=True)

    B, V, C = 1, args.vlen, args.ctx
    v_dim, num_ada, cross_adaln = CFG["v_dim"], 9, CFG["cross_attention_adaln"]
    g = torch.Generator(device="cpu").manual_seed(0)

    def mk(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.5).to(dev, dt)

    vx = mk(B, V, v_dim)
    ax = torch.empty(B, 0, CFG["a_dim"], device=dev, dtype=dt)
    v_context = mk(B, C, CFG["v_context_dim"])
    v_timestep = mk(B, 1, num_ada * v_dim)
    v_prompt_timestep = mk(B, 1, 2 * v_dim) if cross_adaln else None
    cross_mask = torch.zeros(B, V, C, device=dev, dtype=dt) if args.masked_cross else None

    gate = Int8AttnGate()
    override = make_fni8_attn_override(gate)
    topts_i8 = {
        "optimized_attention_override": override,
        "run_vx": True,
        "run_ax": False,
        "a2v_cross_attn": False,
        "v2a_cross_attn": False,
    }
    topts_fp = {"run_vx": True, "run_ax": False, "a2v_cross_attn": False, "v2a_cross_attn": False}

    def blk_inputs():
        return dict(
            v_context=v_context,
            a_context=None,
            attention_mask=cross_mask,
            v_timestep=v_timestep,
            a_timestep=None,
            v_pe=None,
            a_pe=None,
            self_attention_mask=None,
            v_prompt_timestep=v_prompt_timestep,
        )

    def run_i8():
        return i8_block((vx.clone(), ax), transformer_options=topts_i8, **blk_inputs())

    def run_fp():
        return fp_block((vx.clone(), ax), transformer_options=topts_fp, **blk_inputs())

    with torch.no_grad():
        y_i8 = run_i8()[0].float()
        y_fp = run_fp()[0].float()
    cos = torch.nn.functional.cosine_similarity(y_i8.flatten(), y_fp.flatten(), dim=0).item()
    noise = (y_i8 - y_fp).pow(2).mean().item()
    sig = y_fp.pow(2).mean().item()
    sqnr = 10.0 * torch.log10(torch.tensor(sig / max(noise, 1e-30))).item()
    print(
        f"\n[quality] int8-block vs fp-block (random wts, indicative): "
        f"cosine={cos:.6f}  SQNR={sqnr:.1f} dB"
    )
    print(f"[attn-gate] self-attn decisions: {gate.decisions}", flush=True)

    with torch.no_grad():
        med_i8, min_i8 = _time_ms(run_i8, args.iters)
        try:
            med_fp, min_fp = _time_ms(run_fp, args.iters)
        except torch.OutOfMemoryError:
            med_fp = min_fp = None
            torch.cuda.empty_cache()
    print(f"\n[block latency] vlen={V} ctx={C} masked_cross={args.masked_cross}")
    print(f"  int8 dp4a : {med_i8:8.3f} ms (min {min_i8:.3f})")
    if med_fp is None:
        print("  fp (bf16) : OOM (quadratic SDPA score matrix)")
        print("  speedup   : n/a")
    else:
        print(f"  fp (bf16) : {med_fp:8.3f} ms (min {min_fp:.3f})")
        print(f"  speedup   : {med_fp / med_i8:6.2f}x")

    _breakdown(
        i8_block,
        fp_block,
        vx,
        v_context,
        v_timestep,
        v_prompt_timestep,
        cross_mask,
        topts_i8,
        topts_fp,
        gate,
        args.iters,
    )


def _breakdown(
    i8b,
    fpb,
    vx,
    v_context,
    v_timestep,
    v_prompt_timestep,
    cross_mask,
    topts_i8,
    topts_fp,
    gate,
    iters,
):
    import comfy.ldm.common_dit as cdit

    def sub_times(blk, topts):
        with torch.no_grad():
            vshift, vscale = blk.get_ada_values(blk.scale_shift_table, 1, v_timestep, slice(0, 2))
            norm_vx = cdit.rms_norm(vx) * (1 + vscale) + vshift

            def f_self():
                return blk.attn1(norm_vx, pe=None, mask=None, transformer_options=topts)

            def f_cross():
                return blk._apply_text_cross_attention(
                    vx,
                    v_context,
                    blk.attn2,
                    blk.scale_shift_table,
                    getattr(blk, "prompt_scale_shift_table", None),
                    v_timestep,
                    v_prompt_timestep,
                    cross_mask,
                    topts,
                )

            vshift_mlp, vscale_mlp = blk.get_ada_values(
                blk.scale_shift_table, 1, v_timestep, slice(3, 5)
            )
            vx_scaled = cdit.rms_norm(vx) * (1 + vscale_mlp) + vshift_mlp

            def f_ffn():
                return blk.ff(vx_scaled)

            out = {}
            for name, fn in [
                ("self-attn(attn1)", f_self),
                ("text-cross(attn2)", f_cross),
                ("FFN", f_ffn),
            ]:
                try:
                    out[name] = _time_ms(fn, iters)[0]
                except torch.OutOfMemoryError:
                    out[name] = None
                    torch.cuda.empty_cache()
            return out

    ti = sub_times(i8b, topts_i8)
    tf = sub_times(fpb, topts_fp)
    print("\n[sub-op breakdown]  op                 int8 ms   fp ms   speedup")
    for k in ti:
        i8_text = f"{ti[k]:7.3f}" if ti[k] is not None else "    OOM"
        fp_text = f"{tf[k]:7.3f}" if tf[k] is not None else "    OOM"
        speedup = f"{tf[k] / ti[k]:5.2f}x" if ti[k] is not None and tf[k] is not None else "  n/a "
        print(f"                    {k:18s} {i8_text} {fp_text}   {speedup}")
    print(f"[attn-gate final] {gate.decisions}", flush=True)


if __name__ == "__main__":
    main()
