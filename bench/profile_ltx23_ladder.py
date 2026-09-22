# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 resolution x frames time/VRAM ladder (video analogue of the Z-Image ladder).

The full int8 LTX-2.3 DiT does NOT fit one 16 GB card (`Lightricks__LTX-2.3.dit.b4.fni8`
= 18.9 GB, `.b8` = 27.6 GB), so it cannot run end-to-end on a single GPU. The honest
method is **block-level extrapolation** (same as `docs/ltx23-block-profile.md`): measure
ONE `BasicAVTransformerBlock` at a given video-token count N, then

    full-DiT time per denoise step  =  ms/block x 48 blocks
    full-DiT time (whole sample)    =  ms/block x 48 x num_steps

Random int8 weights give bit-for-bit the SAME kernel latency as real weights (per-row
int8 GEMM + dp4a FlashAttention are data-independent in timing), so we never need the
18.9 GB resident to measure honest per-block latency.

Video token count for LTX-2.3 (confirmed from comfy `EmptyLTXVLatentVideo` +
`SymmetricPatchifier(1)`, `vae_scale_factors=(8,32,32)`):

    latent = [B, 128, ((length-1)//8)+1, height//32, width//32]     # DiT patch_size = 1
    N_video = ((length-1)//8 + 1) * (height//32) * (width//32)

    python3 bench/profile_ltx23_ladder.py [--iters 15] [--steps 30]

Video-only path (empty audio tensor). RoPE skipped (pe=None): cheap elementwise, identical
on both paths. Attention is dp4a FlashAttention-2 (O(N) memory, O(N^2) compute), so peak
VRAM stays linear in N.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

# Real LTX-2.3 (22b) block config, read off Lightricks__LTX-2.3.dit.b8.fni8.
CFG = dict(
    v_dim=4096, a_dim=2048, v_heads=32, a_heads=32, vd_head=128, ad_head=64,
    v_context_dim=4096, a_context_dim=2048,
    apply_gated_attention=True, cross_attention_adaln=True,
)
NUM_BLOCKS = 48
DIT_B4_GIB = 18887633664 / (1024 ** 3)   # 17.59 GiB resident int8 weights (b4/W4A8)
DIT_B8_GIB = 27604642048 / (1024 ** 3)   # 25.71 GiB resident int8 weights (b8/W8A8)

# LTX-2.3 latent geometry (comfy EmptyLTXVLatentVideo): spatial /32, temporal /8 (+1).
SPATIAL_PX = [512, 768, 1024]
FRAMES = [25, 49, 97]   # pixel frames; all == 1 (mod 8); latent frames 4, 7, 13


def latent_tokens(px: int, frames: int) -> tuple[int, int, int]:
    lat_f = ((frames - 1) // 8) + 1
    lat_hw = px // 32
    return lat_f, lat_hw, lat_f * lat_hw * lat_hw


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

    return BasicAVTransformerBlock(
        v_dim=CFG["v_dim"], a_dim=CFG["a_dim"],
        v_heads=CFG["v_heads"], a_heads=CFG["a_heads"],
        vd_head=CFG["vd_head"], ad_head=CFG["ad_head"],
        v_context_dim=CFG["v_context_dim"], a_context_dim=CFG["a_context_dim"],
        apply_gated_attention=CFG["apply_gated_attention"],
        cross_attention_adaln=CFG["cross_attention_adaln"],
        dtype=dtype, device=dev, operations=operations,
    )


def _init_and_quantize(fp_block, int8_block):
    import torch.nn as nn

    from comfyui_superl8.superl8_tensor import FNI8Tensor
    from comfyui_superl8.int8_linear import quantize_linear_weight

    g = torch.Generator(device="cpu").manual_seed(1234)

    def randn_like(p):
        return (torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.02).to(p.dtype)

    fp_lin = {n: m for n, m in fp_block.named_modules() if isinstance(m, nn.Linear)}
    i8_mods = dict(int8_block.named_modules())
    fp_params = dict(fp_block.named_parameters())
    i8_params = dict(int8_block.named_parameters())
    ref_dev = int8_block.attn1.to_q.weight.device
    with torch.no_grad():
        for name, p in fp_params.items():
            newv = randn_like(p)
            p.copy_(newv.to(p.device))
            if name in i8_params:
                i8_params[name].copy_(newv.to(i8_params[name].device))
        for name, lin in fp_lin.items():
            w = lin.weight.data
            qt = quantize_linear_weight(w.float())
            fni = FNI8Tensor(qt.data.to(ref_dev), qt.scale.to(ref_dev))
            sub = i8_mods[name]
            sub._parameters["weight"] = fni
            if lin.bias is not None:
                sub._parameters["bias"].data.copy_(lin.bias.data.to(sub.bias.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30,
                    help="denoise steps assumed for full-DiT extrapolation (LTX-2.3 base default)")
    ap.add_argument("--allow-outlier-gate", action="store_true",
                    help="do NOT force the dp4a path; let the SageAttention outlier gate "
                         "(detect_q_outlier_domination) fire on random weights, which "
                         "demotes self-attn to fp SDPA (O(N^2)) non-deterministically")
    args = ap.parse_args()

    # RANDOM-WEIGHT ARTIFACT CONTROL. `superl8.attn_int8_fwd` has a SageAttention accuracy
    # gate: if a Q row has a dominant outlier channel it transparently falls back to fp
    # SDPA. On real LTX-2.3 weights the self-attn dp4a path ENGAGES and clears the SQNR
    # gate (docs/ltx23-block-profile.md: block int8-vs-fp cosine 0.9996). On RANDOM probe
    # weights the outlier gate fires arbitrarily, which would (a) make the dp4a latency
    # non-deterministic and (b) route self-attn through O(N^2) SDPA. To measure the
    # intended dp4a fast path honestly and reproducibly, force the outlier gate OFF for
    # the sweep (the fp SDPA O(N^2) fallback is reported separately as the OOM boundary).
    if not args.allow_outlier_gate:
        import superl8.quant as _q
        _q.detect_q_outlier_domination = lambda q: False

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
    torch.cuda.synchronize()
    weights_bytes = torch.cuda.memory_allocated()
    print(f"config: {CFG}", flush=True)
    print(f"gpu: {torch.cuda.get_device_name(0)}  1-block weights resident: "
          f"{weights_bytes/2**20:.0f} MiB (int8+fp probe blocks)", flush=True)
    print(f"assumed denoise steps for full-DiT: {args.steps}\n", flush=True)

    C = args.ctx
    v_dim, num_ada = CFG["v_dim"], 9
    cross_adaln = CFG["cross_attention_adaln"]

    # Two overrides:
    #  - gate (SQNR-gated) is used ONCE at a mid-N point to report the honest gate
    #    verdict (does int8 self-attn engage + at what SQNR). Its first-sighting builds
    #    an O(N^2) fp SDPA reference, which OOMs at the top of the ladder — so we do NOT
    #    use it for the latency/VRAM sweep.
    #  - raw (gate=None) is the pure dp4a FlashAttention path (no fp reference built): it
    #    is the exact kernel work a passing gate runs, and it is O(N) memory, so it
    #    measures every rung including the largest clip that the fp path can't fit.
    gate = Int8AttnGate()
    override_raw = make_fni8_attn_override(None)
    topts_i8 = {"optimized_attention_override": override_raw, "run_vx": True, "run_ax": False,
                "a2v_cross_attn": False, "v2a_cross_attn": False}
    topts_gate = {"optimized_attention_override": make_fni8_attn_override(gate),
                  "run_vx": True, "run_ax": False, "a2v_cross_attn": False, "v2a_cross_attn": False}
    topts_fp = {"run_vx": True, "run_ax": False, "a2v_cross_attn": False, "v2a_cross_attn": False}

    def mk_io(N):
        g = torch.Generator(device="cpu").manual_seed(0)

        def mk(*shape):
            return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.5).to(dev, dt)

        return dict(
            vx=mk(1, N, v_dim),
            ax=torch.empty(1, 0, CFG["a_dim"], device=dev, dtype=dt),
            kw=dict(v_context=mk(1, C, CFG["v_context_dim"]), a_context=None,
                    attention_mask=None, v_timestep=mk(1, 1, num_ada * v_dim), a_timestep=None,
                    v_pe=None, a_pe=None, self_attention_mask=None,
                    v_prompt_timestep=mk(1, 1, 2 * v_dim) if cross_adaln else None),
        )

    def measure(N):
        """Everything for one rung inside a function so ALL activation tensors free on
        return (avoids cross-rung caching-allocator buildup that otherwise OOMs 16 GiB)."""
        io = mk_io(N)

        def run(blk, topts):
            return blk((io["vx"].clone(), io["ax"]), transformer_options=topts, **io["kw"])

        # int8 dp4a steady-state peak activation VRAM. With the outlier gate forced off
        # (default) self-attn stays on the O(N) FlashAttention kernel, so block activation
        # is O(N) (~0.18 MiB/token). If the fp SDPA fallback engages instead it is O(N^2).
        with torch.no_grad():
            run(i8_block, topts_i8)  # warm (autotune/workspace) so peak is steady-state
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            run(i8_block, topts_i8)
        torch.cuda.synchronize()
        act_bytes = torch.cuda.max_memory_allocated() - base

        with torch.no_grad():
            med_i8, _ = _time_ms(lambda: run(i8_block, topts_i8), args.iters)

        # fp SDPA path is O(N^2) on Volta (no flash kernel) — record OOM honestly.
        med_fp = None
        try:
            with torch.no_grad():
                med_fp, _ = _time_ms(lambda: run(fp_block, topts_fp), max(4, args.iters // 2))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        return med_i8, med_fp, act_bytes

    # --- one honest gate check at a mid rung where the fp reference fits (N=4096) ---
    io = mk_io(4096)
    with torch.no_grad():
        i8_block((io["vx"].clone(), io["ax"]), transformer_options=topts_gate, **io["kw"])
    print(f"[attn-gate @N=4096] self-attn decisions (int8 engages if passed): "
          f"{gate.decisions}\n", flush=True)
    del io
    torch.cuda.empty_cache()

    rows = []
    for px in SPATIAL_PX:
        for frames in FRAMES:
            lat_f, lat_hw, N = latent_tokens(px, frames)
            med_i8, med_fp, act_bytes = measure(N)
            torch.cuda.empty_cache()
            step_ms = med_i8 * NUM_BLOCKS
            full_s = step_ms * args.steps / 1e3
            rows.append(dict(px=px, frames=frames, lat_f=lat_f, lat_hw=lat_hw, N=N,
                             ms_i8=med_i8, ms_fp=med_fp, step_ms=step_ms, full_s=full_s,
                             act_mib=act_bytes / 2**20))
            fp_s = f"{med_fp:8.2f}" if med_fp is not None else "  OOM  "
            sp = f"{med_fp/med_i8:.2f}x" if med_fp is not None else " n/a "
            print(f"  {px}px x{frames}f  N={N:6d} (lf={lat_f} {lat_hw}^2)  "
                  f"int8={med_i8:8.2f} ms/blk  fp={fp_s}  spd={sp}  "
                  f"step={step_ms/1e3:6.3f}s  full({args.steps})={full_s:7.1f}s  "
                  f"act={act_bytes/2**20:6.0f} MiB", flush=True)

    _print_table(rows, args.steps)
    _fit(rows)


def _print_table(rows, steps):
    print("\n=== LTX-2.3 ladder (int8 dp4a, ms/block extrapolated x48 blocks) ===")
    print(f"| res | frames | lat (f x hw^2) | N tokens | ms/block int8 | ms/step (x48) | "
          f"full-DiT ({steps} steps) | fp ms/block | dp4a speedup | dp4a block act VRAM (O(N)) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        fp = f"{r['ms_fp']:.1f}" if r['ms_fp'] is not None else "OOM"
        sp = f"{r['ms_fp']/r['ms_i8']:.2f}x" if r['ms_fp'] is not None else "n/a"
        print(f"| {r['px']}^2 | {r['frames']} | {r['lat_f']}x{r['lat_hw']}^2 | {r['N']} | "
              f"{r['ms_i8']:.1f} | {r['step_ms']/1e3:.2f} s | {r['full_s']:.0f} s | "
              f"{fp} | {sp} | {r['act_mib']:.0f} MiB |")


def _fit(rows):
    # Least-squares fit ms_i8(N) = a*N + b*N^2 (GEMM linear + attention quadratic).
    import numpy as np

    N = np.array([r["N"] for r in rows], dtype=float)
    y = np.array([r["ms_i8"] for r in rows], dtype=float)
    A = np.stack([N, N * N], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    resid = y - A @ coef
    print(f"\n[scaling law] ms/block(N) ~= {a:.4g}*N + {b:.4g}*N^2   "
          f"(rms resid {np.sqrt((resid**2).mean()):.2f} ms)")
    # GEMM (linear) vs attention (quadratic) crossover: a*N == b*N^2 -> N = a/b.
    if b > 0:
        Nx = a / b
        print(f"[crossover] GEMM-linear == attention-quadratic at N ~= {Nx:.0f} tokens "
              f"(below: GEMM-bound; above: attention-bound)")


if __name__ == "__main__":
    main()
