# SPDX-License-Identifier: Apache-2.0
"""Profile the Z-Image int8 dp4a DiT forward on a real denoise step.

Loads the published `Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8` exactly as
`UnetLoaderFNI8` does (int8 dp4a linears + fused-qkv int8 attention + int8 FA,
SQNR-gated), encodes a real prompt with the Qwen3-4B TE, and runs an N-step Turbo
denoise under `torch.profiler`, emitting:

  * a ranked CUDA-kernel table (self CUDA time) -> where the time actually goes;
  * op-class counters (int8 linear calls, fp-fallback linears, int8 attn, fp SDPA);
  * per-denoise-step wall latency (the real optimization metric).

Run inside the e2e image on a Volta/CMP sm_70 GPU::

    FNI8_GPU=8 docker compose run --rm --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
       custom_nodes/ComfyUI-superl8/bench/profile_zimage_dit.py --steps 8 --size 64'
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bench.full_pipeline_zimage import (  # noqa: E402
    WEIGHTS_DIR,
    DEFAULT_PROMPT,
    encode_prompt,
    load_int8_model,
)

# ---- counters installed by monkeypatching the compute hot paths -------------
COUNTS: Counter = Counter()


def _install_counters():
    import comfyui_superl8.int8_linear as il
    import comfyui_superl8.attention as at

    _orig_int8_linear = il.int8_linear
    _orig_dp4a = at._int8_dp4a
    _orig_sdpa = at._sdpa

    def counted_int8_linear(x, w, bias=None):
        COUNTS["int8_linear_calls"] += 1
        return _orig_int8_linear(x, w, bias)

    def counted_dp4a(q, k, v):
        COUNTS["attn_int8_calls"] += 1
        return _orig_dp4a(q, k, v)

    def counted_sdpa(q, k, v, mask=None):
        COUNTS["attn_sdpa_calls"] += 1
        return _orig_sdpa(q, k, v, mask)

    il.int8_linear = counted_int8_linear
    # ops.py imported int8_linear by value -> patch there too.
    import comfyui_superl8.ops as ops
    ops.int8_linear = counted_int8_linear
    at._int8_dp4a = counted_dp4a
    at._sdpa = counted_sdpa


def _gate_report(model):
    """Count int8 vs fp-fallback linears from the per-layer SQNR gate decisions."""
    from comfyui_superl8.superl8_tensor import FNI8Tensor

    dm = model.model.diffusion_model
    n_int8_weight = n_fp_weight = 0
    n_pass = n_fail = n_unchecked = 0
    fail_names = []
    for name, m in dm.named_modules():
        if not hasattr(m, "_fni8_sqnr_pass"):
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, FNI8Tensor):
            n_int8_weight += 1
            d = m._fni8_sqnr_pass
            if d is True:
                n_pass += 1
            elif d is False:
                n_fail += 1
                fail_names.append((name, tuple(w.shape)))
            else:
                n_unchecked += 1
        else:
            n_fp_weight += 1
    print("  fp-fallback (gate FAIL) linears:")
    from collections import Counter as _C
    leaf = _C(n.rsplit(".", 1)[-1] for n, _ in fail_names)
    for k, c in leaf.most_common():
        print(f"    {k:14s}: {c}")
    for n_, sh in fail_names[:20]:
        print(f"      {n_}  {sh}")
    return dict(
        int8_weight_linears=n_int8_weight,
        fp_weight_linears=n_fp_weight,
        gate_pass_int8=n_pass,
        gate_fail_fp_fallback=n_fail,
        gate_unchecked=n_unchecked,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=64, help="latent H=W (px=8x)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--warmup-steps", type=int, default=2)
    args = ap.parse_args()

    import folder_paths
    folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)
    from comfyui_superl8.gate import is_sm70
    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return

    _install_counters()

    print(f"[1/3] encode prompt with Qwen3-4B TE: {args.prompt!r}")
    cond, empty = encode_prompt(args.prompt)
    emb = cond[0][0]                      # [1, S, cap_feat_dim] real Qwen3-4B conditioning
    S = emb.shape[1]

    print("[2/3] load int8 Z-Image DiT (UnetLoaderFNI8 path) + load onto GPU")
    import comfy.model_management as mm
    model = load_int8_model()
    in_ch = model.model.diffusion_model.in_channels
    mm.load_model_gpu(model)             # ModelPatcher load: weights->GPU, patches applied
    dev = mm.get_torch_device()
    dtype = model.model.get_dtype()
    topts = dict(model.model_options.get("transformer_options", {}))
    emb = emb.to(dev, dtype)

    # A representative mid-schedule sigma (Turbo flow-match). DiT compute is
    # sigma-invariant in cost, so any in-range value profiles the same step.
    sig = model.model.model_sampling.sigmas
    sigma = sig[len(sig) // 2].to(dev).reshape(1)

    def run_forward():
        x = torch.randn(1, in_ch, args.size, args.size, device=dev, dtype=dtype)
        return model.model.apply_model(
            x, sigma, c_crossattn=emb, num_tokens=S,
            transformer_options=dict(topts),
        )

    # -- per-forward wall latency ----------------------------------------------
    step_times = []

    print(f"[3/3] warmup {args.warmup_steps}, then profile {args.steps} DiT forwards @ {args.size*8}px "
          f"(seq S={S})")
    with torch.no_grad():
        for _ in range(max(args.warmup_steps, 1)):
            run_forward()
    torch.cuda.synchronize()
    warmup_n = args.warmup_steps

    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_forward()
                torch.cuda.synchronize()
                step_times.append((time.perf_counter() - t0) * 1e3)

    n = len(step_times) or 1
    print("=" * 78)
    print(f"Z-Image int8 DiT profile  ({args.size*8}px, {args.steps} Turbo steps)")
    print("=" * 78)
    print(f"  DiT forward calls (profiled) : {len(step_times)} (warmup engaged {warmup_n})")
    print(f"  per-forward wall latency     : mean {sum(step_times)/n:.1f} ms  "
          f"min {min(step_times):.1f}  max {max(step_times):.1f}")
    print()
    print("  op-class counters (per full denoise, {} forwards):".format(len(step_times)))
    for k in ("int8_linear_calls", "attn_int8_calls", "attn_sdpa_calls"):
        per = COUNTS[k] / n
        print(f"    {k:22s}: {COUNTS[k]:6d}  ({per:.1f}/forward)")
    print()
    g = _gate_report(model)
    print("  linear gate census:")
    for k, v in g.items():
        print(f"    {k:22s}: {v}")
    print()
    print("  top CUDA kernels by self time (whole {}-step denoise):".format(args.steps))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
