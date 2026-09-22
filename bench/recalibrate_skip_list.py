# SPDX-License-Identifier: Apache-2.0
"""Per-op SQNR recalibration for the bf16 skip-list (issue #103).

The loader's `_SKIP_HINTS` keeps norms, modulations, and output projections as native
bf16 — a heuristic borrowed from Nunchaku that assumes bf16 ≈ int8 in speed. On this
fleet the int8 dp4a path is 4–40× faster than bf16 tensor-core matmul (Volta has no
bf16 TC, so bf16 runs on CUDA cores at ~1/40 the throughput of dp4a). Keeping the
candidate projections in bf16 is therefore costing us wall-time, not saving quality.

This script measures per-operation SQNR (cosine similarity of int8 vs fp output) for
each candidate layer type with realistic weight/activation distributions, and reports
which pass the sqnr_gate bar (cos >= 0.99). The results are used to update
`_LOADER_SKIP_OVERRIDE` in loader.py.

Usage:
    python3 bench/recalibrate_skip_list.py [--device cuda] [--seed 42]

Output: a pass/fail table for each candidate + a latency speedup estimate.
"""

from __future__ import annotations

import argparse
import time
import textwrap
from typing import NamedTuple

import torch

from comfyui_superl8.int8_linear import quantize_linear_weight, sqnr_gate, int8_linear


SHAPES: dict[str, tuple[int, int]] = {
    # Patterns from _SKIP_HINTS that we're recalibrating (candidate layer type -> (out, in))
    "guidance_in":       (3072, 256),     # guidance conditioning projection
    "context_embedder":  (4096, 4096),    # text-embedding -> hidden-dim projection
    "final_layer":       (64,   4096),    # output head: hidden -> patch predictions
    "proj_out":          (4096, 4096),    # attention output projection (same as qkv dim)
}

BATCH_SHAPES: list[int] = [1, 2, 4]
SEQ_LENS: list[int] = [256, 512, 1024, 4096]
WARMUP_ITERS = 5
BENCH_ITERS = 20


class Result(NamedTuple):
    pattern: str
    shape: tuple[int, int]
    cos_fp: float
    cos_bf16: float
    sqnr_db: float
    passed: bool
    int8_ms: float
    bf16_ms: float
    speedup: float


def _measure_candidate(
    pattern: str,
    out_dim: int,
    in_dim: int,
    batch: int,
    seq: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Result:
    """Measure int8 vs fp quality for one weight pattern at one activation shape."""
    torch.manual_seed(42)
    # Realistic weight distribution: DiT linears are ~N(0, 0.02 / sqrt(in_dim))
    w_std = 0.02 / (in_dim ** 0.5)
    w_fp = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * w_std
    # Activations: also small normal
    x = torch.randn(batch, seq, in_dim, device=device, dtype=dtype) * w_std * 10

    # int8 path (fp reference via the same dequantized weight so we isolate
    # activation-quantization error from weight-quantization error).
    qt = quantize_linear_weight(w_fp)
    y_int8 = int8_linear(x, qt)

    # FP reference: same weight, matmul in native dtype
    y_fp = torch.nn.functional.linear(x, w_fp)

    # bf16 reference (the current baseline: same as fp on modern GPUs, but
    # on Volta this is the slow CUDA-core path)
    y_bf16 = torch.nn.functional.linear(x.to(torch.bfloat16), w_fp.to(torch.bfloat16))

    cos_vs_fp = torch.nn.functional.cosine_similarity(
        y_int8.float().flatten(), y_fp.float().flatten(), dim=0
    ).item()
    cos_vs_bf16 = torch.nn.functional.cosine_similarity(
        y_int8.float().flatten(), y_bf16.float().flatten(), dim=0
    ).item()
    passed = sqnr_gate(y_int8, y_fp)

    # SQNR in dB
    signal = y_fp.float().pow(2).mean()
    noise = (y_int8.float() - y_fp.float()).pow(2).mean()
    sqnr_db = (10.0 * torch.log10(signal / noise)).item() if noise.item() > 0 else float("inf")

    # Latency: int8 dp4a vs bf16 matmul
    torch.cuda.synchronize(device)
    for _ in range(WARMUP_ITERS):
        _ = int8_linear(x, qt)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        _ = int8_linear(x, qt)
    torch.cuda.synchronize(device)
    int8_ms = (time.perf_counter() - t0) / BENCH_ITERS * 1000.0

    w_bf16 = w_fp.to(torch.bfloat16)
    x_bf16 = x.to(torch.bfloat16)
    torch.cuda.synchronize(device)
    for _ in range(WARMUP_ITERS):
        _ = torch.nn.functional.linear(x_bf16, w_bf16)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        _ = torch.nn.functional.linear(x_bf16, w_bf16)
    torch.cuda.synchronize(device)
    bf16_ms = (time.perf_counter() - t0) / BENCH_ITERS * 1000.0

    speedup = bf16_ms / int8_ms if int8_ms > 0 else float("inf")

    return Result(
        pattern=pattern,
        shape=(out_dim, in_dim),
        cos_fp=cos_vs_fp,
        cos_bf16=cos_vs_bf16,
        sqnr_db=sqnr_db,
        passed=passed,
        int8_ms=int8_ms,
        bf16_ms=bf16_ms,
        speedup=speedup,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Recalibrate the bf16 skip-list against the fleet's 40:1 int8 cost"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: needs CUDA for dp4a kernels")
        return

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device={device} seed={args.seed}")
    print(f"sqnr_gate threshold: cos >= 0.99")
    print()

    all_results: list[Result] = []

    for pattern, (out_dim, in_dim) in SHAPES.items():
        print(f"\n{'='*72}")
        print(f"Candidate: {pattern}  weight=({out_dim}, {in_dim})")
        print(f"{'='*72}")
        pattern_results: list[Result] = []
        for batch in BATCH_SHAPES:
            for seq in SEQ_LENS:
                r = _measure_candidate(pattern, out_dim, in_dim, batch, seq,
                                        device, torch.bfloat16)
                pattern_results.append(r)
                all_results.append(r)
                tag = "PASS" if r.passed else "FAIL"
                print(
                    f"  B={batch:2d} S={seq:5d}  "
                    f"cos_fp={r.cos_fp:.6f}  cos_bf16={r.cos_bf16:.6f}  "
                    f"SQNR={r.sqnr_db:7.2f} dB  "
                    f"[{tag}]  "
                    f"int8={r.int8_ms:.2f}ms  bf16={r.bf16_ms:.2f}ms  "
                    f"{r.speedup:.1f}x"
                )

        pass_ratio = sum(1 for r in pattern_results if r.passed) / len(pattern_results)
        print(f"  >> pass rate: {pass_ratio*100:.0f}% ({sum(1 for r in pattern_results if r.passed)}/{len(pattern_results)})")

    # Summary table
    print(f"\n{'='*72}")
    print("SUMMARY: per-candidate pass rate + representative speedup")
    print(f"{'='*72}")
    print(f"  {'Candidate':<20} {'Pass Rate':>10} {'Avg Cos(fp)':>12} {'Avg SQNR':>10} {'Speedup':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")
    for pattern in SHAPES:
        pat_results = [r for r in all_results if r.pattern == pattern]
        pass_rate = sum(1 for r in pat_results if r.passed) / len(pat_results) * 100
        avg_cos = sum(r.cos_fp for r in pat_results) / len(pat_results)
        avg_sqnr = sum(r.sqnr_db for r in pat_results) / len(pat_results)
        avg_speedup = sum(r.speedup for r in pat_results) / len(pat_results)
        print(f"  {pattern:<20} {pass_rate:>8.0f}%  {avg_cos:>12.6f}  {avg_sqnr:>8.1f}dB  {avg_speedup:>6.1f}x")

    # Verdict
    print()
    all_pass = sum(1 for r in all_results if r.passed)
    all_total = len(all_results)
    print(f"Overall: {all_pass}/{all_total} tests pass (gate bar: cos >= 0.99)")
    print()
    print("Verdict for _LOADER_SKIP_OVERRIDE update:")
    passing_patterns = set()
    failing_patterns = set()
    for pattern in SHAPES:
        pat_results = [r for r in all_results if r.pattern == pattern]
        pat_pass = sum(1 for r in pat_results if r.passed)
        pat_total = len(pat_results)
        pct = pat_pass / pat_total * 100
        if pct >= 90:
            print(f"  ADD to _LOADER_SKIP_OVERRIDE: {pattern} ({pct:.0f}% pass)")
            passing_patterns.add(pattern)
        else:
            print(f"  KEEP in _SKIP_HINTS:         {pattern} ({pct:.0f}% pass — below threshold)")
            failing_patterns.add(pattern)

    if failing_patterns:
        print()
        print("NEGATIVE RESULT — honest report:")
        for p in failing_patterns:
            print(f"  {p}: does NOT clear the cos>=0.99 bar across all tested activation shapes. "
                  "Staying bf16 for now.")


if __name__ == "__main__":
    main()
