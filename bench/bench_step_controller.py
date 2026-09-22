#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the unified step-controller: do the three techniques STACK or CANCEL?

For a given DiT this runs each stack configuration and reports wall time, effective
DiT-forward count (measured skip fraction), and final-latent quality (cos / SQNR vs an
fp reference), so the "stacks vs cancels" matrix in
``docs/step-controller-stacks-vs-cancels.md`` is backed by numbers, not just analysis.

Configs compared:
  1. uniform-int8            (baseline — cost axis only)
  2. teacache-only          (temporal axis only)
  3. step-adaptive-only     (cost axis, W4A8 early)
  4. teacache + adaptive    (controller RICH plan — disjoint-range guard)
  5. controller-auto        (whatever plan_steps() picks for this model)

GATED: needs GPU 10 (deployment-representative), the fni8 wheel, a real .fni8 DiT, and
PRs #94/#109/AYS merged. Disk-polite: prefer Z-Image-Turbo (5.9 GB).

Usage:
    CUDA_VISIBLE_DEVICES=10 python3 bench/bench_step_controller.py \\
        --unet Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8 --arch zimage --steps 8
    CUDA_VISIBLE_DEVICES=10 python3 bench/bench_step_controller.py \\
        --unet <qwen-image>.fni8 --arch qwen_image --steps 20
"""
from __future__ import annotations

import argparse
import time

from comfyui_superl8.step_controller import ModelProfile, QualityGate, apply_plan, plan_steps


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True)
    ap.add_argument("--arch", default="zimage")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--distilled", action="store_true")
    ap.add_argument("--hw", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _configs(prof: ModelProfile, steps: int):
    """Return (label -> StepPlan) for the matrix rows."""
    auto = plan_steps(prof, steps)
    int8_only = plan_steps(prof, steps, QualityGate(allow_w4a8=False, allow_cache=False))
    cache_only = plan_steps(prof, steps, QualityGate(allow_w4a8=False, allow_cache=True))
    adaptive_only = plan_steps(prof, steps, QualityGate(allow_w4a8=True, allow_cache=False))
    full = plan_steps(prof, steps, QualityGate(allow_w4a8=True, allow_cache=True))
    return {
        "uniform-int8": int8_only,
        "teacache-only": cache_only,
        "step-adaptive-only": adaptive_only,
        "teacache+adaptive": full,
        "controller-auto": auto,
    }


def main():
    args = parse_args()
    import torch  # noqa: F401  (import here so --help works without torch)

    from comfyui_superl8.gate import is_sm70

    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU — pin GPU 10 via CUDA_VISIBLE_DEVICES")
        return

    prof = ModelProfile(
        arch=args.arch,
        distilled=args.distilled,
        is_video=args.arch in ("ltx", "wan"),
        memory_bound=args.arch in ("ltx", "wan"),
    )
    configs = _configs(prof, args.steps)

    print(f"\n=== step-controller matrix: {args.arch} steps={args.steps} distilled={args.distilled} ===")
    print(f"{'config':>20}  {'plan':<64}")
    for label, plan in configs.items():
        print(f"{label:>20}  {plan!r}")
    print()

    # --- e2e run (needs the loader + a real model) -------------------------
    try:
        from comfyui_superl8.nodes import UnetLoaderFNI8
    except Exception as e:
        print(f"SKIP e2e: loader import failed ({e!r}); plan table above is still valid.")
        return

    print("Loading model ...")
    (base_model,) = UnetLoaderFNI8().load(args.unet, arch=args.arch)

    print(f"{'config':>20}  {'wall_s':>8}  {'eff_fwd':>8}  {'cos':>7}  {'sqnr_dB':>8}")
    print("-" * 60)
    ref_latent = None
    for label, plan in configs.items():
        patched, handles = apply_plan(base_model, plan)
        t0 = time.perf_counter()
        latent = _denoise(patched, args, plan)  # user-supplied sampler call
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        ctrl = handles.get("cache_controller")
        eff = ctrl.total_steps - ctrl.cache_hits if ctrl else plan.base_steps
        if ref_latent is None:
            ref_latent = latent
            cos = sqnr = float("nan")
        else:
            from comfyui_superl8.metrics import cosine, sqnr as _sqnr

            cos, sqnr = cosine(latent, ref_latent), _sqnr(latent, ref_latent)
        print(f"{label:>20}  {wall:>8.2f}  {eff:>8.1f}  {cos:>7.4f}  {sqnr:>8.1f}")


def _denoise(model, args, plan):
    """Run a full denoise with `plan.base_steps`. Wire to comfy.sample.sample /
    the e2e harness in tests/e2e/_common.py."""
    raise NotImplementedError(
        "Wire to comfy.sample.sample(model, ..., steps=plan.base_steps, "
        "scheduler=plan.scheduler) — see tests/e2e/test_full_pipeline_zimage.py."
    )


if __name__ == "__main__":
    main()
