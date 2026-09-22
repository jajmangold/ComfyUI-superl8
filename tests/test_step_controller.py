# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified DiT step-controller.

Pure-Python planning logic — no torch / fni8 / CUDA / ComfyUI needed (tested on CPU).
Asserts the "stacks vs cancels" policy: the temporal-pool guard (few-step regime
drops caching + W4A8) and the disjoint-range guard (cache lives only in the late
int8 phase so its error can't co-align with the early W4A8 error).
"""

from __future__ import annotations

import pytest

from comfyui_superl8.step_controller import (
    MIN_STEPS_FOR_SCHEDULE,
    ModelProfile,
    QualityGate,
    Regime,
    estimate_compute_steps,
    plan_steps,
)


# ---- regime selection -------------------------------------------------------


def test_rich_regime_stacks_all_three():
    p = plan_steps(ModelProfile.qwen_image_edit(), 20)
    assert p.regime is Regime.RICH
    assert p.precision_mode == "step_adaptive"
    assert p.teacache_enabled is True
    assert p.scheduler.startswith("ays")


def test_medium_regime_precision_on_cache_tight():
    p = plan_steps(ModelProfile(arch="flux1", native_steps=12), 12)
    assert p.regime is Regime.MEDIUM
    assert p.precision_mode == "step_adaptive"
    # medium uses a tighter cache threshold than rich (less quality headroom)
    if p.teacache_enabled:
        assert p.teacache_threshold <= 0.06


def test_few_step_regime_is_uniform_int8():
    p = plan_steps(ModelProfile(arch="flux1"), 4)
    assert p.regime is Regime.FEW_STEP
    assert p.precision_mode == "uniform_int8"
    assert p.teacache_enabled is False
    assert p.switch_step is None
    assert p.expected_compute_steps == 4.0


# ---- temporal-pool guard: distillation cancels the other two ----------------


def test_distilled_forces_few_step_even_at_high_target():
    # A distilled checkpoint asked to run 20 steps must NOT stack cache + W4A8.
    p = plan_steps(ModelProfile.zimage_turbo(), 20)
    assert p.regime is Regime.FEW_STEP
    assert p.precision_mode == "uniform_int8"
    assert p.teacache_enabled is False
    # clamps the wasteful extra steps back toward the native count
    assert p.base_steps <= ModelProfile.zimage_turbo().native_steps + 2
    assert any("distilled" in w.lower() or "distilled" in r.lower()
               for w in p.warnings for r in [w]) or any("Distilled" in r for r in p.rationale)


def test_distilled_native_count_kept():
    p = plan_steps(ModelProfile.zimage_turbo(), 8)
    assert p.regime is Regime.FEW_STEP
    assert p.base_steps == 8
    assert p.precision_mode == "uniform_int8"


def test_low_native_steps_implies_distilled_regime():
    p = plan_steps(ModelProfile(arch="flux1", native_steps=6), 6)
    assert p.regime is Regime.FEW_STEP


# ---- disjoint-range guard: cache only in the late int8 phase ----------------


def test_cache_starts_at_precision_switch():
    p = plan_steps(ModelProfile.qwen_image_edit(), 20)
    assert p.teacache_enabled
    # cache must not touch the early W4A8 region -> start == switch_step
    assert p.teacache_start_step == p.switch_step
    assert p.switch_step == 10


def test_cache_and_w4a8_ranges_are_disjoint():
    p = plan_steps(ModelProfile(arch="qwen_image"), 24)
    if p.teacache_enabled and p.switch_step is not None:
        # W4A8 owns [0, switch); cache owns [start, N) with start == switch => disjoint
        assert p.teacache_start_step >= p.switch_step


# ---- quality-gate overrides -------------------------------------------------


def test_forbid_w4a8_gives_uniform_int8():
    p = plan_steps(ModelProfile.qwen_image_edit(), 20, QualityGate(allow_w4a8=False))
    assert p.precision_mode == "uniform_int8"
    assert p.switch_step is None


def test_forbid_cache_disables_teacache():
    p = plan_steps(ModelProfile.qwen_image_edit(), 20, QualityGate(allow_cache=False))
    assert p.teacache_enabled is False
    # precision axis still active (independent axis)
    assert p.precision_mode == "step_adaptive"


# ---- compute-step estimate (sub-linear temporal savings) --------------------


def test_estimate_no_cache_is_identity():
    assert estimate_compute_steps(20, 10, cache_on=False) == 20.0


def test_estimate_cache_only_skips_late_phase():
    est = estimate_compute_steps(20, cache_start_step=10, cache_on=True)
    # can never skip below the early (always-computed) phase
    assert est >= 10.0
    assert est < 20.0


def test_expected_compute_steps_below_base_when_cache_on():
    p = plan_steps(ModelProfile.qwen_image_edit(), 20)
    if p.teacache_enabled:
        assert p.expected_compute_steps < p.base_steps


# ---- video / arch scheduler selection ---------------------------------------


def test_video_ltx_uses_flow_scheduler_and_stacks():
    p = plan_steps(ModelProfile.ltx2(), 24)
    assert p.regime is Regime.RICH
    assert p.scheduler == "ays_flow"
    assert p.precision_mode == "step_adaptive"


def test_distilled_keeps_native_scheduler():
    # distilled checkpoints carry a trained schedule; don't override with AYS
    p = plan_steps(ModelProfile.zimage_turbo(), 8)
    assert p.scheduler == "default"


# ---- input validation -------------------------------------------------------


def test_zero_steps_raises():
    with pytest.raises(ValueError):
        plan_steps(ModelProfile(arch="flux1"), 0)


def test_min_steps_boundary():
    # exactly MIN_STEPS_FOR_SCHEDULE is the first non-few-step budget
    p = plan_steps(ModelProfile(arch="flux1", native_steps=MIN_STEPS_FOR_SCHEDULE),
                   MIN_STEPS_FOR_SCHEDULE)
    assert p.regime in (Regime.MEDIUM, Regime.RICH)
    assert p.precision_mode == "step_adaptive"


# ---- plan is serializable / reportable --------------------------------------


def test_summary_and_repr():
    p = plan_steps(ModelProfile.qwen_image_edit(), 20)
    s = p.summary()
    assert set(s) >= {"regime", "base_steps", "scheduler", "teacache", "precision"}
    assert "StepPlan" in repr(p)
    assert p.rationale  # non-empty reasoning
