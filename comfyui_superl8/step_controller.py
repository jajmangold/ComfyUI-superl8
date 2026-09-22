# SPDX-License-Identifier: Apache-2.0
"""Unified DiT step-controller — make the three step-cutting techniques STACK, not CANCEL.

Three techniques all spend the *same* denoise-step budget, and they were built as
separate, uncoordinated PRs:

  * **TeaCache / TaylorSeer** (``cache.py``, PR #94) — skips whole DiT forwards on
    steps where features barely change.  1.5–3.5× on Flux-class *at high step counts*.
  * **Step-adaptive precision** (``precision_schedule.py``, PR #109 / issue #59) —
    W4A8 on the coarse early steps, tightening to int8 on the fine late steps.
    Requires ≥8 steps (``MIN_STEPS_FOR_SCHEDULE``).
  * **AYS / Lightning / distillation** (``ays.py``, issue #62; Turbo/distilled
    checkpoints) — trains/reshapes the schedule down to 4–8 steps.

The load-bearing insight is that these live on **two independent savings axes**:

  1. **Temporal axis** — *which / how many* steps actually run.  TeaCache, AYS, and
     distillation ALL draw from one finite "cross-step redundancy" pool.  They
     **compete**: their speedups are *sub-multiplicative*, and at a low step budget
     the pool is already empty (a distilled 4-step model has nothing for TeaCache to
     skip and no coarse phase for W4A8).  PR #94's own node warns: *"Do NOT stack
     multiple cache nodes — temporal caches harvest the same cross-step redundancy."*
     Distillation and AYS are exactly "other caches" in that sense.
  2. **Cost axis** — *how cheap* each executed step is.  int8 dp4a (always on) and
     W4A8 (step-adaptive) draw from a separate per-step-compute pool.  This axis
     **multiplies cleanly** with the temporal axis — *provided* enough real steps
     remain to run a schedule.

The controller reasons about the shared budget and emits ONE plan (the single knob):
it picks a step count, a scheduler, whether/where to cache, and a precision schedule,
with two hard guards against the anti-synergies:

  * **Disjoint-range guard** (TeaCache × step-adaptive): W4A8 owns the *early* noisy
    steps (always computed — features move too fast to skip); TeaCache owns the *late*
    refined steps (int8, where adjacent steps are redundant).  ``teacache_start_step
    = switch_step`` puts them on non-overlapping step ranges, so the quant error and
    the stale-cache error never land on the same latent (the non-co-alignment that
    issue #53 requires).
  * **Temporal-pool guard** (TeaCache × / step-adaptive × distillation): below the
    step thresholds the pool is spent — the controller drops caching and the
    precision schedule and runs uniform int8, instead of stacking things that cancel.

Pure-Python planning (no torch / fni8 / ComfyUI needed — unit-tested on CPU).  The
runtime wiring (``apply``) lazily imports ``cache``, ``precision_schedule`` and
``ays`` so the controller degrades gracefully before those PRs land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Mirror precision_schedule.MIN_STEPS_FOR_SCHEDULE — a schedule needs a coarse and a
# fine phase; distilled few-step models lack both (issue #59, sota DiT-tricks #5).
MIN_STEPS_FOR_SCHEDULE = 8

# Below this the cross-step redundancy pool is too thin for temporal caching to earn
# its keep: the first step always computes, adjacent steps differ a lot, and the
# cache SQNR gate rejects most skips.  Distilled/turbo checkpoints sit here by design.
MIN_STEPS_FOR_CACHE = 10

# Default TeaCache metric threshold (matches cache.TimestepCacheController default).
DEFAULT_CACHE_THRESHOLD = 0.1


class Regime(Enum):
    """Which step-budget regime the plan falls into."""

    FEW_STEP = "few_step"  # distilled or N < MIN_STEPS_FOR_SCHEDULE: cost axis only
    MEDIUM = "medium"  # MIN_STEPS_FOR_SCHEDULE ≤ N < 16: precision + light caching
    RICH = "rich"  # N ≥ 16: all three, disjoint ranges


@dataclass
class ModelProfile:
    """What the controller needs to know about the DiT / checkpoint.

    ``distilled`` is the single most important field: a Lightning/Turbo/DMD checkpoint
    has already spent the temporal-redundancy pool (and reshaped its activations), so
    the controller must NOT stack TeaCache or a W4A8 schedule on top of it.
    """

    arch: str
    distilled: bool = False
    native_steps: int | None = None  # sampler steps the checkpoint is trained for
    is_video: bool = False
    memory_bound: bool = False  # weight-bytes dominate (video / multi-GPU split)

    # Convenience constructors for the three shipped DiTs.
    @classmethod
    def zimage_turbo(cls) -> "ModelProfile":
        # Tongyi-MAI Z-Image-Turbo — distilled, ~8 native steps, 5.9 GB.
        return cls(arch="zimage", distilled=True, native_steps=8)

    @classmethod
    def qwen_image_edit(cls) -> "ModelProfile":
        # Qwen-Image-Edit — full (non-distilled) image DiT, ~20 steps.
        return cls(arch="qwen_image", distilled=False, native_steps=20)

    @classmethod
    def ltx2(cls) -> "ModelProfile":
        # LTX-2.3 video DiT — 23.5 GB, PP-split, flow-matching, ~20–30 steps.
        return cls(
            arch="ltx", distilled=False, native_steps=24, is_video=True, memory_bound=True
        )


@dataclass
class QualityGate:
    """Accuracy floors the plan must respect (defaults mirror ``metrics`` / #109)."""

    sqnr_floor_db: float = 20.0
    cos_floor: float = 0.99
    transition_sqnr_floor_db: float = 18.0
    transition_cos_floor: float = 0.985
    # A quality-first caller can forbid the aggressive knobs entirely.
    allow_w4a8: bool = True
    allow_cache: bool = True


@dataclass
class StepPlan:
    """The single emitted plan — feeds the sampler, the cache, and the precision hook."""

    regime: Regime
    base_steps: int
    scheduler: str  # "default" | "ays" | "ays_flow"

    # Temporal axis (caching)
    teacache_enabled: bool
    teacache_strategy: str  # "taylor" | "tea"
    teacache_threshold: float
    teacache_start_step: int

    # Cost axis (precision)
    precision_mode: str  # "uniform_int8" | "step_adaptive"
    switch_step: int | None  # W4A8 → int8 boundary (step_adaptive only)

    expected_compute_steps: float  # steps after estimated cache skips
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "regime": self.regime.value,
            "base_steps": self.base_steps,
            "scheduler": self.scheduler,
            "teacache": (
                f"{self.teacache_strategy}@{self.teacache_threshold}"
                f"(start={self.teacache_start_step})"
                if self.teacache_enabled
                else "off"
            ),
            "precision": (
                f"step_adaptive(switch={self.switch_step})"
                if self.precision_mode == "step_adaptive"
                else "uniform_int8"
            ),
            "expected_compute_steps": round(self.expected_compute_steps, 2),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"StepPlan(regime={s['regime']}, steps={s['base_steps']}, "
            f"sched={s['scheduler']}, cache={s['teacache']}, "
            f"precision={s['precision']}, ~compute={s['expected_compute_steps']})"
        )


# --- caching-yield model -----------------------------------------------------
#
# Fraction of the LATE (post-switch) steps that TeaCache can safely skip, as a
# function of how many late steps exist.  Empirically the skip ratio saturates:
# TeaCache reports ~40–60% skips on 20–50-step Flux runs, and ~0% once you drop
# into the few-step regime.  We model the *late-phase* skip fraction only, because
# the controller never lets the cache touch the early (W4A8) phase.


def _cache_skip_fraction(late_steps: int) -> float:
    """Estimated fraction of late steps TeaCache can skip (0..~0.55)."""
    if late_steps < 4:
        return 0.0
    # smooth ramp: ~0.15 at 4 late steps → ~0.5 asymptote
    frac = 0.55 * (1.0 - 1.0 / (1.0 + 0.25 * (late_steps - 3)))
    return max(0.0, min(0.55, frac))


def estimate_compute_steps(plan_steps: int, cache_start_step: int, cache_on: bool) -> float:
    """Expected number of *real* DiT forwards after cache skips.

    Only the cacheable region ``[cache_start_step, N)`` can be skipped; every step
    before ``cache_start_step`` (the early W4A8 / always-compute phase) runs in full.
    """
    if not cache_on:
        return float(plan_steps)
    late = plan_steps - cache_start_step
    skipped = late * _cache_skip_fraction(late)
    return float(plan_steps - skipped)


# --- the planner -------------------------------------------------------------


def plan_steps(
    model: ModelProfile,
    target_step_budget: int,
    quality: QualityGate | None = None,
) -> StepPlan:
    """Pick the optimal stack for ``target_step_budget`` on ``model``.

    This is the single knob.  ``target_step_budget`` is the number of denoise steps
    the user asks for (or the checkpoint's native count for distilled models); the
    controller decides the scheduler, caching and precision schedule around it,
    guarding against the temporal-pool and error-co-alignment anti-synergies.
    """
    q = quality or QualityGate()
    N = int(target_step_budget)
    rationale: list[str] = []
    warnings: list[str] = []

    if N < 1:
        raise ValueError(f"target_step_budget must be ≥1 (got {N})")

    # A distilled/turbo checkpoint has already cashed in the whole temporal pool and
    # reshaped its activations — force the few-step regime regardless of N.
    distilled = model.distilled or (model.native_steps is not None and model.native_steps < MIN_STEPS_FOR_SCHEDULE)
    effective_N = N
    if distilled and model.native_steps:
        # Never run a distilled model far off its trained step count.
        if N > model.native_steps + 2:
            warnings.append(
                f"target {N} steps >> distilled native {model.native_steps}; "
                f"clamping to {model.native_steps} (extra steps waste compute on a "
                f"distilled schedule)."
            )
            effective_N = model.native_steps
        N = effective_N

    sched = _pick_scheduler(model)

    # ---- Regime selection -------------------------------------------------
    if distilled or N < MIN_STEPS_FOR_SCHEDULE:
        regime = Regime.FEW_STEP
    elif N < 16:
        regime = Regime.MEDIUM
    else:
        regime = Regime.RICH

    # ======================================================================
    # FEW-STEP: the temporal pool is spent.  Cost axis ONLY (uniform int8).
    # ======================================================================
    if regime is Regime.FEW_STEP:
        if distilled:
            rationale.append(
                f"Distilled/turbo checkpoint (native {model.native_steps}): the "
                f"temporal-redundancy pool is already spent — TeaCache would find "
                f"nothing to skip and W4A8 has no coarse phase to exploit. Stacking "
                f"either would CANCEL (or hurt quality), so: uniform int8 throughout."
            )
        else:
            rationale.append(
                f"N={N} < {MIN_STEPS_FOR_SCHEDULE}: too few steps for a W4A8→int8 "
                f"schedule (precision_schedule would raise) and too few for the cache "
                f"to earn skips. Uniform int8 throughout."
            )
        # AYS still legal & non-distilling — but at few steps its trimming overlaps
        # the (empty) pool; keep the default schedule to avoid over-cutting.
        return StepPlan(
            regime=regime,
            base_steps=N,
            scheduler="default",
            teacache_enabled=False,
            teacache_strategy="taylor",
            teacache_threshold=DEFAULT_CACHE_THRESHOLD,
            teacache_start_step=0,
            precision_mode="uniform_int8",
            switch_step=None,
            expected_compute_steps=float(N),
            rationale=rationale,
            warnings=warnings,
        )

    # ======================================================================
    # MEDIUM / RICH: two independent axes, coordinated on disjoint ranges.
    # ======================================================================
    # Cost axis: W4A8 owns the early noisy steps (quant error masked by noise,
    # issue #59), int8 owns the late fine steps.
    precision_mode = "uniform_int8"
    switch_step: int | None = None
    if q.allow_w4a8:
        switch_step = max(1, N // 2)
        precision_mode = "step_adaptive"
        rationale.append(
            f"Cost axis: step-adaptive precision, W4A8 on steps [0,{switch_step}) "
            f"(coarse/high-noise — quant error is masked), int8 on [{switch_step},{N}). "
            f"Transition gated at SQNR≥{q.transition_sqnr_floor_db}dB / "
            f"cos≥{q.transition_cos_floor}."
        )
    else:
        rationale.append("Cost axis: uniform int8 (quality gate forbids W4A8).")

    # Temporal axis: TeaCache, but ONLY in the late int8 phase (disjoint-range guard).
    cache_on = False
    cache_start = 0
    cache_threshold = DEFAULT_CACHE_THRESHOLD
    late_steps = N - (switch_step or 0)
    if q.allow_cache and late_steps >= (MIN_STEPS_FOR_CACHE - MIN_STEPS_FOR_SCHEDULE + 2):
        # Enough late steps to skip. Start the cache at the precision boundary so the
        # W4A8 early phase is never skipped and cache-error / quant-error never
        # co-align on the same latent (issue #53 non-compounding requirement).
        cache_on = True
        cache_start = switch_step or max(1, N // 2)
        # RICH regime tolerates the default threshold; MEDIUM uses a tighter one
        # because it has fewer late steps and less quality headroom.
        cache_threshold = DEFAULT_CACHE_THRESHOLD if regime is Regime.RICH else 0.06
        rationale.append(
            f"Temporal axis: TeaCache/TaylorSeer active only on the late int8 phase "
            f"[{cache_start},{N}) — start_step={cache_start} == switch_step keeps the "
            f"skip region DISJOINT from the W4A8 region, so the two error sources stay "
            f"independent and STACK (they do not co-align → do not compound)."
        )
        if regime is Regime.RICH and sched == "default":
            # AYS improves step placement at fixed N (quality headroom that lets the
            # cache skip more safely). It reshapes only the schedule, not activations,
            # so it's int8-safe (unlike distillation).
            pass
    else:
        rationale.append(
            f"Temporal axis: TeaCache OFF — only {late_steps} late steps, below the "
            f"skip-worthiness floor. The precision schedule already covers this budget."
        )

    if cache_on and sched.startswith("ays"):
        warnings.append(
            "AYS scheduler + TeaCache both harvest the temporal pool: combined "
            "speedup is SUB-MULTIPLICATIVE, not the product of the two. AYS is kept "
            "for step-placement quality (fixed N), not for extra step-cutting."
        )

    expected = estimate_compute_steps(N, cache_start, cache_on)

    return StepPlan(
        regime=regime,
        base_steps=N,
        scheduler=sched,
        teacache_enabled=cache_on,
        teacache_strategy="taylor",
        teacache_threshold=cache_threshold,
        teacache_start_step=cache_start,
        precision_mode=precision_mode,
        switch_step=switch_step,
        expected_compute_steps=expected,
        rationale=rationale,
        warnings=warnings,
    )


def _pick_scheduler(model: ModelProfile) -> str:
    """Choose the AYS variant for the arch (flow-matching vs discrete), or default."""
    if model.distilled:
        # Distilled checkpoints carry their own trained schedule — don't override.
        return "default"
    if model.arch in ("flux1", "flux", "flux2", "sd3", "qwen_image", "ltx", "wan", "zimage"):
        # These are flow-matching DiTs.
        return "ays_flow"
    return "ays"


# --- runtime wiring (lazy; needs cache.py / precision_schedule.py / ays.py) ---


def apply_plan(model, plan: StepPlan):
    """Wire ``plan`` onto a ComfyUI ``ModelPatcher``.

    Returns ``(patched_model, handles)`` where ``handles`` is a dict with any of
    ``{"cache_controller", "precision_schedule", "precision_hook"}`` that were wired.
    Lazily imports the sibling modules so this is a no-op-friendly integration point
    before PRs #94 / #109 / AYS land — missing pieces are recorded in
    ``handles["skipped"]`` instead of raising.
    """
    handles: dict = {"skipped": []}
    patched = model

    # Precision schedule (cost axis).
    if plan.precision_mode == "step_adaptive" and plan.switch_step is not None:
        try:
            from .precision_schedule import PrecisionSchedule, make_precision_schedule_hook

            sched = PrecisionSchedule(total_steps=plan.base_steps, switch_step=plan.switch_step)
            hook = make_precision_schedule_hook(sched)
            patched = _attach_pre_cfg_hook(patched, hook)
            handles["precision_schedule"] = sched
            handles["precision_hook"] = hook
        except Exception as e:  # pragma: no cover - depends on unmerged PR
            handles["skipped"].append(f"precision_schedule: {e!r}")

    # TeaCache (temporal axis) — patched last so it wraps apply_model outermost.
    if plan.teacache_enabled:
        try:
            from .cache import patch_model

            patched, ctrl = patch_model(
                patched,
                strategy=plan.teacache_strategy,
                threshold=plan.teacache_threshold,
                start_step=plan.teacache_start_step,
            )
            handles["cache_controller"] = ctrl
        except Exception as e:  # pragma: no cover - depends on unmerged PR
            handles["skipped"].append(f"cache: {e!r}")

    return patched, handles


def _attach_pre_cfg_hook(model, hook):
    """Register a per-step ``sampler_pre_cfg_function`` hook on a cloned patcher."""
    m = model.clone()
    opts = dict(getattr(m, "model_options", {}) or {})
    existing = list(opts.get("sampler_pre_cfg_function", []))
    existing.append(hook)
    opts["sampler_pre_cfg_function"] = existing
    m.model_options = opts
    return m


def apply_scheduler(plan: StepPlan):
    """Return the sigma-schedule callable for ``plan.scheduler`` (or None for default).

    Importing ``ays`` registers its schedulers into ComfyUI's ``SCHEDULER_HANDLERS``;
    this just resolves the generator so a caller can build sigmas directly.
    """
    if plan.scheduler == "default":
        return None
    try:
        from . import ays  # noqa: F401  (import registers the schedulers)

        return {
            "ays": getattr(ays, "ays_scheduler_ms", None),
            "ays_flow": getattr(ays, "ays_flow_scheduler_ms", None),
        }.get(plan.scheduler)
    except Exception:  # pragma: no cover - depends on unmerged AYS work
        return None
