# DiT step-controller — "what stacks vs what cancels"

Three techniques all spend the **same denoise-step budget** and were built as
uncoordinated PRs:

| technique | PR / issue | what it cuts | module |
|---|---|---|---|
| **TeaCache / TaylorSeer** | #94 (#53) | *number of steps* — skips whole DiT forwards where features barely change | `cache.py` |
| **Step-adaptive precision** | #109 (#59) | *cost per step* — W4A8 on early steps → int8 on late | `precision_schedule.py` |
| **AYS / Lightning / distillation** | #93 (#62) / Turbo ckpts | *number of steps* — reshapes/distills the schedule to 4–8 steps | `ays.py`, distilled weights |

## The one idea: two axes, one shared pool

Savings come from **two independent axes**:

- **Temporal axis** — *which / how many* steps run. **TeaCache, AYS and distillation
  all draw from a single finite "cross-step redundancy" pool.** They **compete**:
  combined speedup is *sub-multiplicative*, and once the pool is spent (a distilled
  4-step model) there is nothing left for the others to take. PR #94's own node says
  it outright: *"Do NOT stack multiple cache nodes — temporal caches harvest the same
  cross-step redundancy."* Distillation and AYS are "other caches" in that exact sense.
- **Cost axis** — *how cheap* each executed step is. **int8 dp4a (always) and W4A8
  (step-adaptive)** draw from a separate per-step-compute pool. This axis **multiplies
  cleanly** with the temporal axis — *provided enough real steps remain to run a
  schedule* (≥8, `MIN_STEPS_FOR_SCHEDULE`).

So the rule is: **stack ONE temporal technique with the cost axis; never stack two
temporal techniques expecting the product.**

## The stacks-vs-cancels matrix

Verdict as a function of step budget `N` (and whether the checkpoint is distilled).

### TeaCache × Lightning/distillation — **CANCEL** (same pool)

| step budget | verdict | why |
|---|---|---|
| distilled 4-step | **CANCEL (hard)** | per-step feature delta is huge → cache metric always trips → ~0 skips; the ones it does skip fail the 20 dB cache-SQNR gate. Distillation already took the pool. |
| distilled 8-step | **CANCEL** | ≤1 skip, not worth the metric overhead; quality risk. |
| non-distilled 12 | **weak COMPOSE** | a few late-phase skips; sub-multiplicative. |
| non-distilled ≥16 | **COMPOSE** | 40–60% late-phase skips — TeaCache's home turf. |

**Controller action:** if `distilled or N<10`, TeaCache **OFF**. Never run a distilled
checkpoint *and* TeaCache.

### Step-adaptive precision × Lightning/distillation — **CANCEL / CONFLICT**

| step budget | verdict | why |
|---|---|---|
| distilled or N<8 | **CONFLICT (hard)** | `PrecisionSchedule` *raises* below 8 steps; distilled models have **no coarse phase** for W4A8 and their **reshaped activations** break int4 calibration (issue #59, DiT-tricks #5). |
| non-distilled 8–15 | **COMPOSE** | switch at N/2; W4A8 early, int8 late. Orthogonal axis. |
| non-distilled ≥16 | **COMPOSE (best)** | wide coarse phase → W4A8 pays maximally. |

**Controller action:** distilled → **uniform int8**. Non-distilled ≥8 → `switch=N//2`.

### TeaCache × Step-adaptive — **COMPOSE (with guard)** — orthogonal axes

These are the two techniques that *should* stack (temporal × cost), but naively they
share the same *steps* and their errors can **co-align** — issue #53 requires the int8
error and the cache-skip error stay **independent so they don't compound**.

| step budget | verdict | guard |
|---|---|---|
| N<8 | degenerates to few-step (both effectively off) | — |
| 8–15 | **COMPOSE** | disjoint ranges; tighter cache threshold (0.06) |
| ≥16 | **COMPOSE (full stack)** | disjoint ranges; default threshold (0.1) |

**The guard (the core coordination):** partition the step axis into **disjoint ranges**
that each technique is naturally suited to, so no step is touched by both error sources:

```
step:      0 ........ switch_step ........ N
precision: |   W4A8 (coarse, noise-masked)  |   int8 (fine)      |
cache:     |     always compute (skip unsafe)|  TeaCache skips    |
                                             ^
                              teacache_start_step == switch_step
```

- **Early (high-noise):** W4A8 precision, **always computed** — features move too fast
  to skip anyway, and W4A8 error is masked by diffusion noise (issue #59).
- **Late (low-noise):** int8 precision, **TeaCache skips** the redundant steps — where
  adjacent outputs are nearly identical and precision is already highest.

The two regions are disjoint, so W4A8 quant error and stale-cache error **never land on
the same latent** → they stay independent → they **stack instead of compound**. This is
exactly the non-co-alignment issue #53 asks validation to confirm.

## Speedup is not the product (worked estimate)

For a non-distilled 20-step DiT (Qwen-Image-Edit), controller plan
`RICH: switch=10, cache start=10, taylor@0.1`:

- Cost axis: 10 steps at W4A8 (~2× cheaper linears) + 10 at int8.
- Temporal axis: ~35% of the 10 **late** steps skipped → ~3.5 forwards saved →
  **~16.5 effective forwards** (not 20).
- **Do not** multiply "TeaCache 2.25×" by "AYS 1.6×": both pull the temporal pool.
  The controller uses AYS only for *step-placement quality at fixed N*, and warns that
  the combination is sub-multiplicative.

`estimate_compute_steps()` encodes the late-phase-only skip model; the actual skip
fraction is filled by `bench/bench_step_controller.py` on GPU 10.

## Recommended step-config per shipped DiT

| DiT | profile | recommended plan | rationale |
|---|---|---|---|
| **Z-Image-Turbo** (5.9 GB) | distilled, ~8 native | **8 steps, default schedule, uniform int8, no cache, no W4A8** | distilled → temporal pool already spent; stacking cancels/hurts. Cost axis (int8) only. This is already near the step floor. |
| **Qwen-Image-Edit** | full, ~20 steps | **20 steps + `ays_flow` + step-adaptive(switch=10) + TeaCache(taylor@0.1, start=10)** | RICH: full stack on disjoint ranges. W4A8 early / int8+cache late. |
| **LTX-2.3 video** (23.5 GB, PP-split) | full, ~24 steps, memory-bound | **24 steps + `ays_flow` + step-adaptive(switch=12) + TeaCache(taylor@0.1, start=12)** | RICH; W4A8 doubly valuable (halves weight bytes on a memory-bound, PCIe-1.0-x1 PP split); strong video temporal redundancy feeds late-phase cache. If a *distilled* LTX variant is used, drop to uniform int8. |

## Validation status

- **Design + matrix + planner**: complete, unit-tested on CPU (`tests/test_step_controller.py`).
- **End-to-end speedup/quality numbers**: **GATED** on disk availability + GPU 10 +
  the three source PRs landing (#94, #109, AYS). Run
  `bench/bench_step_controller.py --arch zimage --unet Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8`
  on GPU 10 to fill the measured skip-fraction and final-latent cos/SQNR per config.
  Z-Image (small, 5.9 GB) is the disk-polite validation target.
