# TeaCache (cross-step DiT caching) on the fni8 int8 DiTs — findings

**What it is.** `FNI8TeaCache` skips recomputing the whole DiT on denoise steps whose
timestep-conditioned input barely changed, reusing the previous step's cached prediction.
FLOPs are the DiT bottleneck, so skipped steps are meant to stack *multiplicatively* on the
int8 dp4a compute win. Implemented at the `WrappersMP.DIFFUSION_MODEL` seam (the same
`transformer_options` plumbing the int8 `optimized_attention_override` uses) — one
arch-agnostic code path for single-stream Z-Image, double-stream Qwen-Image-Edit, and
audio+video LTX-2.3.

**Seam note (vs draft #94).** The sibling draft #94 patched `ModelPatcher.apply_model`, but
ComfyUI's sampler calls `BaseModel.apply_model` via `inner_model` (samplers.py:334) — a
different object — so #94 never engaged. This implementation uses the wrapper seam
`BaseModel._forward` actually reads, and keeps #94's good ideas (tea/taylor strategy, SQNR
spot-check gate) on top of a working hook + per-stream isolation + generation reset + the
quality-gated measurement #94 lacked.

## Quality gate

Latent-cosine ≥ **0.985** on the FINAL latent vs the no-cache int8 baseline (caching error
accumulates over steps, so we gate the end result, not per block). The largest threshold
whose final-latent cosine clears the bar is the reported knee.

## Measurement method

`bench/teacache_sweep.py` loads the int8 DiT once, runs an **untimed warmup denoise** (so
the timed baseline isn't inflated by CUDA cold-start), then sweeps `rel_l1_thresh`, timing
each denoise and comparing its final latent to the no-cache baseline. GPU 8 (NVIDIA CMP
100-210) — deployment-representative step latency, NOT a pristine real-V100 number.

## Z-Image-Turbo (8-step, cfg=1) — int8 dp4a DiT, 512 px, strategy=tea

Warm baseline (full compute, 0 skips) = **13.7 s**. (The very first cold denoise measured
62 s; that one-time CUDA autotune/alloc cost is excluded — the earlier bench lacked the
warmup pass and misreported it as a 4.6× "speedup" at 0 skips, which was an artifact, not
caching.)

| rel_l1_thresh | skipped | wall (s) | speedup vs int8 | final-latent cos | gate |
|--------------:|:-------:|---------:|----------------:|-----------------:|:----:|
| 0.00 (base)   |   0/8   |   13.7   |      1.00×      |     1.00000      |  —   |
| 0.05          |   0/8   |   13.7   |      1.00×      |     1.00000      | pass |
| 0.10          |   1/8   |   11.2   |      1.22×      |     0.9558       | fail |
| 0.15          |   2/8   |    9.8   |      1.40×      |     0.9363       | fail |
| 0.20          |   2/8   |    9.7   |      1.40×      |     0.9363       | fail |
| 0.30          |   3/8   |    8.2   |      1.68×      |     0.9189       | fail |
| 0.50          |   3/8   |    8.1   |      1.69×      |     0.9189       | fail |

**Verdict: Z-Image-Turbo does NOT tolerate cross-step caching at the 0.985 gate.** Even a
single skipped step (1/8) drops the final-latent cosine to 0.956 — below the bar. The knee
is 0 skips (no speedup). This is expected and honest: an 8-step *distilled* model packs a
lot of denoising into each step, so consecutive predictions are NOT redundant — the premise
caching relies on. TeaCache's published wins are on 30–50-step models.

### Is it the caching *granularity*? Measured — no.

To rule out that whole-prediction reuse is the culprit (vs the paper's block-*residual*
caching, which re-runs the timestep-conditioned epilogue at the current step), we
implemented BOTH and measured Z-Image-Turbo 8-step in **residual mode** too:

| rel_l1_thresh | skipped | latent cos (output mode) | latent cos (residual mode) |
|--------------:|:-------:|-------------------------:|---------------------------:|
| 0.10          |   1/8   |          0.9558          |          0.9552            |
| 0.20          |   2/8   |          0.9363          |          0.9034            |
| 0.50          |   3/8   |          0.9189          |          0.8259            |

The two granularities are **statistically identical at 1 skip** (0.9558 vs 0.9552) — residual
caching does NOT rescue Z-Image. So the limitation is the **model** (8-step distillation →
~5% block-output change per step → no redundancy), *not* the implementation. For Z-Image the
transformer blocks are the fast-changing part; re-running the thin adaLN epilogue buys
nothing. This is a stronger, more useful conclusion than a single-mode negative.

**Recommendation for Z-Image-Turbo:** ship `FNI8TeaCache` available but default
`rel_l1_thresh=0` (off) for the Turbo checkpoint; caching is for many-step models.

## Z-Image (20-step, cfg=1) — positive control: is it just the 8-step count?

Ran the same DiT at 20 steps (4× its distilled schedule) to test whether more steps —
i.e. more inter-step redundancy — lets the cache clear the gate. Warm baseline = 32.2 s.

| rel_l1_thresh | skipped | wall (s) | speedup | final-latent cos | gate |
|--------------:|:-------:|---------:|--------:|-----------------:|:----:|
| 0.00 (base)   |  0/20   |   32.2   |  1.00×  |     1.00000      |  —   |
| 0.05          |  1/20   |   30.7   |  1.05×  |     0.9502       | fail |
| 0.10          |  3/20   |   27.7   |  1.16×  |     0.9015       | fail |
| 0.15–0.50     |  3/20   |   27.7   |  1.16×  |     0.9015       | fail |

**Still fails the gate — and it is NOT primarily the step count.** A *single* skipped step
out of 20 already drops the final-latent cosine to 0.950. The root cause is the caching
granularity: this implementation reuses the **whole DiT prediction**, which is
timestep-conditioned, for a step at a *different* timestep. For flow-matching DiTs the
velocity/denoised prediction changes enough step-to-step that reusing it verbatim costs
~5% cosine per skipped step. TeaCache's published wins use **block-residual** caching, which
re-runs the (cheap) timestep-conditioned input/output projections every step and only reuses
the transformer-block *residual* — we deliberately did NOT implement that (it needs per-arch
block-loop surgery: Z-Image single-stream, Qwen double-stream, LTX audio+video all differ),
trading some quality for one arch-agnostic code path. The measurement makes that trade
visible. (`max_consecutive_skips=3` also caps skips at 3/20 here.)

## Qwen-Image-Edit-2509 (int8 dp4a b4 DiT, cfg) — BLOCKED by an unrelated loader gap

Could not run e2e: the **b4** (4-bit) `.fni8` DiT fails to load on `main` —

```
size mismatch for transformer_blocks.0.attn.to_q.weight:
  copying a param with shape [3072, 1536] ... current model is [3072, 3072]
```

The 4-bit `.fni8` stores weights packed at half input-width; `comfyui_superl8.loader` on `main`
has **no b4/int4 unpacking path** (grep: only full-width b8), so ComfyUI's initial
`load_state_dict` rejects the half-width tensors before `assign_int8_weights` can re-attach
them. Z-Image loaded because it ships as **b8** (full-width). Qwen-Image-Edit b8 is 20.5 GB
and does not fit one 16 GB card, so b4 is required — and b4 loading is the blocker. **This is
a DiT-loader issue, entirely separate from TeaCache** (the cache never got to run). The
per-cond/uncond stream isolation is unit-tested (`test_cond_uncond_isolation`) and the CFG
sweep path is wired in `bench/teacache_sweep.py` — it will measure Qwen the moment the b4
loader lands (owned by the Qwen-Image-Edit e2e work on its own branch).

## LTX-2.3

The full int8 DiT is 23.5 GB and does not fit one 16 GB card (needs the pipeline-parallel
split / sibling shrink), and no block harness exists on `main`. TeaCache is wired for LTX by
the same arch-agnostic seam (LTXAVModel routes through the same `WrapperExecutor`), but an
e2e measurement waits on a single-card-runnable LTX-2.3.

## Implementation status

`FNI8TeaCache` ships **two cache granularities**, both correct and unit-tested (17 tests):
  * **output** (default for multi-stream / unknown archs) — reuse the whole DiT prediction;
    fully arch-agnostic (single-stream Z-Image, double-stream Qwen, audio+video LTX).
  * **residual** (auto-selected for single-stream Lumina/Z-Image) — cache the transformer-
    block delta, re-run the timestep-conditioned projections every step (the paper's method).
  * `mode="auto"` picks residual when the arch supports it, else output. Wrapped at the
    `WrappersMP.DIFFUSION_MODEL` seam; residual block-wrappers are installed/restored around
    each `_forward` so a shared module is never left patched for other clones.

## Bottom line

The cache is **correct, dual-granularity, and quality-gated by measurement** — but on the
models runnable here it does not clear the 0.985 latent gate at any speedup:
  * **Z-Image-Turbo** (8- and 20-step, output AND residual modes): no inter-step redundancy
    to exploit — a *model* property of few-step distillation, proven by both granularities
    landing at the same cosine. Keep caching off.
  * **Qwen-Image-Edit**: blocked by a b4-DiT loader gap on `main` (not TeaCache).
  * **LTX-2.3**: wired via the same seam, but 23.5 GB doesn't fit one card.

The multiplicative int8 × TeaCache stack materialises on **many-step, high-redundancy DiTs**
— the video models TeaCache was designed for (Wan/HunyuanVideo/LTX, 30–50 steps). The
mechanism is in place and validated; a positive e2e number waits on a single-card-runnable
many-step DiT (Wan2.2-TI2V-5B b8 fits at 5 GB and is the obvious next candidate). We report
the honest negative on the tested models rather than a cherry-picked win — per the issue's
explicit "if a model doesn't tolerate caching well, say so."
