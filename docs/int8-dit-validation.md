# int8 DiT quality validation (issue #82)

## Why this document exists

Before **PR #81**, ComfyUI DiTs loaded with **0 int8 weights** — four arch-wide bugs
(`assign=False` dropping the int8 codes+scale, `FNI8Tensor.clone` recursion, `q_scale`
left on the wrong device, and the attention heads-arg mismatch) meant the "int8
acceleration" silently ran the **fp** path end to end. Every earlier "int8 DiT
validated" claim was therefore validating fp, not int8. #81 fixed the engagement
(proven: 300 dp4a linears on Wan2.2-5B, finite denoise). This document records the
**first real int8-engaged quality validation** (post-#81) plus the **int8-attention
SQNR gate** added alongside it.

Hardware: NVIDIA CMP 100-210 / Tesla V100 (GV100, sm_70), 16 GB. int8 = `fni8` W8A8
dp4a on the CUDA cores (the only int8 path on Volta — no int8 tensor cores).

## Task 1 — image-DiT int8 quality (int8 ENGAGED)

**Target: Tongyi-MAI/Z-Image-Turbo** (6B, the stated first validation arch), loaded via
`FNI8Ops` with int8 dp4a weights actually resident, compared against the **same model
with its weights dequantized to fp16** (identical architecture, only the matmul path
differs — int8 dp4a GEMM vs torch fp matmul). Test:
`tests/e2e/test_quality_int8_vs_fp16.py`.

| Metric | int8 vs fp | Floor | Verdict |
|---|---|---|---|
| Latent cosine similarity | **1.000000** | ≥ 0.985 | PASS — int8 matches fp to 6 dp |
| Denoise output finite | yes | — | no fp16/int8 overflow → no NaN/black |
| Denoise output non-constant | yes | — | not collapsed to a black latent |

**Verdict: int8 does NOT degrade Z-Image image quality.** The int8 dp4a DiT is
numerically indistinguishable from the fp reference on the denoise latent
(cosine 1.000000). This is the first time this has been measured with int8 genuinely
engaged.

*Constraint note (now lifted — see Task 3):* on the 16 GB cards a full multi-step +
VAE-decode pass on the 6B DiT was previously memory-tight; the latent-space cosine
comparison above was the smallest viable apples-to-apples validation (same model
instance, weights swapped in place) and is the metric the existing quality gate asserts.
With the tiled VAE decode (#97) that full pass now fits in 16 GB, so the decoded-image
PSNR/SSIM comparison is measured directly in **Task 3** below.

## Task 3 — full pipeline on 16 GB (Qwen3-4B → int8 DiT → tiled VAE → image)

With the tiled VAE decode (#97) landed, the full generation pipeline the earlier
constraint note deferred now runs end to end on a single 16 GB Volta/CMP card:

    Qwen3-4B text encoder  ->  int8 dp4a Z-Image DiT (8-step Turbo denoise, 512 px)
                           ->  tiled VAE decode (#97)  ->  a decoded RGB image

Driver: `bench/full_pipeline_zimage.py` (real Qwen3-4B TE, real Z-Image VAE, the
published `Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8`); e2e gate:
`tests/e2e/test_full_pipeline_zimage.py`. The pipeline loads each stage sequentially
(TE freed before the DiT, DiT freed before the VAE), so peak HBM is the max of any one
stage, not their sum:

| Stage | Peak HBM (measured, GV100/sm_70, 16 GB card) |
|---|---|
| int8 dp4a DiT — 8-step denoise @ 512 px | **8.25 GiB** |
| tiled VAE decode (#97), 4 tiles @ 512 px | **9.03 GiB** |

Both stages sit well under the 16 GB budget: **the full pipeline fits and runs**.

**The decoded image first exposed a real int8 regression the latent-only gate missed —
now diagnosed and FIXED.** The fp reference (same conditioning + same VAE, DiT weights
dequantized to bf16) decodes to a coherent, prompt-faithful photograph. The int8 path
*originally* collapsed over the 8 steps to an incoherent 2×2-patch-grid image (PSNR ≈
7.6 dB) — **but that was a qkv-fusion loader bug, not a precision limit.** ComfyUI's
Z-Image S3-DiT **fuses** the `.fni8`'s separate `to_q/to_k/to_v` int8 projections into one
`attention.qkv`; three independently per-row-scaled int8 blocks can't share one tensor, so
the fused key missed `assign_int8_weights` and attention ran on garbage. The loader now
**losslessly re-fuses** the per-row int8 codes+scales (`DiTArch.fuse_attn_qkv` for
`zimage`), so int8 attention engages correctly (FFN was always int8).

**Reconciled multi-step decoded-image result — a fresh clean run on current comfy +
fni8-master, POST-#130 (worst-case multi-timestep SQNR calibration) and POST-#132 (fp16
attn routing).** This is the measurement that REPLACES the doc's previously-conflicting
number:

| Metric (int8 vs fp, 8-step Turbo, 512 px decoded image) | Value | Bar | Verdict |
|---|---|---|---|
| PSNR (decoded image) | **29.39 dB** | ≥ 28 dB "same image" | PASS |
| latent cosine (full 8-step) | **0.9932** | ≥ 0.985 | PASS |
| SSIM | **0.9546** | ≥ 0.90 | PASS |
| int8 attention on dp4a | **qkv=34 / out=34 blocks, 0 fp fallback** | engaged | PASS |
| image | coherent red fox, matches fp | — | PASS |
| peak HBM (whole pipeline) | 12.05 GiB | 16 GiB card | fits |

The old 7.6 dB figure was the **pre-fix collapse**, not a property of int8 — retained above
only to show what the qkv-fusion bug did. The **29.39 dB** here reproduces the earlier
pre-#132 finding of ~31 dB (`zimage-full-pipeline-findings.md`) within run-to-run variance
(the small delta is #130's stricter worst-case SQNR gating + #132's attention routing +
comfy version) — **both are comfortably "same image" (> 28 dB / SSIM > 0.95).**
Images saved: `.pipe_out/zimage_int8.png` (int8 fox) and `.pipe_out/zimage_fp.png` (fp fox).
Full images, per-op isolation, and the fix:
[`zimage-full-pipeline-findings.md`](zimage-full-pipeline-findings.md).

*Reproducing on current comfy + fni8-master (2026-07):* two integration fixes were needed
that this validation surfaced by actually running the pipeline end to end — (1) the pack's
guidance-distillation `sampler_pre_cfg_function` hook returned the `args` dict instead of the
`conds_out` list, which comfy indexes as `out[0]` → `KeyError(0)` on every distilled denoise
(Z-Image/FLUX/…); fixed in `guidance_harvest.py` (+ `tests/test_guidance_harvest.py`). (2) the
#132 fp16-attn fallback calls `fni8.attn_fp16_fwd`, a kernel that lives on an unmerged fni8
branch and is **absent from fni8 master** — the node path now degrades to torch SDPA when the
kernel is missing (`attention.py`) rather than `AttributeError`. The proper long-term fix is to
merge the `attn_fp16_fwd` kernel into fni8 master.

*int8 VAE compute note:* the VAE stays fp (per the README's "VAE always stays fp" rule
and issue #55) — the tiled decode just caps its activation working set so it fits; it is
not an int8-VAE path. The VAE is **not** implicated in the regression (it decodes the fp
latent to a perfect image).

## Task 2 — int8 self-attention SQNR gate

The int8 **linears** already have a per-layer SQNR fp-fallback gate (`ops.py`, #29): the
first call compares int8 vs fp and demotes the layer to fp if it fails. The int8
**attention** kernel had **no** such gate — un-gated int8 self-attention quality was
unvalidated (especially for video). This PR adds the equivalent
(`comfyui_superl8/attention.py`):

- `attn_sqnr(y_int8, y_fp)` — signal-to-quantization-noise ratio in dB.
- `Int8AttnGate` — one gate per patched model (created in `nodes.FNI8AttentionPatch`).
  Keyed by the attention call-site's shape signature `(B, heads, S, D, dtype)`: the
  first call at a signature measures SQNR of the int8 dp4a output vs the fp SDPA
  reference and caches whether it clears `ATTN_SQNR_FLOOR_DB` (default **20 dB** ≈ 1%
  noise power, at/above the strictness of the linear gate's cos ≥ 0.99 bar, which is
  17.0 dB). A passing signature runs int8-only afterward; a failing one falls back to
  fp SDPA for the rest of the run.

**Multi-timestep calibration (correcting the "stable across denoise steps" assumption).**
The gate originally locked its verdict from the FIRST activation at a signature, on the
premise that activation distributions are stable across the trajectory. This repo's own
Task 3 evidence refutes that premise: the Task 1 latent cosine of **1.000000 was measured
at a single denoise step** and did **not** see the multi-step image collapse. A DiT's
activation distribution shifts across early/mid/late timesteps (and across resolutions),
so a first-activation verdict can wrongly keep int8 on a layer that collapses later. Both
gates (`attention.Int8AttnGate` and `int8_linear.LinearSqnrCalibrator`, used by `ops.py`)
now support **worst-case multi-timestep calibration**: with `FNI8_SQNR_CALIB_SAMPLES > 1`
they sample that many forwards (successive timesteps — different resolutions land on
different attention signatures) and finalize the int8/fp decision on the **WORST** SQNR
seen, which is conservative. `FNI8_SQNR_CALIB_SAMPLES = 1` (default) keeps the legacy
first-activation behavior, so the change is opt-in and backward-compatible;
`FNI8_SQNR_REVALIDATE_EVERY > 0` additionally re-checks passing attention sites and demotes
one whose activation later shifts unusually. Covered by `tests/test_sqnr_calib.py`.

### Measured per-call-site int8-attention SQNR (Z-Image, 8-step denoise, 32×32 latent)

Captured by `bench/validate_int8_attn_sqnr.py` (spies on `attn_sqnr` during a real
`UnetLoaderFNI8` denoise, int8 attention wired + gate active):

```
int8 attn_int8_fwd invocations : 18  (ENGAGED)
denoise output finite          : True
denoise output non-constant    : True
SQNR gate floor                : 20.0 dB
distinct call-sites gated      : 3
bidirectional self-attn SQNR   : 47.2 dB   (>> 20 dB floor -> int8 KEPT)
```

**Reading:** the real Z-Image bidirectional self-attention call-site measures **47.2 dB**
SQNR (int8 dp4a vs fp SDPA) — more than double the 20 dB floor and comfortably above the
linear gate's cos ≥ 0.99 (17 dB) bar. int8 self-attention is **high quality** on Z-Image
and is kept. The remaining two call-sites (cap-token cross paths under this run's random
placeholder conditioning) produced a non-finite fp SDPA reference, so `attn_sqnr` returns
`-inf` and the gate correctly demotes them to fp SDPA — the safe direction, and exactly
what the gate exists to do. Output is finite and non-constant end to end (18 int8 FA calls
over the 8-step denoise).

> This validation was previously **impossible to run**: `UnetLoaderFNI8` on Z-Image
> crashed in `pad_zimage` because the int8 `cap_pad_token` buffer's `q_scale` stayed on
> CPU while its codes moved to CUDA (`dequantize_weight` device mismatch — the maintained
> `tests/e2e/test_smoke_attn_int8.py` was red on `main`). This PR fixes that co-location
> (`int8_linear.dequantize_weight`), which is what let int8 attention run end-to-end at
> all. The smoke test is now green.

## Re-labeling the "int8 DiT validated" claim

- **As of #81 + #82:** image-DiT int8 is *engaged* and *quality-validated* on Z-Image
  (latent cosine 1.000000 vs fp). Prior to #81, "validated" meant "ran (in fp)".
- int8 self-attention is now **SQNR-gated**: it falls back to fp SDPA anywhere the int8
  output is below the noise floor, so a badly-quantized attention block can no longer
  silently ship garbage.
- Video DiTs (Wan/LTX) load via #81 but remain WIP for full end-to-end quality; the
  attention gate applies to them too once they denoise.

## Reproduce

```
# Task 1 — int8-vs-fp latent quality gate
docker compose run --rm e2e python3 -m pytest -m comfy_e2e --import-mode=importlib \
  custom_nodes/ComfyUI-superl8/tests/e2e/test_quality_int8_vs_fp16.py -s

# Task 2 — per-call-site int8-attention SQNR on a live denoise
docker compose run --rm e2e \
  python3 custom_nodes/ComfyUI-superl8/bench/validate_int8_attn_sqnr.py --steps 8 --hw 32

# gate unit tests (CPU, no GPU needed)
docker compose run --rm e2e python3 -m pytest \
  custom_nodes/ComfyUI-superl8/tests/test_attn_gate.py -v --import-mode=importlib
```
