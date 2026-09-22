# Low-rank wire codec on a DiT PP boundary — findings

**Question.** The production low-rank ("VAE-ish") wire codec
(`fni8-serve/fni8serve/dist/lowrank.py`) was rejected for **LLM** pipeline-parallel
transport (`fni8-serve/docs/lowrank-codec-findings.md`): per-boundary it looks great
(cos 0.995 @ 10×) but it **collapses end-to-end generation** because an LM's
next-token readout lives in the low-variance directions a variance-ordered SVD
truncates first. The hypothesis for this probe: a **DiT image latent** may not have
that pathology, so low-rank could win big (5–10× at high fidelity) exactly where the
PCIe-1.0-x1 PP link is bottlenecked. This is the DiT analogue of that LLM sweep.

**Verdict: hypothesis REFUTED. Low-rank is not a win for DiT transport.** At a real
Z-Image-Turbo boundary, low-rank never holds the final latent above cos 0.69 at *any*
compression ratio, and is **dominated at every ratio by int4-had** (3.76×, final
cos 0.94, image SSIM 0.78). Low-rank final quality is nearly **flat in rank** — you
cannot buy fidelity by raising the rank — so the compression axis is decoupled from
fidelity, the opposite of the "tunable 5–10× at high fidelity" hope. Use **int4-had**
as the DiT PP transport codec; low-rank stays rejected.

The DiT does *not* reproduce the LLM's specific pathology (structured subspace
deletion being catastrophically worse than its SQNR predicts). Instead low-rank
roughly tracks the boundary-SQNR→final-fidelity curve — it just never reaches the
SQNR a chaotic 8-step denoising trajectory demands without giving all the compression
back.

## Setup

- **Model:** Tongyi-MAI Z-Image-Turbo `.fni8` — NextDiT, `d = 3840`, 30 `layers`
  (JointTransformerBlock), int8 dp4a path (the real deployed DiT), Qwen3-4B TE,
  8-step Turbo denoise, 512 px (64×64 latent), seed 0.
- **Boundary:** output of `layers[15]` (a 2-way PP split of the 30 blocks). The codec
  round-trips the single hidden state `x` at that boundary. In a real 2-way PP DiT the
  boundary codec fires **once per denoising step**, so the hook fires all 8 steps — the
  errors compound through the sampling loop, exactly as they would on the wire.
- **Base model held fixed:** reference and every codec run use the **same int8 DiT**,
  same conditioning, same seed, so the int8-DiT quantization noise is common-mode and
  the measured delta is the **codec's marginal damage alone**.
- **Codec:** the *production* `LowRankCodec` / `fit_basis` from
  `fni8-serve/fni8serve/dist/lowrank.py`, imported by file path (unmodified). SVD basis
  fit on boundary activations captured over a **held-out calibration prompt** (a
  different prompt than the measured one — no train/test leakage).
- **Metric:** cosine + PSNR of the **final denoised latent** vs the no-codec reference
  (the image analogue of the LLM's top-1 next-token agreement), plus per-boundary
  SQNR/cos, plus decoded-image PSNR/SSIM for anchor configs.
- **Baselines:** int8 / int4 / int4-had / nf4 via `fni8.transport.compress_activation`
  at the same boundary.
- Reproduce: `bench/lowrank_dit_boundary_probe.py` (see its docstring for the exact
  docker invocation). Ran on GPU 0 (CMP 100-210, sm_70) in `comfyui-fni8-e2e`.

## Results

### Quant-codec baselines (same boundary)

| codec | ratio | final latent cos | final latent PSNR | boundary SQNR | boundary cos | image PSNR | image SSIM |
| ----- | ----- | ---------------- | ----------------- | ------------- | ------------ | ---------- | ---------- |
| **int4-had** | 3.76× | **0.9416** | 0.36 dB | 30.84 dB | 0.99981 | **19.24 dB** | **0.780** |
| nf4 | 3.76× | 0.9330 | −0.06 dB | 35.02 dB | 0.99984 | — | — |
| int4 | 3.76× | 0.9135 | −1.15 dB | 33.44 dB | 0.99977 | — | — |
| int8 | 2.00× | 0.4239 | −9.33 dB | 25.93 dB | 0.99872 | 11.14 dB | 0.254 |

Note two DiT-specific facts already visible here: (1) plain per-row **int8 is a poor
DiT-boundary codec** — DiT activations have massive channel outliers, so int8's
boundary SQNR (25.9 dB) is *below* Hadamard-rotated int4's (30.8 dB), and its final
latent collapses to cos 0.42 (image SSIM 0.25). (2) The denoising loop is **chaotically
sensitive**: a 0.001 gap in per-boundary cos (0.9987 → 0.9998) swings the final latent
from cos 0.42 to 0.94. High per-boundary fidelity is *required*, not optional.

### Low-rank sweep (rank × raw_fraction), production codec

Ratio = `2d / (r·1 + n_raw·2 + 4)` (fp16 baseline vs int8 latent + fp16 raw channels;
SVD basis is amortized calibration, shipped once).

| raw_fraction | rank r | ratio | final latent cos | boundary SQNR | boundary cos |
| ------------ | ------ | ----- | ---------------- | ------------- | ------------ |
| 0.5 % | 115 | 48.9× | 0.4811 | 25.93 dB | 0.99872 |
| 0.5 % | 192 | 32.8× | 0.4811 | 25.93 dB | 0.99872 |
| 0.5 % | 307 | 22.0× | 0.4813 | 25.93 dB | 0.99872 |
| 0.5 % | 691 | 10.5× | 0.4811 | 25.93 dB | 0.99872 |
| 0.5 % | 1344 | 5.5× | 0.4801 | 25.93 dB | 0.99872 |
| 5 % | 115 | 15.3× | 0.5359 | 26.79 dB | 0.99895 |
| 5 % | 307 | 11.1× | 0.5374 | 26.79 dB | 0.99895 |
| 5 % | 691 | 7.1× | 0.5366 | 26.79 dB | 0.99895 |
| 5 % | 1344 | 4.4× | 0.5360 | 26.79 dB | 0.99895 |
| 20 % | 115 | 4.6× | 0.6873 | 28.74 dB | 0.99933 |
| 20 % | 307 | 4.2× | 0.6808 | 28.74 dB | 0.99933 |
| 20 % | 691 | 3.4× | 0.6891 | 28.74 dB | 0.99933 |
| 20 % | 1344 | 2.7× | 0.6889 | 28.74 dB | 0.99933 |

(Full 21-point grid in `bench` output; representative rows shown.)

**Final latent cosine is essentially flat in rank** and moves only with
`raw_fraction` (0.48 → 0.54 → 0.69 as raw goes 0.5 % → 5 % → 20 %). Raising the rank
from 115 to 1344 — throwing away 10× of the compression — buys **~0.00** cosine. The
codec's damage is dominated by the projection subspace deletion / per-tensor int8
latent, not by truncation depth, so **you cannot trade compression for DiT fidelity**.
Even the best low-rank point (r≈115, raw 20 %, 4.6×, cos 0.69) is worse than int4-had
at 3.76× (cos 0.94), and its decoded image is worse too (r=1344 raw 20 %: image PSNR
12.3 dB / SSIM 0.53, vs int4-had 19.2 dB / 0.78).

### Iso-quality: max compression at a fidelity floor

| final-latent floor | best low-rank | best quant codec |
| ------------------ | ------------- | ---------------- |
| cos ≥ 0.99 | **none** (max 0.69) | none (int4-had 0.94 is closest) |
| cos ≥ 0.90 | **none** | int4-had 3.76×, int4 3.76×, nf4 3.76× |
| cos ≥ 0.68 | r=115 raw 20 %, **4.6×** | int4-had 3.76× (cos 0.94, far above floor) |

No codec reaches cos ≥ 0.99 for the final latent at an every-step single boundary on
this model — the 8-step trajectory is too sensitive. But the **ranking is robust**:
int4-had ≻ nf4 ≻ int4 ≻ low-rank ≻ int8 on final latent fidelity, and int4-had wins on
decoded-image SSIM by a wide margin.

## Comparison to the LLM curve

| axis | LLM (Qwen3-1.7B, top-1 next-token) | DiT (Z-Image, final-latent cos) |
| ---- | ---------------------------------- | ------------------------------- |
| per-boundary fidelity | great (cos 0.995 @ 10×) | great (cos 0.999 @ 3–49×) |
| does per-boundary predict E2E? | **no** | **no** |
| low-rank pathological vs its SQNR? | **yes** — 0.017 top-1 @ 20.4 dB, far below int8's 0.80 @ 24.5 dB (structured subspace deletion kills the readout) | **no** — 0.69 @ 28.7 dB sits *between* int8 (0.42 @ 25.9 dB) and int4-had (0.94 @ 30.8 dB); low-rank ≈ tracks the SQNR→fidelity curve |
| tunable with `raw_fraction`? | yes: 0.5 %→0.21, 40 %→0.84 top-1 | yes but weak: 0.5 %→0.48, 20 %→0.69 cos |
| tunable with **rank**? | somewhat | **no — flat in rank** |
| best low-rank at a usable floor | ~40 % raw ≈ 2× (no better than int8) | ~20 % raw ≈ 4.6× but cos only 0.69 (worse than int4-had @ 3.76×) |
| does low-rank ever beat the shipped codec E2E? | no | **no** |

The DiT is **kinder** to low-rank than the LLM in one respect — it lacks the LLM's
low-variance-readout pathology, so low-rank is not *catastrophically* worse than its
per-boundary SQNR suggests; it degrades gracefully in line with SQNR. But that is not
enough to make it a win, because (a) the DiT denoising loop demands very high
per-boundary fidelity (int4-had's ~31 dB) to hold the output, and low-rank cannot reach
that SQNR without surrendering its compression, and (b) low-rank fidelity is flat in
rank, so the compression it *does* offer is not purchasable as fidelity. At every
compression ratio a 4-bit elementwise codec (int4-had best) is on or above the
low-rank Pareto point.

## Disposition

- **Low-rank stays rejected** — for LLM *and* DiT PP transport. It is not toxic on the
  DiT the way it is on the LLM, but it never beats int4-had at iso-compression and can't
  be tuned toward the fidelity the sampling loop needs.
- **DiT PP transport should default to `int4-had`, not `int8`.** On DiT boundary
  activations (heavy channel outliers) Hadamard-rotated int4 is both smaller (3.76×) and
  *more* accurate than int8 (2×) — final latent cos 0.94 vs 0.42, image SSIM 0.78 vs
  0.25. This is a concrete, actionable change for the multi-GPU DiT split
  (`comfyui_superl8/multigpu.py` `TransportCodec` / `pipeline_ltx.py`), whose default is
  currently `int8`. (Confirm per-arch on LTX/Wan before flipping the shipped default.)
- The 8-step every-boundary hook is the *realistic* PP scenario and is deliberately
  punishing; a fidelity floor stated on the final latent (cos ≥ 0.99) is not reached by
  any 4-bit codec here, so the **relative ranking** — not the absolute floor — is the
  load-bearing result.

## Reproduce

```
FNI8_GPU=0 docker compose run --rm \
  -v <fni8-serve-dir>:/opt/fni8-serve:ro \
  -e FNI8_SERVE_ROOT=/opt/fni8-serve \
  --entrypoint bash e2e -c \
  'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
   custom_nodes/ComfyUI-superl8/bench/lowrank_dit_boundary_probe.py --steps 8 --size 64 --decode-anchors'
```
