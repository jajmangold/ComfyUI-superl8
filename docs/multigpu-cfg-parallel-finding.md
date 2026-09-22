# CFG-parallel on the int8 DiT path — measured finding (#33)

**Verdict: CFG-parallel by cond/uncond batch-splitting is NOT a validated win on the
int8 DiT path. It fails the cos ≥ 0.99 quality bar (measured ≈ 0.88) and cannot be
wired as originally designed. The `FNI8Multigpu` node therefore safely falls back to
single-GPU for `cfg_parallel`, and `component_parallel` (a pure device move) is the
supported multi-GPU strategy.**

## What CFG-parallel would do

ComfyUI runs classifier-free guidance as one batched forward: the cond and uncond
latents are stacked (`cond_or_uncond=[0,1]`, batch=2) and a single
`model.apply_model(input, timestep, **c)` computes both. CFG-parallel would split
that batch — cond on GPU 0, uncond on GPU 1, concurrently — and gather the two
predictions. The two passes are independent (no per-step wire codec), so on paper it
is "roughly free 2×".

## Measurement (real int8 Z-Image-Turbo, 2× CMP 100-210, in the e2e image)

Reference = the single-GPU **batched** `apply_model` output (`cfg=2.0`). Test =
the same call **split** into two batch-1 forwards, gathered back in order. Cosine of
test vs reference:

| latent | split cos vs single-GPU batched | bar |
|---|---|---|
| 16×16 | 0.8535 | 0.99 |
| 128×128 | 0.8810 | 0.99 |

Speedup at 16×16 was only ~1.36× (small sequence → the DiT step is dominated by
batch-independent cost, so batch-2 is not ~2× batch-1). It would rise toward 2× at
real resolutions, but that is moot while the output fails the quality bar.

## Root-cause isolation (all measured, reproduced)

1. **The raw dp4a kernels are batch-independent.** `fni8.attn_int8_fwd` and
   `int8_linear`: row *i* of a batched-2 call equals the standalone batch-1 call for
   that row, cos = 1.000000.
2. **Within a batch, rows do not contaminate each other.** Two identical rows in a
   batch-2 forward give identical outputs (self-cos = 1.000000).
3. **But the full model prediction is batch-size-dependent.** The *same* input
   produces a different prediction at B=1 vs B=2: cos = 0.8810, max-abs-diff 2.24.
4. **It is not the SQNR attention gate.** Making the gate key batch-independent
   (dropping `B` from `(B,heads,S,D,dtype)`) changed the result by *nothing*
   (bit-identical 0.881040).
5. **It is not the int8 attention.** Forcing every attention call-site to fp SDPA
   left the result bit-identical (0.881040) — so int8 attention isn't even the
   dominant path for Z-Image here, and the divergence survives regardless.

The residual B=1-vs-B=2 dependence lives in the DiT forward at the model's real
shapes and would need deeper kernel / torch-SDPA investigation to eliminate. Until a
batch-invariant int8 forward exists, splitting the CFG batch cannot reproduce the
single-GPU image.

## Secondary blocker

The int8 DiT weight is an `FNI8Tensor` (a `torch.Tensor` subclass) with no
`__deepcopy__`, so `copy.deepcopy(diffusion_model)` — the original PR's mechanism for
building the second-GPU replica — raises. A replica would have to be a *separately
loaded* instance, not a deep copy.

## What ships

* `component_parallel`: validated — moves the DiT to a second GPU to free VRAM on the
  primary card. No batch split, no numeric change.
* `cfg_parallel`: guarded — logs the finding and returns the model unchanged. The
  planning helpers (`plan_cfg_parallel`, `make_cfg_parallel_wrapper`, …) and their
  unit tests remain, and `tests/e2e/test_multigpu_smoke.py::test_cfg_batch_split_diverges`
  measures the divergence so the fallback stays honest. If a future forward makes the
  split match the bar, that test xpasses and the path can be enabled by measurement.
