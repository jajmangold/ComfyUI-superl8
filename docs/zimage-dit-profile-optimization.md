# Z-Image int8 DiT — forward profile + fp-fallback-skip optimization

## What was profiled

The Z-Image-Turbo DiT (`Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8`, a Lumina2 `NextDiT`,
~6 B) loaded exactly as `UnetLoaderFNI8` loads it (int8 dp4a linears + losslessly
re-fused int8 attention qkv (#103) + bidirectional int8 FlashAttention, all SQNR-gated),
profiled on a **real Tesla V100 (sm_70, host idx 9)** — the CMP mining cards lock
hardware counters (`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`), so kernel-level self-CUDA
time needs a real V100. Step latency is additionally measured on a CMP card (idx 8), the
deployment fleet.

Driver: `bench/profile_zimage_dit.py` — real Qwen3-4B conditioning, then N `apply_model`
forwards @ 512 px under `torch.profiler` (the pinned ComfyUI's `comfy.sample.sample`
errors on this build — a sampler version drift unrelated to the int8 path — so the
harness drives `apply_model` directly, which is also the cleaner DiT-only measurement).

## Ranked lever list (self-CUDA time, one 6-step denoise, real V100, BEFORE)

| # | Kernel / op | Self-CUDA | Calls | What it is | Actionable? |
|---|---|---|---|---|---|
| 1 | `fni8::gemm_w8a8_kernel` | **63.7 %** (3.76 s) | 1020 = 170/fwd | the int8 dp4a GEMM — the intended fast path | already optimal |
| 2 | `aten::mm` (`volta_sgemm` fp32) | **13.9 %** (0.82 s) | 90 = 15/fwd | fp32 fallback for 15 SQNR-gate failures (14× FFN `w2` down-proj K=10240 + 1× `attn.out`) | **partly — see below** |
| 3 | `fmha_cutlassF_f16` | **12.6 %** (0.74 s) | 204 = 34/fwd | fp16 SDPA: the fni8 attn kernel's own Q-outlier guard demotes **204/306 (67 %)** of int8-attention calls to fp16 internally | kernel-side (fni8) |
| 4 | `aten::median`+`abs`+`addcmul` | ~2.4 % | 408 | per-call Q-outlier detection inside the attn kernel | kernel-side (fni8) |
| 5 | `aten::copy_` | 3.0 % | 3876 | `.contiguous()` on attn q/k/v + casts | mostly required |
| 6 | `aten::_fused_rms_norm` | 0.9 % | 1230 = 205/fwd | the 6 fp RMSNorms/block (attn/ffn ×2, q/k norm) — numerically load-bearing | keep fp |
| 7 | `quantize_i8_rowwise` | 0.6 % | 1020 | per-linear activation quant (2-launch prefill path) | kernel-side (fni8) |

**Verdict: the Z-Image int8 path is already well-optimized** — 63.7 % of GPU time is the
dp4a GEMM. The two large fp chunks (#2, #3) are **correctness-gated**, not un-quantized
oversights: #2 fails the per-layer cos ≥ 0.99 SQNR gate, #3 fails the kernel's
SageAttention Q-outlier guard. Per AGENTS.md both fp fallbacks are "correct, not a
defeat", and the gates must not be weakened. So the free lunch is small — but there was
one.

## The lever taken — skip the discarded int8 GEMM on demoted linears

`FNI8Ops.Linear.forward_comfy_cast_weights` ran the int8 dp4a GEMM **unconditionally**,
then — for a layer already demoted to the fp32 fallback — threw that result away and
recomputed the fp32 matmul. Every denoise step therefore executed **15 wasted dp4a
GEMMs** (the demoted layers are the *large* FFN `w2` down-projections, K=10240). The fix
(`ops.py`): when `_fni8_sqnr_pass is False`, take the fp32 path **only** — never launch
the GEMM whose output is discarded.

This is **numerically identical** by construction (the emitted value is exactly the same
fp32 fallback either way; only the wasted work is removed). Pinned by
`tests/test_ops_fallback_skip.py` (asserts a demoted layer never calls `int8_linear` and
its output equals the fp32 fallback at `atol=0`).

## Measured (before → after)

| Metric | Before | After | Δ |
|---|---|---|---|
| DiT forward latency — **real V100** (idx 9) | 1044.4 ms | **962.2 ms** | **−7.9 %** |
| DiT forward latency — **CMP fleet** (idx 8) | 1110.8 ms | **1031.2 ms** | **−7.2 %** |
| `gemm_w8a8` calls / denoise | 1020 | 930 | −90 (= 15/fwd × 6) |
| `gemm_w8a8` self-CUDA | 3.76 s | 3.36 s | −0.41 s |
| int8-vs-fp latent cosine (gate, floor 0.985) | 0.9875 | **0.9875** | unchanged (PASS) |

*Fleet caveat (AGENTS.md): these are sm_70 GV100 numbers and do not transfer to a real
data-centre V100 — they measure the actual CMP/V100 deployment target honestly.*

## Left on the table (recommendations, not weakenings)

- **#2 fp32 `w2` fallback (13.9 %):** the published `.fni8` has `smoothed: 0`. K-smoothing
  the FFN down-projections at **convert** time (AGENTS.md flags K-outliers as the dominant
  int8 error source) would likely let `w2` clear cos ≥ 0.99 and run dp4a. A runtime fix is
  not possible — the stored int8 codes can't be re-smoothed without the fp weights.
- **#3 attention fp16 fallback (12.6 % + 2.4 %):** `fni8.attn_int8_fwd` runs its
  Q-outlier gate on the *pre-rotation* Q, so 67 % of Z-Image self-attention calls fall
  back to fp16 even though the call-site's measured SQNR is 47 dB. Evaluating the gate on
  the Hadamard-**rotated** Q (`rotate=True` already exists) — or exposing `rotate` through
  `fni8.attn_int8_fwd` for the wrapper to enable — would keep more attention on dp4a
  (logit-invariant, no quality loss). This is an fni8-kernel change.

## Reproduce

```
FNI8_GPU=9 docker compose run --rm --entrypoint bash e2e -c \
  'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
   custom_nodes/ComfyUI-superl8/bench/profile_zimage_dit.py --steps 6 --size 64'
# quality gate (int8 vs fp latent cosine, no sampler):
FNI8_GPU=9 docker compose run --rm --entrypoint bash e2e -c \
  'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
   custom_nodes/ComfyUI-superl8/bench/quality_zimage_applymodel.py --size 64'
```
