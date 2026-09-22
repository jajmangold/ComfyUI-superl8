# LTX-2.3 DiT block profile — where the int8 dp4a time goes

Profiling of the LTX-2.3 (`comfy.ldm.lightricks.av_model.LTXAVModel`) transformer block
forward on a real Volta/CMP GPU (host GPU 10), to answer: *what in the LTX-2.3 video DiT
is NOT on the int8 dp4a fast path, and what is the top optimization lever?*

## Method + honest scope

The full int8 LTX-2.3 DiT is **26 GB** (`Lightricks__LTX-2.3.dit.b8.fni8`; the W4A8 build
is 18 GB) and does **not** fit a single 16 GB card — and only one GPU was free. So
**LTX-2.3 cannot run end-to-end here** (it needs the 2-GPU pipeline split, see
`pipeline_ltx.py` / issue #106). Additionally the shared weights archive is heavily
disk-contended (~3 MB/s), so even a CPU-resident 26 GB load stalls for hours.

Instead we profile **one `BasicAVTransformerBlock`** built at the exact shipped LTX-2.3
(22b) config — read straight off the `.fni8`: `v_dim=4096`, 32 video heads, head-dim
**128**; `a_dim=2048`, audio head-dim 64; FFN 4096→16384; gated attention;
cross-attention adaLN; **bf16** native. The 48 identical blocks × N steps dominate a
video-DiT denoise, so this block IS the optimization surface. Linear weights are random
then per-row int8-quantized — **for latency this is bit-for-bit the same kernel work as
real weights**; the reported int8-vs-fp SQNR is indicative (random weights). Video-only
path (empty audio tensor). RoPE skipped (`pe=None`): a cheap elementwise op, identical on
both paths. Harness: `bench/profile_ltx23_block.py`, `bench/profile_ltx23_selfattn.py`,
`bench/profile_ltx23_rotate.py`.

Baseline = the same block on the fleet's fp (bf16) path. Per AGENTS.md the fp16/TF32
tensor cores are firmware-dead on the CMP fleet, so this fp baseline runs on the CUDA
cores; the speedups below are **fleet-specific and do not transfer to a real V100**.

## Headline: the block is ALREADY int8-complete

Every compute-heavy op is already routed to int8 dp4a: the video self-attention QK^T/PV
(bidirectional `attn_int8_fwd`, head-dim 128, `mask=None`), all four attention
projections (`to_q/k/v/out`), the text and AV cross-attention, and the FFN. The
self-attention SQNR gate engages int8 (passes) and the block int8-vs-fp cosine is
**0.9996** (SQNR ~31 dB, random weights — indicative).

### Whole-block latency (bf16, ms/block, median)

| video tokens | int8 dp4a | fp (bf16) | speedup |
|---:|---:|---:|---:|
| 2048 | 84.1  | 170.0 | **2.02×** |
| 4096 | 192.7 | 362.0 | **1.88×** |
| 8192 | 503.3 | 841.6 | **1.67×** |

The speedup **shrinks as the video gets longer** because the O(S²) attention kernel — the
one op int8 barely helps — grows to dominate.

### Sub-op breakdown @ 4096 tokens (ms)

| op | int8 | fp | speedup | share of int8 block |
|---|---:|---:|---:|---:|
| video self-attn (attn1) | 96.8 | 142.3 | 1.47× | **50%** |
| video text-cross (attn2) | 24.1 | 49.5 | 2.06× | 12% |
| video FFN | 70.9 | 171.7 | **2.42×** | 37% |

### Inside self-attention @ 4096 (ms)

| component | int8 | fp | note |
|---|---:|---:|---|
| `to_q/k/v` (3 GEMMs) | 24.8 | ~2.4× | fully int8 |
| **dp4a FA2 attention kernel** | **45.9** | 59.3 (SDPA) | **only 1.29×** |
| `to_out` | 8.3 | ~2.4× | fully int8 |
| gated-attn epilogue | 0.66 | — | negligible |
| q/k RMSNorm | 0.21 | — | negligible |

## Ranked levers

1. **The dp4a FA2 attention kernel (the ceiling).** It is O(S²), 50% of the block, and
   only **1.3×** over fp SDPA — because on Volta SDPA also runs on the CUDA cores (the
   tensor cores are dead), so int8's compute win over fp is muted for the
   softmax/PV-bound attention matmuls (vs 2.4× for the pure-GEMM projections/FFN). This
   kernel lives in the **`fni8` sibling repo**, not here; improving it (or an algorithmic
   sparsity like Sliding-Tile / windowed attention, `sta.py`, to cut the O(S²) FLOPs) is
   the only way to move long-video latency. STA is a quality tradeoff (draft / post-1.0).
2. **Everything else is already optimal.** Projections and FFN get the full ~2.4× dp4a
   win. No significant GEMM or attention path is left on fp.

## Candidate levers measured and RULED OUT (no fabricated wins)

| lever | result | verdict |
|---|---|---|
| `rotate=True` Hadamard rotation (issue #74) | SQNR unchanged (already 36–39 dB under injected channel outliers), 0 perf cost | no measurable gain on these activations — the kernel's K-smoothing already handles channel outliers; harmless but not a win |
| `int8_pv=True` full W8A8 attention | 0.98× (slower), SQNR unchanged; accuracy-gate fell back to SDPA under strong outliers | no gain — PV int8 doesn't help the memory/softmax-bound attention on sm_70 |
| runtime `to_q/k/v` fusion (one GEMM) | 0.97–0.99× | no gain — the three int8 GEMMs already saturate; fusion adds output-slicing cost |
| masked text cross-attn → fp fallback | masked block 195 ms vs unmasked 192 ms (+1.5%) | negligible — text context is short (~256), so the fp SDPA fallback is tiny; projections stay int8 regardless |

## Regression risk this protects against

The single thing keeping LTX-2.3 fast on this fleet is that video self-attention **stays
on int8** — a silent demotion to fp SDPA (a wiring regression, or an over-strict SQNR
gate) would cost ~30% on the dominant op. `tests/e2e/test_ltx23_selfattn_int8.py` guards
it: it builds the real-config block, drives the production `optimized_attention_override`,
and asserts the int8 dp4a path engages, clears the SQNR bar, and beats fp latency.
