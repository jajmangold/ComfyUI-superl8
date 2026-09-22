# LTX-2.3 (video) — int8 dp4a resolution x frames ladder

The video analogue of the Z-Image image ladder (`docs/zimage-perf-ladder.md`): how int8
dp4a `fni8` scales the LTX-2.3 audio+video DiT across spatial resolution AND clip length,
measured on a **real Tesla V100 (host GPU 9)** — honest timing for this fleet.

## Method — block-level extrapolation (labelled, not a full-pipeline wall clock)

The full int8 LTX-2.3 DiT does **not** fit one 16 GB card:
`Lightricks__LTX-2.3.dit.b4.fni8` = **18.9 GB** (W4A8), `.b8` = **27.6 GB** (W8A8) of
weights. So it cannot run end-to-end single-card at all — it needs the 2-GPU pipeline
split (`comfyui_superl8/pipeline_ltx.py`, a *memory-fit* strategy, not a speedup). The honest
method (same as `docs/ltx23-block-profile.md`) is to measure **one
`BasicAVTransformerBlock`** at the exact shipped LTX-2.3 config and extrapolate:

    full-DiT ms / denoise step  =  ms/block x 48 blocks
    full-DiT time (whole clip)  =  ms/block x 48 x num_steps

The 48 identical blocks x N steps dominate the denoise, so this block IS the compute
surface. Random int8 weights give **bit-identical kernel latency** to real weights (the
dp4a GEMM + FlashAttention are data-independent in timing), so we never need the 18.9 GB
resident. Config: v_dim 4096 / 32 heads / head-dim 128, a_dim 2048, FFN 4096->16384, gated
attention, cross-attn adaLN, bf16, video-only path. Harness:
`bench/profile_ltx23_ladder.py`. Numbers are **fleet-specific** (firmware-gimped tensor
cores; see below) and do not transfer to a datacenter V100.

## Token count from resolution x frames (confirmed from comfy)

LTX-2.3 is a video DiT, so token count N depends on BOTH spatial resolution and frame
count. From `comfy_extras/nodes_lt.py` (`EmptyLTXVLatentVideo`) + `SymmetricPatchifier(1)`
with `vae_scale_factors=(8, 32, 32)` — VAE compresses **32x spatial, 8x temporal**, DiT
patch size **1**:

    latent = [B, 128, ((length-1)//8)+1, height//32, width//32]
    N_video = ((length-1)//8 + 1) * (height//32) * (width//32)

Frame counts 25 / 49 / 97 (all == 1 mod 8, the comfy default `length` is 97) map to
**4 / 7 / 13 latent frames**; 512 / 768 / 1024 px map to **16 / 24 / 32** latent side.

## The ladder (real V100, bf16, 30-step assumed, int8 dp4a fast path)

| res | frames | latent (f x hw^2) | N tokens | ms/block int8 | ms/step (x48) | full-DiT (30 steps) | fp ms/block | dp4a speedup | dp4a block act VRAM (O(N)) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512^2  | 25 | 4x16^2  | 1024  | 36.1  | 1.73 s  | 52 s   | 79.6  | 2.20x | 180 MiB  |
| 512^2  | 49 | 7x16^2  | 1792  | 64.4  | 3.09 s  | 93 s   | 140.6 | 2.19x | 316 MiB  |
| 512^2  | 97 | 13x16^2 | 3328  | 129.2 | 6.20 s  | 186 s  | 274.2 | 2.12x | 585 MiB  |
| 768^2  | 25 | 4x24^2  | 2304  | 84.9  | 4.08 s  | 122 s  | 185.3 | 2.18x | 405 MiB  |
| 768^2  | 49 | 7x24^2  | 4032  | 164.3 | 7.89 s  | 237 s  | 341.7 | 2.08x | 710 MiB  |
| 768^2  | 97 | 13x24^2 | 7488  | 364.0 | 17.47 s | 524 s  | 731.8 | 2.01x | 1317 MiB |
| 1024^2 | 25 | 4x32^2  | 4096  | 172.2 | 8.26 s  | 248 s  | 352.5 | 2.05x | 720 MiB  |
| 1024^2 | 49 | 7x32^2  | 7168  | 346.1 | 16.61 s | 498 s  | 695.1 | 2.01x | 1260 MiB |
| 1024^2 | 97 | 13x32^2 | 13312 | 830.0 | 39.84 s | 1195 s | OOM   | n/a   | 2340 MiB |

**Assumed 30 denoise steps** (a typical LTX base setting; distilled/turbo LTX configs run
~8 steps — divide the full-DiT column by ~3.75). `ms/step (x48)` is step-count-independent,
so rescale to any step budget. `fp ms/block` is the same block on the fleet's bf16 CUDA-core
path (the fp16 tensor cores are firmware-dead here — see below).

### Headline

**A 97-frame clip on this fleet (30 steps, block-extrapolated): ~3.1 min at 512^2,
~8.7 min at 768^2, ~20 min at 1024^2.** Sweet spot 512-768^2; the practical single-clip
ceiling is **1024^2 x 97f** (N=13312, ~20 min) — beyond it the O(N^2) attention overtakes
the GEMMs and the fp fallback stops fitting 16 GB.

## Scaling law + GEMM-vs-attention crossover

Least-squares fit over the 9 rungs:

    ms/block(N)  ~=  0.0317 * N  +  2.30e-6 * N^2      (rms residual 1.7 ms)

- The **linear term** (0.0317 ms/token) is the int8 dp4a GEMM work: `to_q/k/v`, `to_out`,
  the text cross-attention, and the FFN (4096->16384->4096) — all O(N).
- The **quadratic term** (2.30e-6 ms/token^2) is the O(N^2) self-attention matmuls.
- **Crossover at N ~= 13764 tokens** (a/b): the linear GEMM term dominates the quadratic
  attention term across the *entire practical ladder*. Unlike a normal GPU (where long
  video is attention-bound), here the int8 GEMMs — chiefly the fat FFN — are the largest
  cost right up to ~1024^2 x 97f, because dp4a self-attention is both fast and O(N) memory.
  The dp4a-vs-fp speedup accordingly drifts 2.20x -> 2.01x as N grows (attention's *share*
  rises, muting the GEMM-favoured win), consistent with `docs/ltx23-block-profile.md`.

## VRAM: two regimes (O(N) fast path vs O(N^2) fallback)

- **int8 dp4a FlashAttention self-attn is O(N) memory** (measured on the isolated kernel:
  108 MiB @ N=2304 -> 577 MiB @ N=13312, linear). The whole block activation is then
  O(N) (~0.18 MiB/token, the `dp4a block act VRAM` column) — even the largest clip is
  2.3 GiB, dwarfed by the 18.9 GB of weights.
- **The fp SDPA fallback is O(N^2).** `fni8.attn_int8_fwd` has a SageAttention accuracy
  gate (`detect_q_outlier_domination`): if a Q row has a dominant outlier channel it
  transparently falls back to fp SDPA, and Volta has no flash-SDPA kernel, so it
  materialises the full N x N score matrix. At N=13312 that score buffer alone is ~11 GiB
  and **OOMs a 16 GB card** (the `fp ms/block` = OOM row). On real LTX-2.3 weights the
  self-attn dp4a path engages and clears the gate (block int8-vs-fp cosine 0.9996,
  `docs/ltx23-block-profile.md`), so it stays O(N); the ladder forces the gate off to
  measure that intended fast path deterministically (random probe weights trip the gate
  arbitrarily). Masked / cross-attention paths always use the O(N^2) fp fallback, but their
  context is short (~256), so it is cheap.

## Running it end-to-end (the 2-card path)

Because the 18.9 GB (b4) / 27.6 GB (b8) of **weights** exceed one 16 GB card, a real
LTX-2.3 sample needs the pipeline split (`install_ltxav_pipeline`, issue #73): the 48
blocks are distributed across 2 GPUs with an fni8 transport-codec handoff at the stage
boundary. Note the split fits the *weights*, not the activations — the per-block O(N^2)
fp-SDPA fallback (if the outlier gate fires) still needs its N x N scores on whichever card
runs that block, so 1024^2 x 97f wants the dp4a path to stay engaged regardless. The
pipeline path is LLM-focused today; wiring it into the ComfyUI LTX sampler node is the
remaining step to a real single-box e2e video (flagged, not done here).

## Fleet caveat (AGENTS.md)

Every card here is an NVIDIA CMP 100-210 (GV100 silicon) — the "Tesla V100" ones just have
a V100 VBIOS. NVIDIA firmware-gimped the FP16/TF32 **tensor cores** to ~5-6% of a real
V100, so the fp16-tensor-core path (SDPA-flash, a normal video-DiT baseline) is artificially
slow here; integer/CUDA-core paths are intact and **int8 dp4a (~46 TOP/s) is the fast
path**, ~6.7x the crippled fp16-tensor-core throughput. These numbers are honest for THIS
deployment fleet and **do not transfer to a datacenter V100** (where fp16 tensor cores
would beat dp4a). int8 additionally halves HBM/smem for weights and K/V.
