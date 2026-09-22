# Z-Image (image) — int8 dp4a resolution ladder

How int8 dp4a `fni8` scales the Z-Image (Turbo) image DiT across output resolution,
measured **end-to-end** on a real Tesla V100 (sm_70), 6-step Turbo schedule, int8 dp4a.
Unlike the LTX-2.3 video ladder (`docs/ltx23-perf-ladder.md`, which is block-extrapolated
because the 18.9 GB DiT can't fit one card), Z-Image fits a single 16 GB card, so these are
**full-pipeline wall-clock** numbers with real peak VRAM.

## The ladder (real V100, 6-step Turbo, int8 dp4a)

| resolution | time | peak VRAM |
|---|---:|---:|
| 512^2  | 5.7 s  | 6.4 GiB  |
| 768^2  | 13.9 s | 6.4 GiB  |
| 1024^2 | 28.5 s | 6.5 GiB  |
| 1536^2 | 88 s   | 7.1 GiB  |
| 2048^2 | 203 s  | 7.9 GiB  |
| 3072^2 | 812 s  | 10.2 GiB |

### Headline

**Sweet spot 1024^2 (~28 s); practical max ~2048^2 at ~3.4 min.** Z-Image is
**compute-bound with no OOM boundary** on this card — flash attention keeps VRAM O(N)
linear, so peak grows only 6.4 -> 10.2 GiB from 512^2 to 3072^2 while time grows 140x.
(Tiling the DiT is a **net loss** here — the extra passes cost more than the VRAM headroom
is worth, since VRAM was never the limit.)

## Scaling law + GEMM-vs-attention crossover

    time(N)  ~=  0.84 * N  +  7.8e-5 * N^2   ms      (N = latent tokens)

- Linear term = int8 dp4a GEMMs (projections + FFN), O(N).
- Quadratic term = O(N^2) self-attention.
- **GEMM/attention crossover ~= 1660 px**: below it the run is GEMM-bound; above it the
  O(N^2) attention takes over and the time curve steepens (hence 2048^2 -> 3072^2 costing
  4x for 1.5x the side).

## Fleet caveat (AGENTS.md)

These numbers are for the CMP 100-210 / VBIOS-"V100" fleet, whose FP16/TF32 **tensor
cores** are firmware-gimped to ~5-6% of a real V100. That is why int8 dp4a on the CUDA
cores (~46 TOP/s) is the fast path here — the reverse of a datacenter GPU. **Fleet-specific;
does not transfer to a real V100.**
