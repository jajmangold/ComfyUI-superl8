# Recipe: LTX-2.3 (text → video) — PLACEHOLDER

- **Workflow:** [`example_workflows/ltx-2.3_t2v_fni8.json`](../../example_workflows/ltx-2.3_t2v_fni8.json)
- **Status:** **PLACEHOLDER — PENDING 8-bit e2e validation (task in flight).** The int8 DiT
  loads and its denoise core is validated via a 2-GPU pipeline-parallel split, but there is
  **no single-card render and no full prompt→frames clip** today. Do not treat as working.
  The shipped JSON is the intended *shape* (loader → pipeline-parallel split), not a runnable
  graph through stock `KSampler` (which re-collapses the split).
- **Source:** ComfyUI's LTX-Video templates + this pack's `UnetLoaderFNI8` /
  `FNI8PipelineParallel`. See `docs/ltx23-single-card-findings.md`, `docs/e2e-coverage.md`.

## What is validated (2-card split, real weights)

- DiT-denoise core over a 2-GPU pipeline split (base doesn't fit one card): resident
  **cuda:0 12.32 GB, cuda:1 11.21 GB**; split faithfulness vs non-split **cos ~1.0**.
- Transport codec on the real boundary activation `[1,2048,4096]` bf16: int8 **cos 0.99996,
  SQNR 41 dB, 2.0× on-wire** (8.4 vs 16.8 MB), modelled 33.6 ms vs 67 ms on the 250 MB/s
  PCIe-1.0-x1 link; int4 gives 4× / SQNR 16 dB.
- Source: git commit `c0909a7`, `bench/validate_ltx23_split.py`, `tests/e2e/test_pipeline_ltx.py`.
- Artifacts from an earlier direct-drive: `.pipe_out/ltx23_firstclip/…` frames + `.mp4`.

## Checkpoints

| Role | File | Notes |
|---|---|---|
| DiT (int8) | `Lightricks__LTX-2.3.dit.b8.fni8` (**23.5 GB** = 16.6 video + 6.9 audio) | `> 16 GB` → 2-card split; base 22B (not distilled) |
| Text encoder (fp) | Gemma-3-12B | **NOT on the fleet** — blocker |
| VAE (fp) | LTX 3D causal VAE (+ audio VAE + vocoder) | config dropped from the `.fni8` — blocker |

Build-host: `<weights-dir>/Lightricks__LTX-2.3.dit.b8.fni8`. `arch="ltx_video"`
(`clip_type="ltxv"`). The `b4` file is `per_group_i4` — **not** runnable on the `per_row_i8`
dp4a path.

## Three blockers to a first real clip

1. **Fit:** need a mixed-precision `per_row_i8` DiT ≤ ~14 GB for a single card (distillation
   is a *speed* lever, not a *fit* lever — same 22B / ~23.5 GB).
2. **Text encoder:** no Gemma-3-12B on the fleet (only a wrong `gemma-4` + `zimage_te`).
3. **VAE config** dropped from the `.fni8` → ComfyUI auto-detect builds the wrong VAE; the
   scalar forward flags (`rope_type`, `causal_temporal_positioning`, timestep-scale) also
   need the source diffusers `transformer` config (converter-side fix).

Also: `scale_shift_table` adaLN tables were over-quantized to int8 (should stay fp), and the
split is a memory-**fit** strategy, **not** a speedup. Flip this recipe to ran-e2e only after
a single-card 8-bit render produces good frames.
