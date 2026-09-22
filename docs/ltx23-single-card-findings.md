# LTX-2.3 single-card e2e + distilled-variant evaluation (findings)

Goal: render a real LTX-2.3 video clip end-to-end on ONE 16 GB card (int8 dp4a), and
evaluate the distilled variant as a size/speed lever. This doc records what was built,
what runs today, and the concrete external blockers that gate the first real clip. It
corrects two assumptions carried into the task.

## UPDATE 2026-07-16 — distilled assets found; sampler harness corrected

The archive/NVMe audit found the production assets that the older notes below said were
absent: the full distilled-1.1 safetensors source, a b8 `.fni8`, and Q3_K/Q4_K GGUFs.
The checked-in direct denoise harness was also still using uniform `linspace(1,0)` sigmas.
It now matches ComfyUI's resolution-shifted `LTXVScheduler` and accepts the official
distilled eight-step schedule explicitly:

`1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0`

Earlier finite/structured renders are not prompt-quality evidence: archived base/mixed
and ad-hoc GGUF outputs are visibly garbled. The next production gate is a stock
ComfyUI-GGUF load using the matching external text projection/connectors plus this exact
schedule, followed by a decoded-image/video oracle.

## UPDATE 2026-07-13 — FIRST SINGLE-CARD CLIP DECODED (blockers #1 + #3 closed)

The first real single-card LTX-2.3 clip is on disk. Two of the three blockers below are
now closed; the run is the default pipeline (no distillation LoRA, no re-conversion):

- **Blocker #1 (fit) — CLOSED.** The shrunk `Lightricks__LTX-2.3.dit.b4-conn-i4.fni8`
  (embeddings-connectors → `per_group_i4` W4A8, DiT blocks int4) is **14.12 GB resident**
  and runs on ONE card. The `per_group_i4` W4A8 DiT path is wired to `gemm_w4a8` (PR #115).
- **Blocker #3 (video-VAE config) — CLOSED (this PR, #116).** The already-shipped `.fni8`s
  carry the raw `vae.*` weights (170 keys) but their meta has **no `config`** — the
  converter never persisted the VAE config on any on-disk checkpoint. `comfy.sd.VAE`
  auto-detect then mis-sizes the LTX video VAE (512-vs-256, dies at `decoder.up_blocks.4`).
  Fix: `loader.ltx2_vae_metadata` now falls back to a **packaged canonical LTX-2.3
  `CausalVideoAutoencoder` config** (`comfyui_superl8/data/ltx23_vae_config.json`) for any
  LTX-2.3 bundle lacking a persisted config — building the EXACT VAE (170/170 weights,
  0 missing) so the latent decodes. A converter that persists `meta['config']['vae']` still
  supersedes it (`ltx2_persisted_vae_config` is checked first).
- **Blocker #2 (Gemma-3 TE) — still open.** Conditioning is the placeholder Gemma-3 context
  (deterministic random), so the clip is finite + structured but **not prompt-faithful**.
  Wiring the real Gemma-3 TE (`google__gemma-3-12b-it.text.b8.fni8`, on the fleet) is the
  remaining gap for a *prompt→content* clip; it does not gate the decode.

**Deliverable clip:** `.pipe_out/ltx23_firstclip/` (17 PNGs + `ltx23_firstclip.mp4`).
512×512, 3 video-latent frames → **17 video frames** (8× temporal), 6 flow-match Euler
steps, seed 0, GPU 9 (real Tesla V100). Frames finite + structured (std ≈ 62–67, 92 k
unique colours, temporal Δ ≈ 31). **Wall-time: load 136.8 s / denoise 49.6 s (8.3 s/step) /
VAE 24.5 s = 211 s total. Peak HBM 14.17 GiB (card = 16 GiB).** Repro:
`full_pipeline_ltx.py --unet /dev/shm/ltx_i4.fni8 --hw 16 --frames 3 --steps 6 --no-fp-ref`.

## TL;DR

- **Distillation is a SPEED lever, NOT a fit lever.** LTX-2.3-distilled is the *same 22B*
  parameters as the base (distillation cuts denoise **steps** 40→8, not weights), so its
  int8 DiT is the same ~23.5 GB (b8) / ~18.9 GB-file (b4) as the base. It does **not** help
  the single-card fit; the mixed-precision shrink is required either way. Distilled's value
  is ~**5× fewer denoise steps** on top of whatever fits.
- **The distilled checkpoint is a SEPARATE download** (`Lightricks/LTX-2.3` ships
  `ltx-2.3-22b-distilled` / `-distilled-1.1` alongside the `-dev` base). It is **NOT on the
  fleet** — we only have the base `Lightricks/LTX-2.3`. Evaluating distilled requires
  downloading (~43 GB bf16) + converting to a `.fni8`, then shrinking it (same as base).
- **The `.fni8` we call "LTX-2.3" is the BASE (dev) 22B, not distilled** (meta
  `repo: Lightricks/LTX-2.3`, 48 blocks; `docs/e2e-coverage.md`'s "(distilled)" label was
  wrong — corrected here).
- **No shipped LTX-2.3 checkpoint yields a single-card render today:** b8 int8 DiT = 23.5 GB
  (> 16 GB); the b4 file is `per_group_i4` (4-bit packed) which the pack's `per_row_i8` dp4a
  path does **not** support (loader `_dequant_qtensor` assumes a per-row scale and throws).
- **The harness (`bench/full_pipeline_ltx.py`) is built and its DiT-denoise core is
  validated on the real weights** (2-GPU split, since the base doesn't fit one card). It
  drives the denoise straight through `dit.forward` (the `apply_model`-level path — no
  `comfy.sample.sample`, which is broken on this build for LTX's dual conditioning). It
  auto-selects single-card the moment a DiT that fits is passed.
- **Two more external (converter-side) gaps block the first *prompt→frames* clip**, both
  now pinned down: (a) **no Gemma-3 text encoder on the fleet**; (b) the **LTX-2.3 video
  VAE config is dropped from the `.fni8`** so ComfyUI auto-detect builds the wrong VAE.

## What was built

`bench/full_pipeline_ltx.py` — single-card LTX-2.3 e2e harness:
- Loads the DiT via `UnetLoaderFNI8(arch="ltx_video")` (accepts `--unet <name>`; defaults to
  the base b8, ready to take the shrunk `<=14 GB` per_row_i8 DiT).
- **Fit-aware placement**: measures resident DiT bytes; if `<= --budget-gb` (15 GB) → runs on
  `cuda:0` alone (the deliverable path); else → transparently falls back to the existing
  2-GPU pipeline-parallel split (`install_ltxav_pipeline`) so the pipeline is exercisable on
  today's 23.5 GB weights.
- **Flow-matching Euler denoise** driven through `dit.forward(x=[vx, ax], timestep, context,
  frame_rate, audio_length)` — the same faithful forward `bench/validate_ltx23_split.py`
  exercises, wrapped in a real N-step integration. No `comfy.sample`/`KSampler`.
- **Quality gate**: latent cosine ≥ 0.985 int8-vs-fp (fp = same DiT dequantized in place,
  identical conditioning) + frame coherence (finite, non-constant, structured). Runs on the
  single-card topology (the split mutates `forward`; use `validate_ltx23_split.py` for the
  split's own int8-vs-fp cosine).
- **Per-stage wall-time** (load / denoise / vae) printed and compared to the PR#114 ladder.
- Conditioning: placeholder Gemma-3 context (deterministic random, the stable operating
  point the split validator uses) until a real TE is wired; `--no-decode`, `--no-fp-ref`.

`tests/e2e/test_full_pipeline_ltx.py` — e2e gate (skips cleanly without weights/GPU).

## The three external blockers to the first real clip (ranked)

1. **Fit — the shrunk DiT (sibling agent, pending).** b8 int8 DiT 23.5 GB > 16 GB; b4 is an
   unsupported `per_group_i4` scheme. A mixed-precision **per_row_i8** DiT ≤ ~14 GB drops the
   harness onto the single card with no code change (just `--unet <shrunk name>`).
2. **Text encoder — Gemma-3-12B is not on the fleet.** The fleet has `google__gemma-4-*.fni8`
   (wrong model, and LLM-`.fni8` not a ComfyUI TE) and `zimage_te` only. Real prompt
   conditioning needs Gemma-3 provisioned as a ComfyUI text encoder (`ltxav_te` +
   `LTXAVGemmaTokenizer` in `comfy.text_encoders.lt`). Until then the int8-vs-fp *quality
   gate* is valid (identical conditioning both paths) but there is no *prompt faithfulness*.
3. **Video VAE config — dropped from the `.fni8`.** The bundle carries the raw `vae.` weights,
   but `comfy.sd.VAE` auto-detect (comfy/sd.py ~L638) sizes the LTX video VAE from a single
   conv shape + built-in `version` presets (0/1/2); LTX-2.3's VAE matches none (fails at
   `decoder.up_blocks.4+`: 512-vs-256 / 1024-vs-256 channel mismatches). It needs
   `metadata["config"]["vae"]` — the **same converter-side gap** as the DiT transformer
   config (`loader.ltx2_persisted_transformer_config`). Fix: persist `config.vae` in the
   `.fni8` meta, or infer the full VAE config from shapes. This is the converter's job.

Consequence: item 1 gives the first single-card **int8 denoise** (validate-able against the
fp reference); items 2+3 are additionally required for a **prompt → decoded-frames** clip.

## Base vs distilled — the single-card recommendation

| | base (`-dev`) | distilled (`-distilled-1.1`) |
|---|---|---|
| params / int8 DiT size | 22B / ~23.5 GB (b8) | 22B / ~same (distillation ≠ smaller) |
| denoise steps | ~40 | ~8 |
| CFG / guidance | 4.0 | 1.0 |
| on the fleet? | yes (base) | **no — separate download + convert needed** |
| fits one card? | only after the shrink | only after the shrink (same size) |

**Recommendation:** use **base-mixed-precision (shrunk)** for the *first* single-card LTX-2.3
video — it is the checkpoint we have, and the shrink is the only thing gating fit. Pursue
**distilled** as the *production speed* path: once fit is solved for the base, converting +
shrinking the distilled checkpoint buys ~5× fewer denoise steps (8 vs 40) at the same
per-step cost and same VRAM. Distilled is the fastest single-card path, but it is strictly
*downstream* of the shrink — it does not remove the fit blocker, so it is not a shortcut
around the sibling agent's work.

## Measured wall-time vs the extrapolated ladder

See the "RESULT" block from `full_pipeline_ltx.py` and `bench/_ltx_denoise_walltime.txt`
(per-step DiT time from the 2-card split run). NOTE: timing on GPU 12/13 is on **CMP
100-210** cards (fleet-representative, NOT a pristine real-V100 number), and the 2-card split
serialises stages + pays a codec handoff per boundary per step, so its per-step time is an
**upper bound** on the true single-card int8 per-step cost. The single-card number lands once
the shrunk weights arrive; that is the apples-to-apples comparison to the PR#114 ladder.

## Reproduce

```
# Core DiT denoise (real int8 blocks, multi-step), 2-card split on today's b8:
FNI8_GPU=12,13 docker compose run --rm --entrypoint bash e2e -c \
  'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI:custom_nodes/ComfyUI-superl8 python3 \
   custom_nodes/ComfyUI-superl8/bench/full_pipeline_ltx.py --steps 8 --frames 25 --hw 64 \
   --no-decode --no-fp-ref'

# Single-card full path (once a <=14 GB per_row_i8 DiT exists):
FNI8_GPU=13 docker compose run --rm --entrypoint bash e2e -c \
  'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI:custom_nodes/ComfyUI-superl8 python3 \
   custom_nodes/ComfyUI-superl8/bench/full_pipeline_ltx.py --unet <shrunk-name> --steps 8 \
   --frames 25 --hw 64'
```
