# Recipe: Qwen3-TTS (text → speech) — DESIGN-ONLY (no workflow yet)

- **Workflow:** none. Qwen3-TTS has **no ComfyUI nodes** in this pack yet, so there is no
  JSON to ship (per the `example_workflows/README.md` contract, design-only families get a
  recipe stub and no un-resolvable JSON).
- **Status:** **design / scoping only.** No model code, no kernel, no `.fni8`, **no measured
  numbers.** Source: `docs/qwen3-tts-fni8-design.md` ("Status: design / scoping only").

## What exists

- A design doc proposing target archs `qwen3_tts` / `qwen3_tts_talker` + a code-predictor,
  to be registered in the fni8-serve `convert.py` / `config.py` (not built).
- No bench file. The only numbers in the doc are **estimates / Alibaba-published** figures
  (e.g. ~1.2 GB / ~3.4 GB fp16 for the 0.6B / 1.7B backbones; streaming first-packet
  97–150 ms / RTF 0.25–0.29) — **not** fleet measurements; do not cite them as fni8 results.
- No local `.fni8` weights; only unquantized wan2gp assets under
  `<tts-models-dir>`.

## Design notes (from the doc)

- No new attention/GEMM kernel needed — reuses the existing int8 dp4a linear + attention path.
- Codec / causal-ConvNet decoder, BigVGAN, RVQ lookup, RMSNorm/softmax stay **fp**
  (reconstruction load-bearing).
- M-RoPE (`mrope_section: [24,20,20]`) needs a variant (host cos/sin precompute for an MVP).
- Precedent (other families, not TTS): int8 attention on Qwen2-0.5B GQA cost +0.23 %
  perplexity, 24/24 layers routed, 0 fallbacks.

## To make this a real recipe

1. Register the arch + build a `.fni8` in fni8-serve; add ComfyUI TTS loader/predict nodes.
2. Run an int8-vs-fp audio quality eval (e.g. speaker-sim / WER / mel-cos) and record VRAM +
   RTF on the fleet.
3. Add `example_workflows/qwen3-tts_tts_fni8.json` and flip this stub to structure-validated
   / ran-e2e.
