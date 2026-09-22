# Recipe: FLUX.1-dev (text → image) — NEEDS-WORK (int8 attention)

- **Workflow:** none shipped yet — **withheld on purpose.** FLUX's int8 *attention* is not yet
  correct on this pack (root cause + fix below), so a `flux_t2i_fni8.json` would resolve its
  nodes but produce a degraded image. Per `example_workflows/README.md`'s honesty bar we do not
  ship a workflow we cannot label **ran-e2e** with a good artifact. Ship it once the fix lands.
- **Status:** **needs-work.** The `.fni8` DiT loads and passes the one-step smoke (finite +
  non-constant, `test_smoke_flux1.py`) — but "smoke passes" says nothing about image quality
  (the Z-Image collapse also passed smoke). Two blockers, one correctness + one memory:

## Blocker 1 (correctness) — the Z-Image qkv-fusion hazard, here UN-fixed

Verified by the `.fni8` header inventory (`FQReader`) + ComfyUI source, not assumed:

- The published `black-forest-labs__FLUX.1-dev.dit.b8.fni8` stores attention as **separate**
  diffusers projections — `single_transformer_blocks.#.attn.to_q/to_k/to_v` and
  `transformer_blocks.#.attn.add_q_proj/add_k_proj/add_v_proj`.
- ComfyUI's native FLUX model uses a **FUSED** `self.qkv = Linear(dim, dim*3)` per attention
  (`comfy/ldm/flux/layers.py`: `img_attn.qkv`, `txt_attn.qkv`). Its diffusers→native load
  **concatenates** `to_q/to_k/to_v` into that one `qkv` tensor.
- That concat is exactly the Z-Image bug: three **independently per-row-scaled** int8 code
  blocks cannot share one tensor/scale, and the fused `qkv` key matches **no** int8 tensor kept
  aside, so `assign_int8_weights` misses the attention projections → garbage int8 attention →
  the same 2×2-patch-grid collapse Z-Image showed pre-fix (`docs/zimage-full-pipeline-findings.md`).
- `arch.py`'s `flux1` has **neither** `fuse_attn_qkv=True` **nor** attention `fp_dequant`, so the
  Z-Image fix is **not engaged** for FLUX.

**The fix (arch-scoped, same class as Z-Image):** give `flux1` either
`fuse_attn_qkv=True` with a FLUX-keyed re-fuser (losslessly concat the separate per-row int8
codes+scales into `img_attn.qkv`/`txt_attn.qkv`, so attention stays dp4a) **or**, as the safe
interim, `fp_dequant=("to_q","to_k","to_v","add_q_proj","add_k_proj","add_v_proj")` so the
attention projections dequantize to fp at load (FFN stays int8) — identical to Z-Image's
minimal-fp set. This is a load-time arch fix, no csrc change. Contrast: **Qwen-Image /
Qwen-Image-Edit keep q/k/v separate in comfy too, so they need no such fix** (see their recipe).

## Blocker 2 (memory) — 17 GB int8 > 16 GB card

`FLUX.1-dev.dit.b8.fni8` is **17 GB** int8; the fp-dequant reference is ~34 GB. Neither fits a
single 16 GB fleet card, so even after Blocker 1 is fixed, the decoded int8-vs-fp image metric
needs a ≥24 GB GPU or a 2-card split (`FNI8PipelineParallel` + `TiledVAEDecode`). FLUX.1-dev is
guidance-distilled (cfg=1), so the uncond branch is auto-skipped (`GUIDANCE_DISTILLED_ARCHS`).

## Checkpoints (for when the fix lands)

| Role | File | Place in |
|---|---|---|
| DiT (int8) | `black-forest-labs__FLUX.1-dev.dit.b8.fni8` (~17 GB) | `models/diffusion_models/` |
| Text encoders (fp) | CLIP-L + T5-XXL (dual) | `models/text_encoders/` |
| VAE (fp) | FLUX 16-ch AE — `ae.safetensors` | `models/vae/` |

`arch="flux1"`. The `b4` file is `per_group_i4` — not on the `per_row_i8` dp4a path.
Flip this recipe to **ran-e2e** only after the attention fix + a ≥24 GB/2-card render produce a
coherent image with a measured int8-vs-fp PSNR/SSIM.
