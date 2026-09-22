# Z-Image-Turbo full-pipeline findings (Qwen3-4B → int8 DiT → tiled VAE → image)

## What was run

With the tiled VAE decode (#97) landed, the full generation pipeline that
`int8-dit-validation.md` had deferred ("a full multi-step + VAE-decode + LPIPS pass
wouldn't fit on the 16 GB card") now runs end to end on one 16 GB Volta/CMP (sm_70) card:

    Qwen3-4B text encoder  ->  int8 dp4a Z-Image DiT (8-step Turbo denoise, 512 px)
                           ->  tiled VAE decode (#97)  ->  a decoded RGB image

- **DiT:** the published `Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8` (present on the fleet),
  loaded by `UnetLoaderFNI8` (int8 dp4a linears + bidirectional int8 FlashAttention).
- **Text encoder:** the real Qwen3-4B (Z-Image's TE), loaded through ComfyUI's own
  Z-Image TE (`<|im_start|>user…` template, hidden layer −2) — real conditioning
  `[1, S, 2560]`, not a random placeholder.
- **VAE:** the real Z-Image AutoencoderKL (16-ch, Flux-style, `scaling_factor` 0.3611),
  decoded through the pack's tiled VAE primitive (#97).
- **fp reference:** the *same* conditioning and the *same* VAE, DiT weights dequantized
  to bf16 (Z-Image is bf16-native — fp16 overflows to black). Only the DiT compute path
  differs (int8 dp4a GEMM + int8 attention vs bf16 torch matmul + fp SDPA).

Driver: `bench/full_pipeline_zimage.py`. e2e gate: `tests/e2e/test_full_pipeline_zimage.py`.

## Result 1 — the pipeline fits 16 GB (the #97 memory goal: PASS)

Stages load sequentially (TE freed before the DiT, DiT freed before the VAE), so peak HBM
is the max of any single stage, not the sum:

| Stage | Peak HBM (measured, GV100/sm_70, 16 GB card) |
|---|---|
| int8 dp4a DiT — 8-step denoise @ 512 px | **8.25 GiB** |
| tiled VAE decode (#97), 4 tiles @ 512 px | **≤ 9.03 GiB** |

The int8 denoise latent is finite and non-constant across all 8 steps, and the tiled VAE
decodes it to a full 512×512×3 image. **The full multi-step + VAE-decode pass the earlier
doc said "wouldn't fit" now fits and runs.**

## Result 2 — int8 image quality: was a collapse, now FIXED (coherent)

Originally the **int8 path collapsed to an incoherent 2×2-patch-grid image** (PSNR ≈
7.6 dB vs the coherent fp fox) — structured and finite (so the old "finite + non-constant"
smoke and the 1-step latent-cosine gate both PASS), but not a real picture. The
root cause below was found and fixed; the int8 path now decodes to a coherent,
prompt-faithful fox that matches the fp reference.

int8 image vs fp image (decoded, 512 px, 8-step Turbo):

| Metric | before (collapse) | AFTER FIX |
|---|---|---|
| PSNR (decoded image) | **≈ 7.6 dB** | **31.18 dB** (≥ 28 dB "same image") |
| latent cosine (8-step) | ≈ 0.01 | **0.9952** |
| SSIM | — | **0.9664** |
| image | patch-grid noise | coherent red fox (matches fp) |
| peak HBM (whole pipeline) | — | 12.05 GiB (card = 16 GiB) |

## Root cause — NOT precision: a qkv-fusion loader bug (corrected diagnosis)

The original hypothesis (int8 `final_layer`/`proj_out` head) was **wrong** — those layers
are not even int8 in the published Z-Image `.fni8` (they load raw/fp already). A
controlled bisection (`bench/diag_zimage_isolate.py`, engaging int8 exactly as
`UnetLoaderFNI8` does) settled it, one knob at a time, all measured on GPU 4:

- **int8 attention is NOT the cause.** Attention-patch ON vs OFF gave a *bit-identical*
  collapsed latent (cos 0.0093, PSNR 7.37 dB) — int8 self-attention makes no difference.
- **Dtype is NOT the cause.** Running the int8 DiT in bf16 vs fp16 was bit-identical
  (both 7.37 dB). (The `.fni8` meta records `native_dtype: float16`, but forcing bf16
  changed nothing.)
- **The int8 linears are the cause — specifically the attention projections.** Keeping
  the attention `to_q`/`to_k`/`to_v`/`to_out` fp while FFN stays int8 → **PSNR 22.1 dB,
  cos 0.97, coherent**; the reverse (attention int8, FFN fp) stayed collapsed.

The mechanism (`bench/diag_zimage_keys.py`): the published `.fni8` stores attention in
**diffusers layout — separate `to_q`/`to_k`/`to_v`/`to_out` projections**, but ComfyUI's
native Z-Image (S3-DiT) model uses a **FUSED `attention.qkv.weight`**. ComfyUI's
diffusers→native conversion **concatenates** the three projections into `qkv` during
load. That concat is correct for fp weights (which is why the fp reference is perfect) but
**corrupts int8** — three independently per-row-scaled int8 code blocks cannot share one
tensor — and leaves the fused `qkv` key unmatched by `assign_int8_weights`, so the
attention weights end up as garbage int8 codes copied into fp params. Of the 240 int8
tensors in the file, only the 104 FFN/token weights matched a model param; all 136
attention projections missed. Garbage attention destroys spatial token-mixing → each
output patch is independent → the 2×2-patch-grid artifact, compounded over the 8
(non-self-correcting) Turbo steps into full collapse.

## The fix — round 1 (#103): keep the attention projections fp

The first fix loaded the four attention projections fp: `UnetLoaderFNI8` dequantized their
int8 weights back to fp at load so comfy's qkv-concat operated on fp tensors — identical to
the (correct) fp path. FFN stayed int8. This made the image coherent (**PSNR 31.18 dB**) but
left attention OFF the dp4a path — "full W8A8" only for the FFN.

## The fix — round 2 (this PR): int8 attention via LOSSLESS qkv re-fusion

The round-1 note below called for a converter-side "pre-fuse + re-quantize". It turns out no
re-quantization is needed — the fusion is **lossless at load time**, done in
`loader.fuse_attention_qkv_int8` (gated per-arch by `DiTArch.fuse_attn_qkv`, True for
`zimage`):

- **Why the naive concat corrupted int8:** three `per_row_i8` blocks each carry their own
  per-row fp32 scale; concatenating the raw int8 CODES while forcing ONE shared scale is
  garbage — which is exactly what comfy's `convert_diffusers_mmdit` narrow-copy produces.
- **Why the re-fusion is lossless:** `per_row_i8` stores one scale PER OUTPUT ROW. The fused
  `attention.qkv.weight` is `[3*dim, dim]`, and its rows are just q's rows, then k's, then v's
  (comfy's own offsets `q=[0:dim], k=[dim:2dim], v=[2dim:3dim]`). So the fused weight = concat
  of the three code blocks AND concat of their scale vectors — **bit-identical** to the three
  shipped tensors, zero added error. `to_out.0` is a 1:1 `[dim,dim]` linear → just re-keyed to
  native `attention.out`.
- **Plumbing:** the loader still dequantizes the diffusers projections to fp so comfy's convert
  BUILDS the model on the proven fp path; the fused int8 weights (native-keyed) then OVERWRITE
  those fp params via `assign_int8_weights` — exactly how the int8 FFN is re-attached today.
  All **34** attention blocks (`layers` + `noise_refiner` + `context_refiner`) come up as int8
  `attention.qkv`/`attention.out` FNI8Tensors on the dp4a path.

**Measured (GPU 4, real `Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8`, 8-step/512px):**

| Build | attention | image PSNR vs fp | latent cos | SSIM |
|---|---|---|---|---|
| #103 | fp projections, int8 FFN | 31.18 dB | 0.9952 | 0.9664 |
| this PR | **int8 dp4a projections + int8 FFN** | **27.81 dB** | 0.9907 | 0.9419 |

int8 attention costs ~3.4 dB vs fp attention (the honest price of int8-quantizing the qkv/out
weights) but stays a coherent "same image", comfortably above the 20 dB "same image" floor.
"Full W8A8" (the compute-dominant attention projections + FFN on dp4a) is now real; the tiny
always-fp head/norms stay fp per the accuracy-gate doctrine.

Video DiTs (Wan/LTX) keep **separate** q/k/v natively (loader `_WAN_RULES`/`_LTX_RULES`), so
they never needed this and already ran int8 attention. Other fused-qkv image DiTs
(Flux/SD3/Qwen-Image MMDiTs) can adopt the same lossless re-fusion by setting
`fuse_attn_qkv=True` once verified per-arch.

> Note (superseded): round 1 speculated the converter should pre-fuse + re-quantize the qkv
> (with K-smoothing, since the published weight has `smoothed: 0`). The load-time LOSSLESS
> re-fusion above makes converter changes unnecessary for correctness — the fused int8 is
> bit-identical to the shipped per-block int8, so no second quantization (and thus no
> smoothing step) is introduced. A converter that ships a pre-smoothed qkv could still
> RAISE quality above 27.81 dB, but is now an optimization, not a prerequisite.

## Verdict

- **Full pipeline plumbing (encoder → DiT → tiled VAE → image): works, and fits 16 GB**
  (12.05 GiB peak).
- **Z-Image-Turbo is COHERENT end-to-end on the FULL int8 path** — attention projections AND
  FFN on dp4a (PSNR 27.81 dB, cos 0.991, SSIM 0.94 vs the fp reference). The earlier collapse
  was purely the corrupt qkv concat; a lossless per-row re-fusion runs int8 attention with no
  added quantization error and keeps the image well above the "same image" floor.

## Reproduce

```
FNI8_GPU=<free non-server GPU> docker compose run --rm --entrypoint bash e2e -c \
  'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
   custom_nodes/ComfyUI-superl8/bench/full_pipeline_zimage.py --steps 8 --size 64 \
   --outdir /opt/ComfyUI/custom_nodes/ComfyUI-superl8/.pipe_out'
# writes zimage_int8.png (coherent fox) and zimage_fp.png (coherent fox) + prints
# peak HBM and int8-vs-fp PSNR/SSIM/cosine (~31 dB after the fix).
#
# Isolation / bisection harness (attention vs FFN vs dtype, keep-fp toggles):
#   custom_nodes/ComfyUI-superl8/bench/diag_zimage_isolate.py
# Which int8 keys match a model param (the qkv-fusion miss):
#   custom_nodes/ComfyUI-superl8/bench/diag_zimage_keys.py
```
