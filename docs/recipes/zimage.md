# Recipe: Z-Image-Turbo (text → image)

- **Workflow:** [`example_workflows/zimage_t2i_fni8.json`](../../example_workflows/zimage_t2i_fni8.json)
- **Status:** **ran-e2e** — produced a coherent image on a 16 GB Volta/CMP (sm_70).
- **Source workflow:** ComfyUI's official *Text to Image (Z-Image-Turbo)* template
  (`ComfyUI/blueprints/`), and the community `nunchaku-z-image-turbo` example; the DiT
  loader is swapped for `UnetLoaderFNI8` and the CLIP/VAE loaders for `FNI8ComponentLoader`.

## Node graph

```
UnetLoaderFNI8 ──► ModelSamplingAuraFlow ──► KSampler ──► VAEDecode ──► SaveImage
FNI8ComponentLoader ─CLIP─► CLIPTextEncode ─► KSampler.positive
                                           └► ConditioningZeroOut ─► KSampler.negative
FNI8ComponentLoader ─VAE──────────────────────────────────────────► VAEDecode
EmptySD3LatentImage ──► KSampler.latent_image
```

8-step Turbo, `cfg=1` (negative conditioning zeroed), `euler`/`simple`. The DiT runs int8
dp4a (incl. losslessly re-fused int8 `qkv` attention); TE (Qwen3-4B) and VAE stay fp.

## Checkpoints

| Role | File | Place in |
|---|---|---|
| DiT (int8) | `Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8` (~5.9 GB) | `models/diffusion_models/` |
| Text encoder (fp) | Qwen3-4B — ComfyUI-Org packaged `qwen_3_4b.safetensors` | `models/text_encoders/` |
| VAE (fp) | Z-Image AE — `ae.safetensors` | `models/vae/` |

Build-host source: `<weights-dir>/Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8`;
TE shards under `.../zimage_te/`, VAE under `.../zimage_vae/`. (The workflow references the
Comfy-Org single-file `qwen_3_4b.safetensors` / `ae.safetensors` a ComfyUI user actually
drops in `models/`, not the sharded build-host copies.) `arch="zimage"`.

## Measured (CMP 100-210 / V100 fleet — not real-V100)

- int8-vs-fp **decoded image PSNR 29.39 dB**, latent **cosine 0.9932**, **SSIM 0.9546**
  (fresh 8-step, 512 px run on current comfy + fni8-master, post-#130/#132; int8 attention
  fully on dp4a — qkv=34/out=34 blocks, 0 fp fallback). Corroborates the earlier ~31 dB
  (`docs/zimage-full-pipeline-findings.md`); both are comfortably "same image" (> 28 dB).
- Single-step latent **cosine 1.000000**; self-attention **SQNR 47.2 dB** (kept int8) —
  `docs/int8-dit-validation.md`, `bench/validate_int8_attn_sqnr.py`.
- Peak HBM whole pipeline **12.05 GiB** (DiT stage 8.25 GiB; tiled VAE decode 9.03 GiB).
- Artifacts: `.pipe_out/zimage_int8.png` (+ `zimage_fp.png`).

## Known limits

- The published `.fni8` stores attention as separate `to_q/to_k/to_v`; ComfyUI's S3-DiT
  fuses them to `attention.qkv`. Early int8 collapse (PSNR 7.6 dB) was that loader fusion
  bug — fixed by re-fusing per-row int8 losslessly (`DiTArch.fuse_attn_qkv`) / dequantizing
  the projections to fp; **FFN stays int8**. Do not "fix" this by weakening tolerances.
- Published weight has `smoothed: 0` (no K-smoothing). Native dtype is bf16 — fp16 would
  overflow to black; `UnetLoaderFNI8` forces bf16 for this arch.
