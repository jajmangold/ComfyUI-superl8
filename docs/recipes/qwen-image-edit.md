# Recipe: Qwen-Image-Edit-2509 (instruction image edit)

- **Workflow:** [`example_workflows/qwen-image-edit_edit_fni8.json`](../../example_workflows/qwen-image-edit_edit_fni8.json)
- **Status:** **structure-validated + int8-attention structurally verified** — JSON loads and
  every node type resolves; the DiT loads via `UnetLoaderFNI8` and passes a one-step-denoise
  smoke (finite + non-constant). Its int8 **attention keys match cleanly** (see "Known limits"),
  so int8 attention engages without the Z-Image qkv-fusion hazard. **Not** run e2e to a finished
  edited image on this fleet: the **20 GB int8 DiT does not fit a 16 GB card** (nor does the
  ~40 GB fp reference), so a single-card decoded int8-vs-fp image metric is **memory-blocked**
  here — it needs a ≥24 GB GPU or a multi-GPU split. No PSNR/SSIM/cosine yet for this reason.
- **Source workflow:** ComfyUI's official *Image Edit (Qwen 2511)* blueprint and the
  `nunchaku-qwen-image-edit-2509` example; the `NunchakuQwenImageDiTLoader` is swapped for
  `UnetLoaderFNI8` and the CLIP/VAE loaders for `FNI8ComponentLoader`.

## Node graph

```
UnetLoaderFNI8 ─► ModelSamplingAuraFlow ─► CFGNorm ─► KSampler ─► VAEDecode ─► SaveImage
LoadImage ─► ImageScaleToTotalPixels ─┬─► TextEncodeQwenImageEditPlus(pos).image1
                                      └─► VAEEncode ─► KSampler.latent_image
FNI8ComponentLoader ─CLIP/VAE─► TextEncodeQwenImageEditPlus (pos + neg) ─► KSampler
FNI8ComponentLoader ─VAE──────► VAEEncode, VAEDecode
```

Single reference-image edit (`TextEncodeQwenImageEditPlus` also accepts `image2`/`image3`
for multi-image edits). DiT runs int8 dp4a; Qwen2.5-VL TE and VAE stay fp.

## Checkpoints

| Role | File | Place in |
|---|---|---|
| DiT (int8) | `Qwen__Qwen-Image-Edit-2509.dit.b8.fni8` (~19 GB) | `models/diffusion_models/` |
| Text encoder (fp) | `qwen_2.5_vl_7b_fp8_scaled.safetensors` (Qwen2.5-VL-7B) | `models/text_encoders/` |
| VAE (fp) | `qwen_image_vae.safetensors` | `models/vae/` |

Build-host: `<weights-dir>/Qwen__Qwen-Image-Edit-2509.dit.b8.fni8` (a `-2511`
build and `b4` variants also exist — `b4` is `per_group_i4`, **not** runnable on the
`per_row_i8` dp4a path). TE + VAE fp sources under
`<model-root>/{text_encoders,vae}/`.
`arch="qwen_image_edit"`.

## Measured (CMP 100-210 / V100 fleet)

- Smoke: one-step denoise output is **finite + non-constant** (`test_smoke_qwen_image.py`,
  `docs/e2e-coverage.md`).
- **int8 attention keys verified to MATCH** (build-only probe `bench/diag_qkv_keymatch.py` +
  `.fni8` header inventory): the published `.fni8` stores attention as **separate**
  `to_q`/`to_k`/`to_v`/`add_q_proj`/… and ComfyUI's `QwenImageTransformer2DModel` **also keeps
  them separate** (`comfy/ldm/qwen_image/model.py` — `self.to_q/to_k/to_v`, NOT a fused `qkv`).
  So every int8 attention projection lands on a model param and runs dp4a — **no** qkv-fusion
  corruption. This is the concrete correction to the earlier "may need fp_dequant" worry below.
- **No decoded int8-vs-fp image metric yet** (PSNR/SSIM/cosine): the 20 GB int8 DiT (+ ~40 GB
  fp reference) exceeds a single 16 GB fleet card — see Known limits. The quantitative decoded
  metric that exists for Z-Image needs a ≥24 GB card or a 2-card split to reproduce here.

## Known limits

- **20 GB `.fni8`** does not fit a 16 GB card (int8 DiT alone > 16 GB; the fp-dequant reference
  is ~40 GB) — run on a ≥24 GB GPU, or split with `FNI8Multigpu`/`FNI8PipelineParallel` across
  two sm_70 cards (+ `TiledVAEDecode`). This is the sole blocker to a decoded image metric here.
- **Correction (was a false alarm):** the earlier note that Qwen-Image, being an MMDiT, "may
  need per-arch `fp_dequant` like Z-Image" does **NOT** apply. The Z-Image hazard was that its
  S3-DiT **fuses** `to_q/to_k/to_v` into one `attention.qkv` (three per-row int8 scales can't
  share one tensor). ComfyUI's Qwen-Image attention keeps q/k/v **separate**, matching the
  `.fni8`'s separate keys, so there is no fusion and no fix is needed. Verified, not assumed.
- Treat sampler settings (20 steps, cfg 2.5) as ComfyUI-template defaults, not a quality claim
  (no decoded-image quality run on this fleet — memory-blocked).
