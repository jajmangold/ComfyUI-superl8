# Z-Image Turbo resident service

Issue #181 standardizes the content-factory Z-Image production lane on the
quality-validated `.fni8` artifact. The service is a bounded, single-slot HTTP
daemon with exact provenance, safe GPU election, sequential TE/DiT/VAE
residency, and fail-closed startup checks.

## Canonical artifacts

Production reads only the real files under this NVMe root:

```text
<SUPERL8_MODEL_ROOT>/
  unet/Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8
  text_encoders/qwen_3_4b.safetensors
  vae/ae.safetensors
```

`comfyui_superl8/zimage_profile.py` pins the exact paths and checksums. Startup
rejects a different root, missing files, symlinks, or any checksum mismatch.
The DiT checksum is:

```text
1051c94040c0b831c5f1a307d66d51d19a812da45a7742e814782b2027a0355b
```

The native GGUF `unet/z_image_turbo-Q5_K_M.gguf` is retained as a separate
candidate. It is not selected by the production profile, resident service, or
benchmark defaults. Earlier claims that no Z-Image `.fni8` existed were wrong;
the artifact had remained in the archive and is now restored to NVMe.

## Locked workflow

The request API cannot override production sampling:

| Property | Value |
|---|---|
| Steps | 8 |
| CFG | 1.0 |
| Flow shift | 3.0, read back from the loaded model |
| Sampler | `res_multistep` |
| Scheduler | `simple` |
| Negative conditioning | empty |
| Cache | disabled |

This matches the official ComfyUI Z-Image Turbo workflow. Width and height
must be multiples of the VAE stride (8); fractional latent geometry is rejected
instead of silently cropped.

## Readiness gates

The service does not report ready until all of these pass:

1. The external launcher elects and locks an eligible GPU by UUID before Torch
   starts, and the child verifies that it sees exactly that one GPU.
2. All three canonical files pass SHA-256 verification from NVMe.
3. ComfyUI detects Z-Image and reports `model_sampling.shift == 3.0`.
4. The exact `.fni8` inventory contains 172 `FNI8Tensor` parameters: 170 Linear
   module weights (34 fused QKV, 34 attention-output, and 102 FFN) plus exactly
   the non-Linear `x_pad_token` and `cap_pad_token` embeddings.
5. After `load_models_gpu(..., force_full_load=True)` handles ordinary model
   state, `move_fni8_tensors_` explicitly replaces every direct custom
   parameter with `param.to(elected_device)`. This moves all 172 int8 code and
   sidecar `q_scale` tensors while preserving scheme/group/codebook metadata.

After each denoise, the request fails if any of the 170 Linear modules is pending or
if an fp fallback lacks a finalized SQNR verdict, complete calibration samples, and a
measured cosine below the configured quality threshold. An evidenced fp fallback is
the intentional quality-preserving route, not failed int8 engagement. `/ready` reports
each fallback layer and its calibration evidence alongside structural and last-request
runtime engagement.
The TE/VAE eviction path keeps the resident DiT's `LoadedModel` by object
identity; post-request readiness and e2e tests verify all 172 custom tensors
remain resident after those evictions. Readiness recomputes device placement
rather than returning the startup inventory snapshot.

The full-pipeline oracle remains the acceptance gate: real Qwen3-4B
conditioning, eight denoise steps, tiled VAE decode, and decoded-image metrics
against the dequantized BF16 reference. Recorded validation for this artifact
is 12.05 GiB peak HBM, latent cosine 0.9907, PSNR 27.81 dB, and SSIM 0.9419.

## Launch and benchmark

Never invoke the service module directly. The launcher performs GPU election
and holds the per-UUID lock for the child lifetime:

```bash
cd <ComfyUI-root>
PYTHONPATH=<ComfyUI-root> python3 \
  custom_nodes/ComfyUI-superl8/scripts/launch_resident_service.py -- --port 8410
```

The benchmark uses the same canonical profile and safe election path:

```bash
python3 bench/resident_service_benchmark.py --outdir /tmp/zimage-resident-bench
```

The runtime container must launch from `<ComfyUI-root>` with
`PYTHONPATH=<ComfyUI-root>`; otherwise ComfyUI's top-level `comfy` package is not
importable when the resident child starts.

`GET /health` is liveness. `GET /ready` is the fail-closed model gate.
`POST /generate` accepts `width`, `height`, `prompt`, `seed`, `timeout_s`,
`tile_size`, and `overlap`; unknown fields and sampling overrides are rejected.

The service checks cancellation between stages, after every denoise step, and
before every VAE tile. It is cooperative: the maximum overrun is one in-flight
TE pass, denoise step, or VAE tile. A supervisor must provide a process-level
hard deadline when that bound is insufficient.
