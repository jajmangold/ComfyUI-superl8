# LTX-2.3 distilled native-GGUF one-card validation

Date: 2026-07-16. Hardware: fleet-specific CMP 100-210 / flashed GV100 card,
16 GiB HBM2. These timings do not transfer to a normal V100.

## Result

The stock ComfyUI-GGUF loader can run the official LTX-2.3 22B distilled Q4_K_S
transformer on one card when its matching text projection and AV connector assets are
loaded once and evicted before denoising. The fni8 patch keeps GGUF k-quant weights in
their native representation and routes eligible linears and attention through DP4A.

Validated inputs:

- `ltx-2.3-22b-distilled-1.1-Q4_K_S.gguf`
- `ltx-2.3-22b_embeddings_connector.safetensors`
- `ltx-2.3-22b_text_embedding_projection.safetensors`
- cached Gemma hidden states for the fox prompt
- the official distilled 8-step sigma sequence

At video latent `(1, 128, 9, 32, 32)` and audio length 24:

| Run | Denoise | Peak HBM | DP4A attention |
|---|---:|---:|---:|
| stale fni8 ABI, 8 steps | 465.6 s | 14.00 GiB | 0/0 (safe fp rescue) |
| merged fni8, calibration step | 105.4 s | 14.00 GiB | 96/96 pass |
| merged fni8, 2 steps total | 140.6 s | 14.00 GiB | 96/96 pass |
| merged fni8, steady state (difference) | 35.2 s/step | 14.00 GiB | int8 only |
| synchronized AV, 8 steps, 65 audio latents | 369.7 s | 14.00 GiB | 96/96 pass |
| synchronized AV, full-row calibration step | 105.6 s | 14.00 GiB | 96/96 pass |
| synchronized AV, 256-row calibration step | 69.5 s | 12.65 GiB | 96/96 pass |

Steady-state DP4A is 39.5% faster than the stale-ABI fp path's 58.2 s/step. Including
the one-time accuracy calibration, the measured values project to about 352 s for the
official 8-step denoise versus 465.6 s before the image refresh (24% overall). This is
an inference from the measured one- and two-step runs; it is not presented as a full
8-step timed measurement.

Q3_K_S used 11.57 GiB but took 95.5 s for one long-sequence step and recorded no DP4A
attention decisions under the stale image. It is rejected as the speed tier: its native
unpack/linear path was substantially slower than Q4_K_S, despite the extra headroom.

## Correctness finding

The old direct harness multiplied sigma by 1000 before the LTX forward. LTX-2.3 uses
ComfyUI's `ModelSamplingFlux`, whose `timestep()` is the identity. The bad timestep
produced finite tensors and colorful structured noise, so finite/SQNR-only checks did
not catch it. A regression test now pins the exact timesteps seen by the model.

With identity timestep scaling, the decoded output is a coherent orange fox moving
through autumn leaves, matching the cached prompt. The decoded tensor was
`(1, 65, 1024, 1024, 3)` with standard deviation 0.2982. Representative artifact hashes:

- frame 000: `0b295bd8e662f6079fd0fa4165379a4c735b05de78ee48bf1cb46eb1677f6a7c`
- frame 032: `549a3d46e3599afc9446c5f91029ac708b07a0e580da44c4ae83a5cb044af619`
- rejected pre-fix frame 000: `a72c3a4911299a0e991cd6c746a662253031c411045aa4bec0b33f54e54dd289`

Generated PNGs remain outside git under
`<build-artifacts-dir>`.

## Deployment ABI finding

ComfyUI-superl8 calls `fni8.attn_int8_fwd(..., internal_accuracy_gate=False)` so its outer
multi-site SQNR gate can measure the real DP4A output. The old `fni8-built:sm70` image
did not accept that keyword; the adapter caught the exception and correctly rescued to
the tiled fp kernel, but performance silently regressed. Both `fni8-built:sm70` and
`comfyui-superl8-e2e:latest` were refreshed from fni8 merge `ce12d982` and labeled with
`fni8.git_commit=ce12d982`.

The benchmark entry point is `bench/full_pipeline_ltx_gguf.py`.

## Synchronized audio

The video VAE expands 9 latent frames to 65 decoded frames. At 25 fps the official
LTX audio VAE contract requires 65 audio latents (`ceil(video_seconds * 25 audio
latents/second)`), not the earlier ad-hoc length of 24. The AV harness now derives this
length by default, retains the denoised audio trajectory, selectively loads only the
`audio_vae.` and `vocoder.` subtrees from the official checkpoint, and writes PCM16 WAV
without requiring TorchCodec.

The exact synchronized run produced:

- video: `(1, 65, 1024, 1024, 3)`, std 0.2945, coherent fox prompt result;
- audio: stereo 48 kHz, 123,360 samples (2.57 s), RMS 0.0886;
- audio range `[-0.4309, 0.3874]`, zero fraction 0.018%, clipped fraction 0%;
- video decode 87.1 s and audio decode 7.2 s after the 369.7 s denoise;
- WAV SHA-256 `21af84b77c8be2ae611a039daa9b7eeb7de8902dc54662713f09fc7ec5b1fbf8`;
- frame 032 SHA-256 `960c974249c4a5be45c25064cecce333400ff90565e3e5f4f6fd3f74aaf38d76`.

The waveform gate proves synchronized duration, finite non-silent output, valid stereo
serialization, and no clipping. It is not a semantic audio-quality oracle. Artifacts are
outside git under `<build-artifacts-dir>`.

## Bounded GGUF calibration

Native `.fni8` linears already bound their one-time fp accuracy reference to 256 evenly
spread activation rows. Native GGUF linears accidentally evaluated the fp reference on
every row, making the first long-sequence step slow and temporarily consuming another
1.35 GiB. GGUF now uses the same deterministic row sampler while still computing and
returning the full native output. A sampled failure still recomputes and returns the full
fp output.

Exact one-step synchronized AV A/B, with only
`FNI8_SQNR_CALIB_MAX_ROWS` changed from 256 to 999999:

- bounded: 69.5 s, 12.65 GiB;
- full reference: 105.6 s, 14.00 GiB;
- video latent SHA-256, both runs:
  `56e57e6cd8c64ea495ac309b254fe0c033714352622336fbf1259d00b28d1a37`;
- audio latent SHA-256, both runs:
  `78820433156c5df9add054dfbcc12e520ee3289661a6aee52a5afb0fa4113988`;
- 96/96 attention sites passed in both runs.

The first step is 34.2% faster with byte-identical video and audio outputs. Steady-state
steps are unchanged because calibration runs only once per linear.
