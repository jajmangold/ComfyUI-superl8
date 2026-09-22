# LTX-2.3 portrait resolution sweep (CMP 100-210)

Date: 2026-07-16

## Result

The default coherent-fast portrait configuration is **512x896 with a 129-frame
output window**, generated at 12 fps and finished at 24 fps by applying the
learned temporal x2 upscaler before the two-evaluation coherent-fast refinement.
With the resident FP32 VAE default, it projects a synchronized 30-second clip
at about 21-24 minutes on one card.

Use **576x1024** when native 9:16 framing and additional spatial detail matter
more than throughput. It is the resolution-only quality/time knee, but it is no
longer the overall pipeline default. Resolutions above it add substantial
denoise time without a consistent visual gain.

These results are fleet-specific. The cards are CMP 100-210 devices with GV100
silicon and firmware-crippled tensor cores, not ordinary V100s.

## Method

- LTX-2.3 22B distilled Q4_K_M GGUF plus the fni8 VAE bundle
- one card per job, fni8 attention enabled, 96/96 eligible calls on DP4A
- 8 official distilled denoise steps, seed 0, identical prompt
- 9 video latent frames, producing 65 decoded frames at 25 fps
- synchronized audio generated for every sample
- model loads were warm from the archive drive cache
- three independent cards ran concurrently; denoise is GPU-local, but VAE and
  model-load timings are noisier because the jobs shared host and storage

Resolution is reported as decoded width x height. `tokens` is the video token
count presented to each transformer block.

| Resolution | Tokens | Denoise | Denoise/step | VAE decode | Audio | Peak allocator |
|---|---:|---:|---:|---:|---:|---:|
| 384x672 | 2,268 | 78.3 s | 9.79 s | 75.6 s | 8.0 s | 13.44 GiB |
| 448x800 | 3,150 | 105.3 s | 13.16 s | 51.8 s | 5.5 s | 13.45 GiB |
| 512x896 | 4,032 | 132.8 s | 16.60 s | 57.3 s | 5.2 s | 13.45 GiB |
| **576x1024** | **5,184** | **173.8 s** | **21.73 s** | **50.7 s** | **7.7 s** | **13.46 GiB** |
| 640x1120 | 6,300 | 217.9 s | 27.24 s | 99.3 s | 8.8 s | 13.47 GiB |
| 704x1248 | 7,722 | 278.7 s | 34.84 s | 99.5 s | 5.1 s | 13.73 GiB |
| 768x1344 | 9,072 | 353.9 s | 44.24 s | 94.8 s | 5.0 s | 13.98 GiB |

All seven samples were coherent on visual inspection. Composition changes with
resolution because the seeded noise tensor has a different shape, so this is
not a paired super-resolution comparison. The useful conclusion is the cost of
each adequate portrait tier, not a claim that one composition is objectively
better. At 768x1344, driver-observed use was about 15.5 GiB during compute,
leaving too little operational margin for a default.

## Step-rate experiment at 576x1024

| Steps | Denoise | Speedup vs 8 | Frame tensor std | Visual result |
|---|---:|---:|---:|---|
| 8 | 173.8 s | 1.00x | 0.2805 | detailed reference |
| 6 | 135.2 s | 1.29x | 0.2738 | coherent, good structure |
| 4 | 94.3 s | 1.84x | 0.2229 | coherent, visibly softer |

The 6- and 4-step runs are experiments, not production presets yet. The current
harness uses LTX's generic resolution-shifted schedule whenever the step count
is not eight; only the 8-step run uses the official distilled sigma sequence.
Before promoting a fast preset, test a deliberately selected/subsampled
distilled schedule against the 8-step reference and across multiple prompts.

The timing difference fits an approximately 14-second per-process calibration
cost plus 20 seconds per steady denoise step at 576x1024. Reusing the loaded
model and calibration in a sliding-window process should therefore approach a
2x denoise speedup at four steps.

## Estimated 30-second sliding-window cost

### Production two-stage correction

The initial sweep used native full-resolution denoising. LTX-2.3's production
path instead generates at half resolution for eight steps, applies the learned
x2 spatial latent upscaler, and refines at full resolution for three steps. An
exact 576x1024 run of that Wan2GP-compatible path measured:

| Component | Time |
|---|---:|
| Load and context | 99.6 s |
| 288x512 stage-one denoise (8 steps) | 50.4 s |
| Learned spatial x2 | 19.6 s |
| 576x1024 refinement (3 steps) | 67.5 s |
| VAE decode | 36.7 s |
| Audio decode | 5.2 s |

The output was coherent across the 65-frame clip, audio remained synchronized,
all 144 eligible attention sites selected DP4A, and peak allocator use was
13.52 GiB. Keeping the loaded model, calibration decisions, and a CPU-resident
upsampler across windows projects the conservative 30-second sliding render at
about **35-40 minutes**. This replaces the native-full-resolution estimate as
the production baseline.

### Temporal x2 placement

The published temporal x2 latent upscaler was tested with five input latent
frames conditioned at 12 fps, producing nine output latent frames / 65 decoded
frames at 24 fps. Audio length was derived from the final duration.

| Placement | Transformer stages | Upscalers | Decode + audio | Finding |
|---|---:|---:|---:|---|
| none | 117.9 s | 19.6 s | 41.9 s | coherent quality baseline |
| after refinement | 69.3 s | 16.8 s | 41.4 s | faster, visible leaf/leg smearing |
| **before refinement** | **102.3 s** | **18.7 s** | **41.6 s** | coherent; acceptable balanced path |

Post-refinement temporal upscaling is rejected as a quality default because its
synthesized latents receive no diffusion cleanup. Applying temporal x2 before
the three-step full-resolution refinement removes that failure, but saves only
about 9% of measured warm-path time for a 65-frame window because refinement
still processes all final temporal tokens. It remains useful as a balanced
preset and as the correct basis for larger-window experiments, not as a route
to a tenfold speedup.

### Recommended coherent fast preset

The larger-window operating point was validated at **512x896**, with nine base
latent frames conditioned at 12 fps, temporal x2 before refinement, and 17 final
latent frames / 129 decoded frames at 24 fps:

| Component | Time |
|---|---:|
| Stage-one denoise | 42.6 s |
| Spatial x2 | 15.8 s |
| Temporal x2 | 3.8 s |
| Three-step refinement | 108.7 s |
| VAE load + stock dynamic decode | 75.1 s |
| Audio decode | 5.0 s |

The run stayed on one card, selected DP4A for all 144 eligible attention sites,
and peaked at 13.70 GiB allocated (about 15.3 GiB driver-observed). The saved
inspection span remained coherent and artifact-free. This is close enough to
the memory ceiling that 129 output frames should be the production maximum.

An exact-shape follow-up reduced decode from 67.9 to 36.8 seconds by retaining
the 2.705 GiB FP32 VAE on device and raising its native chunk budget to 192 MiB;
the output was bitwise identical with fixed VAE noise. Dropping only the final
interior refinement sigma (`0.4219`) reduced stage two from 108.7 to 78.3 seconds.
Across the saved 64-frame inspection span it remained 34.7 dB from the official
result, retained 97.2% of its temporal motion magnitude, and retained 95.4% of
its second-difference motion. The alternative that dropped the `0.725` midpoint
was worse (33.5 dB and 92.0% motion) and was rejected.
The measured rows are recorded in
[`bench/ltx23_refinement_sweep.csv`](../bench/ltx23_refinement_sweep.csv).

A four-card component-staged run now keeps the GGUF DiT, both learned
upscalers, FP32 video VAE, and audio VAE/vocoder on logical devices 0/1/2/3.
At the vertical default it measured 43.1 s for stage one, 16.6 + 5.3 s for
the upscalers, 79.2 s for refinement, 68.5 s for first-use VAE load + decode,
and 5.5 s for audio. Peak allocated HBM by role was 13.70 / 1.40 / 12.41 /
0.85 GiB. The final video latent hash matched the one-card reference and all
64 saved PNGs were bitwise identical. This proves placement and residency; it
does not yet claim an end-to-end speedup because this harness still executes
one window serially. Overlapping decode with the next window is separate work.

With a 17-frame overlap, 721 final frames need six 129-frame windows plus a
49-frame tail. Reusing the loaded model, gate decisions, and CPU-resident
upsamplers now projects **about 21-24 minutes for a synchronized 30-second clip**.
This is the recommended coherent fast preset. The 576x1024 / 65-frame path
remains the higher-resolution option; post-refinement temporal x2 remains a
preview-only speed mode.

For the conservative 65-frame window with 17-frame overlap at 24 fps, 721
frames require 15 windows (65 initial frames, thirteen 48-frame advances, and a
49-frame tail). Based on this sweep rather than a completed 30-second render:

| Preset | Estimated denoise | Estimated serial end-to-end |
|---|---:|---:|
| 576x1024, native 8 steps | about 39 min | about 60-70 min |
| 576x1024, two-stage 8+3 | about 22 min | about 35-40 min |
| 576x1024, 6 steps | about 30 min | about 50-55 min |
| 576x1024, 4 steps | about 20 min | about 40-45 min |

End-to-end estimates include repeated window VAE decode, audio, initial load,
overlap handling, and muxing. They should be replaced by an actual sliding-run
measurement. Keeping decoded overlap on device and overlapping decode/audio
with the next denoise are the largest remaining pipeline opportunities; they do
not change the transformer step rate itself.

## Default reproduction

The GGUF benchmark accepts rectangular decoded dimensions through
`--width`/`--height`; both must be multiples of 32. For example:

```bash
python bench/full_pipeline_ltx_gguf.py \
  --width 512 --height 896 --frames 9 --steps 8 \
  --frame-rate 12 --output-frame-rate 24 \
  --temporal-before-refine \
  --refinement-preset coherent-fast \
  --stage-devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --vae-dtype fp32 --vae-chunk-mib 192 --vae-resident \
  --gguf /models/ltx-2.3-22b-distilled-Q4_K_M_light.gguf \
  --connector /models/ltx-2.3-22b_embeddings_connector.safetensors \
  --projection /models/ltx-2.3-22b_text_embedding_projection.safetensors \
  --upscaler /models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --temporal-upscaler /models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors \
  --vae-stats /models/ltx-2.3-22b_vae.safetensors \
  --vae-bundle /models/Lightricks__LTX-2.3.dit.b4-conn-i4.fni8 \
  --audio-bundle /models/ltx-2.3-22b-audio-vae-vocoder.safetensors
```
