# LTX-2.3 one-card optimization plan

Date: 2026-07-16

## Current default and immediate result

The default remains the quality-preserving two-stage path: generate 9 latent frames
at 256x448 and 12 fps, apply the official learned spatial and temporal x2 upscalers,
then run three refinement steps at 512x896 and 24 fps. This matches the official
recommendation to use a two-stage distilled pipeline and the model's supported
temporal upscaler.

The first decoder sweep found that Comfy's automatic LTX VAE dtype is already FP32
on this fleet. BF16 is not a usable alternative on Volta: cuDNN cannot select an
engine for the causal 3D convolution. FP32 is therefore explicit in the harness.

Comfy's stock 16 GiB heuristic limits native decoder intermediates to about 85 MiB.
At the default 17x28x16 latent shape, the measured speed/memory sweep on locked GPU 7
was:

| Native chunk budget | Decode | Peak allocator | Result |
| ---: | ---: | ---: | --- |
| stock (~85 MiB) | 67.9 s | 5.41 GiB | baseline |
| 192 MiB | 59.3 s | 10.31 GiB | default knee |
| 256 MiB | 58.8 s | 13.00 GiB | only 0.5 s faster; poor headroom |
| 192 MiB, resident weights | 36.8 s | 12.41 GiB | fleet default |

The tuned and resident outputs were both bitwise identical to stock on fixed-noise
real-VAE checks covering 33 decoded frames (`max_abs=0`, `relative-L1=0`). The VAE is
2.705 GiB, so full residency fits after the DiT is evicted. A Torch profile of dynamic
mode attributed 15.0 of 20.4 seconds sampled CUDA time to 208 pageable H-to-D copies,
versus 4.3 seconds to 94 3D convolutions. Full residency removes that PCIe 1.0 x1
streaming pathology. The exact-shape delta projects the measured 251-second window to
about 228 seconds and the 30-second, 17-frame-overlap clip to roughly 24-27 minutes.
These numbers are specific to the CMP 100-210 fleet. The focused Nsight Compute
artifact is [`bench/ltx23_vae_resident_192mib_ncu.csv`](../bench/ltx23_vae_resident_192mib_ncu.csv).

## Primary-source review mapped to this pipeline

1. **Official two-stage and second-order sampling.** Lightricks calls its two-stage
   pipeline the production path, its distilled two-stage pipeline the fastest path,
   and its `res_2s` variant a way to use fewer steps at comparable quality. We already
   use the distilled two-stage structure and official learned upscalers. The next
   source shows that every `res_2s` step performs two DiT evaluations. Its 15-step HQ
   default therefore replaces 30 Euler evaluations with 30 second-order evaluations:
   it is a quality method, not a raw step-rate win for our three-evaluation distilled
   refinement. Two `res_2s` steps would cost four evaluations here, so it was rejected
   for the coherent-fast default.
   [Official pipeline guide](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md)

2. **TeaCache and TaylorSeer.** TeaCache predicts output change from modulated inputs
   and reports up to 4.41x acceleration on much longer video schedules. TaylorSeer
   forecasts features instead of reusing them and reports about 5x on HunyuanVideo.
   Our schedule has only eight stage-one and three refinement evaluations, so the
   headline ratios cannot transfer: stage two has almost no safe cache horizon and
   stage one is only 43 seconds. Existing whole-output caching must remain opt-in until
   it clears final-latent and decoded-motion gates; blockwise forecasting is the more
   promising follow-up.
   [TeaCache paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html),
   [TaylorSeer paper](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_From_Reusing_to_Forecasting_Accelerating_Diffusion_Models_with_TaylorSeers_ICCV_2025_paper.html)

3. **Block-wise caching.** BWCache observes that block outputs vary differently over
   time and reports up to 2.24x without training. It is a better fit than reusing the
   entire LTX audio-video prediction because sensitive blocks can still run. It needs
   LTX-specific block telemetry and decoded A/V quality gates before implementation.
   [BWCache paper](https://openreview.net/pdf?id=5bJZtzTFYy)

4. **Decoder replacement.** Flash-VAED targets the bottleneck we now measure directly:
   causal 3D video decoding. It reports roughly 6x VAE and up to 36% end-to-end gains
   on Wan and LTX through channel pruning, operator optimization, and distillation.
   This is the highest-ceiling path, but it requires trained replacement weights; it
   is not a safe inference-only patch to the official VAE. Until compatible LTX-2.3
   weights exist, larger native chunks and operator profiling are the correct local
   work.
   [Flash-VAED paper](https://arxiv.org/abs/2602.19161)

5. **Long-video noise rescheduling.** FreeNoise improves long-range consistency by
   rescheduling initial noise and using local temporal windows. It addresses seam and
   drift quality, not the number of one-card window evaluations, so it belongs in the
   30-second assembly gate rather than the step-rate optimization.
   [FreeNoise paper](https://openreview.net/pdf?id=ijoqFqSC7p)

## Ordered experiments

1. **Completed:** retain the VAE fully, use a 192 MiB decoder chunk, and profile it.
2. **Completed:** compare two-evaluation refinement subsets. Keeping `[0.85, 0.725,
   0.0]` reduced refinement from 108.7 to 78.3 seconds at 34.7 dB versus the official
   result, with 97.2% motion magnitude. Keeping the low-sigma point instead lost more
   motion and was rejected. The coherent-fast preset now projects about 198 seconds
   after load per window and 21-24 minutes for 30 seconds. The raw comparison is in
   [`bench/ltx23_refinement_sweep.csv`](../bench/ltx23_refinement_sweep.csv).
3. Test temporal-upscale-after-refine plus one low-sigma full-temporal polish step;
   reject it unless motion coherence matches temporal-before-refine.
4. Instrument per-block LTX feature deltas, then evaluate BWCache/Taylor forecasting
   only where the decoded A/V oracle stays green.
5. Evaluate a distilled Flash-VAED-compatible LTX-2.3 decoder when weights and license
   are available; do not substitute an untrained pruned decoder.
