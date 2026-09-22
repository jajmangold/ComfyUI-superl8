# Per-architecture e2e coverage (issue #12)

Tracks `tests/e2e/`'s per-arch matrix: load a real published `.fni8` DiT via
`UnetLoaderFNI8`, run one denoise step, assert the output latent is finite and
non-constant (the black-image/fp16-overflow guard — see `tests/e2e/_common.py`).
Run with `docker compose run --rm e2e` (needs the weights archive mounted at
`<weights-dir>` and a Volta/CMP sm_70 GPU; each test skips cleanly
without either).

**CI lane (not a per-PR blocker):** this matrix loads real 17-26 GB `.fni8` DiTs and
runs a GPU denoise per arch, so it takes 40+ min and is **not** part of the required
`build-test` gate (which runs only the fast unit tests). It runs in the opt-in
`e2e-full` job — add the **`run-e2e`** label to a PR to trigger it (e.g. when a change
touches the load/denoise path). See `.github/workflows/ci.yml`.

**Metric caveat:** "Pass" in the *Test* column means finite + non-constant against a
one-step denoise — a numerical-sanity check that does **not** prove image quality (the
Z-Image collapse below also passed it). A real **quantitative int8-vs-fp comparison** now
exists for the image archs; the honest per-arch picture (see
[`int8-dit-validation.md`](int8-dit-validation.md) + `docs/recipes/`):

| Image arch | Fits 16 GB card? | int8 attention keys | Decoded int8-vs-fp | Verdict |
|---|---|---|---|---|
| **zimage** (5.9 GB) | yes | fused → **re-fused losslessly** (`fuse_attn_qkv`) | **PSNR 29.39 dB, cos 0.9932, SSIM 0.9546** (fresh post-#130/#132 8-step 512 px run, real images saved; corroborates the earlier ~31 dB) | **CERTIFIED** |
| **qwen_image / qwen_image_edit** (20 GB) | **no** (int8 alone > 16 GB) | separate q/k/v → **match cleanly, no fusion bug** | memory-blocked on this fleet | usable-with-caveat (needs ≥24 GB / 2-card) |
| **flux1** (17 GB) | **no** | separate q/k/v in `.fni8` but comfy **fuses to `qkv`** and no fix is engaged → **miss → garbage attn** | not run (would collapse) | **needs-work** (add `fuse_attn_qkv`/`fp_dequant` for `flux1`) |

Only **Z-Image** has a real saved decoded image with a good measured number — it is the
one certified image workflow. Qwen-Image(-Edit)'s int8 attention is structurally clean but
its 20 GB DiT (and ~40 GB fp reference) exceed a single 16 GB fleet card, so the decoded
metric is memory-blocked here. FLUX needs the same arch-scoped qkv fix Z-Image got before
its int8 image can be trusted. The video rows (Wan / LTX-Video v1 / LTX-2.3) **load** with
int8 engaged (sections below); a per-arch quantitative video comparison is still follow-up,
and LTX-2.3's full denoise is memory-bound on a 16 GB card (23.5 GB int8 DiT).

| Arch | Checkpoint(s) | Test | Status | Notes |
|---|---|---|---|---|
| zimage | Tongyi-MAI/Z-Image-Turbo | `test_smoke_zimage.py` | ✅ pass | First validated arch (issue #9 / PR #23) |
| flux1 | black-forest-labs/FLUX.1-dev | `test_smoke_flux1.py` | ✅ pass | `object_patch` attn seam, guidance-distilled (cfg=1) |
| qwen_image | Qwen/Qwen-Image | `test_smoke_qwen_image.py` | ✅ pass | 5-D `[B,C,T,H,W]` latent (not 4-D like Flux/Z-Image) |
| qwen_image_edit | Qwen/Qwen-Image-Edit-2509 | `test_smoke_qwen_image.py` | ✅ pass | Same `QwenImageTransformer2DModel`/unet_config as base; `ref_latents` conditioning is optional at the model level, so a plain denoise step exercises the same DiT path |
| qwen_image_edit | Qwen/Qwen-Image-Edit-2511 | `test_smoke_qwen_image.py` | ✅ pass | Same as above |
| wan22 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | `test_smoke_video_diffusers_gap.py` + `test_smoke_wan.py` | ✅ loads (int8) | diffusers→native remap (#30); 5B fits 16GB, runs a denoise step |
| wan22 | Wan-AI/Wan2.2-T2V-A14B-Diffusers | `test_smoke_video_diffusers_gap.py` | ✅ loads (int8) | Same remap; 14B skips the denoise on a 16GB card (`load_dit_or_skip`) |
| ltx_video | Lightricks/LTX-Video | `test_smoke_video_diffusers_gap.py` | ✅ loads (int8) | diffusers→native remap (#30); verified this pass (was stale xfail) |
| ltx_video | Lightricks/LTX-2.3 (distilled) | `test_smoke_video_diffusers_gap.py` | ✅ loads (int8, structural) | Native `ltxav` bundle (NOT a diffusers-remap case): loader extracts `model.diffusion_model.*` + infers the dropped shape-config; builds native `LTXAVModel`, int8 resident. Forward needs the full diffusers config + >16 GB — see "LTX-2.3" below |

## Wan2.2 / LTX-Video (v1): loaded via the #30 diffusers→native remap

`wan21`, `wan22`, and `ltx_video` are registered in `comfyui_superl8/arch.py` (attn-seam /
key-prefix / text-encoder / VAE metadata — covered by `tests/test_core.py`'s
`test_arch_registry_has_popular_families`). The published `.fni8` DiTs for Wan and
LTX-**Video v1** were quantized from **diffusers** checkpoints, whose state-dict keys
differ from ComfyUI's native `WanModel` / `comfy.ldm.lightricks` layout (e.g. Wan:
`blocks.N.attn1.to_q`/`attn2.*`/`ffn.net.0.proj`/`scale_shift_table` vs. native
`blocks.N.self_attn.q`/`cross_attn.*`/`ffn.0`/`ffn.2`/`modulation`). ComfyUI's
`detect_unet_config` ships a diffusers remap only for a handful of *image* MMDiTs
(`convert_diffusers_mmdit`), none for Wan/LTX — so **on the raw diffusers keys**
detection returns `None`:

```python
>>> comfy.model_detection.detect_unet_config(wan_or_ltxv1_diffusers_keys, "")
None   # still true — this is why the pack's remap exists
```

**This gap is now closed in the pack (issue #30):** `loader.remap_diffusers_to_native`
rewrites the keys to native before `comfy.sd.load_diffusion_model_state_dict`, and
`UnetLoaderFNI8` re-asserts the int8 weights onto the built model. Verified against the
real mounted `.fni8`s: Wan 2.2 (5B/14B) and LTX-Video v1 build as native
`WanModel`/`LTXVModel` with int8 `FNI8Tensor` weights resident. The earlier "❌ xfail"
verdict predated #30 and was stale.

## LTX-2.3: a native `ltxav` bundle — loads structurally; forward blocked on dropped config

LTX-2.3 is **not** a diffusers-remap case. Its published `.fni8` is a *bundled full
checkpoint* (~28 GB, 5947 keys): the DiT lives under `model.diffusion_model.*` in
**already-native ComfyUI `ltxav` (audio+video) layout** — `patchify_proj`,
`adaln_single`, `audio_adaln_single`, `transformer_blocks.*`, `scale_shift_table` —
alongside a bundled `vae.`/`audio_vae.`/`vocoder.`/`text_embedding_projection.`.
ComfyUI v0.27.0 fully supports it (`comfy.ldm.lightricks.av_model.LTXAVModel`, Gemma-3
TE, LTX-2 VAE). The pack now (this PR):

1. **Extracts the DiT sub-tree** from the bundle (`loader.dit_bundle_prefix` +
   `load_fni8_dit(keep_only_prefixed=True)`): strips `model.diffusion_model.`, drops the
   bundled VAE/vocoder. `detect_unet_config` then correctly identifies it as `ltxav`.
2. **Recovers shape-determining config** the quantization dropped
   (`loader.ltx2_detection_metadata`, passed as `metadata={"config": …}`): the `.fni8`
   meta keeps only kind/arch/native_dtype, but ComfyUI needs the diffusers
   `transformer` config to size the model. Inferred from the checkpoint's own shapes:
   `cross_attention_adaln` (9-row vs 6-row `scale_shift_table`), the video/audio
   embeddings-connector head geometry (channel dim from `learnable_registers`, num-heads
   from `to_gate_logits` to disambiguate head_dim — video 32×128, audio 32×64), connector
   layer count, and gated-attention flag.

**Honest remaining gap.** Step 2 recovers only config that manifests as tensor
*shapes*. Pure scalar flags that don't change any shape (`rope_type`,
`causal_temporal_positioning`, timestep-scale multipliers, …) are **not** recoverable
from a `.fni8` and stay at comfy defaults — enough to build the native `LTXAVModel` and
engage int8, but a *numerically faithful forward* still needs the source diffusers
`transformer` config. The correct fix is **converter-side**: the fni8 forge/quantizer
must persist the diffusers `metadata["config"]` into the `.fni8`; `ltx2_detection_metadata`
already passes such a config straight through when present. Separately, the LTX-2.3 DiT
is **23.5 GB int8** (16.6 video + 6.9 audio), so a full resident forward exceeds a 16 GB
card — a denoise step needs ComfyUI weight-streaming (slow on PCIe-1.0) or a >23 GB GPU.

### LTX-2.3 pipeline-parallel: runnable split across 2 cards (issue #73)

The 23.5 GB int8 DiT is made **runnable** by splitting it across 2 GPUs
(`FNI8PipelineParallel` node / `comfyui_superl8.pipeline_ltx.install_ltxav_pipeline`,
`tests/e2e/test_pipeline_ltx.py`). The 48 `transformer_blocks` are distributed
contiguously — stage 0 (`cuda:0`) holds the prologue, the video/audio embedding
connectors, the output head and blocks 0–18; stage 1 (`cuda:1`) holds blocks 19–47 —
so **each card holds ~12 GB of weight, well under 16 GB** (the loader keeps the DiT
CPU-resident, so no single card ever sees the full 23.5 GB). Measured on GPU 6 (CMP)
+ GPU 7 (Tesla V100): `cuda:0` **12.32 GB**, `cuda:1` **11.21 GB** resident
(`torch.cuda.memory_allocated`), distribute 126 s over the PCIe-1.0-x1 link.

At each denoise step the only tensor that evolves block-to-block — the `(vx, ax)` hidden
state — is handed across the GPU boundary **compressed with the fni8 transport codec**
(`fni8.transport`, the same int8/int4 wire codec the serve side uses); the constant
sidecars (context / positional embeddings / timesteps / masks) move once per boundary.
Measured on a real boundary activation (`[1, 2048, 4096]` bf16, `cuda:0`→`cuda:1`):
int8 codec **cos 0.99996, SQNR 41 dB, 2.0× on-wire** (8.4 MB vs 16.8 MB fp16),
modelled link time **33.6 ms vs 67 ms** uncompressed on the 250 MB/s PCIe-1.0-x1 link
(int4 gives 4× / SQNR 16 dB if more compression is wanted). Real LTX-2.3 int8 blocks
(attn/ff dp4a) then execute on **both** cards with the compressed `(vx, ax)` hand-off
between them — once the adaLN modulation tables are dequantised to fp (see gap below).

This is a memory-**FIT** strategy, not a speedup — the stages serialise and each step
pays one codec handoff. The stock ComfyUI `KSampler` moves a `MODEL` to a single
`load_device`, which would re-collapse the split; driving the split through it needs a
model-management bypass (follow-up). A numerically-faithful full denoise still needs the
dropped diffusers config above — and the published `.fni8` also over-quantised the adaLN
modulation tables (`scale_shift_table` &c.) to int8, whose dequant-on-index path is broken
(a pre-existing LTX-2.3 forward gap, single-GPU too; modulation tables should stay fp,
same class as the output-head dequant). So the e2e proves the **split + transport
mechanism** on the real weights and the real boundary activation, which is the mechanism's
bar; a faithful full denoise is gated on those separate converter/loader fixes.
