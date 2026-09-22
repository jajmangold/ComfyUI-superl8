# SAM 3D Objects + SAM 3D Body on fni8 (int8 dp4a, sm_70) — architecture & phased plan

**Status:** research complete; **fp MVP + int8 bring-up in progress on the LOCAL working
runtime** (see §0). Gated by the sm_70 hardware truths in `fused_ni8/AGENTS.md` (no int8/fp16
tensor cores; only fast int8 matmul is `__dp4a` on CUDA cores; softmax/LSE/reconstruction
stay fp; single 16 GiB card).

> **CORRECTION (2026-07-13).** An earlier draft concluded from Meta's official `doc/setup.md`
> that SAM 3D Objects needs ≥32 GB VRAM and a CUDA-12.1 stack with no sm_70 support, and that
> both models were HF-gate-blocked. **Both were wrong for this fleet.** The user has already
> run BOTH models here on real V100s; the weights + a proven Volta runtime are on local disk,
> so the gate is moot. Objects runs single-card via a **rebuilt cu126/torch-2.9 native stack**
> + a **Volta-specific `flash_attn_v100`** (dense attention) + a custom **`comfy_sparse_attn`**
> package (sparse/varlen attention + flex_gemm/spconv/torchsparse conv backend). See §0, §2.

**Verdicts up front.**
- **SAM 3D Body** — clean, tractable int8-encoder target: a plain **ViT-H image backbone
  (embed_dim 1280, depth 32, 16 heads → head_dim 80)** *or* a DINOv3-H+ backbone, feeding a
  small promptable transformer decoder + camera head + MHR (Momentum Human Rig) parametric-
  mesh head. head_dim 80 is fni8-supported; the backbone is the exact non-causal ViT workload
  the SAM 3.1 MVP already accelerates (`comfyui_superl8/sam3_encoder.py`). Decoder / camera head
  / MHR head / MHR body model **stay fp**. **Fits one 16 GiB card easily** (2.1 GB fp16); the
  community pack runs it **fp16 on Volta**. Recommended **first** SAM 3D target.
- **SAM 3D Objects** — **TRELLIS-architecture** two-stage flow-matching generator (DINOv2
  conditioner → sparse-structure flow DiT → structured-latent (SLAT) flow DiT → Gaussian-
  splat & mesh VAE decoders). int8 targets: DINOv2 conditioner + the two flow-DiTs' dense
  linears/attention. **It DOES run single-card on this Volta fleet** via the rebuilt stack;
  stages run sequentially with offload (ss_generator 6.7 GB, slat_generator 4.9 GB loaded one
  at a time) so it fits 16 GiB, and SLAT sparse attention/conv goes through `comfy_sparse_attn`
  (not xformers). Reconstruction (VAE decoders, MoGe) stays fp.

---

## 0. RUNTIME (local, proven) — NOT gate-blocked

The HF gate is **moot**: weights and a working Volta runtime are on local disk (the user has
run both models here). Do not wait on `huggingface.co` acceptance.

**Weights (local):**
- Body: `<model-root>/sam3dbody/model.safetensors`
  (2.1 GB) + `.../assets/mhr_model.pt` + `model_config.yaml`. Mirror `apozz/sam-3d-body-safetensors`.
- Objects: `<model-root>/sam3dobjects/` — full
  TRELLIS set: `ss_generator.safetensors` (6.7 GB), `ss_decoder`, `slat_generator` (4.9 GB),
  `slat_decoder_gs`, `slat_decoder_gs_4`, `slat_decoder_mesh`, `dinov2_vitl14_reg`
  (conditioner), `moge_vitl` (depth) + per-model `.yaml` + `pipeline.yaml`. Mirror
  `apozz/sam-3d-objects-safetensors`.

**Working runtimes (the "how it ran on Volta" answer):**
- ComfyUI packs vendoring Volta-patched model libs:
  `<comfyui-root>/ComfyUI/custom_nodes/ComfyUI-SAM3D{Body,Objects}/`.
  Body: `nodes/process.py` (`SAM3DBodyEstimator.process_one_image`; "auto" → fp16 on Volta),
  vendored `nodes/sam_3d_body/sam_3d_body/`. Objects: `nodes/{load_model,depth_estimate,
  generate_slat,gaussian_decode,mesh_decode,export_ply}.py`, vendored `nodes/sam3d/`, recipe
  `workflows/{mesh,gaussian,full}_generation.json`.
- Pixi envs w/ the rebuilt native stack (torch **2.9.1+cu126**, spconv 2.3.8, gsplat,
  nvdiffrast, pytorch3d, sageattention, flash_attn 2.8.3, **flash_attn_v100 26.4**):
  `<comfyui-root>/.ce/.pixi/envs/sam3d{body,objects}-nodes/bin/python`.
- **`comfy_sparse_attn`** (`<comfyui-root>/sam3d-deps/comfy_sparse_attn/`)
  — the Volta sparse-attention shim: `dispatch_varlen_attention` + conv-backend detection
  (`SPARSE_CONV_BACKEND` ∈ flex_gemm/spconv/torchsparse). Makes the SLAT stage run on sm_70
  without xformers.
- Prior single-card run outputs: `<sam3d-output-dir>/d/sam3d_inference_*/`.

**Remaining real risk** is not feasibility but (a) the int8 win split (ViT attention may stay
fp on image outliers, the int8 *linears* carry it — the SAM 3.1 pattern), and (b) wiring
`fni8`'s sm_70 dp4a kernel into the pixi env (torch 2.9 vs fni8's pinned 2.10 — check ABI /
install the fni8 wheel, or run the int8 pass where torch+fni8 coexist).

**Code repos** for both models are also cloned at `<sam3d-repos-dir>` for
reference; architectures below are confirmed from source.

---

## 1. SAM 3D Body — architecture (confirmed from source)

`sam_3d_body/models/meta_arch/sam3d_body.py` (`SAM3DBody`), backbone `sam_3d_body/
models/backbones/{vit.py,dinov3.py}`, heads `sam_3d_body/models/heads/{mhr_head.py,
camera_head.py}`, decoder `sam_3d_body/models/decoders/promptable_decoder.py`.

**Pipeline:** full image → (optional) human **detector** (ViTDet cascade / SAM3) → crop to
person bbox → **top-down affine to 256×192** → **ViT-H (or DINOv3-H+) backbone** → patch
tokens → **promptable transformer decoder** (SAM-style; optional 2D-keypoint / mask prompts)
regressing MHR pose/shape tokens → **camera head** + **MHR head** → **MHR parametric mesh**
(`assets/mhr_model.pt`, 70-joint "mhr70" rig; body+hands+feet). Output = posed 3D human mesh
+ 3D joints + camera. Renderer: `sam_3d_body/visualization/renderer.py` (pyrender).

**Backbone (the heavy, int8-able compute):**
- **ViT-H:** img (256,192), patch 16 → **16×12 = 192 tokens**, embed_dim 1280, depth 32,
  16 heads → **head_dim 80** (∈ fni8 SUPPORTED_HEAD_DIMS), mlp_ratio 4 (fc1 1280→5120,
  fc2 5120→1280), `LayerNorm32`, non-causal attention. `qkv`/`proj`/`fc1`/`fc2` are plain
  `nn.Linear`. Attention uses `flash_attn_func` if importable, else a plain fallback — on
  sm_70 flash-attn is absent (prints "No Flash Attention!"), so the fp path is already a
  non-flash math attention, which our int8 SDPA shim can slot into.
- **DINOv3-H+ (default ckpt):** `torch.hub.load("facebookresearch/dinov3", …)`, embed_dim
  1280-ish, `get_intermediate_layers`. Same int8 target surface (block qkv/proj + MLP).
  Note: pulls the dinov3 code over the network at load time.

**Component → precision map:**

| Sub-module | Precision | fni8 op / rationale |
|---|---|---|
| **Backbone block qkv/proj/MLP (32×)** | **int8 W8A8** | `linear_w8a8` dp4a; K=1280/5120 (÷4 ✓) |
| **Backbone attention (192 tokens, hd 80)** | **int8 dp4a, SQNR-gated** | `attn_int8_fwd(causal=False)`; likely stays fp on image outliers (see SAM 3.1 finding) — the **linears carry the win** |
| Patch-embed conv (16×16) | fp | 1 conv, no int8 conv primitive |
| LayerNorm32 / GELU | fp (half2 CUDA cores) | `F.layer_norm` / `F.gelu` |
| Promptable decoder (cross-attn + softmax) | **fp** | pose-quality load-bearing (AGENTS.md) |
| Prompt encoder (keypoints/mask) | fp | tiny |
| Camera head / MHR head / MHR body model | **fp** | numerically load-bearing reconstruction — **never quantize** |

**VRAM / fit:** ViT-H ≈ 0.63B (~1.3 GB fp16 → ~0.65 GB int8 backbone) + DINOv3-H+ 840M
(~1.7 GB fp16); decoder+heads small; MHR mesh trivial; 192 tokens → negligible attention
activations. **Trivially single 16 GiB card.**

**Expected int8 win (honest):** the backbone is only **192 tokens** — far smaller than SAM
3.1's 5184-token PE-L. The int8-linear GEMMs are correspondingly small, so the encoder
speedup will be **more modest than SAM 3.1's 1.67×** (the workload is closer to
latency-bound than compute-bound at 192 tokens; batching multiple detected people per image
recovers arithmetic intensity). The correctness story (int8 backbone → same MHR mesh within
a geometric tolerance) is the primary deliverable; the speedup is a secondary, honestly
smaller, win at this token count.

**Validation gate (int8 vs fp oracle):** **MPJPE / PA-MPJPE** on the 70 MHR joints (mm),
**per-vertex mesh error** (mm, mean + Hausdorff), and mesh IoU of the posed volume. Target:
int8-backbone MPJPE within a small delta of the fp oracle (propose ≤ ~2 mm added error,
tuned to the fp16-vs-fp32 baseline gap — do not weaken once measured). Helper implemented in
`comfyui_superl8/sam3d_metrics.py` (`mpjpe`, `pa_mpjpe`, `per_vertex_error`).

---

## 2. SAM 3D Objects — architecture (confirmed from source)

`sam3d_objects/model/backbone/` — this is a **TRELLIS** structured-latent 3D generator.
Checkpoint set (from the gated repo listing): `ss_encoder`/`ss_generator`/`ss_decoder`
(sparse-structure stage) + `slat_encoder`/`slat_generator`/`slat_decoder_gs`/
`slat_decoder_mesh` (structured-latent stage) — matches the prior on-disk artifacts
(`<sam3d-output-dir>/*`: `sparse_structure.pt`, `slat.pt`) which are
outputs of exactly this pipeline.

**Pipeline:** image (+ object mask, + MoGe monocular **pointmap/depth** conditioning) →
**DINOv2 image conditioner** (`dit/embedder/dino.py`, `torch.hub facebookresearch/dinov2`)
→ **sparse-structure flow-matching DiT** (`tdfy_dit/models/sparse_structure_flow.py`,
flow-matching solver `generator/flow_matching/`) generating a coarse voxel structure →
**structured-latent (SLAT) flow-matching DiT** (`structured_latent_flow.py`, operates on
**sparse** voxels via `spconv` + `xformers` sparse attention) → **SLAT VAE decoders**:
Gaussian-splat (`decoder_gs.py`, → `.ply` splat) and mesh (`decoder_mesh.py`, → textured
mesh). Optional layout post-optimization for multi-object scenes.

**Component → precision map (target, if ported):**

| Sub-module | Precision | fni8 op / rationale |
|---|---|---|
| DINOv2 conditioner (ViT) | **int8 W8A8 + int8 attn** | same ViT surface as SAM 3.1 |
| SS flow DiT (dense linears + attn) | **int8 W8A8 + int8 attn** | non-causal DiT — fni8's core regime |
| SLAT flow DiT (sparse) dense linears | int8 W8A8 where GEMM-expressible | attn goes through `comfy_sparse_attn` varlen dispatch — stays fp |
| Timestep/RoPE embedders | fp | cheap |
| SS VAE decoder / SLAT GS+mesh decoders | **fp** | reconstruction load-bearing — never quantize |
| MoGe pointmap conditioner | fp | separate model |

**How it actually runs on this Volta fleet — MEASURED (fp MVP produced 2026-07-13):**
Reproduced end-to-end on one Tesla V100 (sm_70) from local weights, input `elephant.png`
(RGBA, mask=alpha), fp16, seed 42, stage1/2 = 12/12 steps, cfg 7.5/5.0. **Real outputs:**
`.pipe_out/sam3dobject_fp.ply` (Gaussian splat, **280,128 gaussians**, 77 MB) +
`sam3dobject_fp.glb` (mesh, **193,836 verts / 387,664 faces**, 7.7 MB). Driver:
`bench/sam3dobjects_fni8_mvp.py`.
- **VRAM: whole-pipeline peak ≈ 6.4 GiB → single 16 GiB card CONFIRMED.** Stages load/free
  sequentially via ComfyUI ModelPatcher lowvram streaming (ss_generator 6.7 GB and
  slat_generator 4.9 GB never co-resident). Meta's "≥32 GB" is the naive all-resident path;
  it is wrong for this runtime. Stage timings: MoGe depth 58 s (warm), SLAT (both flow-DiTs +
  DINOv2 cond) 528 s, gaussian decode 61 s, mesh decode (sparse FlexiCubes) 24 s.
- **Native stack rebuilt for cu126/torch-2.9** (NOT Meta's cu121/torch-2.5.1): spconv 2.3.8,
  gsplat, nvdiffrast, pytorch3d, sageattention, flash_attn 2.8.3 — all working on sm_70.
- **Backends engaged (from logs):** SLAT sparse conv = **spconv** (`SPARSE_CONV_BACKEND=spconv`,
  `[SPARSE] Backend: spconv`); sparse attention = `comfy_sparse_attn.dispatch_varlen_attention`;
  **dense attention = PyTorch SDPA.** Nuance: the pack's comfy dispatcher tries Dao
  `flash_attn`, which **rejects sm_70** ("FlashAttention only supports Ampere GPUs or newer")
  and falls back to SDPA. `flash_attn_v100` is installed but the current node-pack attention
  path does not route to it — SDPA carries dense attention and the pipeline is correct. (The
  key point stands: SLAT sparse attn via `comfy_sparse_attn`, not xformers, is what unblocks
  sm_70; the earlier "no sm_70 path" verdict was wrong.)
- **Ops gotcha:** `CUDA_DEVICE_ORDER=PCI_BUS_ID` is mandatory — else `CUDA_VISIBLE_DEVICES`
  selects a CMP mining card, not a V100.

**int8 status — wiring proven, kernel BLOCKED by a cross-repo torch-ABI mismatch.**
`patch_backbone_linears_int8` cleanly wrapped the **DINOv2 conditioner ViT-L (110 nn.Linear**,
1 skipped for K%4≠0), SQNR gate armed. But `fni8._C.so` fails to load in this env:
`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`. Root cause (via
`nm -D`): **every on-disk fni8 `_C.so` is built against torch 2.10/cu129** (AGENTS.md pin),
whose `c10_cuda_check_implementation` line-arg ABI is `…S2_jb`, while the SAM3D env's **torch
2.9.1+cu126** exports `…S2_ib`. They cannot co-load in one process, and the SAM3D deps
(spconv/flash_attn_v100/pytorch3d/gsplat/nvdiffrast, all cu126) cannot move to torch 2.10. No
nvcc on host/pixi to rebuild fni8 for cu126 (the PyPI `nvidia-cuda-nvcc` wheels ship only
ptxas+libnvvm, no nvcc frontend/cicc). **Contrast with SAM 3D Body:** Body's int8 pass
succeeded by running in the `comfyui-fni8-sam3` docker image (torch 2.10/cu129) because Body
needs only torch + the vendored `sam_3d_body` lib — no spconv/gsplat. Objects needs the cu126
sparse stack AND fni8's cu129 kernel in one process, which is the incompatibility.

**To unblock Objects int8:** build an fni8 `_C.so` against **torch 2.9.1+cu126** with a full
CUDA-12.6 toolchain (real nvcc+cicc, e.g. a `cuda-toolkit` image + torch 2.9.1). Then
`linear_w8a8` loads in the SAM3D env and the already-wired DINOv2 patch (+ the two flow-DiTs,
the real compute over 24 diffusion steps) runs as-is; `sam3d_metrics.chamfer_distance/
chamfer_fscore` consume the two point sets for the fp-vs-int8 gate. **Honest expectation:**
per the SAM 3D Body result (int8 linears → geometric parity but ~0 speedup because fp
attention dominates on the gimped-TC fleet), the Objects int8 win is likeliest on the
flow-DiT **linears** amortized over the 24 sampling steps, not the small DINOv2 conditioner.

**Output coordinate frame — the reported splat-vs-mesh "mirror" (investigated 2026-07-13).**
A viewer showed `sam3dobject_fp.ply` (Gaussian splat) and `.glb` (mesh) mirrored / in two
positions. **The two files are NOT mirrored — their stored coordinates are co-registered.**
Both decoders apply the *same* z-up→y-up rotation `M = [[1,0,0],[0,0,-1],[0,1,0]]` in
`run_decode` (gaussian via `Gaussian.save_ply(transform=M)`, mesh via `vertices @ M`); the
pack's own default workflow uses `up_axis="Y-up (standard)", world_coordinates=False` for
both, matching our driver. Verified 3 ways (`bench/sam3d_verify_alignment.py`): raw `.glb`
POSITION buffer bbox == `.ply` bbox; **occupancy IoU identity 0.767 vs x/y/z-mirror
0.43/0.21/0.23** (identity clearly wins); and an overlay render
(`.pipe_out/sam3dobject_align_check.png`) shows the gaussian (red) and mesh (blue) points
interleaved into one elephant, not two. (The object's bilateral symmetry is why a naive
chamfer barely separated identity from an x-mirror — occupancy IoU is the reliable test.)
- **Actual cause of the visible "mirror": a viewer-convention mismatch.** 3D-Gaussian-Splat
  `.ply` viewers conventionally render **Y-down** (INRIA/COLMAP 3DGS), while glTF `.glb` is
  **Y-up** — so each file opened in its *native* viewer looks vertically flipped relative to
  the other despite identical stored coordinates. Fix = view both Y-up; deliverable =
  **`sam3dobject_fp_aligned.glb`**, a single Y-up glTF scene containing the mesh + the
  gaussian means as a colored point cloud, provably coincident in any glTF viewer.
- **Convention (do not regress):** both exports are **Y-up, right-handed**, object-centered
  (`world_coordinates=False`); the shared z-up→y-up matrix is `M` above. `sam3d_verify_alignment.py`
  is a regression guard — it fails if any decoder export flips handedness (identity IoU must
  beat every axis-mirror). *Secondary latent bug in the pack's `Gaussian.save_ply` (not ours):
  it rotates the means by M but not the per-splat rotation quaternions / SH DC, so an ellipsoid
  splat renderer shows correctly-placed but mis-oriented splats; the means/points and the mesh
  are correct.*

**Recommendation for Objects:** fp MVP **shipped** (real splat + mesh + co-registered combined
GLB, single-card). int8 is a **follow-up PR gated on a cu126 fni8 build** — not an architecture
problem, a toolchain-ABI one. Body ships first with a complete fp+int8 result.

---

## 3. Node surface (mirrors `comfyui_superl8/sam3_nodes.py`)

Registered via the package `__init__.py` (NOT the contended `nodes.py`), guarded so mappings
are empty outside a ComfyUI process. Scaffold in `comfyui_superl8/sam3d_nodes.py`; int8 backbone
glue reuses a **generalized** ViT-int8 patch (`comfyui_superl8/sam3d_encoder.py`).

- **`SAM3DBodyLoaderFNI8`** → loads `facebook/sam-3d-body-{dinov3,vith}` (fp oracle) and
  optionally flips the ViT/DINOv3 backbone onto the int8 dp4a path (attention + linears,
  SQNR-gated). Decoder/camera/MHR heads stay fp. Returns `SAM3D_BODY_MODEL`.
- **`SAM3DBodyPredict`** → `IMAGE` (+ optional bbox / 2D-keypoint prompt) → posed MHR mesh;
  saves `.obj`/`.glb` to `.pipe_out/` + a pyrender preview PNG; returns the mesh path + an
  overlay `IMAGE`.
- **`SAM3DObjectLoaderFNI8` / `SAM3DObjectPredict`** → drive the TRELLIS pipeline (DINOv2 +
  MoGe → SS flow → SLAT flow → GS/mesh decode) via the local Volta runtime (`comfy_sparse_attn`
  + `flash_attn_v100`); int8 the DINOv2 conditioner + flow-DiT dense linears, decoders fp.
  IMAGE → 3D asset (`.ply` splat / `.glb` mesh) saved to `.pipe_out/`.

Reuse from the SAM 3.1 MVP: `Int8LinearShim` + per-row int8 quant + the SQNR gate + the
module-scoped SDPA proxy (generalized to accept an arbitrary attention module, since Body's
backbone is not `sam3.model.vitdet`).

---

## 4. Phased roadmap

**Phase 0 — SAM 3D Body fp MVP oracle (in progress).** Load the local `model.safetensors` +
`mhr_model.pt` (fp16 on Volta) via the vendored `sam_3d_body` lib, run
`estimator.process_one_image` on a real photo, save a posed MHR mesh (`.obj`) + preview to
`.pipe_out/`. The correctness oracle.

**Phase 1 — int8 the Body ViT/DINOv3 backbone (in progress).** Route backbone block linears
through `linear_w8a8` (SQNR-gated), decoder/heads/MHR fp. Measure MPJPE / per-vertex error vs
the fp oracle, encoder latency, peak VRAM. Direct reuse of `sam3d_encoder.py`.

**Phase 2 — SAM 3D Objects fp MVP + int8 (in progress).** Reproduce the working pack's
pipeline on local weights via the pixi env (`comfy_sparse_attn` + `flash_attn_v100`, staged
offload) → real `.ply`/`.glb`. Then int8 the DINOv2 conditioner + flow-DiT dense linears
(SQNR-gated), SLAT sparse attn + decoders + MoGe fp. Metric: Chamfer / F-score vs fp.

---

## 5. Precedent to lean on

- **SAM 3.1 MVP** (`comfyui_superl8/sam3_encoder.py`, `docs/sam31-fni8-design.md`) — the exact
  non-causal ViT int8 pattern: module-scoped SDPA proxy + `Int8LinearShim`, per-site SQNR
  gate → fp fallback, reversible `.unpatch()`. Measured IoU 0.9995, 1.67× encoder, −0.96 GiB.
  **Key transferable finding:** ViT *attention* often stays fp on image-activation outliers
  (fni8's inner SageAttention gate), and the int8 **linears** carry the speedup — expect the
  same for Body (more so, given only 192 tokens).
- **AGENTS.md** — softmax/reconstruction stay fp; "the accuracy gate decides where int8 is
  allowed, not ideology"; int8 still buys memory/bandwidth on top of any compute win.
- **`comfyui_superl8/sam3d_metrics.py`** — MPJPE/PA-MPJPE/per-vertex/Chamfer helpers for the
  int8-vs-fp geometric gate (tested, weight-independent).
