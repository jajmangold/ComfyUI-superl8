# Recipe: SAM 3D Body (image → posed 3D human mesh)

- **Workflow:** [`example_workflows/sam3d-body_reconstruct_fni8.json`](../../example_workflows/sam3d-body_reconstruct_fni8.json)
- **Status:** **ran-e2e** — validated end-to-end, real 3D output, single 16 GB Volta/CMP (sm_70).
- **Source model:** `facebook/sam-3d-body-vith` (or `-dinov3`) — a top-down human mesh
  recovery estimator (person bbox → 256×192 crop → ViT-H / DINOv3-H+ backbone → MHR head
  → posed MHR mesh). Runs via the vendored `sam_3d_body` lib inside the
  `comfyui-fni8-sam3` image (torch 2.10 / cu129 — Body needs only torch + the vendored lib,
  no spconv / gsplat).

## Node graph

```
LoadImage ──IMAGE──► SAM3DBodyPredict ──overlay(IMAGE)──► SaveImage
SAM3DBodyLoaderFNI8 ─SAM3D_BODY_MODEL─► SAM3DBodyPredict.sam3d_body_model
                                        SAM3DBodyPredict.mesh_path (STRING) → saved .glb / .obj
```

`SAM3DBodyLoaderFNI8` loads the estimator and (with `int8_backbone=True`) flips the ViT-H /
DINOv3 trunk linears onto int8 dp4a (`linear_w8a8`, SQNR-gated); the MHR head, decoder, and
mesh model stay fp. `SAM3DBodyPredict` takes an `IMAGE` (+ optional `bbox_xyxy`,
`inference_type` ∈ {full, body, hand}) and returns the saved mesh path + a render overlay.

## Checkpoints

| Role | File | Place in |
|---|---|---|
| SAM 3D Body estimator (~2.1 GB fp16) | `facebook/sam-3d-body-vith` weights + `mhr_model.pt` + `model_config.yaml` | `models/sam3d_body/` |

Mirror `apozz/sam-3d-body-safetensors` for a safetensors copy. The int8 backbone is applied
in-process at load (reversible `.unpatch()`); no separate `.fni8` conversion is required for
Body.

## Measured (CMP 100-210 / V100 fleet — not real-V100)

- **MPJPE (int8 backbone vs fp oracle) = 2.27 mm** added error on the 70 MHR joints —
  geometric parity (`comfyui_superl8/sam3d_metrics.py`, `bench/sam3dbody_fni8_mvp.py`).
- Real posed 3D mesh output (Y-up), single-card, ≈2.1 GB fp16 footprint — fits one 16 GiB
  card easily.

## Known limits

- **int8 buys geometric parity but ~0 wall-clock win here** — unlike SAM 3.1's ViT, the
  Body backbone GEMMs are **not** the bottleneck (192 tokens; the head/mesh/IO dominate).
  We ship int8 as the memory-neutral, parity-preserving default; the honest speedup is ~1×.
  Set `int8_backbone=False` to run pure fp.
- MPJPE is the primary deliverable (a geometric tolerance), the speedup is secondary and
  honestly small on this workload. See [`docs/sam3d-fni8-design.md`](../sam3d-fni8-design.md).
