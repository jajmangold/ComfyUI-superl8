# Recipe: SAM 3.1 (image → segmentation mask)

- **Workflow:** [`example_workflows/sam3.1_segmentation_fni8.json`](../../example_workflows/sam3.1_segmentation_fni8.json)
- **Status:** **ran-e2e** — validated end-to-end on a 16 GB Volta/CMP (sm_70).
- **Source model:** [`facebook/sam3.1`](https://huggingface.co/facebook/sam3) image model
  (Perception-Encoder ViT trunk + DETR detector + mask decoder), run via the
  `facebookresearch/sam3` runtime. The PE-L ViT backbone is flipped onto the fni8 int8
  dp4a path; the mask decoder, DETR detector, and text encoder stay fp
  (mask-quality load-bearing, per fni8/AGENTS.md).

## Node graph

```
LoadImage ──IMAGE──► SAM3Predict ──overlay(IMAGE)──► SaveImage
SAM3LoaderFNI8 ─SAM3_MODEL─► SAM3Predict.sam3_model
                             SAM3Predict.prompt = "a person"  (concept / text prompt)
                             SAM3Predict.mask (MASK)  ← also available for downstream nodes
```

`SAM3LoaderFNI8` loads the PE-L backbone with `int8_encoder=True` (int8 W8A8 linears on the
32 ViT blocks, SQNR-gated per layer → fp fallback if a layer's SQNR dips); LayerNorm, GELU,
2D RoPE, patch-embed conv, and the mask decoder stay fp. `SAM3Predict` takes an `IMAGE`
plus a text (concept) and/or `box_xyxy` prompt and returns a `MASK` (+ a tinted overlay
`IMAGE`).

## Checkpoints

| Role | File | Place in |
|---|---|---|
| PE-L ViT backbone (int8) | `facebook__sam3.1.pe-l.b8.fni8` (~1 GB) | `models/sam3/` (or set `fni8_path` on the loader) |
| Detector / mask decoder / text encoder (fp) | official `facebook/sam3.1` weights | fetched by the `sam3` runtime |

The int8 backbone `.fni8` is produced offline by `comfyui_superl8/sam3_convert.py`
(`bench/sam31_fni8_convert_validate.py` validates it is **bit-identical** to the on-load
int8 path — mask IoU 1.000000, and saves ~1.67 GiB by storing int8 only, no fp32 `w_fp`
reference). Leave `fni8_path` blank to quantize on load instead.

## Measured (CMP 100-210 / V100 fleet — not real-V100)

- **mask IoU (int8 vs fp oracle) = 0.9995** (target ≥ 0.98 ✓) — `bench/sam31_fni8_eval.py`.
- **ViT encoder latency 890 ms → 533 ms = 1.67× faster** (int8 linears carry the win; the
  numerically-touchy decoder/detector stay fp).
- **int8 backbone VRAM −0.96 GiB** vs the fp backbone (offline `.fni8` −1.67 GiB vs the
  on-load path that also keeps fp32 `w_fp`).
- Offline `.fni8` vs on-load int8: **mask IoU 1.000000** (bit-identical `per_row_i8` math).

Full analysis: [`docs/sam31-fni8-design.md`](../sam31-fni8-design.md) §Validation.

## Known limits

- Only the PE-L **windowed-attention / FFN linears** are int8; the mask decoder and DETR
  detector are the mask-quality load-bearing pieces and stay fp — do not int8 them to chase
  more speedup.
- The 1.67× is the **encoder** speedup on this fleet (int8 dp4a beats the firmware-gimped
  fp16 tensor cores); it is fleet-specific and does not transfer to a real V100.
