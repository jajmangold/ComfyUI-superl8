# SPDX-License-Identifier: Apache-2.0
"""Offline `facebook/sam3.1` -> `.fni8` converter for the int8 dp4a ViT backbone (#120).

Quantizes the Perception-Encoder ViT trunk's linears (qkv / proj / fc1 / fc2 ×32) to
`per_row_i8` ONCE, offline, and writes a `.fni8` whose on-disk bytes ARE the resident
dp4a layout (per the fni8 format contract: load = mmap + place, no dequant/repack).
Attention stays fp (fni8's SageAttention gate already decided that on the ViT's post-
RoPE Q), and the mask decoder / prompt encoder / detector / text encoder / conv stem /
neck / LayerNorm / GELU are NOT carried here — they stay fp and load from the HF
checkpoint. The `.fni8` is an int8 accelerator OVERLAY the loader places onto the
fp-built model (`comfyui_superl8.sam3_encoder.load_vit_linears_from_fni8`), so there is no
per-run quant.

Mirrors the fni8-serve HF->`.fni8` pattern + the comfy DiT converter: per-row symmetric
int8 (`scale = max|w| / 127`, scales fp32), `in % 4 == 0` for dp4a, and the trunk config
+ per-layer int8/fp decisions persisted in `__meta__` so the loader rebuilds without
re-deriving (the LTX "config never persisted" bug, avoided by construction).

CLI:
    python3 -m comfyui_superl8.sam3_convert --out /path/sam3.1.vit.b8.fni8 [--device cuda]
"""

from __future__ import annotations

import argparse
import time

import torch

from .sam3_encoder import _quantize_per_row_i8

# `.fni8` meta schema version for this component (bump on a breaking layout change).
SAM3_FNI8_META_VERSION = 1
BIAS_DTYPE = torch.float16


def _trunk(model):
    return model.backbone.vision_backbone.trunk


def quantize_sam3_trunk(model):
    """Quantize every dp4a-eligible ViT-trunk `nn.Linear` to `per_row_i8`.

    Returns `(tensors, meta)` where `tensors` maps a trunk-relative module path ->
    `superl8.QTensor` (weight `per_row_i8`; bias, if any, as a paired `<path>.bias` raw
    fp16), and `meta` carries the trunk config + int8/fp layer lists."""
    import superl8

    trunk = _trunk(model)
    tensors: dict = {}
    int8_layers: list[str] = []
    fp_layers: list[str] = []
    for name, mod in trunk.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if (mod.in_features % 4) != 0:
            fp_layers.append(name)  # can't dp4a (K%4!=0) — stays fp, not carried
            continue
        q, scale = _quantize_per_row_i8(mod.weight)
        tensors[name] = superl8.QTensor(
            q.cpu().contiguous(), scale.cpu().contiguous(), "per_row_i8")
        if mod.bias is not None:
            tensors[name + ".bias"] = superl8.QTensor(
                mod.bias.detach().to(BIAS_DTYPE).cpu().contiguous(), None, "raw")
        int8_layers.append(name)

    b0 = trunk.blocks[0].attn
    meta = {
        "model": "sam3.1",
        "component": "backbone.vision_backbone.trunk",
        "meta_version": SAM3_FNI8_META_VERSION,
        "trunk": {
            "dim": int(b0.qkv.in_features),
            "num_heads": int(b0.num_heads),
            "head_dim": int(b0.head_dim),
            "num_blocks": int(len(trunk.blocks)),
        },
        "int8_layers": int8_layers,
        "fp_layers": fp_layers,
        "quant": "per_row_i8",
        "bias_dtype": "float16",
    }
    return tensors, meta


def convert_sam3_to_fni8(
    out_path: str, *, checkpoint_path: str | None = None, device: str = "cuda",
    version: str = "sam3.1",
) -> dict:
    """Build the fp sam3.1 image model, quantize its ViT trunk, and write the `.fni8`.

    Returns the persisted `__meta__` (plus timing) for logging."""
    import superl8
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

    t0 = time.perf_counter()
    ckpt = checkpoint_path or download_ckpt_from_hf(version=version)
    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False).eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    tensors, meta = quantize_sam3_trunk(model)
    t_quant = time.perf_counter() - t1

    t2 = time.perf_counter()
    superl8.save_fni8(out_path, tensors, meta=meta)
    t_write = time.perf_counter() - t2

    meta = dict(meta)
    meta["_timing_s"] = {
        "build": round(t_build, 2), "quantize": round(t_quant, 2),
        "write": round(t_write, 2)}
    return meta


def main():
    ap = argparse.ArgumentParser(description="facebook/sam3.1 -> int8 .fni8 (ViT trunk)")
    ap.add_argument("--out", required=True, help="output .fni8 path")
    ap.add_argument("--checkpoint", default=None, help="sam3.1_multiplex.pt (else HF)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--version", default="sam3.1")
    args = ap.parse_args()

    import json
    import os

    meta = convert_sam3_to_fni8(
        args.out, checkpoint_path=args.checkpoint, device=args.device,
        version=args.version)
    size_mb = os.path.getsize(args.out) / 2**20
    print(json.dumps({
        "out": args.out,
        "size_mib": round(size_mb, 1),
        "int8_layers": len(meta["int8_layers"]),
        "fp_layers": len(meta["fp_layers"]),
        "trunk": meta["trunk"],
        "timing_s": meta["_timing_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
