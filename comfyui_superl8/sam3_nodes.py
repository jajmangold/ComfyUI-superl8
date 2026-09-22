# SPDX-License-Identifier: Apache-2.0
"""ComfyUI nodes for SAM 3 / 3.1 segmentation on the int8 dp4a backbone.

`SAM3LoaderFNI8` loads the official `facebook/sam3.1` image model (via the
`facebookresearch/sam3` runtime) and optionally flips its Perception-Encoder ViT
trunk onto the fni8 int8 dp4a path (`comfyui_superl8.sam3_encoder`); the mask decoder,
DETR detector, and text encoder stay fp (mask-quality load-bearing, fni8/AGENTS.md).
`SAM3Predict` takes an `IMAGE` + a text (concept) and/or box prompt and returns a
`MASK` (+ an overlay `IMAGE`).

sam3/torch are imported lazily inside the methods so this module (and the node pack)
loads even where the SAM runtime isn't installed — the node just errors when run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._comfy_compat import COMFY


@dataclass
class Sam3ModelBundle:
    """Opaque `SAM3_MODEL` payload passed between the loader and predictor nodes."""
    model: Any
    processor: Any
    int8_handle: Any  # comfyui_superl8.sam3_encoder.Sam3Int8Handle | None
    precision: str
    gate_summary: dict | None = None


def _image_to_pil(image: "torch.Tensor"):
    """ComfyUI IMAGE [B,H,W,C] float 0..1 -> a list of PIL RGB images."""
    import numpy as np
    from PIL import Image

    if image.dim() == 3:
        image = image.unsqueeze(0)
    out = []
    for i in range(image.shape[0]):
        arr = (image[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        out.append(Image.fromarray(arr, mode="RGB"))
    return out


def _best_masks(state, max_det: int):
    """Top-`max_det` detection masks [N,H,W] bool + scores, sorted by score desc."""
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or scores is None or len(scores) == 0:
        return None, None
    scores = torch.as_tensor(scores).float()
    order = torch.argsort(scores, descending=True)[:max_det]
    sel = []
    for i in order.tolist():
        m = masks[i]
        while m.dim() > 2:
            m = m[0]
        sel.append(m.bool())
    return torch.stack(sel, 0).cpu(), scores[order].cpu()


if COMFY:

    class SAM3LoaderFNI8:
        """Load facebook/sam3.1 with an int8 dp4a Perception-Encoder ViT backbone."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "precision": (["fp16", "bf16", "fp32"],),
                    "int8_encoder": ("BOOLEAN", {"default": True}),
                },
                "optional": {
                    "int8_attention": ("BOOLEAN", {"default": True}),
                    "int8_linears": ("BOOLEAN", {"default": True}),
                    "hadamard_rotate": ("BOOLEAN", {"default": False}),
                    "confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "fni8_path": ("STRING", {"default": "", "multiline": False}),
                },
            }

        RETURN_TYPES = ("SAM3_MODEL",)
        FUNCTION = "load"
        CATEGORY = "fni8/sam3"
        TITLE = "SAM3 Loader FNI8 (int8 dp4a ViT)"

        def load(self, precision, int8_encoder, int8_attention=True, int8_linears=True,
                 hadamard_rotate=False, confidence=0.5, fni8_path=""):
            from .gate import require_sm70
            require_sm70()
            from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
            from sam3.model.sam3_image_processor import Sam3Processor

            # Volta: force math SDP (no flash/mem-efficient on sm_70) for the fp path.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

            ckpt = download_ckpt_from_hf(version="sam3.1")
            model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False).eval().cuda()
            processor = Sam3Processor(model, confidence_threshold=confidence)

            handle = None
            summary = None
            if int8_encoder and (int8_attention or int8_linears):
                from . import sam3_encoder
                if fni8_path.strip() and int8_linears:
                    # Fast path: place PRE-QUANTIZED offline `.fni8` linears (mmap, no
                    # runtime quant). Attention still uses the self-gating SDPA proxy.
                    handle = sam3_encoder.load_sam3_encoder_int8_from_fni8(
                        model, fni8_path.strip(), attention=int8_attention,
                        rotate=hadamard_rotate)
                else:
                    handle = sam3_encoder.patch_sam3_encoder_int8(
                        model, attention=int8_attention, linears=int8_linears,
                        rotate=hadamard_rotate)
                summary = handle.summary()
            return (Sam3ModelBundle(model, processor, handle, precision, summary),)

    class SAM3Predict:
        """Segment an IMAGE with a text (concept) and/or box prompt -> MASK + overlay."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "sam3_model": ("SAM3_MODEL",),
                    "image": ("IMAGE",),
                    "prompt": ("STRING", {"default": "", "multiline": False}),
                },
                "optional": {
                    "box_xyxy": ("STRING", {"default": "", "multiline": False}),
                    "max_detections": ("INT", {"default": 1, "min": 1, "max": 100}),
                    "combine_masks": ("BOOLEAN", {"default": True}),
                },
            }

        RETURN_TYPES = ("MASK", "IMAGE")
        RETURN_NAMES = ("mask", "overlay")
        FUNCTION = "predict"
        CATEGORY = "fni8/sam3"
        TITLE = "SAM3 Predict (int8 dp4a)"

        def predict(self, sam3_model, image, prompt, box_xyxy="",
                    max_detections=1, combine_masks=True):
            bundle: Sam3ModelBundle = sam3_model
            proc = bundle.processor
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                     "fp32": torch.float32}[bundle.precision]
            pil = _image_to_pil(image)[0]
            box = [float(v) for v in box_xyxy.split(",")] if box_xyxy.strip() else None

            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype,
                                                        enabled=dtype != torch.float32):
                state = proc.set_image(pil)
                proc.reset_all_prompts(state)
                if prompt.strip():
                    state = proc.set_text_prompt(prompt=prompt.strip(), state=state)
                if box is not None:
                    w, h = pil.size
                    x0, y0, x1, y1 = box
                    state = proc.add_geometric_prompt(
                        box=[((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h,
                             abs(x1 - x0) / w, abs(y1 - y0) / h],
                        label=True, state=state)

            masks, _ = _best_masks(state, max_detections)
            H, W = pil.size[1], pil.size[0]
            if masks is None:
                mask_t = torch.zeros((1, H, W), dtype=torch.float32)
            elif combine_masks:
                mask_t = masks.any(0, keepdim=True).float()
            else:
                mask_t = masks.float()

            # Overlay (red tint) on the input image.
            base = image[0] if image.dim() == 4 else image
            overlay = base.clone().cpu()
            m = mask_t.any(0).bool()
            tint = torch.tensor([1.0, 0.15, 0.15])
            overlay[m] = 0.55 * overlay[m] + 0.45 * tint
            return (mask_t, overlay.unsqueeze(0))

    SAM3_NODE_CLASS_MAPPINGS = {
        "SAM3LoaderFNI8": SAM3LoaderFNI8,
        "SAM3Predict": SAM3Predict,
    }
else:  # pragma: no cover - outside ComfyUI the nodes are unavailable
    SAM3_NODE_CLASS_MAPPINGS = {}
