# SPDX-License-Identifier: Apache-2.0
"""Validate the offline SAM 3.1 `.fni8` vs the on-load int8 path (issue #120, Phase 2).

Same fp model, same image/prompt, two int8 routes:
  A) on-load quant  — `patch_sam3_encoder_int8` (quantizes the 128 trunk linears at
     runtime, keeps fp reference weights, one-time SQNR gate);
  B) offline `.fni8` — `load_sam3_encoder_int8_from_fni8` (mmap + place the
     pre-quantized resident int8 codes/scales; no runtime quant, no fp weights).

Asserts the offline mask matches the on-load int8 mask (IoU >= 0.999 — identical
`per_row_i8` math), reports the encoder latency for both, and times the int8-install
step for each (the per-run quant that the offline path removes)."""

import argparse
import time

import numpy as np
import torch
from PIL import Image


def _now():
    torch.cuda.synchronize()
    return time.perf_counter()


def _iou(a, b):
    a = a.bool().reshape(-1); b = b.bool().reshape(-1)
    inter = (a & b).sum().item(); union = (a | b).sum().item()
    return inter / union if union else 1.0


def _best_mask(state):
    masks, scores = state.get("masks"), state.get("scores")
    if masks is None or scores is None or len(scores) == 0:
        return None
    i = int(torch.as_tensor(scores).argmax().item())
    m = masks[i]
    while m.dim() > 2:
        m = m[0]
    return m.bool().cpu()


def _mask_for(processor, image, prompt, dtype, warmup, iters):
    enc = []
    state = None
    for k in range(warmup + iters):
        t0 = _now()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            st = processor.set_image(image)
            st = processor.set_text_prompt(prompt=prompt, state=st)
        t1 = _now()
        if k >= warmup:
            enc.append((t1 - t0) * 1e3)
        state = st
    return _best_mask(state), float(np.median(enc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="cat")
    ap.add_argument("--fni8", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.model.sam3_image_processor import Sam3Processor
    from comfyui_superl8 import sam3_encoder

    ckpt = download_ckpt_from_hf(version="sam3.1")
    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False).eval().cuda()
    processor = Sam3Processor(model, confidence_threshold=0.5)
    image = Image.open(args.image).convert("RGB")

    # ---- A) on-load int8 (runtime quant) ----
    torch.cuda.synchronize()
    mem0 = torch.cuda.memory_allocated() / 2**30
    t0 = _now()
    hA = sam3_encoder.patch_sam3_encoder_int8(model, attention=True, linears=True)
    onload_install_ms = (_now() - t0) * 1e3
    onload_mem = torch.cuda.memory_allocated() / 2**30 - mem0
    maskA, encA = _mask_for(processor, image, args.prompt, dtype, args.warmup, args.iters)
    print(f"[on-load int8]  install {onload_install_ms:.0f} ms | +VRAM {onload_mem:.2f} GiB "
          f"| encoder {encA:.1f} ms | mask_px={int(maskA.sum()) if maskA is not None else 0}",
          flush=True)
    hA.unpatch()

    # ---- B) offline .fni8 (mmap + place) ----
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem0 = torch.cuda.memory_allocated() / 2**30
    t0 = _now()
    hB = sam3_encoder.load_sam3_encoder_int8_from_fni8(
        model, args.fni8, attention=True, device="cuda")
    offline_install_ms = (_now() - t0) * 1e3
    offline_mem = torch.cuda.memory_allocated() / 2**30 - mem0
    maskB, encB = _mask_for(processor, image, args.prompt, dtype, args.warmup, args.iters)
    print(f"[offline .fni8] install {offline_install_ms:.0f} ms | +VRAM {offline_mem:.2f} GiB "
          f"| encoder {encB:.1f} ms | mask_px={int(maskB.sum()) if maskB is not None else 0}",
          flush=True)
    print(f"[gate] offline linears: {hB.linears.stats.summary()}", flush=True)

    print("\n=== Validation ===", flush=True)
    if maskA is not None and maskB is not None:
        iou = _iou(maskB, maskA)
        print(f"mask IoU (offline .fni8 vs on-load int8) = {iou:.6f} "
              f"(target >= 0.999)", flush=True)
        assert iou >= 0.999, f"offline mask diverged from on-load int8 (IoU {iou})"
    print(f"int8 install time: on-load-quant {onload_install_ms:.0f} ms | "
          f"offline-place {offline_install_ms:.0f} ms", flush=True)
    print(f"int8 added VRAM:  on-load-quant {onload_mem:.2f} GiB (keeps fp32 w_fp) | "
          f"offline-place {offline_mem:.2f} GiB (int8 only) "
          f"-> {onload_mem - offline_mem:.2f} GiB saved", flush=True)
    print(f"encoder latency: on-load {encA:.1f} ms | offline {encB:.1f} ms", flush=True)
    print("[done] offline .fni8 matches on-load int8", flush=True)


if __name__ == "__main__":
    main()
