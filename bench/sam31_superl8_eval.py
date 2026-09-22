# SPDX-License-Identifier: Apache-2.0
"""SAM 3.1 fp-vs-int8 end-to-end proof on a real image (issue #120, Phase 0 + 1).

Phase 0 (oracle): load the official facebook/sam3.1 image model in fp16, run a real
image + text/box prompt through the whole pipeline (PE-L ViT encoder -> DETR detector
-> mask head), and save a REAL segmentation mask. This is the correctness oracle.

Phase 1 (int8 encoder): flip the ViT trunk (attention + linears) onto fni8's int8
dp4a path via `comfyui_superl8.sam3_encoder`, re-run, and report IoU-vs-fp, per-site SQNR
gate decisions, encoder latency fp-vs-int8, and peak VRAM. The mask decoder + detector
+ text encoder stay fp (mask-quality load-bearing, fni8/AGENTS.md).

Runs in the comfyui-fni8-sam3 image (fni8 built sm_70 + sam3 + weights volume) on a
free Tesla V100. See the PR / docs/sam31-fni8-design.md for the numbers.
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image


def _now():
    torch.cuda.synchronize()
    return time.perf_counter()


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.bool().reshape(-1)
    b = b.bool().reshape(-1)
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 1.0


def _best_mask(state):
    """(mask[H,W] bool, score, box) of the top-scoring detection, or None."""
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or scores is None or len(scores) == 0:
        return None
    i = int(torch.as_tensor(scores).argmax().item())
    m = masks[i]
    while m.dim() > 2:
        m = m[0]
    return m.bool().cpu(), float(torch.as_tensor(scores)[i].item()), state["boxes"][i]


def _save_mask(mask: torch.Tensor, path: str):
    arr = (mask.cpu().numpy().astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(path)


def _save_overlay(image: Image.Image, mask: torch.Tensor, path: str):
    img = np.asarray(image.convert("RGB")).astype(np.float32)
    m = mask.cpu().numpy().astype(bool)
    if m.shape != img.shape[:2]:
        return
    tint = np.array([255.0, 40.0, 40.0])
    img[m] = 0.55 * img[m] + 0.45 * tint
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(path)


def _run(processor, model, image, prompt, box, warmup, iters):
    """Return (state, encoder_ms, e2e_ms). set_image runs the ViT encoder; the prompt
    call runs the fp detector/mask-head."""
    for _ in range(warmup):
        st = processor.set_image(image)
        if box is not None:
            processor.reset_all_prompts(st)
            _prompt(processor, st, prompt, box, image.size)
        else:
            processor.set_text_prompt(prompt=prompt, state=st)
    enc, e2e = [], []
    state = None
    for _ in range(iters):
        t0 = _now()
        st = processor.set_image(image)
        t1 = _now()
        if box is not None:
            processor.reset_all_prompts(st)
            st = _prompt(processor, st, prompt, box, image.size)
        else:
            st = processor.set_text_prompt(prompt=prompt, state=st)
        t2 = _now()
        enc.append((t1 - t0) * 1e3)
        e2e.append((t2 - t0) * 1e3)
        state = st
    return state, float(np.median(enc)), float(np.median(e2e))


def _prompt(processor, state, prompt, box, size):
    if prompt:
        state = processor.set_text_prompt(prompt=prompt, state=state)
    if box is not None:
        w, h = size
        x0, y0, x1, y1 = box
        cx, cy = ((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h
        bw, bh = abs(x1 - x0) / w, abs(y1 - y0) / h
        state = processor.add_geometric_prompt(
            box=[cx, cy, bw, bh], label=True, state=state)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--box", default="", help="x0,y0,x1,y1 in pixels (optional)")
    ap.add_argument("--out", default="/job/out")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--rotate", action="store_true",
                    help="Hadamard-rotate int8 ViT attention (outlier mitigation, #74)")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    box = [float(v) for v in args.box.split(",")] if args.box else None

    # Volta: force math SDP for the fp reference (no flash/mem-efficient on sm_70).
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.model.sam3_image_processor import Sam3Processor

    print("[load] building sam3.1 image model (fp)...", flush=True)
    ckpt = download_ckpt_from_hf(version="sam3.1")
    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False).eval().cuda()
    processor = Sam3Processor(model, confidence_threshold=args.conf)

    # ViT trunk shape report
    trunk = model.backbone.vision_backbone.trunk
    nblk = len(trunk.blocks)
    a0 = trunk.blocks[0].attn
    print(f"[arch] ViT trunk: {nblk} blocks, dim={a0.qkv.in_features}, "
          f"heads={a0.num_heads}, head_dim={a0.head_dim}", flush=True)

    image = Image.open(args.image).convert("RGB")
    print(f"[image] {args.image} {image.size}", flush=True)

    from comfyui_superl8 import sam3_encoder

    def timed_pass(tag):
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            state, enc_ms, e2e_ms = _run(
                processor, model, image, args.prompt, box, args.warmup, args.iters)
        vram = torch.cuda.max_memory_allocated() / 2**30
        best = _best_mask(state)
        print(f"[{tag}] encoder {enc_ms:.1f} ms | e2e {e2e_ms:.1f} ms | "
              f"peak VRAM {vram:.2f} GiB", flush=True)
        if best is None:
            print(f"[{tag}] NO DETECTION for prompt {args.prompt!r}", flush=True)
        else:
            m, sc, bx = best
            print(f"[{tag}] best det score={sc:.3f} mask_px={int(m.sum())} "
                  f"box={[round(float(v),1) for v in bx.tolist()]}", flush=True)
        return state, best, enc_ms, e2e_ms, vram

    # ---- Phase 0: fp oracle ----
    print("\n=== Phase 0: fp oracle ===", flush=True)
    _, fp_best, fp_enc, fp_e2e, fp_vram = timed_pass("fp")
    if fp_best is not None:
        _save_mask(fp_best[0], f"{args.out}/mask_fp.png")
        _save_overlay(image, fp_best[0], f"{args.out}/overlay_fp.png")
        print(f"[fp] saved {args.out}/mask_fp.png + overlay_fp.png", flush=True)

    # ---- Phase 1: int8 ViT encoder ----
    print("\n=== Phase 1: int8 ViT encoder (attention + linears) ===", flush=True)
    handle = sam3_encoder.patch_sam3_encoder_int8(
        model, attention=True, linears=True, rotate=args.rotate)
    _, i8_best, i8_enc, i8_e2e, i8_vram = timed_pass("int8")
    if i8_best is not None:
        _save_mask(i8_best[0], f"{args.out}/mask_int8.png")
        _save_overlay(image, i8_best[0], f"{args.out}/overlay_int8.png")
        print(f"[int8] saved {args.out}/mask_int8.png + overlay_int8.png", flush=True)

    import json
    summ = handle.summary()
    print("\n[int8 gate] " + json.dumps(summ), flush=True)

    # ---- Validation ----
    print("\n=== Validation ===", flush=True)
    if fp_best is not None and i8_best is not None:
        iou = _iou(i8_best[0], fp_best[0])
        print(f"mask IoU (int8 vs fp oracle) = {iou:.4f}", flush=True)
    else:
        iou = None
        print("mask IoU = N/A (a pass produced no detection)", flush=True)
    print(f"encoder latency: fp {fp_enc:.1f} ms -> int8 {i8_enc:.1f} ms  "
          f"({fp_enc / i8_enc:.2f}x)", flush=True)
    print(f"e2e latency:     fp {fp_e2e:.1f} ms -> int8 {i8_e2e:.1f} ms  "
          f"({fp_e2e / i8_e2e:.2f}x)", flush=True)
    print(f"peak VRAM:       fp {fp_vram:.2f} GiB -> int8 {i8_vram:.2f} GiB", flush=True)

    with open(f"{args.out}/result.json", "w") as f:
        json.dump({
            "prompt": args.prompt, "box": box, "precision": args.precision,
            "iou": iou, "fp_encoder_ms": fp_enc, "int8_encoder_ms": i8_enc,
            "fp_e2e_ms": fp_e2e, "int8_e2e_ms": i8_e2e,
            "fp_vram_gib": fp_vram, "int8_vram_gib": i8_vram,
            "gate": summ,
        }, f, indent=2)
    print(f"[done] wrote {args.out}/result.json", flush=True)


if __name__ == "__main__":
    main()
