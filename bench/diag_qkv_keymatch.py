# SPDX-License-Identifier: Apache-2.0
"""Generalized build-only probe (CPU, no GPU): for a given arch + `.fni8`, replicate
exactly what `UnetLoaderFNI8.load` does up to the model build, then report how many
int8 attention weights actually MATCH a built-model parameter (-> dp4a engages) vs MISS
(-> comfy copies raw int8 codes into fp params = garbage attention). This is the
memory-cheap diagnostic behind the Z-Image qkv-fusion finding
(docs/zimage-full-pipeline-findings.md), generalized so Flux/Qwen can be checked without
a full multi-step denoise (which their 17-20 GB int8 DiTs can't fit on a 16 GB card).

    python3 custom_nodes/ComfyUI-superl8/bench/diag_qkv_keymatch.py <arch> <fni8_filename>
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
WEIGHTS_DIR = os.environ.get("FNI8_WEIGHTS_DIR", "")


def fam(n):
    return re.sub(r"\d+", "#", n)


def is_attn(n):
    return any(t in n.lower() for t in ("attn", "to_q", "to_k", "to_v", "qkv",
                                        "add_q", "add_k", "add_v"))


def main():
    arch = sys.argv[1]
    fname = sys.argv[2]
    import folder_paths
    folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)
    import comfy.sd

    from comfyui_superl8.arch import get as arch_get
    from comfyui_superl8.superl8_tensor import FNI8Tensor
    from comfyui_superl8.loader import (
        _HEAD_FP_DEQUANT,
        fni8_dit_native_dtype,
        fuse_attention_qkv_int8,
        load_fni8_dit,
        remap_diffusers_to_native,
    )
    from comfyui_superl8.ops import FNI8Ops

    path = folder_paths.get_full_path("diffusion_models", fname)
    arch_obj = arch_get(arch)
    dequant_fp = tuple(_HEAD_FP_DEQUANT) + (arch_obj.fp_dequant if arch_obj else ())
    native = fni8_dit_native_dtype(path)
    dtype = torch.bfloat16 if native == "bfloat16" else torch.float16
    prefix = "diffusion_model."
    sd = load_fni8_dit(path, device="cpu", strip_prefix=prefix, dequant_fp=dequant_fp)
    sd = remap_diffusers_to_native(sd, arch)
    fused_attn = fuse_attention_qkv_int8(sd, dtype=dtype) if (
        arch_obj and arch_obj.fuse_attn_qkv) else {}
    int8_weights = {k: v for k, v in sd.items() if isinstance(v, FNI8Tensor)}
    int8_weights.update(fused_attn)

    build_sd = {}
    for k, v in sd.items():
        if isinstance(v, FNI8Tensor) and getattr(v, "q_scheme", "") == "per_group_i4":
            n_out, cols = int(v.shape[0]), int(v.shape[1])
            build_sd[k] = torch.zeros((), dtype=dtype).expand(n_out, cols * 2)
        else:
            build_sd[k] = v
    model = comfy.sd.load_diffusion_model_state_dict(
        dict(build_sd), model_options={"custom_operations": FNI8Ops, "dtype": dtype})
    if model is None:
        print(f"BUILD FAILED for arch={arch}")
        return
    names = set(dict(model.model.diffusion_model.named_parameters()))

    def report(tag, keys):
        matched = [k for k in keys if k in names]
        missed = [k for k in keys if k not in names]
        print(f"  {tag}: total={len(keys)}  MATCH={len(matched)}  MISS={len(missed)}")
        return matched, missed

    all_attn = [k for k in int8_weights if is_attn(k)]
    print(f"==== {arch}  {fname} ====")
    print(f"fuse_attn_qkv={bool(arch_obj and arch_obj.fuse_attn_qkv)}  "
          f"fp_dequant={arch_obj.fp_dequant if arch_obj else ()}")
    print(f"int8 tensors kept aside: {len(int8_weights)}")
    _, missed = report("int8 ATTENTION weights", all_attn)
    report("int8 ALL weights", list(int8_weights))
    if missed:
        print("  MISSED attention key families (-> garbage int8 -> collapse risk):")
        for k, c in Counter(fam(k) for k in missed).most_common(12):
            print(f"     MISS {c:4d}  {k}")
        # show what model params exist near a missed key
        base = missed[0].rsplit(".", 2)[0]
        near = sorted(n for n in names if n.startswith(base))[:8]
        print(f"  model params near missed '{missed[0]}':")
        for n in near:
            print(f"     {n}")
    else:
        print("  ALL int8 attention weights matched -> int8 attention ENGAGES cleanly.")


if __name__ == "__main__":
    main()
