# SPDX-License-Identifier: Apache-2.0
"""Fast build-only probe: which int8 .fni8 keys actually MATCH a model parameter name
(and thus engage the dp4a path via assign_int8_weights) vs which fall through to comfy's
assign=False copy (raw int8 codes -> garbage fp weights)."""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import bench.full_pipeline_zimage as F


def fam(n):
    return re.sub(r"\d+", "#", n)


def main():
    import folder_paths
    folder_paths.add_model_folder_path("diffusion_models", F.WEIGHTS_DIR)
    import comfy.sd
    from comfyui_superl8.superl8_tensor import FNI8Tensor
    from comfyui_superl8.loader import load_fni8_dit, remap_diffusers_to_native
    from comfyui_superl8.ops import FNI8Ops

    path = folder_paths.get_full_path("diffusion_models", F.ZIMAGE_DIT)
    # No head dequant, no extra keep-fp: see the raw picture.
    sd = load_fni8_dit(path, device="cpu", strip_prefix="", dequant_fp=())
    sd = remap_diffusers_to_native(sd, None)
    int8_keys = [k for k, v in sd.items() if isinstance(v, FNI8Tensor)]
    print(f"int8 tensors in loaded sd: {len(int8_keys)}")

    model = comfy.sd.load_diffusion_model_state_dict(
        dict(sd), model_options={"custom_operations": FNI8Ops, "dtype": torch.bfloat16})
    names = dict(model.model.diffusion_model.named_parameters())
    matched = [k for k in int8_keys if k in names]
    missed = [k for k in int8_keys if k not in names]
    print(f"int8 keys MATCHED to a model param (-> dp4a): {len(matched)}")
    for k, c in Counter(fam(k) for k in matched).most_common():
        print(f"   MATCH {c:4d}  {k}")
    print(f"int8 keys MISSED (-> comfy copied raw int8 codes = GARBAGE): {len(missed)}")
    for k, c in Counter(fam(k) for k in missed).most_common():
        print(f"   MISS  {c:4d}  {k}")

    # For a couple of missed attention keys, show near-miss model param names.
    if missed:
        sample = missed[0]
        base = sample.rsplit(".", 2)[0]
        print(f"\n  sample missed key: {sample}")
        print(f"  model params under '{base}':")
        for n in names:
            if n.startswith(base):
                print(f"     {n}")


if __name__ == "__main__":
    main()
