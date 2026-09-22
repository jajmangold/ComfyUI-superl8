# SPDX-License-Identifier: Apache-2.0
"""Z-Image int8-vs-fp quality gate via a direct `apply_model` forward.

The repo's e2e quality gate (`tests/e2e/test_quality_int8_vs_fp16.py`) drives
`comfy.sample.sample`, which errors on the current pinned ComfyUI build (a sampler
version drift unrelated to the int8 path). This harness reproduces the SAME metric —
latent cosine of the int8 dp4a DiT vs the fp reference DiT on identical conditioning —
by calling `BaseModel.apply_model` directly, so it is independent of the sampler.

Models are loaded sequentially (int8 freed before fp) to fit 16 GB. Used to confirm the
fp-fallback-skip optimization (`ops.py`) leaves the gate green.

    FNI8_GPU=8 docker compose run --rm --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI python3 \
       custom_nodes/ComfyUI-superl8/bench/quality_zimage_applymodel.py --size 64'
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bench.full_pipeline_zimage import (  # noqa: E402
    ZIMAGE_DIT, WEIGHTS_DIR, DEFAULT_PROMPT,
    encode_prompt, load_int8_model, load_fp_model,
)
from bench.quality import cosine_similarity  # noqa: E402

COS_FLOOR = 0.985


def _forward(model, x, emb, S):
    import comfy.model_management as mm
    mm.load_model_gpu(model)
    dev = mm.get_torch_device()
    dtype = model.model.get_dtype()
    sig = model.model.model_sampling.sigmas
    sigma = sig[len(sig) // 2].to(dev).reshape(1)
    topts = dict(model.model_options.get("transformer_options", {}))
    with torch.no_grad():
        return model.model.apply_model(
            x.to(dev, dtype), sigma, c_crossattn=emb.to(dev, dtype),
            num_tokens=S, transformer_options=topts,
        ).float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    import folder_paths
    folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)
    from comfyui_superl8.gate import is_sm70
    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return
    path = folder_paths.get_full_path("diffusion_models", ZIMAGE_DIT)

    cond, _ = encode_prompt(args.prompt)
    emb = cond[0][0]
    S = emb.shape[1]

    m_int8 = load_int8_model()
    in_ch = m_int8.model.diffusion_model.in_channels
    torch.manual_seed(args.seed)
    x = torch.randn(1, in_ch, args.size, args.size)

    y_int8 = _forward(m_int8, x, emb, S)
    del m_int8
    import comfy.model_management as mm
    mm.unload_all_models()
    torch.cuda.empty_cache()

    m_fp = load_fp_model(path)
    y_fp = _forward(m_fp, x, emb, S)

    cos = cosine_similarity(y_int8, y_fp)
    finite = bool(torch.isfinite(y_int8).all())
    print("=" * 60)
    print("Z-Image int8-vs-fp quality gate (apply_model, no sampler)")
    print(f"  latent cosine (int8 vs fp) : {cos:.6f}   (floor {COS_FLOOR})")
    print(f"  int8 output finite         : {finite}")
    print(f"  VERDICT                    : {'PASS' if cos >= COS_FLOOR and finite else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
