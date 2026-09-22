# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the guidance-distillation harvest hook (CPU, no GPU/fni8).

Regression guard for the comfy-`sampler_pre_cfg_function` contract: ComfyUI's
`sampling_function` does `out = fn(args); cfg_function(model, out[0], out[1], ...)`,
so the hook MUST return the (modified) `conds_out` LIST — not the `args` dict.
Returning `args` made comfy index a dict -> `KeyError(0)` on every distilled denoise
(Z-Image / FLUX / SD3.5-turbo / Wan-distilled) once run against current comfy.
"""
from __future__ import annotations

from comfyui_superl8.guidance_harvest import FNI8_KEY, make_pre_cfg_hook


def _args(conds_out, distilled=True):
    return {
        "conds_out": conds_out,
        "model_options": {FNI8_KEY: True} if distilled else {},
    }


def test_hook_returns_conds_out_list_not_args_dict():
    cond, uncond = object(), object()
    out = make_pre_cfg_hook()(_args([cond, uncond]))
    # comfy will do out[0], out[1]; a dict here would KeyError(0).
    assert isinstance(out, list)
    assert out[0] is cond and out[1] is cond  # uncond replaced by cond (identity CFG)


def test_hook_clamps_uncond_to_cond_when_distilled():
    cond, uncond = "C", "U"
    out = make_pre_cfg_hook()(_args([cond, uncond], distilled=True))
    assert out == [cond, cond]


def test_hook_noop_when_not_distilled():
    cond, uncond = "C", "U"
    out = make_pre_cfg_hook()(_args([cond, uncond], distilled=False))
    assert out == [cond, uncond]  # untouched, still a list


def test_hook_handles_cfg1_dropped_uncond():
    # comfy's own cfg1 optimization sets uncond_ = None -> conds_out[1] is None.
    cond = "C"
    out = make_pre_cfg_hook()(_args([cond, None], distilled=True))
    assert isinstance(out, list) and out[0] == cond and out[1] is None
