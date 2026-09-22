# SPDX-License-Identifier: Apache-2.0
"""Guidance-distillation harvest — remove the 2× cond+uncond batch for
guidance-distilled DiTs.

Guidance-distilled DiTs (FLUX.1-dev, certain SD3.5/PixArt/Wan variants) bake CFG
into a *single* forward pass: the model was fine-tuned so that running with
cfg=1.0 (cond-only, no uncond) matches what the base model would produce at
cfg>1. This removes the 2× compute of the uncond branch — ~2× dp4a on the int8
path.

Architecture:
    ``mark_guidance_distilled(model)``
        Sets ``model.model_options["fni8_guidance_distilled"] = True``. This flag
        tells the sampler to skip the uncond branch unconditionally, regardless
        of the CFG scale the user requests.

    ``patch_sampler(model)``
        Injects a ``sampler_pre_cfg_function`` hook that zeros the uncond output
        when the model is marked guidance-distilled. This is redundant with
        ComfyUI's built-in ``cond_scale == 1.0 → skip uncond`` optimization
        (``sampling_function`` line ~608), but ensures correctness when a user
        accidentally sets ``cfg > 1`` on a distilled model: the hook clamps the
        uncond contribution to zero so the output is purely the cond pass.

    ``GUIDANCE_DISTILLED_ARCHS``
        Known set of (arch_name, checkpoint_hint) tuples. When loading a DiT
        through ``UnetLoaderFNI8``, the loader checks this list and auto-marks
        the model if the arch is listed.

Known guidance-distilled checkpoints:
    - black-forest-labs__FLUX.1-dev (cfg=1, was trained distilled)
    - stabilityai__stable-diffusion-3.5-large-turbo (CFG-distilled variant of SD3.5)
    - PixArt-Sigma (training-free distillation via EDM-style karras schedule)
    - Wan 2.1 (cfg-distilled variant, can run at cfg=1)

Usage:
    from comfyui_superl8.guidance_harvest import mark_guidance_distilled

    # At model load time:
    model = comfy.sd.load_diffusion_model_state_dict(...)
    mark_guidance_distilled(model)
"""

from __future__ import annotations

# Architecture names known to have guidance-distilled checkpoint variants.
# When one of these arches is loaded, the sampler automatically skips the
# unconditional branch for a 2× compute saving.
GUIDANCE_DISTILLED_ARCHS: set[str] = {
    "flux1",
    "sd35",       # SD3.5-large-turbo (CFG-distilled)
    "pixart",     # PixArt-Sigma (EDM training-free distilled)
    "wan21",      # Wan 2.1 (some variants are CFG-distilled)
    "wan22",      # Wan 2.2 (CFG-distilled variant)
    "zimage",     # Z-Image Turbo (8-step, CFG=1)
}

FNI8_KEY = "fni8_guidance_distilled"


def mark_guidance_distilled(model) -> None:
    """Mark ``model`` as guidance-distilled so the uncond branch is always
    skipped. Safe to call multiple times; idempotent."""
    model.model_options[FNI8_KEY] = True


def is_guidance_distilled(model) -> bool:
    """Return True if the model was marked as guidance-distilled."""
    return model.model_options.get(FNI8_KEY, False)


def make_pre_cfg_hook():
    """Build a ``sampler_pre_cfg_function`` hook that zeros the uncond output
    when the model is guidance-distilled.

    The hook signature ComfyUI expects (see ``sampling_function`` in
    comfy/samplers.py):
        ``def hook(args: dict) -> dict``
    where ``args["conds_out"]`` is a list ``[cond_pred, uncond_pred]``.

    When the model is guidance-distilled, we set ``uncond_pred = cond_pred`` so
    the CFG math becomes identity::
        cfg = uncond + scale * (cond - uncond) -> cond

    This is defensive: ComfyUI's built-in ``cond_scale == 1.0 → skip uncond``
    optimization already handles the happy path. This hook catches cases where
    a user inadvertently sets ``cfg > 1`` on a distilled model, preventing the
    double-prediction that would waste compute and potentially distort output.
    """

    def pre_cfg_hook(args):
        conds_out = args["conds_out"]
        model_options = args.get("model_options", {})
        if (
            model_options.get(FNI8_KEY)
            and len(conds_out) == 2
            and conds_out[1] is not None
        ):
            # Replace uncond with cond — CFG becomes identity regardless of scale.
            # (When comfy's own cfg1 optimization already dropped uncond, conds_out[1]
            # is None and there is nothing to clamp — leave it for cfg_function.)
            conds_out[1] = conds_out[0]
        # ComfyUI's `sampler_pre_cfg_function` contract is `out = fn(args); out[0]` —
        # the hook must return the (modified) conds_out LIST, not the args dict.
        # Returning `args` made comfy index a dict -> KeyError(0) on every distilled
        # denoise (Z-Image/FLUX/…) once run against current comfy.
        return conds_out

    return pre_cfg_hook


def _patch_model(model) -> bool:
    """Inject the pre-CFG hook if the model is guidance-distilled and not yet
    patched. Returns True if patched, False if already patched or not distilled."""
    marker = f"{FNI8_KEY}_patched"
    if not is_guidance_distilled(model) or model.model_options.get(marker):
        return False

    pre_hooks = model.model_options.setdefault("sampler_pre_cfg_function", [])
    pre_hooks.append(make_pre_cfg_hook())
    model.model_options[marker] = True
    return True
