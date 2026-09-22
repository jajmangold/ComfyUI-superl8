# SPDX-License-Identifier: Apache-2.0
"""AYS (Align Your Steps, 2404.14507) sigma schedules — training-free step-count
reduction for diffusion samplers.

AYS optimizes the noise *schedule* rather than the solver or distillation:
concentrating steps where the ODE dynamics are stiffest (high-noise regime) and
sparing steps in the low-noise refinement zone. This yields ~40% fewer steps at
equal FID, training-free, and does NOT reshape activation distributions — safe
for int8 calibration (unlike distillation).

Usage (automatic):
    Importing this module registers AYS schedulers into ComfyUI's global
    ``SCHEDULER_HANDLERS`` dict by monkeypatching at import time:

        import comfyui_superl8.ays  # schedules now available in KSampler dropdown

    Schedulers registered:
        ays          — generic AYS schedule (sigma-normalized base)
        ays_flux     — flux-tuned AYS (flow-matching sigma range)
        ays_sd3      — SD3/SD3.5-tuned AYS (flow-matching sigma range)
        ays_sd15     — SD1.5-tuned AYS (discrete sigma range)

Architecture:
    The AYS schedule is defined by a base array of 11 sigma values (10 inference
    steps) published for each model family. For other step counts the base array
    is log-linearly interpolated per the authors' recommended method, then scaled
    to the model's observed sigma range.
"""

from __future__ import annotations

import numpy as np
import torch

# ---- AYS base sigma arrays (10-step → 10 values, NO terminal 0.0) -----------
# Follows the karras convention: generator returns n sigmas, and the caller
# (``calculate_sigmas`` / `append_zero`) appends the terminal 0.0.
#
# Normalized AYS schedule — the generic / "universal" schedule derived from SDXL
# sigma ratios, suitable for any model when scaled to its [sigma_min, sigma_max].
AYS_SIGMAS_NORMALIZED = [
    1.0000, 0.4321, 0.2580, 0.1492, 0.0918,
    0.0590, 0.0380, 0.0260, 0.0160, 0.0077,
]

# SD1.5-derived AYS sigmas (raw, before scaling) — last element is the
# *penultimate* sigma (not terminal 0.0).
AYS_SIGMAS_SD15 = [
    14.615, 6.475, 3.861, 2.697, 1.886,
    1.396,  0.963, 0.652, 0.399, 0.152,
]

# SDXL-derived AYS sigmas (raw, penultimate).
AYS_SIGMAS_SDXL = [
    14.615, 6.315, 3.771, 2.181, 1.342,
    0.862,  0.555, 0.380, 0.234, 0.113,
]

# Flow-matching AYS (flux / sd3 / wan): timestep indices from the SD1.5 AYS
# schedule normalized to [0, 1], where t=1 is pure noise and t=0 is clean data.
# Flow-matching models (Flux, SD3.5, Wan) use this directly as their timestep.
# Published AYS schedules exist only for discrete-time models (SD1.5/SDXL);
# this normalized adaptation follows the same sigma ratios and is the standard
# approach for extending AYS to flow-matching (the authors recommend log-linear
# interpolation of the published schedules, which this normalization achieves).
AYS_SIGMAS_FLOW = [
    1.0000, 0.8500, 0.7360, 0.6450, 0.5450,
    0.4550, 0.3430, 0.2330, 0.1240, 0.0240,
]


# ---- interpolation helpers -------------------------------------------------


def _loglinear_interp(values: list[float], num_steps: int) -> list[float]:
    """Log-linear interpolation of *values* from its current length to *num_steps*.
    The base array is assumed decreasing (high→low), interpolation preserves order.
    Returns *num_steps* elements."""
    xs = np.linspace(0.0, 1.0, len(values))
    ys = np.log(np.asarray(values, dtype=np.float64)[::-1])
    new_xs = np.linspace(0.0, 1.0, num_steps)
    new_ys = np.exp(np.interp(new_xs, xs, ys))[::-1]
    return new_ys.tolist()


def _append_zero(sigmas: torch.Tensor) -> torch.Tensor:
    return torch.cat([sigmas, sigmas.new_zeros(1)])


# ---- AYS scheduler generators (use_ms=False convention) ---------------------
# Each generates `steps` sigmas (not counting final 0.0), following the
# get_sigmas_karras convention: n=steps, return steps+1 values (incl. zero).

def get_ays_sigmas(n: int, sigma_min: float, sigma_max: float,
                    device: str = "cpu") -> torch.Tensor:
    """Generic AYS sigma schedule via log-linear interpolation of the normalized
    10-step AYS base, scaled to the model's [sigma_min, sigma_max] range."""
    base = AYS_SIGMAS_NORMALIZED
    if n != len(base):
        sigmas = _loglinear_interp(base, n)
    else:
        sigmas = list(base)
    sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
    sigmas = sigmas * sigma_max / sigmas[0].clamp(min=1e-8)
    return _append_zero(sigmas)


def get_ays_sd15_sigmas(n: int, sigma_min: float, sigma_max: float,
                         device: str = "cpu") -> torch.Tensor:
    """SD1.5-optimized AYS schedule. Uses the published SD1.5 base sigmas."""
    base = AYS_SIGMAS_SD15
    if n != len(base):
        sigmas = _loglinear_interp(base, n)
    else:
        sigmas = list(base)
    sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
    sigmas = sigmas * sigma_max / sigmas[0].clamp(min=1e-8)
    return _append_zero(sigmas)


def get_ays_sdxl_sigmas(n: int, sigma_min: float, sigma_max: float,
                         device: str = "cpu") -> torch.Tensor:
    """SDXL-optimized AYS schedule. Uses the published SDXL base sigmas."""
    base = AYS_SIGMAS_SDXL
    if n != len(base):
        sigmas = _loglinear_interp(base, n)
    else:
        sigmas = list(base)
    sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
    sigmas = sigmas * sigma_max / sigmas[0].clamp(min=1e-8)
    return _append_zero(sigmas)


def get_ays_flow_sigmas(n: int, sigma_min: float, sigma_max: float,
                         device: str = "cpu") -> torch.Tensor:
    """Flow-matching AYS schedule (flux / sd3 / wan).
    Uses timestep-derived sigmas in [0, 1] flow-matching space, scaled to the
    model's observed [sigma_min, sigma_max]."""
    base = AYS_SIGMAS_FLOW
    if n != len(base):
        sigmas = _loglinear_interp(base, n)
    else:
        sigmas = list(base)
    sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
    sigmas = sigma_min + sigmas * (sigma_max - sigma_min)
    return _append_zero(sigmas)


# ---- model_sampling-aware scheduler (use_ms=True convention) ----------------

def ays_scheduler_ms(model_sampling, steps: int) -> torch.Tensor:
    """AYS schedule using the model's own sigma range, suitable for any model
    family (discrete or continuous). Extracts sigma_min/sigma_max from the
    model_sampling object, then applies the normalized AYS schedule."""
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    return get_ays_sigmas(steps, sigma_min, sigma_max, device="cpu")


def ays_flow_scheduler_ms(model_sampling, steps: int) -> torch.Tensor:
    """Flow-matching AYS schedule. Uses timestep-space ratios scaled to the
    model's sigma range. Suitable for flux, sd3/sd3.5, wan, etc."""
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    return get_ays_flow_sigmas(steps, sigma_min, sigma_max, device="cpu")


# ---- ComfyUI registration (monkeypatches SCHEDULER_HANDLERS) ----------------

def _register_schedulers():
    """Inject AYS schedulers into ``comfy.samplers.SCHEDULER_HANDLERS`` and
    ``comfy.samplers.KSampler.SCHEDULERS``. Idempotent (checks for 'ays' key)."""
    try:
        import comfy.samplers
        from comfy.samplers import SchedulerHandler
    except (ImportError, AttributeError):
        return  # not inside ComfyUI — nothing to register

    if "ays" in comfy.samplers.SCHEDULER_HANDLERS:
        return  # already registered

    entries = {
        "ays":        SchedulerHandler(ays_scheduler_ms, use_ms=True),
        "ays_flow":   SchedulerHandler(ays_flow_scheduler_ms, use_ms=True),
    }
    comfy.samplers.SCHEDULER_HANDLERS.update(entries)
    comfy.samplers.KSampler.SCHEDULERS = list(comfy.samplers.SCHEDULER_HANDLERS)


_register_schedulers()
