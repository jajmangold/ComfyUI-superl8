# SPDX-License-Identifier: Apache-2.0
"""ComfyUI is imported lazily so the package (and its unit tests) load without it.
Inside ComfyUI, `comfy` resolves and the real op/patcher bases are used; outside
(CI / kernel tests) COMFY is False and only the pure numerics are exercised."""
from __future__ import annotations

try:
    import comfy.model_patcher  # noqa: F401
    import comfy.ops  # noqa: F401
    import comfy.sd  # noqa: F401

    COMFY = True
    MANUAL_CAST = comfy.ops.manual_cast
    MODEL_PATCHER = comfy.model_patcher.ModelPatcher
except Exception:  # pragma: no cover - exercised only outside ComfyUI
    COMFY = False
    MANUAL_CAST = object
    MODEL_PATCHER = object


def require_comfy():
    if not COMFY:
        raise RuntimeError(
            "ComfyUI not importable — these nodes must run inside a ComfyUI process "
            "(drop this package in ComfyUI/custom_nodes/)."
        )
