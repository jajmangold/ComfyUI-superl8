# SPDX-License-Identifier: Apache-2.0
"""ComfyUI-superl8 — int8 dp4a (sm_70 / Volta) acceleration for diffusion DiTs in
ComfyUI, via the ops-patch route (no fork). Shares the `.superl8` weight format with
the superl8 LLM stack."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Register AYS schedulers into ComfyUI's SCHEDULER_HANDLERS at import time.
# Idempotent (checks for existing 'ays' key); safe to import outside ComfyUI.
from . import ays  # noqa: F401

# SAM 3 / 3.1 segmentation nodes (int8 dp4a Perception-Encoder ViT backbone, #120).
# Merged here (not in nodes.py) to keep the ViT-encoder glue separable from the DiT
# path; the mappings are empty outside a ComfyUI process (guarded in sam3_nodes).
from .sam3_nodes import SAM3_NODE_CLASS_MAPPINGS

# SAM 3D Objects + SAM 3D Body nodes (int8 dp4a ViT/DINOv3 backbone; image -> 3D).
# See docs/sam3d-superl8-design.md. Kept out of nodes.py for the same separability reason.
from .sam3d_nodes import SAM3D_NODE_CLASS_MAPPINGS

# CAS-backed artifact cache nodes (#190). Content-addressed latents and conditioning
# with model-revision provenance. Pure Python core (artifact_cache.py), thin ComfyUI
# wrapper (cache_nodes.py). No CUDA, no model execution.
from .cache_nodes import NODE_CLASS_MAPPINGS as CACHE_NODE_CLASS_MAPPINGS
from .cache_nodes import NODE_DISPLAY_NAME_MAPPINGS as CACHE_NODE_DISPLAY_MAPPINGS

# ComfyUI loads custom nodes in directory order; ComfyUI_GGUF is normally present
# before this package. Patch its public GGMLOps seam when available, while keeping
# this node pack installable on systems that do not use GGUF at all.
try:
    from .gguf_ops import enable_superl8_gguf

    GGUF_NATIVE_ENABLED = bool(enable_superl8_gguf())
except Exception:
    GGUF_NATIVE_ENABLED = False

NODE_CLASS_MAPPINGS.update(SAM3_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SAM3D_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CACHE_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {k: v.TITLE for k, v in SAM3_NODE_CLASS_MAPPINGS.items()}
)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {k: v.TITLE for k, v in SAM3D_NODE_CLASS_MAPPINGS.items()}
)
NODE_DISPLAY_NAME_MAPPINGS.update(CACHE_NODE_DISPLAY_MAPPINGS)

__all__ = ["GGUF_NATIVE_ENABLED", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.0.1"
