# SPDX-License-Identifier: Apache-2.0
"""ComfyUI entry point — ComfyUI imports this package from custom_nodes/ and reads
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS."""
try:
    # ComfyUI loads this dir as a package, so the relative import resolves.
    from .comfyui_superl8 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Under pytest the dir name ("ComfyUI-superl8", a hyphen) isn't a valid package
    # name, so there's no parent package for the relative import — fall back to the
    # absolute import of the pip-installed inner package.
    from comfyui_superl8 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
