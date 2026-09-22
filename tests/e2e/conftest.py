# SPDX-License-Identifier: Apache-2.0
"""Points ComfyUI's `diffusion_models` search path at the mounted weights archive.
Standalone (these tests never run ComfyUI's own main.py startup, so nothing else
would parse an extra_model_paths.yaml for us)."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("folder_paths")

WEIGHTS_DIR = os.environ.get("FNI8_WEIGHTS_DIR", "")


@pytest.fixture(scope="session", autouse=True)
def _register_fni8_weights_dir():
    import folder_paths

    if WEIGHTS_DIR and os.path.isdir(WEIGHTS_DIR):
        folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)
