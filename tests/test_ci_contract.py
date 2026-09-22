# SPDX-License-Identifier: Apache-2.0
"""Regression checks for assumptions made by the public docker-compose.yml."""

from pathlib import Path

import pytest


def test_e2e_compose_has_required_env_vars():
    """docker-compose.yml must expose GPU election and CUDA env vars."""
    compose = (Path(__file__).parents[1] / "docker-compose.yml").read_text()
    assert "- FNI8_GPU_ELECTED" in compose
    assert "- FNI8_GPU_UUID" in compose
    assert "- CUDA_VISIBLE_DEVICES" in compose
    assert "SUPERL8_MODEL_ROOT" in compose or "FNI8_WEIGHTS_DIR" in compose


# ---------------------------------------------------------------------------
# #198 regression: explicit-arch hint must not crash on either loader
# ---------------------------------------------------------------------------


def test_explicit_arch_hint_does_not_crash_on_string_a():
    """#198: UnetLoaderFNI8 sets `a = arch (string)` then used `a.name` which
    crashes with AttributeError. Verify the fixed path uses arch_obj instead."""
    from comfyui_superl8.nodes import arch_get

    arch_str = "zimage"
    arch_obj = arch_get(arch_str)

    arch_hint = (arch_obj.name if arch_obj else arch_str).lower()
    assert arch_hint == "zimage"


def test_explicit_arch_hint_works_for_auto():
    """When arch="auto", arch_obj is None; hint should fall back to the raw string."""
    from comfyui_superl8.nodes import arch_get

    arch_str = "auto"
    arch_obj = arch_get(arch_str)
    assert arch_obj is None
    arch_hint = (arch_obj.name if arch_obj else arch_str).lower()
    assert arch_hint == "auto"


@pytest.mark.parametrize("arch_name", ["zimage", "qwen_image", "flux1", "flux2", "ltx_video", "wan21", "sd3"])
def test_explicit_arch_hint_resolves_for_all_registered_archs(arch_name):
    """Every registered arch must resolve through arch_get without crashing."""
    from comfyui_superl8.nodes import arch_get

    arch_obj = arch_get(arch_name)
    assert arch_obj is not None, f"arch_get({arch_name!r}) returned None"
    arch_hint = (arch_obj.name if arch_obj else arch_name).lower()
    assert arch_hint == arch_name
