# SPDX-License-Identifier: Apache-2.0
"""Verify that FNI8ComponentLoader handles clip_type="" without stranding TE/VAE
(issue #72). Tests the loader's _resolve_clip_type helper with ComfyUI available."""

from __future__ import annotations

import pytest

pytest.importorskip("comfy.sd")

import comfy.sd

from comfyui_superl8 import arch
from comfyui_superl8.nodes import _resolve_clip_type

pytestmark = pytest.mark.comfy_e2e


def test_resolve_clip_type_returns_none_for_empty_clip_type():
    """Archs with clip_type="" (zimage, flux2, etc.) must not raise — instead
    return None so ComfyUI auto-detects the text encoder from the state dict."""
    a = arch.get("zimage")
    assert a is not None and a.clip_type == ""
    result = _resolve_clip_type(a)
    assert result is None, f"Expected None for empty clip_type, got {result!r}"

    a = arch.get("flux2")
    assert a is not None and a.clip_type == ""
    result = _resolve_clip_type(a)
    assert result is None, f"Expected None for empty clip_type, got {result!r}"


def test_resolve_clip_type_returns_enum_for_known_clip_type():
    """Archs with a confirmed clip_type still return the correct CLIPType member."""
    a = arch.get("flux1")
    assert a is not None and a.clip_type == "flux"
    result = _resolve_clip_type(a)
    assert result is not None
    assert isinstance(result, comfy.sd.CLIPType)
    assert result == comfy.sd.CLIPType.FLUX


def test_resolve_clip_type_raises_for_invalid_clip_type():
    """If clip_type is non-empty but doesn't match a CLIPType member, the
    ValueError still fires (existing error-detection preserved)."""
    from dataclasses import replace

    a = arch.get("flux1")
    assert a is not None
    bad = replace(a, clip_type="nonexistent_type")
    with pytest.raises(ValueError, match="CLIPType has no member"):
        _resolve_clip_type(bad)
