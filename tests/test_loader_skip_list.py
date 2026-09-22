# SPDX-License-Identifier: Apache-2.0
"""Validate the bf16 skip-list recalibration (issue #103 — post-1.0).

The skip-list was a Nunchaku heuristic — on this fleet bf16 is 4–40× slower than int8
dp4a. The recalibration moves four projection types (`final_layer`, `proj_out`,
`context_embedder`, `guidance_in`) from always-bf16 to int8-gated (the runtime SQNR
gate in `FNI8Ops` catches per-instance outliers). This test suite validates:
  1. The override list correctly exempts the four patterns from `_SKIP_HINTS`.
  2. Non-override skip-list entries (norms, modulations) still stay raw/bf16.
  3. The `_is_dit_linear` function respects both lists.
  4. The per-weight SQNR of each override pattern clears the gate bar.
"""

from __future__ import annotations

import pytest
import torch

from comfyui_superl8.int8_linear import quantize_linear_weight, sqnr_gate, int8_linear
from comfyui_superl8.loader import _is_dit_linear, _SKIP_HINTS, _LOADER_SKIP_OVERRIDE

# -- Static tests (no CUDA needed) --

# Realistic DiT shapes for each override pattern
_OVERRIDE_WEIGHT_SHAPES: dict[str, tuple[int, int]] = {
    "final_layer":       (64, 4096),
    "proj_out":          (4096, 4096),
    "context_embedder":  (4096, 4096),
    "guidance_in":       (3072, 256),
}

# Patterns that stay in _SKIP_HINTS (always-fp)
_FP_PATTERNS = ("norm", "modulation", "adaln", "ada_ln", "pos_embed", "patch_embed",
                "_embed", "embedder", "time_in", "txt_in", "img_in", "vector_in")


def _make_linear_name(pattern: str) -> str:
    """Build a realistic full weight name from the hint pattern."""
    if pattern == "final_layer":
        return "final_layer.linear.weight"
    if pattern == "proj_out":
        return "blocks.0.attn.proj_out.weight"
    if pattern == "context_embedder":
        return "context_embedder.linear.weight"
    if pattern == "guidance_in":
        return "guidance_in.weight"
    if pattern in ("norm",):
        return "blocks.0.norm1.weight"
    if pattern in ("modulation",):
        return "blocks.0.modulation.lin.weight"
    if pattern in ("time_in",):
        return "time_in.weight"
    if pattern in ("txt_in",):
        return "txt_in.weight"
    if pattern in ("_embed", "embedder"):
        return "some_embedder.weight"
    if pattern in ("pos_embed", "patch_embed", "ada_ln", "adaln",
                   "img_in", "vector_in"):
        return f"blocks.0.{pattern}.weight"
    return f"blocks.0.{pattern}.weight"


# 1. Override patterns are correctly classified as linears (int8-eligible)

def test_override_patterns_are_linears():
    for pattern, (out, inp) in _OVERRIDE_WEIGHT_SHAPES.items():
        name = _make_linear_name(pattern)
        w = torch.randn(out, inp)
        assert _is_dit_linear(name, w), (
            f"{pattern} should be int8-eligible but _is_dit_linear returned False"
        )


def test_override_patterns_actual_hint_names():
    """The patterns as they appear in real state-dict keys: verify substring match."""
    real_names = [
        "final_layer.linear.weight",
        "final_layer.bias",
        "blocks.0.attn.proj_out.weight",
        "context_embedder.weight",
        "context_embedder.linear.weight",
        "guidance_in.weight",
        "guidance_in.bias",
    ]
    for name in real_names:
        n = name.lower()
        assert any(s in n for s in _LOADER_SKIP_OVERRIDE), (
            f"'{name}' should match _LOADER_SKIP_OVERRIDE"
        )
        w = torch.randn(64, 256)  # arbitrary 2-D shape with dim%4==0
        assert _is_dit_linear(name, w), (
            f"'{name}' should pass _is_dit_linear (override active)"
        )


# 2. FP patterns (in _SKIP_HINTS but NOT in override) stay as non-linears

def test_fp_patterns_are_not_linears():
    names = [_make_linear_name(p) for p in _FP_PATTERNS]
    for name in names:
        n = name.lower()
        # Should match _SKIP_HINTS
        assert any(s in n for s in _SKIP_HINTS), (
            f"'{name}' should match _SKIP_HINTS"
        )
        # Should NOT match override
        assert not any(s in n for s in _LOADER_SKIP_OVERRIDE), (
            f"'{name}' should NOT match _LOADER_SKIP_OVERRIDE"
        )
        w = torch.randn(1024, 1024)
        assert not _is_dit_linear(name, w), (
            f"'{name}' should be NOT int8-eligible (fp-only)"
        )


# 3. Override patterns that would ALSO match _SKIP_HINTS via broad patterns
#    (e.g., "context_embedder" contains "_embed" and "embedder") — the override
#    must take priority.

def test_override_takes_priority_over_broad_skip_hints():
    name = "context_embedder.linear.weight"
    n = name.lower()
    # Sanity: broad skip-hints DO match
    assert any(s in n for s in ("_embed", "embedder"))
    # But override also matches
    assert any(s in n for s in _LOADER_SKIP_OVERRIDE)
    w = torch.randn(4096, 4096)
    # Override must win
    assert _is_dit_linear(name, w), (
        "override must win over broad _embed/embedder skip-hints"
    )


# 4. Non-override, non-skip linears still work normally

def test_vanilla_dit_linear_not_affected():
    name = "double_blocks.0.img_attn.qkv.weight"
    w = torch.randn(3072, 1024)
    assert _is_dit_linear(name, w)


# 5. Edge: non-2D weights still rejected even for override patterns

def test_override_still_rejects_non_2d():
    name = "final_layer.bias"
    w_bias = torch.randn(64)          # 1-D
    assert not _is_dit_linear(name, w_bias)
    w_conv = torch.randn(64, 3, 3, 3)  # 4-D
    assert not _is_dit_linear("final_layer.conv.weight", w_conv)


# 6. Edge: dim % 4 != 0 still rejected

def test_override_rejects_non_dp4a_dim():
    name = "final_layer.linear.weight"
    w = torch.randn(64, 66)  # 66 % 4 != 0
    assert not _is_dit_linear(name, w)


# 7. Per-weight SQNR for override patterns (requires CUDA)

def _has_cuda():
    return torch.cuda.is_available()


def test_override_pattern_weight_sqnr():
    """Validate that the override patterns actually pass sqnr_gate with
    representative random weights (static, not activation-sensitive, test)."""
    if not _has_cuda():
        pytest.skip("needs CUDA for dp4a")
    for pattern, (out_dim, in_dim) in _OVERRIDE_WEIGHT_SHAPES.items():
        w = torch.randn(out_dim, in_dim, device="cuda", dtype=torch.float16)
        qt = quantize_linear_weight(w)
        x = torch.randn(2, 256, in_dim, device="cuda", dtype=torch.float16)
        y_int8 = int8_linear(x, qt)
        y_fp = torch.nn.functional.linear(x, w)
        # Assert gate passes for these static weights
        assert sqnr_gate(y_int8, y_fp), (
            f"{pattern} ({out_dim},{in_dim}) failed sqnr_gate — "
            "reconsider this entry in _LOADER_SKIP_OVERRIDE"
        )
