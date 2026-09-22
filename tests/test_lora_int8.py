# SPDX-License-Identifier: Apache-2.0
"""Unit tests for issue #64: LoRA composition onto int8 FNI8Tensor weights.

Tests that LoRA weight deltas can be merged into int8-quantized weights via int8
accumulation, preserving the dp4a GEMM path.  The core function under test is
`apply_patches_to_int8_weight` which dequantizes to fp, applies patch functions,
then re-quantizes with the original scale — never dequantizing the entire model.
"""

import pytest
import torch

pytest.importorskip("superl8")

from comfyui_superl8.int8_linear import (
    apply_patches_to_int8_weight,
    dequantize_weight,
    int8_linear,
    quantize_linear_weight,
)

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="dp4a needs CUDA")


# ---------------------------------------------------------------------------
# Helpers to simulate ComfyUI-style LoRA weight_function callables
# ---------------------------------------------------------------------------


def _make_lora_weight_patch(
    out_dim: int, in_dim: int, rank: int, alpha: float, dtype: torch.dtype, device: str
) -> callable:
    """Return a weight_function callable equivalent to the ComfyUI LowVramPatch
    that applies `weight += (alpha / rank) * (B @ A)`."""
    A = torch.randn(rank, in_dim, device=device, dtype=dtype) * 0.1
    B = torch.randn(out_dim, rank, device=device, dtype=dtype) * 0.1
    lora_diff = (alpha / rank) * (B @ A)

    def patch_fn(w_fp: torch.Tensor) -> torch.Tensor:
        return w_fp + lora_diff.to(device=w_fp.device, dtype=w_fp.dtype)

    return patch_fn


def _make_lora_weight_patch_composed(
    out_dim: int,
    in_dim: int,
    rank: int,
    alpha: float,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> callable:
    """Deterministic LoRA patch (seeded) for repro in assertions."""
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(rank, in_dim, device=device, dtype=dtype, generator=g) * 0.1
    B = torch.randn(out_dim, rank, device=device, dtype=dtype, generator=g) * 0.1

    def patch_fn(w_fp: torch.Tensor) -> torch.Tensor:
        return w_fp + (alpha / rank) * (B @ A).to(device=w_fp.device, dtype=w_fp.dtype)

    return patch_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_apply_patches_preserves_int8_dtype(dtype):
    """apply_patches_to_int8_weight returns int8 data and unchanged scale."""
    out_dim, in_dim = 128, 512
    w = torch.randn(out_dim, in_dim, device="cuda", dtype=dtype) * 0.1
    qt = quantize_linear_weight(w)
    patch = _make_lora_weight_patch(out_dim, in_dim, rank=4, alpha=1.0, dtype=dtype, device="cuda")
    new_data, new_scale = apply_patches_to_int8_weight(
        qt.data.clone(),
        qt.scale,
        dtype,
        [patch],
    )
    assert new_data.dtype == torch.int8, f"expected int8, got {new_data.dtype}"
    assert new_data.shape == qt.data.shape
    assert new_scale is qt.scale or torch.equal(new_scale, qt.scale)
    assert new_scale.shape == qt.scale.shape


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_int8_lora_merge_via_weight_function(dtype):
    """After applying a LoRA patch via weight_function, int8_linear produces a
    different output (LoRA took effect) that matches the fp reference."""
    out_dim, in_dim = 256, 1024
    w = torch.randn(out_dim, in_dim, device="cuda", dtype=dtype) * 0.1
    x = torch.randn(32, in_dim, device="cuda", dtype=dtype)

    qt = quantize_linear_weight(w)

    # Baseline: no patch
    y_base = int8_linear(x, qt)

    # Patched
    patch = _make_lora_weight_patch(out_dim, in_dim, rank=8, alpha=2.0, dtype=dtype, device="cuda")
    new_data, new_scale = apply_patches_to_int8_weight(
        qt.data.clone(),
        qt.scale,
        dtype,
        [patch],
    )
    from superl8 import QTensor

    qt_patched = QTensor(new_data.to("cuda"), new_scale.to("cuda"), scheme="per_row_i8")
    y_patched = int8_linear(x, qt_patched)

    # Patched output must differ from baseline (LoRA changed the weight)
    diff = (y_patched - y_base).abs().mean().item()
    assert diff > 1e-6, f"LoRA patch had no measurable effect (mean diff={diff})"

    # Output dtype preserved
    assert y_patched.dtype == dtype, f"dtype changed: {y_patched.dtype} != {dtype}"

    # Output is valid
    assert torch.isfinite(y_patched).all(), "patched output has NaN/inf"

    # Match fp reference
    w_patched_fp = dequantize_weight(new_data, new_scale, dtype)
    y_fp = torch.nn.functional.linear(x, w_patched_fp.to("cuda"))
    cos = torch.nn.functional.cosine_similarity(
        y_patched.float().flatten(),
        y_fp.float().flatten(),
        dim=0,
    )
    assert cos.item() >= 0.99, f"cos={cos.item():.6f} < 0.99"


@cuda_only
def test_multiple_patches_stack_correctly():
    """Multiple LoRA patches applied sequentially compose correctly (same order as
    how ComfyUI's LowVramPatch chains weight_function entries)."""
    out_dim, in_dim = 128, 512
    dtype = torch.float16
    w = torch.randn(out_dim, in_dim, device="cuda", dtype=dtype) * 0.1
    x = torch.randn(16, in_dim, device="cuda", dtype=dtype)

    qt = quantize_linear_weight(w)

    # Two LoRA-style patches with different seeds
    patch1 = _make_lora_weight_patch_composed(
        out_dim,
        in_dim,
        rank=4,
        alpha=1.0,
        dtype=dtype,
        device="cuda",
        seed=42,
    )
    patch2 = _make_lora_weight_patch_composed(
        out_dim,
        in_dim,
        rank=4,
        alpha=1.0,
        dtype=dtype,
        device="cuda",
        seed=99,
    )

    # Apply both patches
    new_data, new_scale = apply_patches_to_int8_weight(
        qt.data.clone(),
        qt.scale,
        dtype,
        [patch1, patch2],
    )
    from superl8 import QTensor

    qt_patched = QTensor(new_data.to("cuda"), new_scale.to("cuda"), scheme="per_row_i8")
    y_patched = int8_linear(x, qt_patched)

    # Both patches applied in float
    w_fp_ref = dequantize_weight(qt.data.clone(), qt.scale, dtype).to("cuda")
    w_fp_ref = patch1(w_fp_ref)
    w_fp_ref = patch2(w_fp_ref)
    y_fp = torch.nn.functional.linear(x, w_fp_ref)

    cos = torch.nn.functional.cosine_similarity(
        y_patched.float().flatten(),
        y_fp.float().flatten(),
        dim=0,
    )
    assert cos.item() >= 0.99, f"multi-patch cos={cos.item():.6f} < 0.99"
    assert torch.isfinite(y_patched).all()
    assert y_patched.dtype == dtype


@cuda_only
def test_no_patch_is_noop():
    """Calling apply_patches_to_int8_weight with an empty patch list is a no-op."""
    out_dim, in_dim = 64, 256
    dtype = torch.float16
    w = torch.randn(out_dim, in_dim, device="cuda", dtype=dtype) * 0.1
    qt = quantize_linear_weight(w)

    new_data, new_scale = apply_patches_to_int8_weight(
        qt.data.clone(),
        qt.scale,
        dtype,
        [],
    )
    assert torch.equal(new_data, qt.data), "data changed with empty patches"
    assert torch.equal(new_scale, qt.scale) or (new_scale is qt.scale)


@cuda_only
def test_lora_merge_preserves_dp4a_path_accuracy():
    """The int8 dp4a path with a merged LoRA patch still passes the SQNR accuracy
    bar against the fp reference (cos >= 0.99)."""
    out_dim, in_dim = 256, 1024
    dtype = torch.float16
    w = torch.randn(out_dim, in_dim, device="cuda", dtype=dtype) * 0.1
    x = torch.randn(32, in_dim, device="cuda", dtype=dtype)

    qt = quantize_linear_weight(w)

    patch = _make_lora_weight_patch(out_dim, in_dim, rank=8, alpha=1.0, dtype=dtype, device="cuda")
    new_data, new_scale = apply_patches_to_int8_weight(
        qt.data.clone(),
        qt.scale,
        dtype,
        [patch],
    )
    from superl8 import QTensor

    qt_patched = QTensor(new_data.to("cuda"), new_scale.to("cuda"), scheme="per_row_i8")
    y_int8 = int8_linear(x, qt_patched)

    # FP reference: dequantize the patched weight, run fp linear
    w_fp = dequantize_weight(new_data, new_scale, dtype)
    y_fp = torch.nn.functional.linear(x, w_fp.to("cuda"))

    cos = torch.nn.functional.cosine_similarity(
        y_int8.float().flatten(),
        y_fp.float().flatten(),
        dim=0,
    )
    assert cos.item() >= 0.99, f"dp4a accuracy check failed after LoRA merge: cos={cos.item():.6f}"
