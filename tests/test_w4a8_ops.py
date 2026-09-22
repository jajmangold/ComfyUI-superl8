# SPDX-License-Identifier: Apache-2.0
"""W4A8 (per_group_i4) DiT Linear path: the enabler that lets a sub-8-bit DiT RUN
through the ComfyUI-superl8 dp4a forward.

The comfy DiT `Linear` only supported `per_row_i8` (W8A8); a `per_group_i4` (4-bit,
group-wise packed) weight — what the LTX-2.3 b4 checkpoint and the #80 shrink produce —
threw. These tests pin the wiring that closes that gap:

  1. `FNI8Tensor` carries the quant scheme (group_size, codebook) through the ops
     ComfyUI performs on weights (device move / clone / detach / dtype-dequant).
  2. the loader's `_dequant_qtensor` accepts a `per_group_i4` QTensor (unpack + group
     scale) instead of throwing.
  3. `FNI8Ops.Linear` dispatches a `per_group_i4` weight to fni8's W4A8 dp4a GEMM and
     matches the fp reference at the comfy latent gate (cos >= 0.985).

The pure-python metadata/dequant checks run anywhere; the GEMM forward needs a Volta
(sm_70) GPU + a W4A8-capable fni8.
"""
from __future__ import annotations

import pytest

pytest.importorskip("superl8")

import torch  # noqa: E402

from comfyui_superl8.superl8_tensor import FNI8Tensor  # noqa: E402
from comfyui_superl8.int8_linear import (  # noqa: E402
    dequantize_int4_weight,
    dequantize_qtensor_data,
    in_features_for_scheme,
    quantize_weight_int4,
)

GROUP = 128


def _rand_weight(N=256, K=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N, K, generator=g) * 0.02).float()


def _i4_qtensor(w, group_size=GROUP):
    """Quantize an fp weight to a per_group_i4 superl8.QTensor (as the .fni8 stores it)."""
    from superl8 import QTensor

    packed, scale = quantize_weight_int4(w, group_size=group_size)
    return QTensor(packed, scale, scheme="per_group_i4",
                   group_size=group_size, codebook="int4")


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


# ----------------------------- pure-python wiring -----------------------------

def test_in_features_for_scheme_doubles_i4():
    assert in_features_for_scheme(1024, "per_group_i4") == 2048  # packed K//2 -> K
    assert in_features_for_scheme(2048, "per_row_i8") == 2048
    assert in_features_for_scheme(2048, "raw") == 2048  # unknown -> identity


def test_fni8tensor_carries_scheme_through_move_clone_detach():
    qt = _i4_qtensor(_rand_weight())
    t = FNI8Tensor(qt.data, qt.scale, scheme="per_group_i4",
                   group_size=GROUP, codebook="int4")
    for made, how in [
        (t.to(device="cpu"), "to(device)"),
        (t.clone(), "clone"),
        (t.detach(), "detach"),
        (t.contiguous(), "contiguous (torch_function)"),
    ]:
        assert made.q_scheme == "per_group_i4", how
        assert made.q_group_size == GROUP, how
        assert made.q_codebook == "int4", how
        assert made.q_scale is not None, how
        assert made.dtype == torch.uint8, how  # never cast the packed codes to float


def test_fni8tensor_i4_to_dtype_dequantizes():
    """`.to(dtype=...)` on a per_group_i4 weight must unpack+dequant (ModelPatcher's
    eager-dtype path), not multiply the packed bytes by a mis-shaped scale."""
    w = _rand_weight()
    qt = _i4_qtensor(w)
    t = FNI8Tensor(qt.data, qt.scale, scheme="per_group_i4",
                   group_size=GROUP, codebook="int4")
    deq = t.to(dtype=torch.float32)
    assert deq.shape == w.shape and deq.dtype == torch.float32
    ref = dequantize_int4_weight(qt.data, qt.scale, GROUP, w.shape[0], w.shape[1],
                                 torch.float32)
    torch.testing.assert_close(deq, ref)
    assert _cos(deq, w) > 0.99  # int4 round-trip is faithful


def test_loader_dequant_qtensor_accepts_i4():
    """The gap the LTX e2e agent found: `_dequant_qtensor` threw on per_group_i4."""
    from comfyui_superl8.loader import _dequant_qtensor

    w = _rand_weight()
    qt = _i4_qtensor(w)
    out = _dequant_qtensor(qt, dtype=torch.float32)  # used to raise
    assert out.shape == w.shape
    assert _cos(out, w) > 0.99


def test_dequantize_qtensor_data_dispatches_by_scheme():
    w = _rand_weight()
    qt = _i4_qtensor(w)
    out = dequantize_qtensor_data(qt.data, qt.scale, "per_group_i4", GROUP, torch.float32)
    assert _cos(out, w) > 0.99


# ----------------------------- W4A8 GEMM forward ------------------------------

def _gpu_ready():
    if not torch.cuda.is_available():
        return False, "no CUDA GPU"
    from superl8 import _C

    if not hasattr(_C, "gemm_w4a8"):
        return False, "installed fni8 lacks the gemm_w4a8 W4A8 kernel"
    return True, ""


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_w4a8_ops_linear_matches_fp_reference(dtype):
    """A per_group_i4 FNI8Ops.Linear forward matches the fp reference (dequant weight)
    at the comfy latent gate (cos >= 0.985) — the DiT-block-level correctness gate."""
    ok, why = _gpu_ready()
    if not ok:
        pytest.skip(why)
    pytest.importorskip("comfy")
    from comfyui_superl8.ops import FNI8Ops

    dev = torch.device("cuda")
    N, K = 4096, 2048  # a real LTX-2.3 attn projection shape
    w = _rand_weight(N, K, seed=7)
    qt = _i4_qtensor(w)

    lin = FNI8Ops.Linear(K, N, bias=False, dtype=dtype, device=dev)
    lin._parameters["weight"] = FNI8Tensor(
        qt.data.to(dev), qt.scale.to(dev),
        scheme="per_group_i4", group_size=GROUP, codebook="int4",
    )

    g = torch.Generator().manual_seed(3)
    x = (torch.randn(64, K, generator=g) * 0.5).to(dev, dtype)
    y = lin.forward_comfy_cast_weights(x)
    assert torch.isfinite(y).all(), "W4A8 DiT Linear produced non-finite output"

    w_deq = dequantize_int4_weight(qt.data.to(dev), qt.scale.to(dev), GROUP, N, K, dtype)
    y_ref = torch.nn.functional.linear(x, w_deq, None)
    cos = _cos(y, y_ref)
    assert cos >= 0.985, f"W4A8 DiT Linear diverged from fp reference: cos {cos:.4f}"
    # The gate must ADMIT int8 (per_group_i4 through the dp4a path), not demote to fp.
    assert lin._fni8_sqnr_pass is True, "W4A8 linear was demoted to the fp fallback"
