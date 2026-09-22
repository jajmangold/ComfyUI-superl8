# SPDX-License-Identifier: Apache-2.0
"""FNI8Ops.Linear fp-fallback must NOT run the discarded int8 dp4a GEMM.

When the per-layer SQNR gate has demoted a linear to the fp32 CUDA-core path
(`_fni8_sqnr_pass is False`), the earlier implementation still executed the int8
`int8_linear` GEMM every forward and then threw its result away in favour of the
fp32 matmul. On the Z-Image DiT the demoted layers are the large FFN `w2`
down-projections (K=10240), so that discarded GEMM was a measurable slice of every
denoise step. These tests pin the fix: a demoted layer runs the fp path ONLY, and its
output is numerically the fp32 fallback (byte-identical to the old behaviour's emitted
value — only the wasted work is gone).

Runs wherever ComfyUI is importable (the e2e image); CPU-only, no GPU needed, because
the whole point is that the CUDA dp4a GEMM is never entered on the demoted path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("comfy")
pytest.importorskip("superl8")

import torch  # noqa: E402

import comfyui_superl8.ops as ops_mod  # noqa: E402
from comfyui_superl8.superl8_tensor import FNI8Tensor  # noqa: E402
from comfyui_superl8.int8_linear import dequantize_weight  # noqa: E402
from comfyui_superl8.ops import FNI8Ops  # noqa: E402


def _make_int8_linear(N=16, K=8, seed=0):
    """A FNI8Ops.Linear (bias=False) with a resident int8 FNI8Tensor weight on CPU."""
    torch.manual_seed(seed)
    lin = FNI8Ops.Linear(K, N, bias=False)
    codes = torch.randint(-127, 127, (N, K), dtype=torch.int8)
    scale = torch.rand(N, dtype=torch.float32) * 0.02 + 0.001
    # Assign into _parameters directly (as loader.assign_int8_weights does) so the
    # FNI8Tensor subclass survives nn.Module's Parameter type-check.
    lin._parameters["weight"] = FNI8Tensor(codes, scale)
    return lin


def _fp32_reference(lin, x):
    w = dequantize_weight(lin.weight.int8_data(), lin.weight.q_scale, torch.float32)
    return torch.nn.functional.linear(x.float(), w, None).to(x.dtype)


def test_demoted_layer_skips_int8_gemm(monkeypatch):
    """A layer already demoted (`_fni8_sqnr_pass is False`) must never call
    int8_linear on subsequent forwards, and must return the fp32 fallback."""
    lin = _make_int8_linear()
    lin._fni8_sqnr_pass = False  # pretend the gate already demoted this layer

    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("int8_linear must not run on a demoted (fp-fallback) layer")

    monkeypatch.setattr(ops_mod, "int8_linear", _boom)

    x = torch.randn(4, 8, dtype=torch.float32)
    for _ in range(3):
        y = lin.forward_comfy_cast_weights(x)
    assert calls["n"] == 0
    torch.testing.assert_close(y, _fp32_reference(lin, x), rtol=0, atol=0)


def test_passing_layer_still_runs_int8(monkeypatch):
    """A passing layer keeps using int8_linear (no accidental fp demotion)."""
    lin = _make_int8_linear()
    lin._fni8_sqnr_pass = True

    seen = {"n": 0}
    sentinel = torch.zeros(4, 16, dtype=torch.float32)

    def _fake_int8(x, qt, bias=None):
        seen["n"] += 1
        return sentinel

    monkeypatch.setattr(ops_mod, "int8_linear", _fake_int8)
    x = torch.randn(4, 8, dtype=torch.float32)
    y = lin.forward_comfy_cast_weights(x)
    assert seen["n"] == 1
    torch.testing.assert_close(y, sentinel)


def test_first_call_gate_fail_demotes_and_then_skips(monkeypatch):
    """With the default single-sample calibration ($FNI8_SQNR_CALIB_SAMPLES unset), the
    first call runs int8 ONCE to measure the gate; when the cosine fails the layer is
    demoted and later forwards skip int8 entirely."""
    monkeypatch.delenv("FNI8_SQNR_CALIB_SAMPLES", raising=False)  # default = 1 (legacy)
    lin = _make_int8_linear()
    assert lin._fni8_sqnr_pass is None

    seen = {"n": 0}

    def _fake_int8(x, qt, bias=None):
        seen["n"] += 1
        # Return garbage uncorrelated with the fp reference so the SQNR cosine fails
        # (cos << 0.99) and the layer is demoted on the single calibration sample.
        return torch.full((x.shape[0], qt.data.shape[0]), 1e3, dtype=x.dtype)

    monkeypatch.setattr(ops_mod, "int8_linear", _fake_int8)

    x = torch.randn(4, 8, dtype=torch.float32)
    y0 = lin.forward_comfy_cast_weights(x)  # first call: int8 once (for the gate)
    assert seen["n"] == 1
    assert lin._fni8_sqnr_pass is False
    y1 = lin.forward_comfy_cast_weights(x)  # demoted: no more int8
    y2 = lin.forward_comfy_cast_weights(x)
    assert seen["n"] == 1  # still 1 — no discarded GEMMs
    ref = _fp32_reference(lin, x)
    for y in (y0, y1, y2):
        torch.testing.assert_close(y, ref, rtol=0, atol=0)


def test_multistep_calibration_demotes_on_late_collapse(monkeypatch):
    """With FNI8_SQNR_CALIB_SAMPLES>1 the ops Linear calibrates across forwards and
    demotes when a LATER activation collapses — the multi-timestep failure mode a
    single-call (first-activation) gate would miss. It stays UNlocked while calibrating,
    so it keeps measuring until the collapse is seen."""
    monkeypatch.setenv("FNI8_SQNR_CALIB_SAMPLES", "3")
    lin = _make_int8_linear()
    step = {"n": 0}

    def _staged_int8(x, qt, bias=None):
        # Accurate (== fp reference) on the first two steps, then a collapse on the third.
        w = dequantize_weight(lin.weight.int8_data(), lin.weight.q_scale, x.dtype)
        ref = torch.nn.functional.linear(x, w, None)
        out = ref if step["n"] < 2 else torch.full_like(ref, 1e3)
        step["n"] += 1
        return out

    monkeypatch.setattr(ops_mod, "int8_linear", _staged_int8)
    x = torch.randn(4, 8, dtype=torch.float32)

    lin.forward_comfy_cast_weights(x)
    assert lin._fni8_sqnr_pass is None  # step 0 accurate — not locked yet
    lin.forward_comfy_cast_weights(x)
    assert lin._fni8_sqnr_pass is None  # step 1 accurate — still calibrating
    lin.forward_comfy_cast_weights(x)
    assert lin._fni8_sqnr_pass is False  # step 2 collapse -> worst-case demote to fp


def test_calibration_fp_reference_uses_bounded_rows(monkeypatch):
    """Large DiT activations must not allocate a full-size fp reference output."""
    monkeypatch.setenv("FNI8_SQNR_CALIB_MAX_ROWS", "3")
    lin = _make_int8_linear(N=16, K=8)
    original_linear = torch.nn.functional.linear
    fp_reference_rows = []

    def _accurate_int8(x, qt, bias=None):
        w = dequantize_weight(lin.weight.int8_data(), lin.weight.q_scale, x.dtype)
        return original_linear(x, w, bias)

    def _record_fp_reference(x, weight, bias=None):
        fp_reference_rows.append(x.shape[0])
        return original_linear(x, weight, bias)

    monkeypatch.setattr(ops_mod, "int8_linear", _accurate_int8)
    monkeypatch.setattr(torch.nn.functional, "linear", _record_fp_reference)

    x = torch.randn(17, 8, dtype=torch.float32)
    y = lin.forward_comfy_cast_weights(x)

    assert y.shape == (17, 16)
    assert fp_reference_rows == [3]
    assert lin._fni8_sqnr_pass is True
