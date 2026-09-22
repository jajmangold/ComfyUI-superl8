# SPDX-License-Identifier: Apache-2.0
"""Native GGUF k-quant DiT Linear (comfyui_superl8.gguf_ops) — correctness gate.

Verifies the fni8 fused-dp4a path selected by ``_make_forward`` matches the stock
ComfyUI-GGUF dequant reference on REAL LTX-2.3 (``ltxv``) GGUF Q4_K/Q5_K/Q6_K
weights, and that the per-layer accuracy gate demotes a deliberately-broken layer to
the dequant path. int8/gguf paths use cosine/SQNR, not allclose (project convention).

gguf-py can dequantize but not *quantize* k-quants, so the real native super-block
bytes are sourced from an actual GGUF file (``FNI8_LTX_GGUF``, default = the LTX-2.3
distilled Q4_K_S on the fleet). Skips cleanly without CUDA + a gguf-branch fni8
(``superl8.linear_q4k``) + the ``gguf`` package + that model file.
"""
import os
import statistics
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
fni8 = pytest.importorskip("superl8")

_MODEL = os.environ.get(
    "FNI8_LTX_GGUF",
    "/models/ltx-2.3-22b-distilled-1.1-Q4_K_S.gguf",
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and hasattr(fni8, "linear_q4k") and os.path.exists(_MODEL)),
    reason="needs CUDA + gguf-branch fni8 (linear_q4k) + FNI8_LTX_GGUF model file",
)

from comfyui_superl8.gguf_ops import _make_forward  # noqa: E402


_SUPPORTED = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]


def _pick_real_tensors():
    """First 2-D native-eligible weight (in%256==0) of each supported k-quant type
    present in the real LTX GGUF (whatever mix the file uses)."""
    reader = gguf.GGUFReader(_MODEL)
    out = {}
    for t in reader.tensors:
        n = gguf.GGMLQuantizationType(int(t.tensor_type)).name
        if len(t.shape) == 2 and n in _SUPPORTED and n not in out and int(t.shape[0]) % 256 == 0:
            out[n] = t
        if len(out) == len(_SUPPORTED):
            break
    return out


_REAL = _pick_real_tensors() if os.path.exists(_MODEL) else {}


class _StubGGMLTensor(torch.Tensor):
    """Stand-in for ComfyUI-GGUF's GGMLTensor: raw bytes + type/shape carried on .to."""

    @staticmethod
    def __new__(cls, data, tensor_type, tensor_shape):
        t = torch.Tensor._make_subclass(cls, data)
        t.tensor_type = tensor_type
        t.tensor_shape = tensor_shape
        return t

    def to(self, *a, **k):
        out = super().to(*a, **k)
        out.tensor_type = self.tensor_type
        out.tensor_shape = self.tensor_shape
        return out


class _StubLayer:
    """Mimics GGMLOps.Linear enough for _make_forward, from a real GGUF tensor."""

    def __init__(self, t):
        in_f, out_f = int(t.shape[0]), int(t.shape[1])  # gguf shape is [in, out]
        qtype = gguf.GGMLQuantizationType(int(t.tensor_type))
        data = torch.from_numpy(np.ascontiguousarray(t.data).view(np.uint8).reshape(-1))
        self.weight = _StubGGMLTensor(data, qtype, torch.Size([out_f, in_f]))
        self.bias = None
        self._deq = torch.from_numpy(
            gguf.quants.dequantize(t.data, qtype).astype(np.float32).reshape(out_f, in_f)
        )

    def _base_forward(self, x):  # == ComfyUI-GGUF stock dequant path
        return torch.nn.functional.linear(x, self._deq.to(x.device).to(x.dtype))

    def get_weight(self, t, dtype):
        return t.to(dtype)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.mark.parametrize("tname", _SUPPORTED)
@pytest.mark.parametrize("M", [1, 8, 64])
def test_native_matches_dequant(tname, M):
    if tname not in _REAL:
        pytest.skip(f"no {tname} tensor in {_MODEL}")
    torch.manual_seed(0)
    layer = _StubLayer(_REAL[tname])
    in_f = layer.weight.tensor_shape[1]
    fwd = _make_forward(_StubLayer._base_forward)
    x = torch.randn(M, in_f, device="cuda", dtype=torch.float16)
    ref = layer._base_forward(x)
    out = fwd(layer, x)
    assert out.shape == ref.shape
    assert not torch.equal(out, torch.zeros_like(out)), "native output must not be all-zeros"
    assert not torch.equal(ref, torch.zeros_like(ref)), "dequant reference must not be all-zeros"
    assert layer._fni8_sqnr_pass is True, "layer should pass the accuracy gate"
    assert not getattr(layer, "_fni8_demoted", False)
    assert _cos(ref, out) >= 0.98, f"{tname} M={M} cos too low"


def test_gate_demotes_bad_layer(monkeypatch):
    """If the fni8 path is forced to garbage, the gate must demote to dequant."""
    if "Q4_K" not in _REAL:
        pytest.skip("no Q4_K tensor")
    import comfyui_superl8.gguf_ops as g

    torch.manual_seed(1)
    layer = _StubLayer(_REAL["Q4_K"])
    in_f = layer.weight.tensor_shape[1]
    out_f = layer.weight.tensor_shape[0]
    monkeypatch.setattr(g, "_native_linear", lambda *a, **k: torch.randn(
        a[2].shape[0], out_f, device="cuda", dtype=torch.float16))
    fwd = _make_forward(_StubLayer._base_forward)
    x = torch.randn(8, in_f, device="cuda", dtype=torch.float16)
    ref = layer._base_forward(x)
    out = fwd(layer, x)
    assert not torch.equal(out, torch.zeros_like(out)), "demoted output must not be all-zeros"
    assert layer._fni8_demoted is True
    assert _cos(ref, out) >= 0.999, "demoted layer must return the dequant reference"


@pytest.mark.parametrize("tname", ["Q4_K", "Q5_K"])
@pytest.mark.parametrize("M", [1, 4, 16])
def test_perf_native_vs_dequant_throughput(tname, M):
    """Throughput/latency benchmark: native GGUF k-quant DiT Linear vs stock dequant.

    Uses CUDA-synchronised wall-clock over warmup + 30 iterations per path with
    median ms/iteration. Asserts native path is not a slowdown (speedup >= 0.9x)."""
    if tname not in _REAL:
        pytest.skip(f"no {tname} tensor in {_MODEL}")
    torch.manual_seed(0)
    layer = _StubLayer(_REAL[tname])
    in_f = layer.weight.tensor_shape[1]
    out_f = layer.weight.tensor_shape[0]
    fwd = _make_forward(_StubLayer._base_forward)
    x = torch.randn(M, in_f, device="cuda", dtype=torch.float16)

    _sync = torch.cuda.synchronize

    def _time_ms(fn, iters=30, warmup=5):
        for _ in range(warmup):
            fn()
        _sync()
        ts = []
        for _ in range(iters):
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    native_ms = _time_ms(lambda: fwd(layer, x.clone()))
    dequant_ms = _time_ms(lambda: layer._base_forward(x.clone()))
    speedup = dequant_ms / native_ms if native_ms > 0 else float("inf")

    print(
        f"\n[{tname} M={M} in={in_f} out={out_f}] "
        f"native={native_ms:.3f}ms  dequant={dequant_ms:.3f}ms  "
        f"speedup={speedup:.2f}x"
    )
    assert speedup >= 0.9, (
        f"{tname} M={M}: native {native_ms:.2f}ms slower than dequant {dequant_ms:.2f}ms "
        f"(speedup {speedup:.2f}x < 0.9x floor)"
    )
