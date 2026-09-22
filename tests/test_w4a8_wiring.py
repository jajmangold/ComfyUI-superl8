# SPDX-License-Identifier: Apache-2.0
"""W4A8 wiring regression: the DiT W4A8 linear path must reach fni8's native
``per_group_i4`` (``codebook='int4'``) dp4a kernel — NOT the ~2x-slower int4->int8
W8A8 fallback.

Root cause guarded (blocker B2, "W4A8 kernel status unknown"): the ``per_group_i4``
QTensor was built without ``codebook="int4"``. ``QTensor.validate()`` isn't auto-run,
so construction succeeded with ``codebook=""``; then ``superl8.linear`` raised
``ValueError: per_group_i4 GEMM needs codebook 'int4'`` which a broad
``except (RuntimeError, TypeError, ValueError)`` silently swallowed, demoting every
W4A8 linear to the slow Python fallback. fni8's ``gemm_w4a8`` kernel is verified on a
real V100 (24/24 tests, ~41 dB); this test asserts we actually use it."""
import pytest
import torch

pytest.importorskip("superl8")

from comfyui_superl8.int8_linear import (
    W4_GROUP_SIZE,
    dequantize_int4_weight,
    quantize_weight_int4,
    w4a8_linear,
)

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="dp4a needs CUDA")


@cuda_only
def test_w4a8_linear_reaches_kernel_and_matches_fp():
    torch.manual_seed(0)
    N, K, gs = 512, 1024, W4_GROUP_SIZE
    weight = torch.randn(N, K) * 0.05
    packed, scale = quantize_weight_int4(weight, group_size=gs)
    # fp reference from the SAME int4-quantized weight, so this isolates the GEMM
    # dispatch (not the quantizer): dequantize the packed int4 and matmul in fp32.
    w_deq = dequantize_int4_weight(packed, scale, gs, N, K, torch.float32).cuda()
    x = torch.randn(8, K, dtype=torch.float16, device="cuda")

    # Spy on superl8.linear to record which weight scheme reaches the GEMM and whether the
    # call raised. Buggy code: [("per_group_i4", "raised"), ("per_row_i8", "ok")] —
    # the ValueError is swallowed and the fallback fires. Fixed code: exactly one
    # ("per_group_i4", "ok") — the native W4A8 kernel runs and the fallback never does.
    import superl8

    calls: list[tuple] = []
    real = superl8.linear

    def spy(xx, qt, **kw):
        scheme = getattr(qt, "scheme", None)
        try:
            out = real(xx, qt, **kw)
        except Exception:
            calls.append((scheme, "raised"))
            raise
        calls.append((scheme, "ok"))
        return out

    superl8.linear = spy
    try:
        y = w4a8_linear(x, packed.cuda(), scale.cuda(), gs, (N, K))
    finally:
        superl8.linear = real

    assert calls == [("per_group_i4", "ok")], (
        f"W4A8 mis-dispatched — expected one native per_group_i4 GEMM, got {calls}"
    )

    # Numerics at the int8-path bars (NOT allclose): cos >= 0.99, rel-L1 <= 0.02.
    ref = x.float() @ w_deq.t()
    cos = torch.nn.functional.cosine_similarity(
        y.float().flatten(), ref.flatten(), dim=0
    ).item()
    rel_l1 = (y.float() - ref).abs().sum().item() / ref.abs().sum().item()
    assert cos >= 0.99, f"cos {cos:.4f} below 0.99"
    assert rel_l1 <= 0.02, f"rel-L1 {rel_l1:.4f} above 0.02"


@cuda_only
def test_w4a8_qtensor_validates_with_int4_codebook():
    # The exact one-line regression guard: the constructed per_group_i4 QTensor must
    # carry codebook='int4' so fni8 routes it to gemm_w4a8 instead of raising.
    from superl8 import QTensor

    weight = torch.randn(256, 512)
    packed, scale = quantize_weight_int4(weight, group_size=W4_GROUP_SIZE)
    qt = QTensor(
        packed.cuda(), scale.cuda(), scheme="per_group_i4",
        group_size=W4_GROUP_SIZE, codebook="int4",
    )
    qt.validate()  # raises if the codebook is missing/invalid
    assert qt.codebook == "int4"


def test_quantize_dit_state_dict_w4_sets_int4_codebook():
    # The loader construction site: quantized DiT linears must be per_group_i4 QTensors
    # that carry codebook='int4' (else superl8.linear rejects them at runtime). CPU-only.
    from comfyui_superl8.loader import quantize_dit_state_dict_w4

    sd = {
        "blocks.0.attn.to_q.weight": torch.randn(256, 512),
        "blocks.0.norm.weight": torch.randn(256),  # 1-D -> stays raw
    }
    out = quantize_dit_state_dict_w4(sd)

    qt = out["blocks.0.attn.to_q.weight"]
    assert qt.scheme == "per_group_i4"
    assert qt.codebook == "int4"
    qt.validate()
    assert out["blocks.0.norm.weight"].scheme == "raw"
