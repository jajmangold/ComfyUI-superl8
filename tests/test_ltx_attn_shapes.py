# SPDX-License-Identifier: Apache-2.0
"""Fast, render-free unit tests for the two LTX-2.3 attention-wrapper bugs that
block single-card LTX video (findings from the render logs):

  BUG 1 — long-attention routing. LTX's self-attention is UNMASKED and long (17408
  tokens). The DP4A kernel supports this shape and is substantially faster than the
  half2 fallback, but an 8192-token policy cutoff bypassed it because the SQNR gate used
  an O(N^2) torch-SDPA reference. The gate must use the O(N)-memory fp16 kernel
  (`_fp16_attn` -> `superl8.attn_fp16_fwd`) so long self-attention can use DP4A safely.
  ANY exception from the int8 path must still fall to `_fp16_attn`, never torch SDPA.

  BUG 2 — reshape token-count mismatch. `RuntimeError: shape '[1,17408,32,128]' is
  invalid for input of size 4194304` (= 1024*32*128). LTX cross-attention has video
  queries (17408 tokens) attending a 1024-token Gemma context, so K/V have a DIFFERENT
  sequence length than Q. The wrapper reshaped K/V with Q's sequence length. N/heads/D
  must be derived per-tensor from the actual K/V tensor.

These tests monkeypatch the CUDA kernels (`_int8_dp4a`, `_fp16_attn`) so they run on
CPU in seconds with no GPU/fni8 — the wrapper's routing + reshape logic is pure Python.
Correctness (cos ~ 1.0 vs fp32 SDPA) is checked on SMALL proxy shapes that fit; the real
17408/1024 LTX shapes are exercised only for "no exception + right output shape" using a
cheap kernel stub (running true SDPA at 17408 tokens would itself OOM — the whole point).
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

import comfyui_superl8.attention as A
from comfyui_superl8.attention import fni8_attention


def _sdpa_ref(q, k, v, heads):
    """fp32 SDPA reference from ComfyUI-layout [B, S, heads*D] q/k/v (K/V may have a
    different S than Q), returned as [B, S_q, heads*D]."""
    B, Sq, inner = q.shape
    D = inner // heads
    Skv = k.shape[1]
    qh = q.float().view(B, Sq, heads, D).transpose(1, 2)
    kh = k.float().view(B, Skv, k.shape[-1] // D, D).transpose(1, 2)
    vh = v.float().view(B, Skv, v.shape[-1] // D, D).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
    return out.transpose(1, 2).reshape(B, Sq, inner)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


# A CPU stand-in for the fp16 kernel: superl8.attn_fp16_fwd returns [B, heads, M, D].
def _fp16_spy(record):
    def _impl(q, k, v, mask=None):
        record.append((tuple(q.shape), tuple(k.shape)))
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=mask
        ).to(q.dtype)

    return _impl


# ---------------------------------------------------------------- BUG 1: routing


def test_int8_dp4a_requests_raw_output_for_external_sqnr_gate(monkeypatch):
    """The wrapper owns an output-SQNR gate, so the core must return real DP4A output."""
    seen = {}

    def fake_fwd(q, k, v, **kwargs):
        seen.update(kwargs)
        return q

    monkeypatch.setitem(sys.modules, "superl8", types.SimpleNamespace(attn_int8_fwd=fake_fwd))
    q = torch.randn(1, 2, 8, 64)
    A._int8_dp4a(q, q, q)
    assert seen["internal_accuracy_gate"] is False

def test_int8_failure_falls_to_fp16_not_sdpa(monkeypatch):
    """When the int8 kernel raises (as it does on the 17408-token LTX self-attn), the
    wrapper must catch it and route to the O(N) fp16 kernel — not let it escape."""
    fp16_calls: list = []

    def _int8_boom(q, k, v):
        raise RuntimeError("attn_int8_fwd unsupported shape (simulated 17408-token fail)")

    monkeypatch.setattr(A, "_int8_dp4a", _int8_boom)
    monkeypatch.setattr(A, "_fp16_attn", _fp16_spy(fp16_calls))

    B, S, heads, D = 1, 512, 8, 128  # unmasked self-attn, supported D
    q = torch.randn(B, S, heads * D)
    k = torch.randn(B, S, heads * D)
    v = torch.randn(B, S, heads * D)

    out = fni8_attention(q, k, v, heads)  # must NOT raise
    assert out.shape == (B, S, heads * D)
    assert len(fp16_calls) == 1, "int8 failure did not fall through to the fp16 kernel"
    assert _cos(out, _sdpa_ref(q, k, v, heads)) > 0.999


def test_long_seq_self_attn_uses_int8_with_fp16_gate(monkeypatch):
    """Long self-attention uses dp4a and an O(N)-memory fp16 gate reference."""
    int8_calls: list = []
    fp16_calls: list = []

    monkeypatch.setattr(A, "_fp16_attn", _fp16_spy(fp16_calls))

    def _accurate_int8(q, k, v):
        int8_calls.append(1)
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        return (ref + 1e-3 * torch.randn_like(ref)).to(q.dtype)

    monkeypatch.setattr(A, "_int8_dp4a", _accurate_int8)

    B, S, heads, D = 1, 512, 8, 128
    q = torch.randn(B, S, heads * D)
    gate = A.Int8AttnGate()
    out = fni8_attention(q, q.clone(), q.clone(), heads, gate=gate)
    assert out.shape == (B, S, heads * D)
    assert int8_calls == [1]
    assert len(fp16_calls) == 1
    assert gate.cached((B, heads, S, D, str(q.dtype))) is True


def test_short_self_attn_still_uses_int8(monkeypatch):
    """Guard against regressing the image-DiT path: a SHORT unmasked self-attention with
    a supported head dim must still take the int8 kernel."""
    int8_calls: list = []
    monkeypatch.setattr(
        A, "_int8_dp4a",
        lambda q, k, v: (int8_calls.append(1),
                         torch.nn.functional.scaled_dot_product_attention(q, k, v))[1],
    )
    B, S, heads, D = 1, 64, 4, 64
    q = torch.randn(B, S, heads * D)
    out = fni8_attention(q, q.clone(), q.clone(), heads)  # gate=None -> raw int8
    assert out.shape == (B, S, heads * D)
    assert int8_calls == [1], "short self-attn should still use the int8 kernel"


# ---------------------------------------------------------------- BUG 2: reshape

def test_cross_attention_reshape_small(monkeypatch):
    """Cross-attention where K/V sequence length differs from Q must reshape K/V by
    THEIR OWN length, produce the right shape, and match fp32 SDPA. On the old code this
    raised `shape '[1,Sq,heads,D]' is invalid ...` because K/V used Q's length."""
    fp16_calls: list = []
    monkeypatch.setattr(A, "_fp16_attn", _fp16_spy(fp16_calls))

    B, heads, D = 1, 8, 128
    Sq, Skv = 512, 256
    q = torch.randn(B, Sq, heads * D)
    k = torch.randn(B, Skv, heads * D)
    v = torch.randn(B, Skv, heads * D)

    out = fni8_attention(q, k, v, heads)  # must NOT raise
    assert out.shape == (B, Sq, heads * D)
    assert len(fp16_calls) == 1, "cross-attention must route to the fp16 kernel"
    # The fp16 kernel got Q with Sq tokens and K with Skv tokens (per-tensor reshape).
    qshape, kshape = fp16_calls[0]
    assert qshape == (B, heads, Sq, D)
    assert kshape == (B, heads, Skv, D)
    assert _cos(out, _sdpa_ref(q, k, v, heads)) > 0.999


def test_cross_attention_real_ltx_shape_no_oom(monkeypatch):
    """The exact LTX shape from the render log: 17408 video queries attending a
    1024-token context. Only checks 'no exception + correct output shape' using a cheap
    kernel stub (true attention at this size would OOM — which is the bug we fixed)."""
    seen: list = []

    def _cheap(q, k, v, mask=None):
        seen.append((tuple(q.shape), tuple(k.shape)))
        B, H, M, D = q.shape
        return torch.zeros(B, H, M, D, dtype=q.dtype)

    monkeypatch.setattr(A, "_fp16_attn", _cheap)

    B, heads, D = 1, 32, 128
    Sq, Skv = 17408, 1024
    q = torch.zeros(B, Sq, heads * D, dtype=torch.bfloat16)
    k = torch.zeros(B, Skv, heads * D, dtype=torch.bfloat16)
    v = torch.zeros(B, Skv, heads * D, dtype=torch.bfloat16)

    out = fni8_attention(q, k, v, heads)  # old code: RuntimeError in k.view(...)
    assert out.shape == (B, Sq, heads * D)
    assert seen == [((B, heads, Sq, D), (B, heads, Skv, D))]


def test_self_attention_real_ltx_shape_no_oom(monkeypatch):
    """The exact LTX self-attn shape must take DP4A without an O(N^2) allocation."""
    int8_seen: list = []
    fp16_seen: list = []

    def _cheap_int8(q, k, v):
        int8_seen.append((tuple(q.shape), tuple(k.shape)))
        B, H, M, D = q.shape
        return torch.zeros(B, H, M, D, dtype=q.dtype)

    monkeypatch.setattr(A, "_int8_dp4a", _cheap_int8)
    monkeypatch.setattr(A, "_fp16_attn", lambda *args: fp16_seen.append(1))

    B, heads, D = 1, 32, 128
    S = 17408
    q = torch.zeros(B, S, heads * D, dtype=torch.bfloat16)
    out = fni8_attention(q, q.clone(), q.clone(), heads)
    assert out.shape == (B, S, heads * D)
    assert int8_seen == [((B, heads, S, D), (B, heads, S, D))]
    assert fp16_seen == []


# ---------------------------------------------------------------- skip_reshape layout

@pytest.mark.parametrize("skip_reshape", [False, True])
def test_self_attn_both_layouts(monkeypatch, skip_reshape):
    """LTX drives attention in both layouts (skip_reshape True = already [B,H,S,D]).
    Both must produce [B, S, heads*D] and match fp32 SDPA."""
    fp16_calls: list = []
    monkeypatch.setattr(A, "_fp16_attn", _fp16_spy(fp16_calls))
    monkeypatch.setattr(
        A,
        "_int8_dp4a",
        lambda q, k, v: (_ for _ in ()).throw(RuntimeError("force fp16 rescue")),
    )

    B, heads, D, S = 1, 4, 128, 256
    if skip_reshape:
        q = torch.randn(B, heads, S, D)
        k = torch.randn(B, heads, S, D)
        v = torch.randn(B, heads, S, D)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).transpose(1, 2).reshape(B, S, heads * D)
    else:
        q = torch.randn(B, S, heads * D)
        k = torch.randn(B, S, heads * D)
        v = torch.randn(B, S, heads * D)
        ref = _sdpa_ref(q, k, v, heads)

    out = fni8_attention(q, k, v, heads, skip_reshape=skip_reshape)
    assert out.shape == (B, S, heads * D)
    assert _cos(out, ref) > 0.999


def test_cross_attn_skip_reshape_layout(monkeypatch):
    """skip_reshape cross-attention ([B,H,M,D] q vs [B,H,N,D] k/v with N != M) must also
    produce [B, M, heads*D] with no reshape error."""
    fp16_calls: list = []
    monkeypatch.setattr(A, "_fp16_attn", _fp16_spy(fp16_calls))

    B, heads, D = 1, 8, 128
    M, N = 512, 256
    q = torch.randn(B, heads, M, D)
    k = torch.randn(B, heads, N, D)
    v = torch.randn(B, heads, N, D)
    out = fni8_attention(q, k, v, heads, skip_reshape=True)
    assert out.shape == (B, M, heads * D)
    assert len(fp16_calls) == 1
