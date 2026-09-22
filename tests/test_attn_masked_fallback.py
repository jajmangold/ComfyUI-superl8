# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the masked / memory-efficient fp16 attention routing (issue #97).

The guided/reference video-DiT path (LTX's additive `self_attention_mask`) and the
outlier-gate demotion used to fall to torch SDPA, which has no flash backend on Volta
and OOMs on ~10k-17k-token attention. They now route to fni8's O(N)-memory fp16 kernel
(`_fp16_attn` -> `superl8.attn_fp16_fwd`). These tests exercise the ROUTING + mask-prep on
CPU/fp32 (where `_fp16_attn` correctly degrades to SDPA without touching the CUDA
extension), so they need no fni8/GPU."""
from __future__ import annotations

import pytest
import torch

import comfyui_superl8.attention as A
from comfyui_superl8.attention import _fp16_attn, _prep_additive_mask, fni8_attention


# ---- _prep_additive_mask ----

def test_prep_mask_2d_to_broadcast():
    m = torch.randn(8, 8)
    out = _prep_additive_mask(m, B=2, heads=4, M=8, N=8, dtype=torch.float32)
    assert out.shape == (1, 1, 8, 8)


def test_prep_mask_3d_batch_and_bh():
    mb = torch.randn(2, 8, 8)  # [B,M,N] -> [B,1,M,N]
    assert _prep_additive_mask(mb, 2, 4, 8, 8, torch.float32).shape == (2, 1, 8, 8)
    mbh = torch.randn(8, 8, 8)  # [B*heads,M,N] -> [B,heads,M,N]
    assert _prep_additive_mask(mbh, 2, 4, 8, 8, torch.float32).shape == (2, 4, 8, 8)


def test_prep_mask_dtype_mismatch_returns_none():
    # A dtype cast would materialize an expanded mask -> defeat the memory win -> None
    m = torch.randn(1, 1, 8, 8, dtype=torch.float32)
    assert _prep_additive_mask(m, 2, 4, 8, 8, torch.float16) is None


def test_prep_mask_bool_to_additive():
    m = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    m[..., 0] = False
    out = _prep_additive_mask(m, 1, 1, 8, 8, torch.float32)
    assert out.dtype == torch.float32
    assert torch.isneginf(out[..., 0]).all() and (out[..., 1:] == 0).all()


def test_prep_mask_wrong_trailing_returns_none():
    m = torch.randn(1, 1, 8, 9)  # N != M given M=N=8
    assert _prep_additive_mask(m, 1, 1, 8, 8, torch.float32) is None


def test_prep_mask_no_head_expansion():
    # broadcast head axis stays size-1 (a view) — never expanded to `heads`
    m = torch.randn(2, 1, 8, 8)
    out = _prep_additive_mask(m, 2, 4, 8, 8, torch.float32)
    assert out.shape[1] == 1  # not 4


# ---- _fp16_attn CPU fallback (no fni8/CUDA) ----

def _sdpa_ref(q, k, v, mask=None):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def test_fp16_attn_cpu_matches_sdpa_masked():
    # On CPU/fp32 `_fp16_attn` degrades to SDPA (no fni8 import) — but with the mask.
    B, heads, S, D = 1, 2, 16, 64
    q = torch.randn(B, heads, S, D)
    k = torch.randn(B, heads, S, D)
    v = torch.randn(B, heads, S, D)
    mask = torch.randn(B, 1, S, S)
    out = _fp16_attn(q, k, v, mask)
    assert torch.allclose(out, _sdpa_ref(q, k, v, mask), atol=1e-5)


def test_fni8_attention_masked_routes_through_fp16(monkeypatch):
    # A masked self-attention call goes through _fp16_attn (NOT _sdpa directly).
    seen = {"n": 0, "mask_was": None}

    def spy(q, k, v, mask=None):
        seen["n"] += 1
        seen["mask_was"] = mask
        return _sdpa_ref(q, k, v, mask)

    monkeypatch.setattr(A, "_fp16_attn", spy)
    B, S, heads, D = 1, 16, 2, 64
    q = torch.randn(B, S, heads * D)
    mask = torch.randn(B, 1, S, S)
    out = fni8_attention(q, q.clone(), q.clone(), heads, mask=mask)
    assert out.shape == (B, S, heads * D)
    assert seen["n"] == 1 and seen["mask_was"] is mask  # mask threaded through


def test_gate_demotion_routes_through_fp16(monkeypatch):
    # A demoted (cached-False) call-site uses _fp16_attn, not O(N^2) SDPA.
    from comfyui_superl8.attention import Int8AttnGate

    fp16_calls = {"n": 0}
    monkeypatch.setattr(A, "_fp16_attn", lambda q, k, v, mask=None: _sdpa_ref(q, k, v, mask))

    def counting(q, k, v, mask=None):
        fp16_calls["n"] += 1
        return _sdpa_ref(q, k, v, mask)

    monkeypatch.setattr(A, "_fp16_attn", counting)
    monkeypatch.setattr(A, "_int8_dp4a", lambda q, k, v: torch.randn_like(_sdpa_ref(q, k, v)) * 10.0)
    B, S, heads, D = 1, 16, 2, 64
    q = torch.randn(B, S, heads * D)
    gate = Int8AttnGate()
    fni8_attention(q, q.clone(), q.clone(), heads, gate=gate)  # 1st: tiled fp reference
    key = (B, heads, S, D, str(q.dtype))
    assert gate.cached(key) is False
    fni8_attention(q, q.clone(), q.clone(), heads, gate=gate)  # 2nd: cached-False path
    # One reference for calibration, then one direct call for the cached demotion.
    assert fp16_calls["n"] == 2


# ---- override: masked routing + no silent except ----

def test_override_routes_masked_call(monkeypatch):
    from comfyui_superl8.attention import make_fni8_attn_override

    captured = {}

    def fake_attn(q, k, v, heads, mask=None, **kw):
        captured["mask"] = mask
        return torch.zeros(q.shape[0], q.shape[1], heads * (q.shape[-1] // heads))

    monkeypatch.setattr(A, "fni8_attention", fake_attn)
    override = make_fni8_attn_override(gate=None)
    B, S, heads, D = 1, 16, 2, 64
    q = torch.randn(B, S, heads * D)
    mask = torch.randn(B, 1, S, S)

    def func(*a, **k):  # ComfyUI's own attention — must NOT be called for supported D
        raise AssertionError("fell back to torch SDPA on a supported masked call")

    override(func, q, q.clone(), q.clone(), heads, mask=mask)
    assert captured["mask"] is mask


def test_override_no_silent_except(monkeypatch):
    # The silent `except: pass` was removed (#97): a real kernel error must propagate,
    # not be hidden behind an fp fallback.
    from comfyui_superl8.attention import make_fni8_attn_override

    def boom(*a, **k):
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(A, "fni8_attention", boom)
    override = make_fni8_attn_override(gate=None)
    q = torch.randn(1, 16, 2 * 64)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        override(lambda *a, **k: None, q, q.clone(), q.clone(), 2)


def test_override_skip_output_reshape_falls_back(monkeypatch):
    # skip_output_reshape callers keep torch's own path (fni8_attention can't produce
    # that output layout) — a benign, explicit fallback (not the removed silent one).
    from comfyui_superl8.attention import make_fni8_attn_override

    monkeypatch.setattr(A, "fni8_attention", lambda *a, **k: pytest.fail("should not route"))
    override = make_fni8_attn_override(gate=None)
    q = torch.randn(1, 16, 2 * 64)
    called = {"func": False}

    def func(*a, **k):
        called["func"] = True
        return q

    override(func, q, q.clone(), q.clone(), 2, skip_output_reshape=True)
    assert called["func"]
