# SPDX-License-Identifier: Apache-2.0
"""Multi-timestep worst-case SQNR calibration for the int8 gates.

Regression guard for the "1-step looks fine, multi-step collapses" failure mode: a DiT's
activation distribution shifts across the denoise trajectory, so a gate that locks its
int8/fp verdict from the FIRST activation it sees can keep int8 on a layer/site that
collapses at a later timestep — exactly the miss documented in docs/int8-dit-validation.md
(a one-step latent cosine of 1.0 that hid a severe multi-step image collapse).

These tests need no CUDA/fni8: the linear calibrator is pure torch, and the attention gate
is driven with a monkeypatched `_int8_dp4a` whose accuracy changes across "denoise steps"."""
from __future__ import annotations

import torch

import comfyui_superl8.attention as A
from comfyui_superl8.attention import Int8AttnGate, fni8_attention
from comfyui_superl8.int8_linear import LinearSqnrCalibrator, sqnr_calib_samples


# --------------------------------------------------------------------------------------
# LinearSqnrCalibrator (the per-linear gate core used by ops.FNI8Ops.Linear)
# --------------------------------------------------------------------------------------

def _good(y_fp):
    """int8 output that matches fp (cos ~ 1.0) -> passes the 0.99 bar."""
    return y_fp + 1e-3 * torch.randn_like(y_fp)


def _collapsed(y_fp):
    """int8 output uncorrelated with fp (cos ~ 0) -> fails the bar (a 'collapse')."""
    return torch.randn_like(y_fp)


def test_linear_calib_single_sample_is_legacy_first_activation():
    torch.manual_seed(0)
    c = LinearSqnrCalibrator(calib_samples=1)
    y_fp = torch.randn(2048)
    assert c.needs_measurement()
    assert c.observe(_good(y_fp), y_fp) is True
    assert c.passed is True
    # Locked after ONE sample: no further measurement (legacy behavior preserved).
    assert not c.needs_measurement()


def test_linear_calib_multistep_catches_late_collapse():
    # The exact failure mode: two good steps, then a collapse. calib_samples=1 would have
    # locked PASS at step 0 and shipped the collapse; calib_samples=3 takes the WORST.
    torch.manual_seed(0)
    c = LinearSqnrCalibrator(calib_samples=3)
    y_fp = torch.randn(2048)

    assert c.observe(_good(y_fp), y_fp) is True   # step 0 interim pass
    assert c.passed is None                       # not locked yet
    assert c.needs_measurement()
    assert c.observe(_good(y_fp), y_fp) is True    # step 1 interim pass
    assert c.passed is None
    assert c.observe(_collapsed(y_fp), y_fp) is False  # step 2 collapse -> worst-case fail

    assert c.passed is False                       # LOCKED to fp on the worst activation
    assert not c.needs_measurement()


def test_linear_calib_worstcase_is_order_independent():
    # Decision is STABLE regardless of which timestep collapses — worst-case, not first.
    torch.manual_seed(1)
    y_fp = torch.randn(2048)
    for order in ([_collapsed, _good, _good], [_good, _collapsed, _good],
                  [_good, _good, _collapsed]):
        c = LinearSqnrCalibrator(calib_samples=3)
        verdicts = [c.observe(mk(y_fp), y_fp) for mk in order]
        assert c.passed is False, f"a collapse anywhere must demote (order {order})"
        assert verdicts.count(False) >= 1


def test_sqnr_calib_samples_env(monkeypatch):
    monkeypatch.delenv("FNI8_SQNR_CALIB_SAMPLES", raising=False)
    assert sqnr_calib_samples() == 1                       # default = legacy
    monkeypatch.setenv("FNI8_SQNR_CALIB_SAMPLES", "8")
    assert sqnr_calib_samples() == 8
    monkeypatch.setenv("FNI8_SQNR_CALIB_SAMPLES", "0")
    assert sqnr_calib_samples() == 1                       # clamped to >= 1
    monkeypatch.setenv("FNI8_SQNR_CALIB_SAMPLES", "garbage")
    assert sqnr_calib_samples() == 1                       # bad value -> default


# --------------------------------------------------------------------------------------
# Int8AttnGate multi-timestep calibration (attention analogue)
# --------------------------------------------------------------------------------------

def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _accurate_int8(ref):
    noise = torch.randn_like(ref)
    noise *= torch.sqrt(1e-3 * ref.pow(2).mean() / noise.pow(2).mean())  # ~30 dB >> floor
    return ref + noise


def _make_qkv(B=1, S=16, heads=4, D=64):
    return (torch.randn(B, S, heads * D), torch.randn(B, S, heads * D),
            torch.randn(B, S, heads * D), (B, heads, S, D))


def test_attn_gate_single_sample_misses_late_collapse(monkeypatch):
    # Baseline documenting the flaw the fix addresses: with calib_samples=1 the gate locks
    # PASS from the first (accurate) denoise step and never sees the later collapse.
    torch.manual_seed(0)
    step = {"n": 0}

    def staged_int8(q, k, v):
        ref = _sdpa(q, k, v)
        out = _accurate_int8(ref) if step["n"] < 2 else _collapsed(ref)
        step["n"] += 1
        return out

    monkeypatch.setattr(A, "_int8_dp4a", staged_int8)
    q, k, v, (B, heads, S, D) = _make_qkv()
    key = (B, heads, S, D, str(q.dtype))

    gate = Int8AttnGate(calib_samples=1)
    fni8_attention(q, k, v, heads, gate=gate)
    assert gate.cached(key) is True  # locked PASS from step 0 — the collapse is unseen


def test_attn_gate_multistep_catches_late_collapse(monkeypatch):
    # The fix: calibrating across 3 denoise steps and deciding on the WORST demotes the
    # site to fp once the step-2 collapse is observed.
    torch.manual_seed(0)
    step = {"n": 0}

    def staged_int8(q, k, v):
        ref = _sdpa(q, k, v)
        out = _accurate_int8(ref) if step["n"] < 2 else _collapsed(ref)
        step["n"] += 1
        return out

    monkeypatch.setattr(A, "_int8_dp4a", staged_int8)
    q, k, v, (B, heads, S, D) = _make_qkv()
    key = (B, heads, S, D, str(q.dtype))

    gate = Int8AttnGate(calib_samples=3)
    outs = [fni8_attention(q, k, v, heads, gate=gate) for _ in range(3)]
    assert gate.cached(key) is False  # worst-case (step 2 collapse) -> demoted to fp

    # The demoted forward returns the fp SDPA reference, not the garbage int8.
    ref = _sdpa(q.view(B, S, heads, D).transpose(1, 2),
                k.view(B, S, heads, D).transpose(1, 2),
                v.view(B, S, heads, D).transpose(1, 2)).transpose(1, 2).reshape(B, S, heads * D)
    assert torch.allclose(outs[-1], ref, atol=1e-5)


def test_attn_gate_decision_stable_across_sampled_timesteps(monkeypatch):
    # A site that is genuinely accurate at EVERY sampled step stays int8; the verdict does
    # not flip between steps once finalized.
    torch.manual_seed(0)

    def accurate_int8(q, k, v):
        return _accurate_int8(_sdpa(q, k, v))

    monkeypatch.setattr(A, "_int8_dp4a", accurate_int8)
    q, k, v, (B, heads, S, D) = _make_qkv()
    key = (B, heads, S, D, str(q.dtype))

    gate = Int8AttnGate(calib_samples=3)
    for _ in range(5):
        fni8_attention(q, k, v, heads, gate=gate)
    assert gate.cached(key) is True  # stable PASS across all sampled timesteps


def test_attn_gate_revalidation_demotes_on_unusual_shift(monkeypatch):
    # A passing site whose activation later shifts unusually is demoted by periodic
    # revalidation (opt-in via revalidate_every).
    torch.manual_seed(0)
    step = {"n": 0}

    def staged_int8(q, k, v):
        ref = _sdpa(q, k, v)
        out = _accurate_int8(ref) if step["n"] == 0 else _collapsed(ref)
        step["n"] += 1
        return out

    monkeypatch.setattr(A, "_int8_dp4a", staged_int8)
    q, k, v, (B, heads, S, D) = _make_qkv()
    key = (B, heads, S, D, str(q.dtype))

    gate = Int8AttnGate(calib_samples=1, revalidate_every=1)
    fni8_attention(q, k, v, heads, gate=gate)      # step 0 -> PASS
    assert gate.cached(key) is True
    fni8_attention(q, k, v, heads, gate=gate)      # step 1 revalidation -> shifted -> demote
    assert gate.cached(key) is False
