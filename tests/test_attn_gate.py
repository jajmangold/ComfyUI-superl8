# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the int8 self-attention SQNR fallback gate (issue #82).

These test the gate MECHANISM without needing fni8/CUDA: the int8 dp4a kernel call
(`_int8_dp4a`) is monkeypatched so the test controls the "int8" output and drives the
gate's accept/reject/cache logic on CPU. The attention analogue of the per-layer
linear gate in ops.py: first call at a shape signature measures SQNR vs fp SDPA and
either keeps int8 or demotes the call-site to fp SDPA for the rest of the run."""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

import comfyui_superl8.attention as A
from comfyui_superl8.attention import Int8AttnGate, attn_sqnr, fni8_attention
from comfyui_superl8.nodes import _attention_sites_per_signature


# ---- attn_sqnr ----

def test_attn_sqnr_identical_is_inf():
    y = torch.randn(2, 4, 8, 64)
    assert attn_sqnr(y, y.clone()) == float("inf")


def test_attn_sqnr_nonfinite_is_definitive_fail():
    # A non-finite reference/output (e.g. fp SDPA overflow) must fail the gate, never
    # pass it. -inf < any floor -> demote to fp.
    y = torch.randn(64)
    bad = y.clone()
    bad[0] = float("inf")
    assert attn_sqnr(y, bad) == float("-inf")
    assert attn_sqnr(bad, y) == float("-inf")
    gate = Int8AttnGate(sqnr_floor_db=20.0)
    assert gate.record(("k",), attn_sqnr(y, bad)) is False


def test_attn_sqnr_known_noise_level():
    # noise power = 0.01 * signal power  ->  SQNR = 20 dB exactly.
    torch.manual_seed(0)
    y_fp = torch.randn(4096)
    sig_pow = y_fp.pow(2).mean()
    noise = torch.randn(4096)
    noise = noise * torch.sqrt(0.01 * sig_pow / noise.pow(2).mean())
    y_int8 = y_fp + noise
    assert math.isclose(attn_sqnr(y_int8, y_fp), 20.0, abs_tol=0.3)


# ---- Int8AttnGate ----

def test_gate_records_and_caches_decision():
    gate = Int8AttnGate(sqnr_floor_db=20.0)
    key = (1, 4, 8, 64, "torch.float16")
    assert gate.cached(key) is None
    assert gate.record(key, 25.0) is True   # above floor -> pass
    assert gate.cached(key) is True
    assert gate.decisions[key] == (True, 25.0)

    key2 = (1, 4, 8, 128, "torch.float16")
    assert gate.record(key2, 12.0) is False  # below floor -> demote
    assert gate.cached(key2) is False


def test_gate_keeps_same_shape_blocks_as_independent_sites():
    """A shared shape must not let block 0's verdict silently authorize block 1."""
    gate = Int8AttnGate(sites_per_signature=2)
    signature = (1, 32, 9216, 128, "torch.bfloat16")

    block0 = gate.site_key(signature)
    block1 = gate.site_key(signature)
    next_step_block0 = gate.site_key(signature)

    assert block0 != block1
    assert next_step_block0 == block0
    gate.record(block0, 30.0)
    gate.record(block1, 10.0)
    assert gate.cached(block0) is True
    assert gate.cached(block1) is False


def test_attention_site_count_uses_transformer_block_count():
    dit = SimpleNamespace(transformer_blocks=[object()] * 48)
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=dit))
    assert _attention_sites_per_signature(model) == 48


# ---- fni8_attention gated path (CPU, monkeypatched kernel) ----

def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def test_gate_keeps_int8_when_accurate(monkeypatch):
    # A GENUINELY-engaged int8 kernel is accurate but NOT bit-identical: it returns fp +
    # a small quantization noise (finite SQNR well above the floor). The gate keeps it and
    # reuses the int8 kernel on every subsequent call.
    calls = {"n": 0}
    torch.manual_seed(0)

    def fake_int8(q, k, v):
        calls["n"] += 1
        ref = _sdpa(q, k, v)
        noise = torch.randn_like(ref)
        # scale noise to ~0.1% of signal power -> ~30 dB SQNR (>> 20 dB floor)
        noise *= torch.sqrt(1e-3 * ref.pow(2).mean() / noise.pow(2).mean())
        return ref + noise

    monkeypatch.setattr(A, "_int8_dp4a", fake_int8)
    B, S, heads, D = 1, 16, 4, 64
    q = torch.randn(B, S, heads * D)
    k = torch.randn(B, S, heads * D)
    v = torch.randn(B, S, heads * D)
    gate = Int8AttnGate()

    out = fni8_attention(q, k, v, heads, gate=gate)
    assert out.shape == (B, S, heads * D)
    key = (B, heads, S, D, str(q.dtype))
    assert gate.cached(key) is True
    assert gate.decisions[key][1] < float("inf")  # finite SQNR == int8 genuinely engaged
    # Second call reuses the cached PASS: int8 kernel runs again.
    fni8_attention(q, k, v, heads, gate=gate)
    assert calls["n"] == 2  # int8 kernel used on both calls


def test_gate_routes_selffallback_to_fp(monkeypatch):
    # When fni8's kernel hits its OWN SageAttention gate (outlier-heavy Q, e.g.
    # Qwen-Image ~5e4 max/median) it quantizes nothing and returns plain fp SDPA — output
    # BIT-IDENTICAL to our fp reference, SQNR = +inf. The gate must NOT cache such a site
    # as int8 (it would re-dispatch the kernel + its per-call outlier median every step
    # only to fall back again); it pins the site to fp and skips the kernel thereafter.
    calls = {"n": 0}

    def selffallback_int8(q, k, v):  # mimics attn_int8_fwd's inner fp fallback
        calls["n"] += 1
        return _sdpa(q, k, v)

    monkeypatch.setattr(A, "_int8_dp4a", selffallback_int8)
    B, S, heads, D = 1, 16, 4, 64
    q = torch.randn(B, S, heads * D)
    k = torch.randn(B, S, heads * D)
    v = torch.randn(B, S, heads * D)
    gate = Int8AttnGate()

    out = fni8_attention(q, k, v, heads, gate=gate)
    key = (B, heads, S, D, str(q.dtype))
    assert gate.decisions[key][1] == float("inf")  # kernel produced fp-identical output
    assert gate.cached(key) is False               # routed to fp, not cached as int8

    # Output equals the fp SDPA reference (bit-identical either way).
    ref = _sdpa(q.view(B, S, heads, D).transpose(1, 2),
                k.view(B, S, heads, D).transpose(1, 2),
                v.view(B, S, heads, D).transpose(1, 2)).transpose(1, 2).reshape(B, S, heads * D)
    assert torch.allclose(out, ref, atol=1e-5)

    # Second call: the (redundant) int8 kernel is NOT dispatched again.
    n_after_first = calls["n"]
    fni8_attention(q, k, v, heads, gate=gate)
    assert calls["n"] == n_after_first


def test_gate_demotes_to_fp_when_garbage(monkeypatch):
    # "int8" kernel returns garbage -> low SQNR -> gate demotes the site to fp SDPA.
    int8_calls = {"n": 0}

    def fake_int8(q, k, v):
        int8_calls["n"] += 1
        return torch.randn_like(_sdpa(q, k, v)) * 10.0  # unrelated garbage

    monkeypatch.setattr(A, "_int8_dp4a", fake_int8)
    B, S, heads, D = 1, 16, 4, 64
    q = torch.randn(B, S, heads * D)
    k = torch.randn(B, S, heads * D)
    v = torch.randn(B, S, heads * D)
    gate = Int8AttnGate()

    out = fni8_attention(q, k, v, heads, gate=gate)
    key = (B, heads, S, D, str(q.dtype))
    assert gate.cached(key) is False  # demoted

    # Output must equal the fp SDPA reference, not the garbage int8.
    ref = _sdpa(q.view(B, S, heads, D).transpose(1, 2),
                k.view(B, S, heads, D).transpose(1, 2),
                v.view(B, S, heads, D).transpose(1, 2)).transpose(1, 2).reshape(B, S, heads * D)
    assert torch.allclose(out, ref, atol=1e-5)

    # Second call: cached FAIL -> int8 kernel NOT invoked again (straight to fp).
    n_after_first = int8_calls["n"]
    fni8_attention(q, k, v, heads, gate=gate)
    assert int8_calls["n"] == n_after_first  # int8 kernel skipped on the demoted site


def test_ungated_path_unchanged(monkeypatch):
    # gate=None keeps the raw behavior: int8 kernel always used for supported dims.
    calls = {"n": 0}

    def fake_int8(q, k, v):
        calls["n"] += 1
        return _sdpa(q, k, v)

    monkeypatch.setattr(A, "_int8_dp4a", fake_int8)
    B, S, heads, D = 1, 8, 2, 64
    q = torch.randn(B, S, heads * D)
    out = fni8_attention(q, q.clone(), q.clone(), heads)  # gate=None default
    assert out.shape == (B, S, heads * D)
    assert calls["n"] == 1


# ---- persistence (issue #196) ----

def test_gate_save_load_roundtrip(tmp_path):
    gate = Int8AttnGate(sqnr_floor_db=20.0)
    key1 = (1, 4, 8, 64, "torch.float16")
    key2 = (1, 4, 16, 128, "torch.float16")
    gate.record(key1, 25.0)   # pass
    gate.record(key2, 12.0)   # fail

    path = str(tmp_path / "sqnr-attn-test.json")
    written = gate.save(path)
    assert written == path
    assert (tmp_path / "sqnr-attn-test.json").exists()

    # Load into a fresh gate
    gate2 = Int8AttnGate(sqnr_floor_db=20.0)
    assert gate2.cached(key1) is None  # not yet loaded
    assert gate2.load(path) is True
    assert gate2.cached(key1) is True
    assert gate2.cached(key2) is False
    assert gate2.decisions[key1] == (True, 25.0)
    assert gate2.decisions[key2] == (False, 12.0)


def test_gate_load_rejects_stale_gpu(tmp_path):
    gate = Int8AttnGate(sqnr_floor_db=20.0)
    gate.record((1, 4, 8, 64, "torch.float16"), 25.0)
    path = str(tmp_path / "stale.json")
    gate.save(path)

    # Tamper with the gpu field to simulate a different GPU
    import json
    with open(path) as f:
        data = json.load(f)
    data["gpu"] = "SomeOther_GPU"
    with open(path, "w") as f:
        json.dump(data, f)

    gate2 = Int8AttnGate(sqnr_floor_db=20.0)
    assert gate2.load(path) is False  # stale -> rejected
    assert gate2.cached((1, 4, 8, 64, "torch.float16")) is None


def test_gate_load_rejects_floor_mismatch(tmp_path):
    gate = Int8AttnGate(sqnr_floor_db=20.0)
    gate.record((1, 4, 8, 64, "torch.float16"), 25.0)
    path = str(tmp_path / "floor.json")
    gate.save(path)

    gate2 = Int8AttnGate(sqnr_floor_db=25.0)  # different floor
    assert gate2.load(path) is False


def test_gate_load_missing_file():
    gate = Int8AttnGate(sqnr_floor_db=20.0)
    assert gate.load("/nonexistent/path/sqnr-attn.json") is False


def test_gate_load_corrupt_file(tmp_path):
    path = str(tmp_path / "corrupt.json")
    with open(path, "w") as f:
        f.write("{bad json!!!")
    gate = Int8AttnGate(sqnr_floor_db=20.0)
    assert gate.load(path) is False


def test_maybe_save_triggers_after_calib_done(monkeypatch, tmp_path):
    """_maybe_save persists when all calibrations are finalized."""
    torch.manual_seed(0)
    calls = {"n": 0}

    def fake_int8(q, k, v):
        calls["n"] += 1
        ref = _sdpa(q, k, v)
        noise = torch.randn_like(ref)
        noise *= torch.sqrt(1e-3 * ref.pow(2).mean() / noise.pow(2).mean())
        return ref + noise

    monkeypatch.setattr(A, "_int8_dp4a", fake_int8)
    B, S, heads, D = 1, 16, 4, 64
    q = torch.randn(B, S, heads * D)
    k = torch.randn(B, S, heads * D)
    v = torch.randn(B, S, heads * D)

    gate = Int8AttnGate(calib_samples=1)
    # Monkeypatch save to use tmp_path
    path = str(tmp_path / "sqnr-attn-test.json")
    monkeypatch.setattr(gate, "save", lambda p=path: gate.__class__.save(gate, p))

    fni8_attention(q, k, v, heads, gate=gate)
    assert gate._dirty is False  # save was triggered
    assert (tmp_path / "sqnr-attn-test.json").exists()
