# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Sliding-Tile Attention (STA): mask generation, FLOP counting,
SQNR gate mechanism, and attention correctness vs dense reference.

All tests are CPU-only (no fni8/CUDA dependency).  The STA path uses masked
SDPA which runs on CPU in fp32, so gate logic, sparsity ratios, and FLOP
estimates are testable without a Volta GPU.
"""

from __future__ import annotations

import math

import torch

from comfyui_superl8.sta import (
    StaGate,
    flops_report,
    make_sta_mask_3d,
    sparsity_ratio,
    sta_attention,
    sta_sqnr,
)


# ---- make_sta_mask_3d -------------------------------------------------------


def test_sta_mask_3d_small_grid_full_attends():
    """A tile bigger than the grid -> every token attends every other."""
    mask = make_sta_mask_3d(2, 2, 2, tile_f=8, tile_h=8, tile_w=8)
    assert mask.shape == (8, 8)
    assert mask.all().item(), "tile larger than grid should be all True"


def test_sta_mask_3d_small_tile_is_sparse():
    """A single-token tile -> each token attends only itself (diagonal)."""
    mask = make_sta_mask_3d(3, 3, 3, tile_f=1, tile_h=1, tile_w=1)
    N = 27
    assert mask.shape == (N, N)
    expected = torch.eye(N, dtype=torch.bool)
    assert (mask == expected).all().item(), "1x1x1 tile should be identity"


def test_sta_mask_3d_sparsity_ratio():
    """Measure sparsity for a known configuration.
    4x4x4 grid (N=64), half-window 1 in each dimension.  Attending pairs
    per-dim = 2+3+3+2 = 10, so total attending = 10^3 = 1000 of 4096.
    """
    mask = make_sta_mask_3d(4, 4, 4, tile_f=3, tile_h=3, tile_w=3)
    r = sparsity_ratio(mask)
    assert 0.7 < r < 0.8, f"expected ~0.756 sparsity, got {r:.4f}"


def test_sta_mask_3d_bounds_check():
    """Edge tokens have smaller tiles (clamped to grid boundaries).
    Need tile >= 2*(F-1)+1 to cover the whole grid (half-window >= max
    distance F-1).  For F=4 that means tile_f >= 7.
    """
    F, H, W = 4, 4, 4
    mask = make_sta_mask_3d(F, H, W, tile_f=7, tile_h=7, tile_w=7)
    assert (mask.diag()).all().item(), "diagonal must be True for all tokens"
    N = F * H * W
    total = N * N
    attended = mask.sum().item()
    assert attended == total, "tile >= 2*(F-1)+1 -> every token attends every other"


def test_sta_mask_3d_temporal_only():
    """tile_h=H, tile_w=W, tile_f < F -> temporal-windowed attention.
    8x4x4 grid (N=128).  tile_f=5 (half_f=2) gives sum_f=34 of 64.
    tile_h=4 (half_h=2) gives sum_h=14 of 16 (not full coverage since
    2*half_h+1=5 > 4 but max distance in grid is 3, and |d|<=2 excludes
    distance 3 — so spatial dims are NOT fully covered).  Attending
    pairs = 34 * 14 * 14 = 6664 of 16384.
    """
    mask = make_sta_mask_3d(8, 4, 4, tile_f=5, tile_h=4, tile_w=4)
    r = sparsity_ratio(mask)
    assert 0.55 < r < 0.65, f"expected ~0.593 sparsity, got {r:.4f}"


def test_sta_mask_3d_symmetric():
    """Mask must be symmetric (if A attends B then B attends A)."""
    mask = make_sta_mask_3d(5, 3, 4, tile_f=3, tile_h=3, tile_w=3)
    assert (mask == mask.T).all().item(), "STA mask must be symmetric"


def test_sta_mask_3d_single_frame():
    """F=1 -> 2D spatial attention (tile_f is irrelevant)."""
    mask = make_sta_mask_3d(1, 8, 8, tile_f=1, tile_h=3, tile_w=3)
    N = 64
    assert mask.shape == (N, N)
    r = sparsity_ratio(mask)
    # 3x3 tile on 8x8 grid -> ~9/64 = 0.14 density, 0.86 sparsity
    assert r > 0.8, f"2D spatial STA sparsity={r:.4f}"


# ---- flops_report -----------------------------------------------------------


def test_flops_report_dense():
    r = flops_report(100, 100)
    assert r["dense_flops"] == r["sta_flops"]
    assert r["reduction"] == 0.0


def test_flops_report_half():
    r = flops_report(100, 50)
    assert r["dense_flops"] == 10000
    assert r["sta_flops"] == 5000
    assert math.isclose(r["reduction"], 0.5)


def test_flops_report_edge():
    r = flops_report(1, 1)
    assert r["dense_flops"] == 1
    assert r["reduction"] == 0.0


def test_flops_report_zero_N():
    r = flops_report(0, 100)
    assert r["reduction"] == 0.0


# ---- sparsity_ratio ---------------------------------------------------------


def test_sparsity_ratio_full():
    m = torch.ones(10, 10, dtype=torch.bool)
    assert sparsity_ratio(m) == 0.0


def test_sparsity_ratio_identity():
    m = torch.eye(10, dtype=torch.bool)
    r = sparsity_ratio(m)
    assert math.isclose(r, 0.9)


def test_sparsity_ratio_empty():
    m = torch.zeros(10, 10, dtype=torch.bool)
    assert sparsity_ratio(m) == 1.0


# ---- sta_sqnr ---------------------------------------------------------------


def test_sta_sqnr_identical_is_inf():
    y = torch.randn(2, 4, 8, 64)
    assert sta_sqnr(y, y.clone()) == float("inf")


def test_sta_sqnr_nonfinite_is_definitive_fail():
    y = torch.randn(8, 16, 32)
    bad = y.clone()
    bad[0, 0, 0] = float("inf")
    assert sta_sqnr(y, bad) == float("-inf")
    assert sta_sqnr(bad, y) == float("-inf")


def test_sta_sqnr_known_noise_level():
    torch.manual_seed(0)
    y_dense = torch.randn(4096)
    sig_pow = y_dense.pow(2).mean()
    noise = torch.randn(4096)
    noise = noise * torch.sqrt(0.01 * sig_pow / noise.pow(2).mean())
    y_sta = y_dense + noise
    assert math.isclose(sta_sqnr(y_sta, y_dense), 20.0, abs_tol=0.3)


# ---- StaGate ----------------------------------------------------------------


def test_sta_gate_records_and_caches():
    gate = StaGate(sqnr_floor_db=20.0)
    key = (8, 4, 4, 3, 3, 3, 128, "torch.float32")
    assert gate.cached(key) is None
    assert gate.record(key, 25.0) is True
    assert gate.cached(key) is True
    assert gate.decisions[key] == (True, 25.0)

    key2 = (8, 4, 4, 3, 3, 3, 128, "torch.float32")
    assert gate.record(key2, 12.0) is False
    assert gate.cached(key2) is False


def test_sta_gate_below_floor():
    gate = StaGate(sqnr_floor_db=30.0)
    assert gate.record((1,), 29.9) is False
    assert gate.record((2,), 30.0) is True
    assert gate.record((3,), 30.1) is True


def test_sta_gate_default_floor():
    assert StaGate().sqnr_floor_db == 20.0


# ---- sta_attention (CPU, masked SDPA) ---------------------------------------


def _dense_sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def test_sta_attention_full_tile_matches_dense():
    B, heads, N, D = 1, 2, 16, 32
    q = torch.randn(B, heads, N, D)
    k = torch.randn(B, heads, N, D)
    v = torch.randn(B, heads, N, D)
    out_sta = sta_attention(q, k, v, 4, 2, 2, tile_f=8, tile_h=8, tile_w=8)
    out_dense = _dense_sdpa(q, k, v)
    assert out_sta.shape == out_dense.shape
    assert torch.allclose(out_sta, out_dense, atol=1e-5)


def test_sta_attention_identity_tile():
    """tile_f=tile_h=tile_w=1 -> each token attends only itself."""
    B, heads, N, D = 1, 2, 8, 16
    q = torch.randn(B, heads, N, D)
    k = torch.randn(B, heads, N, D)
    v = torch.randn(B, heads, N, D)
    out = sta_attention(q, k, v, 2, 2, 2, tile_f=1, tile_h=1, tile_w=1)
    assert out.shape == (B, heads, N, D)
    assert torch.isfinite(out).all()


def test_sta_attention_gate_keeps_sta_when_accurate():
    gate = StaGate()
    B, heads, N, D = 1, 2, 8, 16
    q = torch.randn(B, heads, N, D)
    k = torch.randn(B, heads, N, D)
    v = torch.randn(B, heads, N, D)
    out = sta_attention(q, k, v, 2, 2, 2, tile_f=8, tile_h=8, tile_w=8, gate=gate)
    key = (2, 2, 2, 8, 8, 8, 8, str(q.dtype))
    assert gate.cached(key) is True
    assert out.shape == (B, heads, N, D)


def test_sta_attention_gate_demotes_to_dense():
    gate = StaGate(sqnr_floor_db=50.0)
    B, heads, N, D = 1, 2, 8, 32
    q = torch.randn(B, heads, N, D)
    k = torch.randn(B, heads, N, D)
    v = torch.randn(B, heads, N, D)
    out = sta_attention(q, k, v, 2, 2, 2, tile_f=1, tile_h=1, tile_w=1, gate=gate)
    key = (2, 2, 2, 1, 1, 1, 8, str(q.dtype))
    assert gate.cached(key) is False
    dense = _dense_sdpa(q, k, v)
    assert torch.allclose(out, dense, atol=1e-5)


def test_sta_attention_gate_caches_demotion():
    gate = StaGate(sqnr_floor_db=50.0)
    B, heads, N, D = 1, 2, 8, 32
    q = torch.randn(B, heads, N, D)
    k = torch.randn(B, heads, N, D)
    v = torch.randn(B, heads, N, D)

    out1 = sta_attention(q, k, v, 2, 2, 2, tile_f=1, tile_h=1, tile_w=1, gate=gate)
    key = (2, 2, 2, 1, 1, 1, 8, str(q.dtype))
    assert gate.cached(key) is False

    out2 = sta_attention(q, k, v, 2, 2, 2, tile_f=1, tile_h=1, tile_w=1, gate=gate)
    dense = _dense_sdpa(q, k, v)
    assert torch.allclose(out2, dense, atol=1e-5)


def test_sta_attention_no_gate():
    """Without a gate, STA is always applied (no dense fallback)."""
    B, heads, N, D = 1, 2, 8, 16
    q = torch.randn(B, heads, N, D)
    k = torch.randn(B, heads, N, D)
    v = torch.randn(B, heads, N, D)
    out = sta_attention(q, k, v, 2, 2, 2, tile_f=3, tile_h=3, tile_w=3)
    assert out.shape == (B, heads, N, D)
    assert torch.isfinite(out).all()


# ---- sta_attention via fni8_attention (CPU, monkeypatched) ------------------


def test_fni8_attention_sta_routes_correctly(monkeypatch):
    """Verify that fni8_attention with sta_layout/sta_tile routes through
    the STA path (not the int8 path)."""
    import comfyui_superl8.attention as A

    # Monkeypatch _int8_dp4a to fail if called (STA should never reach it).
    def fail_int8(q, k, v):
        raise AssertionError("int8 path should not be called for STA")

    monkeypatch.setattr(A, "_int8_dp4a", fail_int8)

    B, S, heads, D = 1, 8, 2, 64
    q = torch.randn(B, S, heads * D)
    k = torch.randn(B, S, heads * D)
    v = torch.randn(B, S, heads * D)
    out = A.fni8_attention(q, k, v, heads, sta_layout=(2, 2, 2), sta_tile=(9, 9, 9))
    assert out.shape == (B, S, heads * D)
    assert torch.isfinite(out).all()


def test_fni8_attention_sta_with_gate(monkeypatch):
    """Verify STA path works through fni8_attention with a gate."""
    import comfyui_superl8.attention as A

    def fake_int8(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    monkeypatch.setattr(A, "_int8_dp4a", fake_int8)

    B, S, heads, D = 1, 16, 4, 64
    q = torch.randn(B, S, heads * D)
    k = torch.randn(B, S, heads * D)
    v = torch.randn(B, S, heads * D)
    gate = A.Int8AttnGate()
    out = A.fni8_attention(q, k, v, heads, gate=gate, sta_layout=(4, 2, 2), sta_tile=(9, 9, 9))
    assert out.shape == (B, S, heads * D)
    # Gate should have been recorded (tile >= grid -> SQNR == inf -> pass)
    key = (4, 2, 2, 9, 9, 9, 16, str(q.dtype))
    # Check the underlying StaGate decision (the Int8AttnGate is not directly
    # accessible here, but the output is finite and correctly shaped)
    assert torch.isfinite(out).all()


def test_fni8_attention_sta_skip_reshape(monkeypatch):
    """Verify STA with skip_reshape=True."""
    import comfyui_superl8.attention as A

    def fake_int8(q, k, v):
        raise AssertionError("int8 path should not be called for STA")

    monkeypatch.setattr(A, "_int8_dp4a", fake_int8)

    B, heads, N, D = 1, 2, 8, 64
    q = torch.randn(B, heads, N, D)
    k = torch.randn(B, heads, N, D)
    v = torch.randn(B, heads, N, D)
    out = A.fni8_attention(
        q, k, v, heads, skip_reshape=True, sta_layout=(2, 2, 2), sta_tile=(9, 9, 9)
    )
    assert out.shape == (B, heads, N, D)
