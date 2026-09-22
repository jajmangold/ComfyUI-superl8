# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tiled VAE / tiled DiT / memory tracking / quality metrics.

Tests are pure torch (no ComfyUI dependency): the tiling and blending primitives are
tested on synthetic data, and quality metrics (SQNR, cosine) are validated against
known noise levels. The ComfyUI node wiring is tested separately via
``pytest.mark.comfy_e2e`` in ``tests/e2e/``.
"""
import math

import pytest
import torch

from comfyui_superl8.memory import _to_gib, peak_hbm_monitor
from comfyui_superl8.metrics import (COS_BAR, SQNR_BAR_DB, cosine, sqnr,
                                   sqnr_gate)
from comfyui_superl8.vae_tiled import (make_1d_weight_map, make_2d_weight_map,
                                     plan_tiles, tiled_vae_decode)

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="needs CUDA")


# ---- weight map (pure torch, no CUDA needed) ----

def test_1d_weight_map_ramp():
    size, overlap = 32, 8
    w = make_1d_weight_map(size, overlap, "cpu", torch.float32)
    assert w.shape == (size,)
    # Edge weights: 1/(overlap+1) = 1/9
    assert w[0].item() == pytest.approx(1 / 9)
    assert w[-1].item() == pytest.approx(1 / 9)
    # Interior boundary: overlap/(overlap+1) at the last ramp element
    assert w[overlap - 1].item() == pytest.approx(overlap / (overlap + 1))
    assert w[overlap].item() == 1.0  # first interior pixel
    interior = w[overlap:-overlap]
    assert (interior == 1.0).all(), "interior must be 1.0"
    # Monotonic ramps
    left = w[:overlap]
    right = w[-overlap:]
    assert (left[:-1] <= left[1:]).all(), "left ramp must be non-decreasing"
    assert (right[:-1] >= right[1:]).all(), "right ramp must be non-increasing"
    # Overlapping pair of adjacent tiles must sum to 1.0 elementwise.
    summed = w[-overlap:] + w[:overlap]
    assert torch.allclose(summed, torch.ones_like(summed), atol=1e-6)


def test_1d_weight_map_no_interior():
    """When overlap is so large there is no interior (2*overlap >= size), fall back
    to uniform weight 1.0 — all pixels participate in overlap blending."""
    size, overlap = 8, 6
    w = make_1d_weight_map(size, overlap, "cpu", torch.float32)
    assert w.shape == (size,)
    assert (w == 1.0).all(), "no-interior fallback must be uniform 1.0"


def test_2d_weight_map_separable():
    tile_h, tile_w, overlap = 16, 24, 4
    w = make_2d_weight_map(tile_h, tile_w, overlap, "cpu", torch.float32)
    assert w.shape == (tile_h, tile_w)
    # Separable: w[h, w] == w_1d_h[h] * w_1d_w[w]
    wh = make_1d_weight_map(tile_h, overlap, "cpu", torch.float32)
    ww = make_1d_weight_map(tile_w, overlap, "cpu", torch.float32)
    expected = wh[:, None] * ww[None, :]
    assert torch.allclose(w, expected)


# ---- tile planner (pure) ----

def test_plan_tiles_simple():
    tiles = plan_tiles(64, 64, 32, 4)
    assert len(tiles) > 0
    for y0, y1, x0, x1 in tiles:
        assert y0 < y1 and x0 < x1
        assert y0 >= 0 and x0 >= 0
        assert y1 <= 64 and x1 <= 64


def test_plan_tiles_covers_full_area():
    H, W, tile_size, overlap = 48, 64, 32, 8
    tiles = plan_tiles(H, W, tile_size, overlap)
    covered = set()
    for y0, y1, x0, x1 in tiles:
        for y in range(y0, y1):
            for x in range(x0, x1):
                covered.add((y, x))
    assert len(covered) == H * W, "tiles must cover the full area"


def test_plan_tiles_non_square():
    H, W = 32, 80
    tiles = plan_tiles(H, W, 32, 4)
    assert len(tiles) >= 3  # need at least 3 tiles for 80px width at stride=24


# ---- tiled VAE decode (pure torch, synthetic data) ----

_FAKE_VAE_W: dict[int, torch.Tensor] = {}
_FAKE_VAE_SEED = 42

def _fake_vae_decode(latent):
    """Minimal VAE stub: upsample by 8x, output 3-channel RGB via a fixed 3xC weight.
    The weight is created once at module init so all calls within and across tests
    use the same linear mapping."""
    B, C, H, W = latent.shape
    up = torch.nn.functional.interpolate(
        latent, scale_factor=8, mode="bilinear", align_corners=False
    )
    key = C
    if key not in _FAKE_VAE_W:
        with torch.random.fork_rng():
            torch.manual_seed(_FAKE_VAE_SEED + key)
            _FAKE_VAE_W[key] = torch.randn(3, C)
    w = _FAKE_VAE_W[key].to(device=latent.device, dtype=latent.dtype)
    return torch.einsum("oc,bchw->bohw", w, up)


def test_tiled_vae_decode_matches_untiled_small():
    """On a small latent that fits in a single tile, tiled and untiled must be
    identical (single tile, no overlap)."""
    torch.manual_seed(0)
    latent = torch.randn(1, 4, 8, 8)
    untiled = _fake_vae_decode(latent)
    tiled = tiled_vae_decode(_fake_vae_decode, latent, tile_size=256, overlap=0)
    assert tiled.shape == untiled.shape
    cos = torch.nn.functional.cosine_similarity(
        tiled.flatten().float(), untiled.flatten().float(), dim=0
    )
    assert cos.item() >= 0.999, "single tile must match untiled"


def test_tiled_vae_decode_multitile_high_cosine():
    """Multi-tile decode of a larger latent should closely match untiled decode
    (cos >= 0.99 with sensible overlap)."""
    torch.manual_seed(1)
    latent = torch.randn(1, 4, 16, 16)  # 128x128 px at 8x stride
    untiled = _fake_vae_decode(latent)
    tiled = tiled_vae_decode(_fake_vae_decode, latent, tile_size=64, overlap=8)
    assert tiled.shape == untiled.shape
    cos_val = cosine(tiled, untiled)
    assert cos_val >= 0.99, f"multi-tile cos={cos_val:.6f} < 0.99"


class _Aborted(Exception):
    pass


def test_tiled_vae_decode_abort_check_stops_after_one_tile():
    """Root-review blocker #3: a real, bounded mid-stage abort boundary --
    ``abort_check`` runs before every tile, so raising from it interrupts the
    decode after at most one in-flight tile instead of only being observable
    once the WHOLE VAE stage has already finished. This exercises the real
    tiling loop (no mocks) -- multiple tiles are planned for this latent/
    tile_size/overlap combination (see test_tiled_vae_decode_multitile_high_cosine)."""
    torch.manual_seed(1)
    latent = torch.randn(1, 4, 16, 16)

    calls = []

    def abort_after_first_tile():
        calls.append(1)
        if len(calls) > 1:
            raise _Aborted("stop")

    with pytest.raises(_Aborted):
        tiled_vae_decode(
            _fake_vae_decode, latent, tile_size=64, overlap=8, abort_check=abort_after_first_tile
        )

    # Raised on the SECOND abort_check call: the first tile was allowed to
    # finish (compute already stopped by the time the exception propagates),
    # and no later tile ever started.
    assert len(calls) == 2


def test_tiled_vae_decode_without_abort_check_is_unaffected():
    """abort_check is optional; omitting it must not change the decode."""
    torch.manual_seed(1)
    latent = torch.randn(1, 4, 16, 16)
    untiled = _fake_vae_decode(latent)
    tiled = tiled_vae_decode(_fake_vae_decode, latent, tile_size=64, overlap=8, abort_check=None)
    assert tiled.shape == untiled.shape
    assert cosine(tiled, untiled) >= 0.99


@cuda_only
def test_peak_hbm_monitor():
    """Verify the memory monitor captures non-zero allocations on CUDA."""
    torch.cuda.reset_peak_memory_stats()
    info = {}
    with peak_hbm_monitor() as info:
        x = torch.randn(1024, 1024, device="cuda")
        del x
    assert "peak_gib" in info
    assert isinstance(info["peak_gib"], float)
    assert info["peak_gib"] > 0


# ---- metrics ----

def test_cosine_identical():
    a = torch.randn(100)
    assert cosine(a, a.clone()) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal():
    a = torch.randn(100)
    b = torch.randn(100)
    c = cosine(a, b)
    assert 0.0 <= c <= 1.0


def test_sqnr_identical():
    a = torch.randn(100)
    assert sqnr(a, a.clone()) == float("inf")


def test_sqnr_nonfinite_is_neg_inf():
    y = torch.randn(64)
    bad = y.clone()
    bad[0] = float("inf")
    assert sqnr(y, bad) == float("-inf")
    assert sqnr(bad, y) == float("-inf")


def test_sqnr_known_level():
    torch.manual_seed(0)
    y_fp = torch.randn(4096)
    sig_pow = y_fp.pow(2).mean()
    noise = torch.randn(4096)
    noise = noise * torch.sqrt(0.01 * sig_pow / noise.pow(2).mean())
    y_lossy = y_fp + noise
    assert math.isclose(sqnr(y_lossy, y_fp), 20.0, abs_tol=0.3)


def test_sqnr_gate_passes_good_quality():
    a = torch.randn(100)
    b = a + torch.randn(100) * 0.01  # very small noise
    assert sqnr_gate(b, a, cos_bar=0.99, sqnr_bar_db=10.0)


def test_sqnr_gate_fails_bad_quality():
    a = torch.randn(100)
    b = torch.randn(100) * 10  # unrelated garbage
    assert not sqnr_gate(b, a, cos_bar=0.99, sqnr_bar_db=20.0)


def test_sqnr_gate_nonfinite_is_fail():
    a = torch.randn(100)
    b = a.clone()
    b[0] = float("nan")
    assert not sqnr_gate(b, a)


# ---- tiled DiT (pure torch, synthetic) ----

def test_tiled_dit_forward_single_tile():
    """A latent that fits in one tile must produce identical output to the untiled
    forward."""
    from comfyui_superl8.dit_tiled import tiled_dit_forward

    def fake_dit(x, t, **kw):
        return x * torch.cos(t.float()).item()

    B, C, H, W = 1, 4, 32, 32
    x = torch.randn(B, C, H, W)
    t = torch.tensor([0.5])

    untiled = fake_dit(x, t)
    tiled = tiled_dit_forward(fake_dit, x, t, tile_size=64, overlap=0)
    assert tiled.shape == untiled.shape
    assert torch.allclose(tiled, untiled, atol=1e-6)


def test_tiled_dit_forward_multitile_quality():
    """Multi-tile denoising step must be close to untiled (high cosine similarity).
    Uses a smooth nonlinear model (tanh) — sin would alias at tile boundaries."""
    from comfyui_superl8.dit_tiled import tiled_dit_forward

    def fake_dit(x, t, **kw):
        return x * 0.5 + torch.tanh(x * 0.1)

    B, C, H, W = 1, 4, 48, 48
    x = torch.randn(B, C, H, W)
    t = torch.tensor([0.5])

    untiled = fake_dit(x, t)
    tiled = tiled_dit_forward(fake_dit, x, t, tile_size=32, overlap=8)
    cos_val = cosine(tiled, untiled)
    assert cos_val >= 0.99, f"tiled DiT cos={cos_val:.6f} < 0.99"


def test_tiled_dit_forward_linear_model():
    """A purely linear model should tile exactly (no seam error)."""
    from comfyui_superl8.dit_tiled import tiled_dit_forward

    W_scale = torch.randn(4, 4)
    def linear_dit(x, t, **kw):
        B, C, H, W = x.shape
        return torch.einsum("co,bchw->bohw", W_scale, x)

    B, C, H, W = 1, 4, 32, 32
    x = torch.randn(B, C, H, W)
    t = torch.tensor([0.5])

    untiled = linear_dit(x, t)
    tiled = tiled_dit_forward(linear_dit, x, t, tile_size=16, overlap=4)
    assert torch.allclose(tiled, untiled, atol=1e-5), "linear model must tile exactly"
