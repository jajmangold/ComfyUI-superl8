# SPDX-License-Identifier: Apache-2.0
"""Sliding-Tile Attention (STA) for video DiTs — 3D block-sparse attention.

Implements mask generation, quality measurement, and SQNR gating for
Sliding-Tile Attention (arXiv 2502.04507). Video DiT attention is O(N^2) over
F x H x W tokens; STA restricts each token to a local 3D tile, cutting 50-80%
of attention FLOPs with no quality loss.

This is POST-1.0 / draft-PR work: quality is currently measured via masked
SDPA (the existing int8 kernel does not accept masks). A CUDA realisation would
gather the attended tile KV pairs and feed them to the existing int8 block
kernel (score/select on CUDA cores -> gather -> int8 FA).

Usage (standalone quality measurement)::

    mask = make_sta_mask_3d(F, H, W, tile_f=9, tile_h=8, tile_w=8)
    out = sta_attention(q, k, v, F, H, W, tile_f=9, tile_h=8, tile_w=8)

The quality-vs-dense benchmark at ``bench/validate_sta_quality.py`` uses this
to measure SQNR and cosine similarity per attention layer on a real video DiT.
"""

from __future__ import annotations

import torch


def make_sta_mask_3d(
    F: int,
    H: int,
    W: int,
    tile_f: int,
    tile_h: int,
    tile_w: int,
    *,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build a 3D sliding-tile attention mask.

    Each token at grid position (f, h, w) attends only to tokens within a
    local tile of size ``(tile_f, tile_h, tile_w)`` centred on it.  Tiles are
    clamped to the grid boundaries (shrink at edges).

    Args:
        F, H, W: token grid dimensions (after patch embedding).
        tile_f, tile_h, tile_w: tile size in each dimension (odd-ish;
            even values divide by 2 and the half-window rounds down).
        device: output device.

    Returns:
        ``[N, N]`` bool tensor, ``True`` where attention is allowed,
        ``N = F * H * W``.
    """
    f = torch.arange(F, device=device)
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)
    fg, hg, wg = torch.meshgrid(f, h, w, indexing="ij")
    f_flat = fg.flatten()
    h_flat = hg.flatten()
    w_flat = wg.flatten()

    half_f = tile_f // 2
    half_h = tile_h // 2
    half_w = tile_w // 2

    fd = f_flat[:, None] - f_flat[None, :]
    hd = h_flat[:, None] - h_flat[None, :]
    wd = w_flat[:, None] - w_flat[None, :]

    return (fd.abs() <= half_f) & (hd.abs() <= half_h) & (wd.abs() <= half_w)


def flops_report(N: int, tile_volume: int) -> dict:
    r"""Theoretical attention FLOP comparison (dense vs STA).

    Attention FLOPs scale as :math:`4 N^2 D` for dense and
    :math:`4 N * tile_volume * D` for STA.  The factor
    :math:`4D` cancels in the ratio, so we report relative counts.

    Returns dict with keys ``dense_flops``, ``sta_flops``, ``reduction``
    (fraction saved, 0..1), ``tile_volume``, ``N``.
    """
    dense = N * N
    sparse = N * tile_volume
    saved = 1.0 - sparse / dense if dense > 0 else 0.0
    return dict(dense_flops=dense, sta_flops=sparse, reduction=saved, tile_volume=tile_volume, N=N)


def sparsity_ratio(mask: torch.Tensor) -> float:
    """Fraction of attention entries *not* computed (masked out)."""
    N = mask.shape[0]
    return 1.0 - mask.sum().item() / (N * N)


def sta_sqnr(y_sta: torch.Tensor, y_dense: torch.Tensor) -> float:
    """SQNR (dB) of STA attention output vs dense reference.

    ``-inf`` when either tensor is non-finite (forces fallback for the gate).
    ``+inf`` when bit-identical.
    """
    a = y_sta.detach().float()
    b = y_dense.detach().float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("-inf")
    signal = b.pow(2).mean()
    noise = (a - b).pow(2).mean()
    if noise.item() == 0.0:
        return float("inf")
    return (10.0 * torch.log10(signal / noise)).item()


class StaGate:
    """SQNR fallback gate for STA vs dense attention, per call-site.

    First call at each config measures SQNR against dense attention; if
    below the floor the site falls back to dense attention for the rest
    of the run (stable activation distributions across denoise steps make
    a one-time decision reusable).
    """

    def __init__(self, sqnr_floor_db: float = 20.0):
        self.sqnr_floor_db = sqnr_floor_db
        self.decisions: dict = {}

    def cached(self, key):
        d = self.decisions.get(key)
        return None if d is None else d[0]

    def record(self, key, sqnr_db: float) -> bool:
        passed = sqnr_db >= self.sqnr_floor_db
        self.decisions[key] = (passed, sqnr_db)
        return passed


def sta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    F: int,
    H: int,
    W: int,
    tile_f: int,
    tile_h: int,
    tile_w: int,
    *,
    gate: StaGate | None = None,
) -> torch.Tensor:
    """Sliding-Tile Attention via masked SDPA (quality reference path).

    This uses PyTorch's SDPA with an ``attn_mask`` — the full QK^T is still
    computed internally, so this measures **quality** not speed.  Actual FLOP
    savings require a specialised block-sparse kernel (see module docstring).

    Args:
        q, k, v: ``[B, heads, N, D]`` tensors.
        F, H, W: token grid dimensions.
        tile_f, tile_h, tile_w: STA tile sizes.
        gate: optional ``StaGate`` for quality-gated fallback.

    Returns:
        ``[B, heads, N, D]`` attention output.
    """
    mask = make_sta_mask_3d(F, H, W, tile_f, tile_h, tile_w, device=q.device)

    if gate is not None:
        B, heads, N, D = q.shape
        key = (F, H, W, tile_f, tile_h, tile_w, N, str(q.dtype))
        cached = gate.cached(key)
        if cached is False:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        if cached is True:
            return _masked_sdpa(q, k, v, mask)
        sta_out = _masked_sdpa(q, k, v, mask)
        dense_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        sqnr = sta_sqnr(sta_out, dense_out)
        passed = gate.record(key, sqnr)
        return sta_out if passed else dense_out

    return _masked_sdpa(q, k, v, mask)


def _masked_sdpa(q, k, v, mask):
    attn_mask = mask[None, None, :, :].expand(q.shape[0], q.shape[1], -1, -1)
    attn_mask = torch.where(attn_mask, 0.0, float("-inf"))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
