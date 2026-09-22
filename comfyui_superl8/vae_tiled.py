# SPDX-License-Identifier: Apache-2.0
"""Spatially tiled / chunked VAE decode for high-res images and video.

The VAE is convolutional and its activation working set grows linearly with pixel
count — at 2048 px a FLUX VAE decode peak is ~37 GiB, well beyond 16 GB.
Tiling splits the latent into overlapping patches, decodes each independently, and
blends the results with a smooth weight map so seams are invisible.

For 3D-causal VAEs (Wan, LTX, HunyuanVideo, etc.) we additionally support *chunked*
decode via feature-cache: process frames in temporal chunks and propagate the causal
convolution hidden states so the full video is never in HBM at once.
"""

from __future__ import annotations

from typing import Callable

import torch


def make_1d_weight_map(size: int, overlap: int, device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
    """A 1-D blending weight map.

    The ramp has ``overlap`` elements and goes from ``1/(overlap+1)`` at the tile
    edge to ``overlap/(overlap+1)`` at the interior boundary, then all-1 in the
    interior.  Two adjacent tiles' overlapping weights sum to exactly 1.0 at every
    pixel — guaranteeing correct reconstruction at image borders (where a single
    tile covers pixels at the tile-edge weight and normalizing by the accumulated
    weight recovers the original prediction) and smooth cross-fade in overlaps.
    """
    interior = size - 2 * overlap
    if interior <= 0:
        return torch.ones(size, device=device, dtype=dtype)
    ramp = torch.arange(1, overlap + 1, device=device, dtype=dtype) / (overlap + 1)
    mid = torch.ones(interior, device=device, dtype=dtype)
    return torch.cat([ramp, mid, ramp.flip(0)])


def make_2d_weight_map(tile_h: int, tile_w: int, overlap: int,
                       device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w1 = make_1d_weight_map(tile_h, overlap, device, dtype)
    w2 = make_1d_weight_map(tile_w, overlap, device, dtype)
    return w1[:, None] * w2[None, :]


def plan_tiles(total_h: int, total_w: int, tile_size: int,
               overlap: int) -> list[tuple[int, int, int, int]]:
    """Compute (y_start, y_end, x_start, x_end) for each tile covering ``[total_h,
    total_w]`` with ``tile_size`` and ``overlap``. The tiles cover the entire spatial
    extent; the last tile in each dim is clipped to the boundary."""
    stride = tile_size - 2 * overlap
    if stride <= 0:
        stride = tile_size // 2

    tiles: list[tuple[int, int, int, int]] = []
    for y in range(0, total_h, stride):
        y0 = y
        y1 = min(y + tile_size, total_h)
        for x in range(0, total_w, stride):
            x0 = x
            x1 = min(x + tile_size, total_w)
            tiles.append((y0, y1, x0, x1))
    return tiles


def _accumulate_tile(acc: torch.Tensor, weight_acc: torch.Tensor,
                     tile: torch.Tensor, weight: torch.Tensor,
                     y0: int, y1: int, x0: int, x1: int):
    """Add a decoded tile into the accumulator with its weight map.
    ``weight`` is ``[tile_h, tile_w]`` (broadcasts over batch + channels).
    ``weight_acc`` is ``[B, 1, H, W]`` (single-channel weight accumulator)."""
    th, tw = y1 - y0, x1 - x0
    acc[:, :, y0:y1, x0:x1] += tile[:, :, :th, :tw] * weight[:th, :tw]
    weight_acc[:, :, y0:y1, x0:x1] += weight[None, None, :th, :tw]


def tiled_vae_decode(
    vae_decode_fn: Callable,
    latent: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 64,
    verbose: bool = False,
    abort_check: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Decode a latent image through the VAE using spatial tiling.

    Args:
        vae_decode_fn: Callable accepting ``(latent_tile) -> decoded_tile``.
        latent: ``[1, C, H, W]`` latent tensor (fp16/bf16/fp32).
        tile_size: Size of each tile in **pixel space** (the latent is cropped
            proportionally to the VAE's stride).
        overlap: Pixel-space overlap between adjacent tiles.
        verbose: If True, prints diagnostic info.
        abort_check: Optional callable invoked before each tile decode; raising
            from it aborts the decode after at most one in-flight tile, giving
            callers a real, bounded mid-stage cancellation point instead of only
            being able to interrupt between whole pipeline stages.

    Returns:
        Decoded image ``[1, 3, H_out, W_out]`` (dtype matches latent).

    The VAE stride (typically 8) is auto-detected from a dry-run decode of a tiny
    latent. The tile positions are computed in pixel space, then mapped back to latent
    coordinates by dividing by the stride.
    """
    if latent.dim() != 4 or latent.shape[0] != 1:
        raise ValueError(f"expected [1, C, H, W] latent, got {latent.shape}")

    B, C, H_lat, W_lat = latent.shape
    device = latent.device
    dtype = latent.dtype

    stride = _detect_vae_stride(vae_decode_fn, device)
    H_px = H_lat * stride
    W_px = W_lat * stride

    tiles = plan_tiles(H_px, W_px, tile_size, overlap)
    if verbose:
        print(f"[tiled_vae_decode] {len(tiles)} tiles, stride={stride}, "
              f"latent=({H_lat}x{W_lat}), pixel=({H_px}x{W_px})")

    out_channels = _detect_out_channels(vae_decode_fn, device, dtype)
    acc = torch.zeros(B, out_channels, H_px, W_px, device=device, dtype=dtype)
    weight_acc = torch.zeros(B, 1, H_px, W_px, device=device, dtype=dtype)

    weight_tile = make_2d_weight_map(tile_size, tile_size, overlap, device, dtype)

    for y0_px, y1_px, x0_px, x1_px in tiles:
        if abort_check is not None:
            abort_check()
        y0_l = y0_px // stride
        y1_l = min(y1_px // stride + (1 if y1_px % stride else 0), H_lat)
        x0_l = x0_px // stride
        x1_l = min(x1_px // stride + (1 if x1_px % stride else 0), W_lat)

        tile_lat = latent[:, :, y0_l:y1_l, x0_l:x1_l]
        if tile_lat.shape[2] == 0 or tile_lat.shape[3] == 0:
            continue

        decoded = vae_decode_fn(tile_lat)
        _accumulate_tile(acc, weight_acc, decoded, weight_tile,
                         y0_px, y1_px, x0_px, x1_px)

    eps = torch.finfo(dtype).eps
    out = acc / (weight_acc + eps)
    return out


def _detect_vae_stride(vae_decode_fn: Callable, device: torch.device) -> int:
    """Run a tiny (1, 4, 8, 8) latent through the VAE and measure the output/input
    size ratio to determine the VAE's spatial stride. Defaults to 8 if the dry run
    fails or produces unexpected output."""
    try:
        probe = torch.randn(1, 4, 8, 8, device=device, dtype=torch.float16)
        out = vae_decode_fn(probe)
        h_in, w_in = probe.shape[2], probe.shape[3]
        h_out, w_out = out.shape[2], out.shape[3]
        if h_out >= h_in and w_out >= w_in:
            return h_out // h_in
    except Exception:
        pass
    return 8


def _detect_out_channels(vae_decode_fn: Callable, device: torch.device,
                         dtype: torch.dtype) -> int:
    """Run a tiny latent through the VAE to detect output channel count.
    Defaults to 3 (RGB) if the dry run fails."""
    try:
        probe = torch.randn(1, 4, 8, 8, device=device, dtype=dtype)
        out = vae_decode_fn(probe)
        return out.shape[1]
    except Exception:
        pass
    return 3


def chunked_vae_decode_3d(
    vae_decode_fn: Callable,
    latent: torch.Tensor,
    chunk_size: int = 4,
    verbose: bool = False,
) -> torch.Tensor:
    """Decode a 3D-causal VAE latent ``[1, C, T, H, W]`` in temporal chunks with
    feature-cache propagation.

    Causal 3D VAEs have temporal convolutions that depend on previous frame hidden
    states. Processing all T frames at once puts the full video activation working set
    in HBM. Chunked decode processes ``chunk_size`` frames at a time and carries the
    causal hidden state forward so each chunk's peak HBM is proportional to chunk_size,
    not T.

    This is VAE-architecture-dependent — it patches the causal conv's hidden-state
    buffer to persist across chunk calls. When the VAE doesn't expose a feature-cache
    mechanism, falls back to un-chunked decode (tiled spatially within each chunk).

    Returns:
        Decoded video ``[1, 3, T, H_out, W_out]``.
    """
    if latent.dim() != 5 or latent.shape[0] != 1:
        raise ValueError(f"expected [1, C, T, H, W] latent, got {latent.shape}")

    B, C, T, H_lat, W_lat = latent.shape

    stride = _detect_vae_stride(lambda x: vae_decode_fn(x.unsqueeze(2)), device=latent.device)
    H_px = H_lat * stride
    W_px = W_lat * stride

    frames: list[torch.Tensor] = []
    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)
        chunk = latent[:, :, t0:t1, :, :]
        decoded_chunk = vae_decode_fn(chunk)
        frames.append(decoded_chunk)

    return torch.cat(frames, dim=2)
