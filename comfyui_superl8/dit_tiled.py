# SPDX-License-Identifier: Apache-2.0
"""MultiDiffusion-style (2302.08113) tiled DiT forward pass.

When the DiT's latent activation working set exceeds GPU HBM, the standard remedy is
host offload - but on PCIe 1×1 (~250 MB/s) moving 16 GB takes ~80 seconds per step.
MultiDiffusion instead **caps the activation set with zero host transfer**: run the
denoising step on overlapping spatial tiles of the latent, then blend tile predictions
with a smooth weight map. Peak HBM scales with tile size, not the full latent.

The per-tile forward is a complete denoising step (same timestep, same cond/uc) on a
cropped latent region. Overlapping tile predictions are averaged, which naturally
suppresses seam artifacts.

This module provides the tiling/blending primitives and a ComfyUI model-patch factory
that replaces the diffusion model's forward call with the tiled version.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .vae_tiled import make_2d_weight_map, plan_tiles


@torch.no_grad()
def tiled_dit_forward(
    orig_forward: Callable,
    x: torch.Tensor,
    timestep: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 32,
    context: torch.Tensor | None = None,
    **extra_args: Any,
) -> torch.Tensor:
    """Run a single DiT denoising step with MultiDiffusion-style spatial tiling.

    Args:
        orig_forward: The DiT model's ``__call__`` (or a partialed version).
        x: ``[B, C, H, W]`` latent at current timestep.
        timestep: ``[B]`` timestep tensor.
        tile_size: Tile size in latent pixels (e.g. 256 → 2048 px at 8× VAE stride).
        overlap: Overlap between adjacent tiles in latent pixels.
        context: Conditioning tensor (text embeddings, etc.).
        **extra_args: Forwarded to ``orig_forward`` for each tile.

    Returns:
        ``[B, C, H, W]`` blended noise prediction (same shape as ``x``).
    """
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    tiles = plan_tiles(H, W, tile_size, overlap)
    weight_tile = make_2d_weight_map(tile_size, tile_size, overlap, device, dtype)

    acc = torch.zeros_like(x)
    weight_acc = torch.zeros_like(x)

    for y0, y1, x0, x1 in tiles:
        tile_x = x[:, :, y0:y1, x0:x1]
        th, tw = y1 - y0, x1 - x0
        if th < 1 or tw < 1:
            continue
        kwargs = dict(extra_args)
        if context is not None:
            kwargs["context"] = context
        tile_out = orig_forward(tile_x, timestep, **kwargs)
        w = weight_tile[:th, :tw]
        acc[:, :, y0:y1, x0:x1] += tile_out * w
        weight_acc[:, :, y0:y1, x0:x1] += w

    eps = torch.finfo(dtype).eps
    return acc / (weight_acc + eps)


def make_tiled_model_patch(tile_size: int = 256, overlap: int = 32):
    """Factory: returns a ComfyUI model-patch function that wraps the diffusion
    model's forward with MultiDiffusion tiling.

    Usage in a ComfyUI node::

        model = model.clone()
        model.model_options["tiled_dit"] = {"tile_size": 256, "overlap": 32}
        # The actual wrapping happens via set_model_sampler_cfg_function or
        # by patching the diffusion_model.

    Returns a tuple ``(wrapper_fn, config_dict)`` where ``wrapper_fn`` has the
    signature ComfyUI's sampler expects for a patch callback.
    """
    config = {"tile_size": tile_size, "overlap": overlap}
    return _tiled_denoise_wrapper, config


def _tiled_denoise_wrapper(
    orig_fn: Callable,
    x: torch.Tensor,
    timestep: torch.Tensor,
    cond: list[list[dict]],
    uc: list[list[dict]],
    transformer_options: dict,
    **extra: Any,
) -> torch.Tensor:
    """Wrapper for ``model.model_sampler_cfg_function`` or the diffusion model
    forward. Intercepts the denoising step to apply MultiDiffusion tiling.

    ``transformer_options`` must carry ``tiled_dit`` dict with ``tile_size`` and
    ``overlap`` (set by ``make_tiled_model_patch`` or directly by the node).

    Falls back to the original forward when no tiling config is found (no-op gate).
    """
    tile_opts = transformer_options.get("tiled_dit", {})
    if not tile_opts:
        return orig_fn(x, timestep, cond=cond, **extra)

    tile_size = tile_opts.get("tile_size", 256)
    overlap = tile_opts.get("overlap", 32)

    def forward_tile(tile_x, tile_t, **kw):
        return orig_fn(tile_x, tile_t, cond=cond, **kw)

    return tiled_dit_forward(
        forward_tile, x, timestep,
        tile_size=tile_size, overlap=overlap,
        context=None,
    )
