# SPDX-License-Identifier: Apache-2.0
"""Quantitative int8-vs-fp16 quality metrics for DiT latent and decoded image.

Imports only torch (and torchvision when available for SSIM), no ComfyUI dependency,
so these metrics are usable from both the e2e test and standalone benches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().float()
    b = b.detach().flatten().float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    img1 = img1.detach().float()
    img2 = img2.detach().float()
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0.0:
        return float("inf")
    return 10.0 * torch.log10(max_val**2 / mse).item()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if img1.shape != img2.shape:
        raise ValueError(f"shape mismatch: {img1.shape} vs {img2.shape}")
    if img1.dim() not in (3, 4):
        raise ValueError(f"expected 3/4-D (C,H,W) or (N,C,H,W), got {img1.shape}")
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1 = img1.detach().float()
    img2 = img2.detach().float()
    C = img1.shape[1]

    # _gaussian_kernel returns a 1-D window; form the separable 2-D kernel via outer
    # product before broadcasting to the depthwise conv weight [C,1,k,k].
    g1d = _gaussian_kernel(kernel_size, sigma, device=img1.device)
    gauss = (g1d[:, None] * g1d[None, :]).view(1, 1, kernel_size, kernel_size).repeat(
        C, 1, 1, 1)

    mu1 = F.conv2d(img1, gauss, groups=C)
    mu2 = F.conv2d(img2, gauss, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, gauss, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, gauss, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, gauss, groups=C) - mu1_mu2

    L = max_val
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return (num / den).mean().item()


def _gaussian_kernel(
    size: int, sigma: float, device: torch.device
) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def format_report(
    arch: str,
    cos_latent: float,
    psnr_val: float | None,
    ssim_val: float | None,
) -> str:
    lines = [
        f"int8-vs-fp quality report [{arch}]",
        f"  cosine similarity (latent):  {cos_latent:.6f}",
    ]
    if psnr_val is not None:
        lines.append(f"  PSNR (decoded image):        {psnr_val:.2f} dB")
    if ssim_val is not None:
        lines.append(f"  SSIM (decoded image):        {ssim_val:.6f}")
    return "\n".join(lines)
