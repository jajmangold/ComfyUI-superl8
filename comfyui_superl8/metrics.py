# SPDX-License-Identifier: Apache-2.0
"""SQNR / cosine similarity quality metrics for tiled-vs-untiled output validation.

Every lossy path (int8/int4 quantized ops) must pass a quality gate against its fp
reference — NOT ``torch.allclose`` (which is a bit-exact check inappropriate for
lossy). Instead we use:

- **Cosine similarity** (``cos >= 0.99``) — the metric already established in the
  per-layer linear gate (``sqnr_gate`` in ``int8_linear.py``) and the attention gate.
- **SQNR** (``signal_power / noise_power`` in dB) — the attention gate floor of 20 dB
  (cos ≈ 0.995).

Also provides the tiled-vs-untiled comparison operators used by the test suite to
ensure tiling degrades output below a tolerance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Minimum cosine similarity for a lossy path to be acceptable vs the fp reference.
# Mirrors the per-layer linear gate bar in int8_linear.sqnr_gate.
COS_BAR = 0.99

# Minimum SQNR (dB) for a lossy path. 20 dB corresponds to rel-RMS error ~0.1,
# cosine-sim ~0.995 — at/above the linear gate's strictness.
SQNR_BAR_DB = 20.0


def cosine(y_lossy: torch.Tensor, y_fp: torch.Tensor) -> float:
    a = y_lossy.detach().flatten().float()
    b = y_fp.detach().flatten().float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()


def sqnr(y_lossy: torch.Tensor, y_fp: torch.Tensor) -> float:
    a = y_lossy.detach().float()
    b = y_fp.detach().float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("-inf")
    signal = b.pow(2).mean()
    noise = (a - b).pow(2).mean()
    if noise.item() == 0.0:
        return float("inf")
    return (10.0 * torch.log10(signal / noise)).item()


def sqnr_gate(y_lossy: torch.Tensor, y_fp: torch.Tensor,
              cos_bar: float = COS_BAR, sqnr_bar_db: float = SQNR_BAR_DB) -> bool:
    """Returns True when the lossy path is accurate enough vs fp (both cos AND SQNR
    clear their respective bars)."""
    if not torch.isfinite(y_lossy).all() or not torch.isfinite(y_fp).all():
        return False
    c = cosine(y_lossy, y_fp)
    if c < cos_bar:
        return False
    s = sqnr(y_lossy, y_fp)
    if s < sqnr_bar_db:
        return False
    return True
