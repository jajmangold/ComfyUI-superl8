# SPDX-License-Identifier: Apache-2.0
"""Geometric quality metrics for the SAM 3D int8-encoder-vs-fp-oracle gate.

The int8 backbone (SAM 3D Body ViT/DINOv3, SAM 3D Objects DINOv2 + flow-DiT) must
reproduce the fp oracle's 3D output within a *geometric* tolerance — SQNR/cosine on the
raw tensors is not the right acceptance bar for a posed mesh or a 3D asset. Per
`fused_ni8/AGENTS.md`, the accuracy gate decides where int8 is allowed; here the gate is
expressed on the 3D geometry itself:

- **SAM 3D Body:** MPJPE / PA-MPJPE on the MHR joints (mm), per-vertex mesh error (mm).
- **SAM 3D Objects:** Chamfer distance / F-score between the fp and int8 point clouds
  (sampled from the mesh / Gaussian-splat means).

All functions are pure-torch, CPU-runnable, and weight-independent so they can be unit
tested without the (gated) checkpoints. Inputs are `[..., N, 3]` tensors in a shared
coordinate frame (the caller aligns / uses the same camera for fp and int8).
"""

from __future__ import annotations

import torch


def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean Per-Joint Position Error (mm if inputs are mm). Both `[..., J, 3]`."""
    a = pred.detach().float()
    b = gt.detach().float()
    return (a - b).norm(dim=-1).mean().item()


def _procrustes_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Rigid+scale (similarity) Procrustes alignment of `pred` onto `gt`.

    Both `[J, 3]`. Returns the aligned `pred`. Used for PA-MPJPE so a global
    rotation/translation/scale difference (which the int8 encoder can induce via the
    camera head) is not counted as pose error."""
    a = pred.detach().double()
    b = gt.detach().double()
    mu_a, mu_b = a.mean(0, keepdim=True), b.mean(0, keepdim=True)
    a0, b0 = a - mu_a, b - mu_b
    # optimal rotation via SVD of the cross-covariance; scale via trace ratio.
    h = a0.t() @ b0
    u, s, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.t() @ u.t()))
    dm = torch.eye(3, dtype=a.dtype)
    dm[2, 2] = d
    r = vt.t() @ dm @ u.t()
    scale = s.sum() / a0.pow(2).sum().clamp_min(1e-12)
    aligned = scale * (a0 @ r.t()) + mu_b
    return aligned.to(pred.dtype)


def pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Procrustes-Aligned MPJPE (mm). Both `[J, 3]` (single pose)."""
    return mpjpe(_procrustes_align(pred, gt), gt)


def per_vertex_error(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Per-vertex mesh error for two meshes in vertex correspondence (same topology,
    as fp and int8 share the MHR template). Both `[V, 3]`. Returns mean + max (Hausdorff-
    like) + rms in the input units (mm)."""
    a = pred.detach().float()
    b = gt.detach().float()
    d = (a - b).norm(dim=-1)
    return {
        "mean_mm": d.mean().item(),
        "max_mm": d.max().item(),
        "rms_mm": d.pow(2).mean().sqrt().item(),
    }


def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Symmetric Chamfer distance between two point sets `[N,3]` / `[M,3]` (no
    correspondence needed — for SAM 3D Objects, where fp and int8 meshes differ in
    topology, sample points from each and compare). Squared-L2, averaged both
    directions. Units = input-units²."""
    a = pred.detach().float()
    b = gt.detach().float()
    # pairwise squared distances [N, M]
    d2 = torch.cdist(a, b).pow(2)
    return (d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()).item()


def chamfer_fscore(pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> float:
    """F-score of the two point sets at a distance `threshold` (same units as the
    points). A standard 3D-reconstruction quality metric: fraction of pred points with a
    gt neighbour within `threshold` (precision) and vice-versa (recall), harmonic-mean."""
    a = pred.detach().float()
    b = gt.detach().float()
    d = torch.cdist(a, b)
    precision = (d.min(dim=1).values < threshold).float().mean()
    recall = (d.min(dim=0).values < threshold).float().mean()
    denom = (precision + recall).clamp_min(1e-12)
    return (2 * precision * recall / denom).item()
