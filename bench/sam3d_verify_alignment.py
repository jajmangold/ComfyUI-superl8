# SPDX-License-Identifier: Apache-2.0
"""Verify + reconcile the SAM 3D Objects Gaussian-splat (.ply) and mesh (.glb) frames.

Context: a viewer showed the splat and mesh "mirrored / in two positions". Empirically
(this script) the two files are ALREADY co-registered — both decoders apply the SAME
z-up->y-up rotation `M = [[1,0,0],[0,0,-1],[0,1,0]]` in `run_decode` (gaussian via
`Gaussian.save_ply(transform=M)`, mesh via `vertices @ M`), so their STORED coordinates
coincide (occupancy IoU ~0.77 identity vs ~0.43 x-flip; the elephant's bilateral symmetry
is why a naive chamfer barely separates them). The apparent "mirror" is a **viewer
convention mismatch**: 3D-Gaussian-Splat `.ply` viewers conventionally render **Y-down**
(INRIA/COLMAP 3DGS), while glTF `.glb` is **Y-up** — so each opened in its native viewer
looks vertically flipped relative to the other, even though the numbers agree.

This script:
  1. Loads both, asserts they are co-registered (identity occupancy-IoU must beat every
     axis-mirror) — a regression guard so a future decoder change that flips one export is
     caught. Exits non-zero on a real mirror.
  2. Writes a SINGLE combined **Y-up** `.glb` (mesh + gaussian means as a colored point
     cloud) so both provably coincide in ANY glTF viewer — the unambiguous deliverable.
  3. Writes an overlay render (3 orthographic views) proving alignment.

Secondary correctness note (not fixed here — it lives in the pack's `save_ply`): the
Gaussian export rotates the means by M but NOT the per-splat rotation quaternions / SH DC,
so an ellipsoid splat viewer shows correctly-placed but mis-oriented splats. Points (means)
are unaffected, so this reconciliation and the mesh are correct.

Run (any python with numpy/trimesh/plyfile/matplotlib — e.g. the sam3dobjects-nodes env):
  python bench/sam3d_verify_alignment.py --tag fp
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh
from plyfile import PlyData

PO = os.environ.get("FNI8_PIPE_OUT", "")
_SH_C0 = 0.28209479177387814  # SH band-0 constant: rgb = 0.5 + C0 * f_dc


def _load_gaussian_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    try:
        fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
        rgb = np.clip(0.5 + _SH_C0 * fdc, 0.0, 1.0)
    except ValueError:
        rgb = np.full_like(xyz, 0.6)
    return xyz, rgb


def _occupancy(p, res=24, flip_axis=None):
    q = p.astype(np.float64).copy()
    if flip_axis is not None:
        q[:, flip_axis] = -q[:, flip_axis]
    q = q - q.mean(0)
    span = (q.max(0) - q.min(0)) + 1e-9
    idx = np.floor((q / span + 0.5) * res).astype(int).clip(0, res - 1)
    o = np.zeros((res, res, res), dtype=bool)
    o[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return o


def _iou(a, b):
    return float((a & b).sum()) / float((a | b).sum())


def verify_coregistered(gpts, mpts):
    """Identity alignment must beat every single-axis mirror. Returns (ok, report)."""
    og = _occupancy(gpts)
    om = _occupancy(mpts)
    iou_id = _iou(og, om)
    mirror = {f"flip{ax}": _iou(og, _occupancy(mpts, flip_axis=ax)) for ax in range(3)}
    ok = all(iou_id >= v + 0.05 for v in mirror.values())  # identity clearly best
    return ok, {"identity": iou_id, **mirror}


def write_combined_glb(gpts, grgb, mesh_path, out_path):
    """One Y-up glTF scene: the mesh + the gaussian means as a colored point cloud. Both
    are already in the same (Y-up) frame, so a glTF viewer shows them coincident."""
    scene = trimesh.Scene()
    mesh = trimesh.load(mesh_path, force="mesh")
    scene.add_geometry(mesh, node_name="mesh")
    colors = (np.concatenate([grgb, np.ones((len(grgb), 1))], axis=1) * 255).astype(np.uint8)
    pc = trimesh.points.PointCloud(vertices=gpts, colors=colors)
    scene.add_geometry(pc, node_name="gaussian_means")
    scene.export(out_path)


def render_overlay(gpts, mpts, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    gs = gpts[rng.choice(len(gpts), min(6000, len(gpts)), replace=False)]
    ms = mpts[rng.choice(len(mpts), min(6000, len(mpts)), replace=False)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, b, lbl) in zip(axs, [(0, 1, "front X-Y"), (2, 1, "side Z-Y"), (0, 2, "top X-Z")]):
        ax.scatter(gs[:, a], gs[:, b], s=1, c="red", alpha=0.3, label="gaussian (ply)")
        ax.scatter(ms[:, a], ms[:, b], s=1, c="blue", alpha=0.3, label="mesh (glb)")
        ax.set_title(lbl)
        ax.set_aspect("equal")
        ax.legend(markerscale=6)
    fig.suptitle("SAM 3D Objects: gaussian-splat vs mesh — co-registered (Y-up)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fp")
    args = ap.parse_args()
    ply = os.path.join(PO, f"sam3dobject_{args.tag}.ply")
    glb = os.path.join(PO, f"sam3dobject_{args.tag}.glb")
    gpts, grgb = _load_gaussian_ply(ply)
    mpts = np.asarray(trimesh.load(glb, force="mesh").vertices, dtype=np.float64)

    ok, report = verify_coregistered(gpts, mpts)
    print("occupancy IoU:", {k: round(v, 3) for k, v in report.items()})
    if not ok:
        raise SystemExit(
            f"MIRROR REGRESSION: identity ({report['identity']:.3f}) does not clearly beat "
            f"the axis-mirrors {report} — a decoder export flipped handedness."
        )
    print("OK: gaussian .ply and mesh .glb are co-registered (identity alignment wins).")

    out_glb = os.path.join(PO, f"sam3dobject_{args.tag}_aligned.glb")
    out_png = os.path.join(PO, "sam3dobject_align_check.png")
    write_combined_glb(gpts, grgb, glb, out_glb)
    render_overlay(gpts, mpts, out_png)
    print("wrote", out_glb)
    print("wrote", out_png)


if __name__ == "__main__":
    main()
