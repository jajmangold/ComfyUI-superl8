# SPDX-License-Identifier: Apache-2.0
"""SAM 3D Body bring-up on sm_70 Volta: fp16 oracle + int8-backbone (dp4a) path.

Loads the LOCAL model.safetensors + mhr_model.pt via the Volta-patched vendored
`sam_3d_body` lib, runs `process_one_image` on a real person photo, and writes the
posed MHR mesh as .obj. With --int8 it wraps every backbone nn.Linear with the
SQNR-gated fni8 W8A8 dp4a shim (comfyui_superl8.sam3d_encoder). Decoder / camera head /
MHR head / MHR body model stay fp (load-bearing).

Run (fp oracle):
  CUDA_VISIBLE_DEVICES=11 PYTHONPATH=<comfy_root> \
    <pixi_py> bench/sam3dbody_superl8_mvp.py --out .pipe_out/sam3dbody_fp.obj

Run (int8 backbone) needs a working superl8._C (sm_70 build matching the torch ABI).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

COMFY_ROOT = os.environ.get("COMFY_ROOT", "")
NODEPACK = os.path.join(COMFY_ROOT, "custom_nodes/ComfyUI-SAM3DBody/nodes")
DEPS = os.environ.get("SAM3D_DEPS", "")
SAM3D_DIR = os.environ.get("SAM3D_WEIGHTS", "")
CKPT = os.path.join(SAM3D_DIR, "model.safetensors")
MHR = os.path.join(SAM3D_DIR, "assets/mhr_model.pt")
FNI8_REPO = os.environ.get("FNI8_REPO", "")

# NODEPACK + FNI8_REPO must win (vendored sam_3d_body, comfyui_superl8). COMFY_ROOT/DEPS
# go at the END so a container's own torch-matched `comfy` shadows the archive copy;
# only fall through to the archive comfy when the runtime has none of its own.
for p in (NODEPACK, FNI8_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
for p in (COMFY_ROOT, DEPS):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)


def write_obj(path, verts, faces):
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces).reshape(-1, 3)
    with open(path, "w") as f:
        f.write("# SAM 3D Body posed MHR mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def save_preview(path, verts, faces):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        v = np.asarray(verts).reshape(-1, 3)
        fig = plt.figure(figsize=(5, 7))
        ax = fig.add_subplot(111, projection="3d")
        # frontal view; SAM3D uses y-down image coords -> flip y for upright preview
        mesh = Poly3DCollection(
            v[np.asarray(faces).reshape(-1, 3)][::4],
            alpha=0.35,
            facecolor=(0.4, 0.55, 0.85),
            edgecolor="none",
        )
        ax.add_collection3d(mesh)
        ax.scatter(v[::40, 0], -v[::40, 1], v[::40, 2], s=1, c="k")
        ax.set_box_aspect((1, 1.6, 1))
        ax.view_init(elev=-80, azim=-90)
        ax.set_axis_off()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[preview] skipped: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default=os.path.join(NODEPACK, "../assets/dancing.jpg"))
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--floor-db", type=float, default=20.0)
    ap.add_argument("--metrics-out", default=None, help="write joints+vertices .npz for the gate")
    args = ap.parse_args()

    dev = "cuda"
    dtype = torch.float16  # precision 'auto' -> fp16 on Volta
    torch.cuda.reset_peak_memory_stats()

    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    print(f"[load] ckpt={CKPT}\n[load] mhr={MHR} dtype={dtype}")
    t0 = time.time()
    model, model_cfg, _ = load_sam_3d_body(
        checkpoint_path=CKPT, device=dev, mhr_path=MHR, dtype=dtype
    )
    model.to(dev)
    # MHR JIT rigs must stay fp32 (sparse CUDA ops), keep on device
    model.head_pose.mhr.float()
    model.head_pose_hand.mhr.float()
    print(f"[load] built + on {dev} in {time.time() - t0:.1f}s")

    bb = model.backbone
    n_lin = sum(1 for m in bb.modules() if isinstance(m, torch.nn.Linear))
    print(
        f"[backbone] {type(bb).__name__} embed_dim={getattr(bb, 'embed_dim', '?')} "
        f"nn.Linear count={n_lin}"
    )

    int8_handle = None
    if args.int8:
        import superl8  # noqa: F401  (fail loudly here if the sm_70 kernel is missing)

        from comfyui_superl8.sam3d_encoder import patch_backbone_linears_int8

        int8_handle = patch_backbone_linears_int8(bb, floor_db=args.floor_db)
        print(
            f"[int8] wrapped {int8_handle.stats.n_wrapped} backbone linears "
            f"(skipped K%4 {int8_handle.stats.n_skipped_k})"
        )

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    # time backbone in isolation
    bb_ms = {"t": 0.0, "n": 0}
    orig_bb_fwd = bb.forward

    def timed_fwd(*a, **k):
        torch.cuda.synchronize()
        s = time.time()
        r = orig_bb_fwd(*a, **k)
        torch.cuda.synchronize()
        bb_ms["t"] += (time.time() - s) * 1e3
        bb_ms["n"] += 1
        return r

    bb.forward = timed_fwd

    print(f"[run] process_one_image({args.image})")
    t0 = time.time()
    outputs = estimator.process_one_image(args.image, inference_type="full")
    dt = time.time() - t0
    if not outputs:
        raise SystemExit("no people detected")
    out = outputs[0]
    verts = np.asarray(out["pred_vertices"]).reshape(-1, 3)
    faces = estimator.faces
    joints = np.asarray(out["pred_keypoints_3d"]).reshape(-1, 3)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_obj(args.out, verts, faces)
    png = os.path.splitext(args.out)[0] + ".png"
    have_png = save_preview(png, verts, faces)

    peak = torch.cuda.max_memory_allocated() / (1024**3)
    bbox = np.asarray(out["bbox"]).reshape(-1)
    print("\n===== RESULT =====")
    print(f"people={len(outputs)} verts={verts.shape} faces={faces.shape} joints={joints.shape}")
    print(
        f"vert bbox min={verts.min(0).round(3)} max={verts.max(0).round(3)} "
        f"nonfinite={(~np.isfinite(verts)).sum()}"
    )
    print(f"joints3d min={joints.min(0).round(3)} max={joints.max(0).round(3)}")
    print(f"detect_bbox={bbox.round(1)}")
    print(f"wrote {args.out}" + (f" + {png}" if have_png else ""))
    print(
        f"total_infer_s={dt:.2f}  backbone_fwd_ms={bb_ms['t']:.1f} over {bb_ms['n']} calls "
        f"(avg {bb_ms['t'] / max(bb_ms['n'], 1):.1f} ms)"
    )
    print(f"peak_vram_GiB={peak:.2f}")
    if int8_handle is not None:
        print("[int8] stats:", json.dumps(int8_handle.stats.summary()))

    if args.metrics_out:
        np.savez(
            args.metrics_out,
            vertices=verts,
            joints=joints,
            faces=np.asarray(faces),
            cam_t=np.asarray(out.get("pred_cam_t")),
        )
        print(f"wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
