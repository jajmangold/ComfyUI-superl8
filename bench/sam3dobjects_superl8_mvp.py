# SPDX-License-Identifier: Apache-2.0
"""SAM 3D Objects (TRELLIS-style) bring-up on sm_70 Volta: fp MVP producing a real
Gaussian-splat (.ply) + mesh (.glb) from a single image, via the community
ComfyUI-SAM3DObjects node pack + its sam3dobjects-nodes pixi env (torch 2.9.1+cu126,
spconv 2.3.8, comfy_sparse_attn). Drives the node `.execute()` classmethods headlessly
(no ComfyUI server). See docs/sam3d-fni8-design.md §2 for the int8 path + the torch-ABI
blocker (fni8 _C.so is torch-2.10/cu129, this env is torch-2.9/cu126 — cannot co-load).

Run (host paths):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=14 SPARSE_CONV_BACKEND=spconv \\
  <pixi>/sam3dobjects-nodes/bin/python bench/sam3dobjects_superl8_mvp.py
`CUDA_DEVICE_ORDER=PCI_BUS_ID` is mandatory — else CUDA_VISIBLE_DEVICES selects a CMP card.
"""

import os, sys, time, argparse, gc

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "14")
os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")

JOB = os.environ["CLAUDE_JOB_DIR"]
SCRATCH = os.path.join(JOB, "tmp")
COMFY = os.environ.get("COMFY_ROOT", "")
DEPS = os.environ.get("SAM3D_DEPS", "")
NP = COMFY + "/custom_nodes/ComfyUI-SAM3DObjects"
WEIGHTS = os.environ.get("SAM3D_WEIGHTS", "")
PIPE_OUT = os.environ.get("FNI8_PIPE_OUT", "")

# node pack FIRST so `import nodes` = the package (not ComfyUI's nodes.py)
sys.path.insert(0, DEPS)
sys.path.insert(0, COMFY)
sys.path.insert(0, NP)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Redirect ComfyUI model/output dirs to writable scratch BEFORE importing the node pkg
import folder_paths

folder_paths.models_dir = os.path.join(SCRATCH, "models")
folder_paths.set_output_directory(os.path.join(SCRATCH, "out"))

import torch
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=os.path.join(NP, "assets/elephant.png"))
ap.add_argument("--precision", default="fp16")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--s1", type=int, default=12)
ap.add_argument("--s2", type=int, default=12)
ap.add_argument("--s1cfg", type=float, default=7.5)
ap.add_argument("--s2cfg", type=float, default=5.0)
ap.add_argument("--tag", default="fp")
ap.add_argument("--int8", action="store_true")
args = ap.parse_args()


def vram(tag):
    a = torch.cuda.memory_allocated() / 1024**2
    p = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[VRAM] {tag}: alloc {a:.0f} MB peak {p:.0f} MB", flush=True)


print("=== importing node package ===", flush=True)
import nodes as sam3d_pkg
from nodes.depth_estimate import SAM3D_DepthEstimate
from nodes.generate_slat import SAM3DGenerateSLAT
from nodes.gaussian_decode import SAM3DGaussianDecode
from nodes.mesh_decode import SAM3DMeshDecode
import comfy.model_management as mm
from comfy_sparse_attn import detect as _detect

print("conv backend:", _detect.get_conv_backend(), flush=True)
print("device:", mm.get_torch_device(), "cap:", torch.cuda.get_device_capability(), flush=True)

model = {
    "config_path": os.path.join(WEIGHTS, "pipeline.yaml"),
    "compile": False,
    "precision": args.precision,
}

# ---- Load image + mask (mask = alpha channel) ----
pil = Image.open(args.image).convert("RGBA")
arr = np.asarray(pil).astype(np.float32) / 255.0
rgb = arr[..., :3]
alpha = arr[..., 3]
image_t = torch.from_numpy(rgb)[None]  # [1,H,W,3]
mask_t = torch.from_numpy(alpha)[None]  # [1,H,W]
print("image", image_t.shape, "mask nonzero frac", float((mask_t > 0.5).float().mean()), flush=True)

# ---- Optional int8 patching hook ----
INT8_HANDLES = []
if args.int8:
    fni8_repo = os.environ.get("FNI8_REPO", "")
    fni8_deps = os.environ.get("FNI8_DEPS", "")
    if fni8_repo:
        sys.path.insert(0, fni8_repo)
    if fni8_deps:
        sys.path.insert(0, fni8_deps)
    import superl8  # noqa

    print("[int8] fni8 OK:", superl8.__file__, flush=True)
    from comfyui_superl8.sam3d_encoder import patch_backbone_linears_int8
    import nodes.utils.stages as st

    def _int8_patch_memsafe(m, what):
        """Force all Linear weights resident on cuda, int8-patch them, then free the
        original fp16 weight storage so we hold ~1.25x (int8 + one fp clone) not ~2.25x.
        Robust: on any error, leave the model fp."""
        try:
            lins = [mod for _, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]
            for lin in lins:
                if lin.weight.device.type != "cuda":
                    lin.weight.data = lin.weight.data.cuda()
                    if lin.bias is not None:
                        lin.bias.data = lin.bias.data.cuda()
            handle = patch_backbone_linears_int8(m)
            # free the originals' big tensors (shim already cloned what it needs)
            for parent, attr, orig in handle.replacements:
                orig.weight = None
                if getattr(orig, "bias", None) is not None:
                    orig.bias = None
            torch.cuda.empty_cache()
            s = handle.stats
            print(
                f"[int8] {what}: wrapped {s.n_wrapped} linears (skipped K%4!=0: {s.n_skipped_k})",
                flush=True,
            )
            INT8_HANDLES.append((what, handle))
            return handle
        except Exception as e:
            print(f"[int8] patch {what} FAILED ({e}); keeping fp", flush=True)
            return None

    _orig_lc = st._load_condition_embedder
    _orig_lg = st._load_generator

    def _wrap(fn, what):
        def inner(*a, **k):
            key, m = fn(*a, **k)
            gtype = a[1] if len(a) > 1 else k.get("generator_type") or k.get("embedder_type", "?")
            _int8_patch_memsafe(m, f"{what}:{gtype}")
            return key, m

        return inner

    st._load_condition_embedder = _wrap(_orig_lc, "embedder")
    st._load_generator = _wrap(_orig_lg, "generator")

t0 = time.time()
torch.cuda.reset_peak_memory_stats()

# ===== Stage A: Depth (MoGe) =====
print("\n=== DEPTH ===", flush=True)
td = time.time()
depth_res = SAM3D_DepthEstimate.execute(model, image_t).result
intrinsics_np, pointmap_t, pointcloud_ply, depth_mask = depth_res
print(
    f"depth done {time.time() - td:.1f}s pointmap {tuple(pointmap_t.shape)} pc_ply {pointcloud_ply}",
    flush=True,
)
vram("after depth")

# ===== Stage B+C: SLAT (stage1 sparse + stage2 slat) =====
print("\n=== GENERATE SLAT ===", flush=True)
ts = time.time()
slat_res = SAM3DGenerateSLAT.execute(
    model,
    image_t,
    mask_t,
    pointmap_t,
    seed=args.seed,
    stage1_steps=args.s1,
    stage1_cfg=args.s1cfg,
    stage1_cfg_pm=0.0,
    stage2_steps=args.s2,
    stage2_cfg=args.s2cfg,
    use_distillation=False,
).result
slat_path = slat_res[0]
slat_secs = time.time() - ts
print(f"slat done {slat_secs:.1f}s -> {slat_path}", flush=True)
vram("after slat")

# ===== Stage D: Gaussian decode =====
print("\n=== GAUSSIAN DECODE ===", flush=True)
tg = time.time()
gs_res = SAM3DGaussianDecode.execute(
    model, slat_path, up_axis="Y-up (standard)", world_coordinates=False
).result
ply_path = gs_res[0]
print(f"gaussian done {time.time() - tg:.1f}s -> {ply_path}", flush=True)
vram("after gaussian")

# ===== Stage E: Mesh decode =====
print("\n=== MESH DECODE ===", flush=True)
tm = time.time()
mesh_res = SAM3DMeshDecode.execute(
    model,
    slat_path,
    with_postprocess=False,
    up_axis="Y-up (standard)",
    world_coordinates=False,
    use_sparse_flexicubes=True,
).result
glb_path = mesh_res[0]
print(f"mesh done {time.time() - tm:.1f}s -> {glb_path}", flush=True)
vram("after mesh")

peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"\n=== TOTAL {time.time() - t0:.1f}s  PEAK VRAM {peak:.0f} MB ===", flush=True)

if args.int8:
    print("=== INT8 SUMMARY ===", flush=True)
    for what, h in INT8_HANDLES:
        print(f"[int8] {what}: {h.stats.summary()}", flush=True)

# copy deliverables
import shutil

out_ply = os.path.join(PIPE_OUT, f"sam3dobject_{args.tag}.ply")
out_glb = os.path.join(PIPE_OUT, f"sam3dobject_{args.tag}.glb")
shutil.copy(ply_path, out_ply)
shutil.copy(glb_path, out_glb)
print("SAVED", out_ply, os.path.getsize(out_ply), flush=True)
print("SAVED", out_glb, os.path.getsize(out_glb), flush=True)
print("SLAT_PATH", slat_path, flush=True)
print(
    "RESULT_JSON",
    {
        "tag": args.tag,
        "peak_vram_mb": round(peak),
        "slat_secs": round(slat_secs, 1),
        "total_secs": round(time.time() - t0, 1),
        "ply": out_ply,
        "glb": out_glb,
        "slat": slat_path,
    },
    flush=True,
)
