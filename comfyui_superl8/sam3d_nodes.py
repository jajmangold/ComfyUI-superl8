# SPDX-License-Identifier: Apache-2.0
"""ComfyUI nodes for Meta SAM 3D on the int8 dp4a (sm_70) backbone.

Two model families (see `docs/sam3d-fni8-design.md`):

* **SAM 3D Body** (`facebook/sam-3d-body-{dinov3,vith}`) — image → posed 3D human mesh.
  `SAM3DBodyLoaderFNI8` loads the official estimator and optionally flips the ViT-H /
  DINOv3 image backbone onto the fni8 int8 dp4a path (`comfyui_superl8.sam3d_encoder`); the
  promptable decoder, camera head, MHR head, and MHR parametric body model stay fp
  (pose-/reconstruction-load-bearing, per fni8/AGENTS.md). `SAM3DBodyPredict` takes an
  `IMAGE` (+ optional bbox) and returns the mesh path + an overlay `IMAGE`.

* **SAM 3D Objects** (`facebook/sam-3d-objects`) — image → 3D asset. TRELLIS-style two-stage
  flow-matching. Its reference stack does not run on sm_70 as-is (needs ≥32 GB VRAM and
  CUDA-12.1 native exts — kaolin/spconv/gsplat/xformers/flash-attn — with no Volta support;
  see the design doc §2). The loader/predict here are honest stubs that point at the design
  doc rather than silently failing.

`sam_3d_body` / `torch` are imported lazily inside the methods so the node pack loads even
where the SAM 3D runtime / checkpoints are absent — the node errors only when run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from ._comfy_compat import COMFY

_PIPE_OUT = os.environ.get("FNI8_PIPE_OUT", ".pipe_out")


@dataclass
class Sam3dBodyBundle:
    """Opaque `SAM3D_BODY_MODEL` payload passed loader -> predictor."""

    estimator: Any
    model: Any
    cfg: Any
    int8_handle: Any  # comfyui_superl8.sam3d_encoder.Sam3dBackboneInt8Handle | None
    precision: str
    int8_summary: dict | None = None


def _image_to_rgb_ndarray(image: "torch.Tensor"):
    """ComfyUI IMAGE [B,H,W,C] float 0..1 -> a uint8 RGB numpy array (first image)."""
    import numpy as np

    if image.dim() == 4:
        image = image[0]
    return (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


if COMFY:

    class SAM3DBodyLoaderFNI8:
        """Load facebook/sam-3d-body-* with an int8 dp4a ViT/DINOv3 backbone."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "hf_repo": (["facebook/sam-3d-body-vith", "facebook/sam-3d-body-dinov3"],),
                    "precision": (["fp16", "bf16", "fp32"],),
                    "int8_backbone": ("BOOLEAN", {"default": True}),
                },
                "optional": {
                    "int8_attention": ("BOOLEAN", {"default": False}),
                    "with_detector": ("BOOLEAN", {"default": True}),
                },
            }

        RETURN_TYPES = ("SAM3D_BODY_MODEL",)
        FUNCTION = "load"
        CATEGORY = "fni8/sam3d"
        TITLE = "SAM3D Body Loader FNI8 (int8 dp4a ViT)"

        def load(self, hf_repo, precision, int8_backbone, int8_attention=False, with_detector=True):
            from .gate import require_sm70

            require_sm70()
            from sam_3d_body.build_models import load_sam_3d_body_hf
            from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

            # Volta: force math SDP (no flash/mem-efficient on sm_70) for the fp path.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

            model, cfg = load_sam_3d_body_hf(hf_repo)
            detector = None
            if with_detector:
                try:
                    from tools.build_detector import build_detector

                    detector = build_detector()
                except Exception as e:  # optional — bbox can be supplied manually
                    print(f"[SAM3DBody] detector unavailable ({e}); pass bbox manually")
            estimator = SAM3DBodyEstimator(model, cfg, human_detector=detector)

            handle = None
            summary = None
            if int8_backbone:
                from . import sam3d_encoder as se

                handle = se.Sam3dBackboneInt8Handle(
                    linears=se.patch_backbone_linears_int8(model.backbone),
                )
                # Attention int8 is opt-in: SAM 3D Body's ViT dispatches attention via
                # flash_attn/math, not a module-scoped SDPA symbol, so a naive proxy is
                # not safe until the backbone is confirmed to route through SDPA. The
                # linears carry the win (design doc §1 / SAM 3.1 precedent).
                if int8_attention:
                    print(
                        "[SAM3DBody] int8_attention requested but the ViT backbone "
                        "does not expose a module-scoped SDPA; keeping attention fp "
                        "(see docs/sam3d-fni8-design.md)."
                    )
                summary = handle.summary()
            return (Sam3dBodyBundle(estimator, model, cfg, handle, precision, summary),)

    class SAM3DBodyPredict:
        """Image (+ optional bbox) -> posed 3D human MHR mesh (saved) + overlay IMAGE."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "sam3d_body_model": ("SAM3D_BODY_MODEL",),
                    "image": ("IMAGE",),
                },
                "optional": {
                    "bbox_xyxy": ("STRING", {"default": "", "multiline": False}),
                    "inference_type": (["full", "body", "hand"],),
                    "save_name": ("STRING", {"default": "sam3d_body"}),
                },
            }

        RETURN_TYPES = ("STRING", "IMAGE")
        RETURN_NAMES = ("mesh_path", "overlay")
        FUNCTION = "predict"
        CATEGORY = "fni8/sam3d"
        TITLE = "SAM3D Body Predict (int8 dp4a)"

        def predict(
            self,
            sam3d_body_model,
            image,
            bbox_xyxy="",
            inference_type="full",
            save_name="sam3d_body",
        ):
            import numpy as np

            bundle: Sam3dBodyBundle = sam3d_body_model
            est = bundle.estimator
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
                bundle.precision
            ]
            rgb = _image_to_rgb_ndarray(image)
            bboxes = None
            if bbox_xyxy.strip():
                bboxes = np.array([[float(v) for v in bbox_xyxy.split(",")]], dtype=np.float32)

            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
            ):
                outputs = est.process_one_image(rgb, bboxes=bboxes, inference_type=inference_type)

            os.makedirs(_PIPE_OUT, exist_ok=True)
            mesh_path = os.path.join(_PIPE_OUT, f"{save_name}.obj")
            self._save_first_mesh(est, outputs, mesh_path)

            overlay = image[0] if image.dim() == 4 else image
            return (mesh_path, overlay.unsqueeze(0).cpu())

        @staticmethod
        def _save_first_mesh(est, outputs, path):
            """Write the first detected person's posed MHR mesh as a Wavefront .obj
            (vertices from the model output, faces from the MHR head template)."""
            import numpy as np

            verts = None
            for key in ("vertices", "pred_vertices", "mhr_vertices", "verts"):
                v = outputs.get(key) if isinstance(outputs, dict) else None
                if v is not None:
                    verts = v
                    break
            if verts is None:
                raise RuntimeError(
                    "SAM 3D Body output had no vertices field; inspect `outputs` keys "
                    f"{list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}"
                )
            v = torch.as_tensor(verts)
            while v.dim() > 2:
                v = v[0]
            v = v.detach().cpu().numpy()
            faces = est.faces  # [F,3] int, from head_pose.faces
            with open(path, "w") as fh:
                for x, y, z in v:
                    fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                for a, b, c in np.asarray(faces):
                    fh.write(f"f {a + 1} {b + 1} {c + 1}\n")

    class SAM3DObjectLoaderFNI8:
        """SAM 3D Objects loader. The fp TRELLIS pipeline RUNS single-card on Volta, but in
        its own cu126/torch-2.9 pixi runtime (spconv + comfy_sparse_attn), which cannot
        co-load fni8's torch-2.10/cu129 kernel in one process (design doc §2). So this node
        does not drive the pipeline in-ComfyUI-superl8-process; use the `ComfyUI-SAM3DObjects`
        pack for the fp pipeline. fni8's int8 contribution (DINOv2 conditioner + flow-DiT
        linears) is wired and validated at the code level but gated on a cu126 fni8 build."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"acknowledge": ("BOOLEAN", {"default": False})}}

        RETURN_TYPES = ("SAM3D_OBJECT_MODEL",)
        FUNCTION = "load"
        CATEGORY = "fni8/sam3d"
        TITLE = "SAM3D Object Loader FNI8 (fp via pixi pack; int8 pending cu126 build)"

        def load(self, acknowledge=False):
            raise NotImplementedError(
                "SAM 3D Objects (TRELLIS flow-matching) RUNS single-card on this Volta fleet "
                "(fp MVP produced: real gaussian splat + mesh, ~6.4 GB peak — see "
                "bench/sam3dobjects_fni8_mvp.py + .pipe_out/sam3dobject_fp.{ply,glb}), but "
                "only in the cu126/torch-2.9 sam3dobjects-nodes pixi env (spconv + "
                "comfy_sparse_attn). That native stack cannot co-load fni8's torch-2.10/cu129 "
                "_C.so in one process (ABI mismatch), so this in-process node can't drive it. "
                "Run the fp pipeline via the ComfyUI-SAM3DObjects pack; fni8 int8 (DINOv2 + "
                "flow-DiT linears, wired via sam3d_encoder.patch_backbone_linears_int8) is "
                "gated on building fni8 for cu126. Full analysis: docs/sam3d-fni8-design.md §2."
            )

    class SAM3DObjectPredict:
        """SAM 3D Objects predict — see SAM3DObjectLoaderFNI8 (fp runs via the pixi pack;
        in-process int8 gated on a cu126 fni8 build)."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {"sam3d_object_model": ("SAM3D_OBJECT_MODEL",), "image": ("IMAGE",)}
            }

        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("asset_path",)
        FUNCTION = "predict"
        CATEGORY = "fni8/sam3d"
        TITLE = "SAM3D Object Predict (fp via pixi pack; int8 pending cu126 build)"

        def predict(self, sam3d_object_model, image):
            raise NotImplementedError("See SAM3DObjectLoaderFNI8 / docs/sam3d-fni8-design.md §2")

    SAM3D_NODE_CLASS_MAPPINGS = {
        "SAM3DBodyLoaderFNI8": SAM3DBodyLoaderFNI8,
        "SAM3DBodyPredict": SAM3DBodyPredict,
        "SAM3DObjectLoaderFNI8": SAM3DObjectLoaderFNI8,
        "SAM3DObjectPredict": SAM3DObjectPredict,
    }
else:  # pragma: no cover - outside ComfyUI the nodes are unavailable
    SAM3D_NODE_CLASS_MAPPINGS = {}
