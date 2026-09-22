# Recipe: SAM 3D Objects (image → 3D asset) — PLACEHOLDER (int8 blocked)

- **Workflow:** none shipped. The `SAM3DObjectLoaderFNI8` / `SAM3DObjectPredict` nodes are
  registered but **raise `NotImplementedError`** for the int8 dp4a path, so there is
  deliberately **no example JSON** (a workflow that cannot resolve/run would be dishonest).
- **Status:** **placeholder / not-shipped-as-working.** The int8 acceleration is blocked by a
  cross-repo torch-ABI mismatch (below). Do not represent this as a working int8 path.

## Why there is no int8 workflow (cu126-vs-cu129 ABI blocker)

fni8's int8 dp4a kernel (`_C.so`) is built against **torch 2.10 / cu129** (the fni8/AGENTS.md
Volta toolchain pin — CUDA 13 drops sm_70, so this is the last Volta-capable stack). The
SAM 3D Objects runtime is a **cu126 / torch 2.9.1** native stack (spconv 2.3.8, gsplat,
nvdiffrast, pytorch3d, flash_attn_v100) — its C++ ABI symbol mangling differs
(`…S3_ib` vs `…S2_ib`), so the two shared objects **cannot co-load in one process**, and the
cu126 sparse deps cannot move to torch 2.10. There is no nvcc on the host/pixi env to rebuild
fni8 for cu126.

- **SAM 3D Body int8 works** because Body needs only torch + the vendored `sam_3d_body` lib
  (no spconv / gsplat), so it runs inside the cu129 image. Objects needs the cu126 sparse
  stack **and** fni8's cu129 kernel in one process — the incompatibility.
- An **fp MVP of Objects** (real splat + mesh, combined Y-up GLB) has been produced, but only
  via the **external cu126 pixi pack** — not the fni8 int8 path, and not in-process here.

## To unblock (follow-up work)

Build an fni8 `_C.so` against **torch 2.9.1 + cu126** with a full sm_70 CUDA toolchain, then
the wired-but-gated Objects int8 linears (`comfyui_superl8/sam3d_encoder.py`, SQNR-gated) can run
in the cu126 process. This is a follow-up PR gated on that build, not an architecture change.

Full analysis: [`docs/sam3d-fni8-design.md`](../sam3d-fni8-design.md) §2 (int8 status — wiring
proven, kernel blocked by the cross-repo torch-ABI mismatch).
