# SPDX-License-Identifier: Apache-2.0
"""int8 dp4a acceleration for the SAM 3D image backbones (generalized ViT patch).

Both SAM 3D models are dominated by a plain non-causal ViT image backbone — the exact
workload the SAM 3.1 MVP (`comfyui_superl8.sam3_encoder`) and the fni8 DiT path already
accelerate. This module generalizes that pattern so it can wrap an *arbitrary* backbone
root `nn.Module` (SAM 3D Body's `ViT` / `Dinov3Backbone`, SAM 3D Objects' DINOv2
conditioner and dense flow-DiT), instead of being hard-scoped to `sam3.model.vitdet`.

What gets int8'd (per `fused_ni8/AGENTS.md`):

  * **Linears** — every `nn.Linear` under the backbone root (block qkv/proj, MLP fc1/fc2)
    is wrapped with `Int8LinearShim`: offline per-row int8 weight (SDNQ recipe, fp32
    scales), `superl8.linear_w8a8` dp4a GEMM, a one-time SQNR gate → fp fallback per layer.
    dp4a needs K % 4 == 0; a linear that fails that (or the SQNR bar) stays fp. This is
    the proven win-carrier (SAM 3.1: all trunk linears int8, 1.67× encoder).
  * **Attention** — two optional int8 SDPA routers (SQNR-gated, non-causal only):
    `patch_module_sdpa_int8` swaps a module-level `F.scaled_dot_product_attention` symbol
    (like SAM 3.1's `vitdet`); `patch_global_sdpa_int8` swaps
    `torch.nn.functional.scaled_dot_product_attention` GLOBALLY, for backbones that call it
    **fully-qualified** (SAM 3D Body's DINOv3 `model_dinov3.py` does — the module-scoped
    proxy can't reach it). int8 attention (head_dim 80 ∈ allowlist) is the real speedup
    lever: on the tensor-core-gimped fleet the fp16-TC attention dominates the backbone, so
    int8'ing only the linears gives geometric parity (SAM 3D Body int8-linears: MPJPE
    2.27 mm vs fp) but ~0 wall-clock win — the GEMMs aren't the bottleneck.

Everything OUTSIDE the backbone stays fp — SAM 3D Body's promptable decoder / camera head
/ MHR head / MHR body model, and SAM 3D Objects' VAE decoders — they are pose- and
reconstruction-load-bearing. All patches are reversible (`.unpatch()`), so a loader holds
the fp oracle and the int8 path side by side for the geometric-quality gate
(`comfyui_superl8.sam3d_metrics`).

Kept self-contained (does not import the in-flux `sam3_encoder`) so the SAM 3D lane and
the SAM 3.1 lane evolve independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# fni8 int8-attention head-dim allowlist (must match comfyui_superl8.attention /
# sam3_encoder). SAM 3D Body ViT-H is head_dim 80; DINOv2/v3 ViT-L/H are 64/80.
SUPPORTED_HEAD_DIMS = (32, 64, 72, 80, 128, 256)
# SQNR floor (dB) to keep a site/linear on int8. 20 dB == quant-noise 1% of signal.
DEFAULT_SQNR_FLOOR_DB = 20.0


def _sqnr_db(y_int8: torch.Tensor, y_fp: torch.Tensor) -> float:
    a = y_int8.detach().float()
    b = y_fp.detach().float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("-inf")
    signal = b.pow(2).mean()
    noise = (a - b).pow(2).mean()
    if noise.item() == 0.0:
        return float("inf")
    return (10.0 * torch.log10(signal / noise)).item()


def _passes(sqnr_db: float, floor_db: float) -> bool:
    # finite SQNR at/above floor => int8 engaged AND accurate. +inf => fni8's inner
    # outlier gate already fell back to fp bit-identically => route to fp so we stop
    # re-dispatching (identical rationale to comfyui_superl8.attention / sam3_encoder).
    return floor_db <= sqnr_db < float("inf")


def _quantize_per_row_i8(w: torch.Tensor):
    """Symmetric per-row (per-output-channel) int8 quant, scales fp32 (SDNQ recipe,
    AGENTS.md): scale = max|w| / 127. Returns (int8 [N,K], fp32 scale [N])."""
    w32 = w.detach().float()
    amax = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    q = (w32 / scale).round().clamp_(-127, 127).to(torch.int8)
    return q, scale.squeeze(1).contiguous()


@dataclass
class LinearStats:
    n_wrapped: int = 0
    n_skipped_k: int = 0
    calls_int8: int = 0
    calls_fp: int = 0
    sqnr: list = field(default_factory=list)

    def summary(self) -> dict:
        finite = [v for v in self.sqnr if math.isfinite(v)]
        return {
            "wrapped": self.n_wrapped,
            "skipped_k_not_mult4": self.n_skipped_k,
            "calls_int8": self.calls_int8,
            "calls_fp": self.calls_fp,
            "sqnr_min_db": min(finite) if finite else None,
            "sqnr_med_db": (sorted(finite)[len(finite) // 2] if finite else None),
        }


class Int8LinearShim(torch.nn.Module):
    """Drop-in for an `nn.Linear` that runs the fni8 W8A8 dp4a GEMM on an offline per-row
    int8 quant of the weight, with a one-time SQNR gate → fp fallback. A linear whose
    K % 4 != 0 (dp4a requirement) or that misses the SQNR bar stays fp — the gate decides
    where int8 is allowed, never a weakened bar (AGENTS.md)."""

    def __init__(self, linear: torch.nn.Linear, floor_db: float, stats: LinearStats):
        super().__init__()
        self.out_features = linear.out_features
        self.in_features = linear.in_features
        self._floor = floor_db
        self._stats = stats
        self._sqnr_pass = None  # None=unchecked, True=int8, False=fp
        dev = linear.weight.device
        q, scale = _quantize_per_row_i8(linear.weight)
        self.register_buffer("w_i8", q.to(dev), persistent=False)
        self.register_buffer("w_scale", scale.to(dev), persistent=False)
        self.register_buffer(
            "bias",
            linear.bias.detach().clone() if linear.bias is not None else None,
            persistent=False,
        )
        self.register_buffer("w_fp", linear.weight.detach().clone(), persistent=False)
        self._dp4a_ok = (self.in_features % 4) == 0

    def _fp(self, x):
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, self.w_fp.to(x.dtype), b)

    def forward(self, x):
        if not self._dp4a_ok or self._sqnr_pass is False:
            self._stats.calls_fp += 1
            return self._fp(x)

        import superl8

        out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
        b = self.bias.to(out_dtype) if self.bias is not None else None
        y = superl8.linear_w8a8(
            x.to(out_dtype),
            self.w_i8,
            self.w_scale,
            bias=b,
            out_dtype=out_dtype,
        ).to(x.dtype)

        if self._sqnr_pass is None:
            ref = self._fp(x)
            sqnr = _sqnr_db(y, ref)
            self._sqnr_pass = _passes(sqnr, self._floor)
            self._stats.sqnr.append(sqnr)
            if not self._sqnr_pass:
                self._stats.calls_fp += 1
                return ref
        self._stats.calls_int8 += 1
        return y


@dataclass
class LinearPatchHandle:
    replacements: list  # (parent_module, attr_name, original_linear)
    stats: LinearStats

    def unpatch(self):
        for parent, attr, orig in self.replacements:
            setattr(parent, attr, orig)


def patch_backbone_linears_int8(
    root: torch.nn.Module,
    *,
    floor_db: float = DEFAULT_SQNR_FLOOR_DB,
    name_filter=None,
) -> LinearPatchHandle:
    """Wrap every `nn.Linear` under `root` with an int8 dp4a W8A8 shim.

    `root` is the backbone (e.g. SAM 3D Body `model.backbone` or its `.encoder`) — pass
    ONLY the backbone so the decoder / heads / MHR body model stay fp. `name_filter(name)`
    (optional) can exclude specific linears (e.g. a projection that must stay fp).
    Idempotent-safe per call; `.unpatch()` restores fp."""
    stats = LinearStats()
    replacements = []
    linears = [
        (name, mod) for name, mod in root.named_modules() if isinstance(mod, torch.nn.Linear)
    ]
    for name, lin in linears:
        if name_filter is not None and not name_filter(name):
            continue
        parent = root
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        if (lin.in_features % 4) != 0:
            stats.n_skipped_k += 1
        shim = Int8LinearShim(lin, floor_db, stats)
        setattr(parent, attr, shim)
        replacements.append((parent, attr, lin))
        stats.n_wrapped += 1
    return LinearPatchHandle(replacements=replacements, stats=stats)


# --------------------------------------------------------------------------------------
# Optional module-scoped SDPA attention proxy (for backbones that dispatch via a
# module-level `F.scaled_dot_product_attention`, like SAM 3.1's vitdet). Off by default.
# --------------------------------------------------------------------------------------


@dataclass
class AttnGate:
    decisions: dict = field(default_factory=dict)
    sqnr: dict = field(default_factory=dict)
    n_int8: int = 0
    n_fallback: int = 0

    def summary(self) -> dict:
        finite = [v for v in self.sqnr.values() if math.isfinite(v)]
        return {
            "sites": len(self.decisions),
            "sites_int8": sum(1 for p in self.decisions.values() if p),
            "sites_fp": sum(1 for p in self.decisions.values() if not p),
            "calls_int8": self.n_int8,
            "calls_fp": self.n_fallback,
            "sqnr_min_db": min(finite) if finite else None,
            "sqnr_med_db": (sorted(finite)[len(finite) // 2] if finite else None),
        }


class _SdpaProxy:
    """Delegates everything to the real `torch.nn.functional` EXCEPT
    `scaled_dot_product_attention`, routed through fni8 int8 dp4a (SQNR-gated) for
    eligible non-causal head-dim-{allowlist} calls."""

    def __init__(self, real_F, gate: AttnGate, floor_db: float, rotate: bool = False):
        self._real = real_F
        self._gate = gate
        self._floor = floor_db
        self._rotate = rotate

    def __getattr__(self, name):
        return getattr(self._real, name)

    def scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        **kw,
    ):
        orig = self._real.scaled_dot_product_attention
        lp = value.dtype if value.dtype in (torch.float16, torch.bfloat16) else torch.float16
        eligible = (
            attn_mask is None
            and not is_causal
            and dropout_p == 0.0
            and query.dim() == 4
            and query.shape[-1] in SUPPORTED_HEAD_DIMS
            and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        if not eligible:
            self._gate.n_fallback += 1
            return orig(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kw,
            )

        import superl8

        out_dtype = query.dtype
        key_sig = (tuple(query.shape), str(query.dtype))
        cached = self._gate.decisions.get(key_sig)
        if cached is False:
            self._gate.n_fallback += 1
            return orig(query, key, value, scale=scale)

        q8 = query.to(lp).contiguous()
        k8 = key.to(lp).contiguous()
        v8 = value.to(lp).contiguous()
        out8 = superl8.attn_int8_fwd(
            q8,
            k8,
            v8,
            causal=False,
            scale=scale,
            rotate=self._rotate,
        ).to(out_dtype)
        if cached is True:
            self._gate.n_int8 += 1
            return out8

        ref = orig(query, key, value, scale=scale)
        sqnr = _sqnr_db(out8, ref)
        passed = _passes(sqnr, self._floor)
        self._gate.decisions[key_sig] = passed
        self._gate.sqnr[key_sig] = sqnr
        if passed:
            self._gate.n_int8 += 1
            return out8
        self._gate.n_fallback += 1
        return ref


@dataclass
class AttnPatchHandle:
    module: object
    original_F: object
    gate: AttnGate

    def unpatch(self):
        self.module.F = self.original_F


def patch_module_sdpa_int8(
    attn_module,
    *,
    floor_db: float = DEFAULT_SQNR_FLOOR_DB,
    gate: AttnGate | None = None,
    rotate: bool = False,
) -> AttnPatchHandle:
    """Swap `attn_module.F` for an int8-routing SDPA proxy. `attn_module` must be the
    module object that calls `F.scaled_dot_product_attention` (e.g. the backbone's
    attention submodule module). Idempotent-safe; `.unpatch()` restores fp."""
    gate = gate or AttnGate()
    real_F = getattr(attn_module, "F")
    if isinstance(real_F, _SdpaProxy):
        real_F = real_F._real
    attn_module.F = _SdpaProxy(real_F, gate, floor_db, rotate=rotate)
    return AttnPatchHandle(module=attn_module, original_F=real_F, gate=gate)


@dataclass
class GlobalAttnPatchHandle:
    original_sdpa: object
    gate: AttnGate

    def unpatch(self):
        import torch.nn.functional as F

        F.scaled_dot_product_attention = self.original_sdpa


def patch_global_sdpa_int8(
    *,
    floor_db: float = DEFAULT_SQNR_FLOOR_DB,
    gate: AttnGate | None = None,
    rotate: bool = False,
) -> GlobalAttnPatchHandle:
    """Replace `torch.nn.functional.scaled_dot_product_attention` GLOBALLY with the int8
    dp4a routing proxy (SQNR-gated non-causal only; everything else delegates to the real
    SDPA).

    Use this when a backbone calls `F.scaled_dot_product_attention` **fully-qualified**
    (e.g. SAM 3D Body's DINOv3 `model_dinov3.py` does `torch.nn.functional.
    scaled_dot_product_attention(...)`), so the module-scoped `patch_module_sdpa_int8`
    (which swaps a module's local `F` symbol) cannot intercept it. This is the lever for
    int8 *attention* — head_dim 80 (DINOv3-H+ / ViT-H) is in the fni8 allowlist, and on
    the tensor-core-gimped fleet the fp16-TC attention dominates the backbone, so int8'ing
    the GEMMs alone gives no wall-clock win (measured: SAM 3D Body int8-linears MPJPE
    2.27 mm vs fp, but ~0 ms faster). GLOBAL scope means it also touches non-backbone SDPA
    calls for the duration — call `.unpatch()` immediately after the backbone forward, or
    scope it around just the encode step. Non-causal-only + SQNR-gated keeps it safe
    (the decoder/head softmax attention that must stay fp is causal/masked or trips the
    gate → delegates to the real SDPA)."""
    import torch.nn.functional as F

    gate = gate or AttnGate()
    orig = F.scaled_dot_product_attention
    proxy = _SdpaProxy(F, gate, floor_db, rotate=rotate)
    # bind the real function so the proxy's internal `orig` calls hit the unpatched impl
    # even after we overwrite the module attribute.
    proxy._real = _RealSdpaHolder(orig, F)
    F.scaled_dot_product_attention = proxy.scaled_dot_product_attention
    return GlobalAttnPatchHandle(original_sdpa=orig, gate=gate)


class _RealSdpaHolder:
    """Delegates attribute access to `torch.nn.functional` but pins
    `scaled_dot_product_attention` to the pre-patch implementation, so a globally-installed
    `_SdpaProxy` never recurses into itself."""

    def __init__(self, orig_sdpa, real_F):
        self._orig_sdpa = orig_sdpa
        self._real_F = real_F

    def __getattr__(self, name):
        return getattr(self._real_F, name)

    @property
    def scaled_dot_product_attention(self):
        return self._orig_sdpa


@dataclass
class Sam3dBackboneInt8Handle:
    """Bundle so a caller flips the whole SAM 3D backbone to int8 and back with one
    object. `attn` is None unless the backbone dispatches via a module-scoped SDPA."""

    linears: LinearPatchHandle | None = None
    attn: AttnPatchHandle | None = None

    def unpatch(self):
        if self.attn is not None:
            self.attn.unpatch()
        if self.linears is not None:
            self.linears.unpatch()

    def summary(self) -> dict:
        return {
            "linears": self.linears.stats.summary() if self.linears else None,
            "attention": self.attn.gate.summary() if self.attn else None,
        }
