# SPDX-License-Identifier: Apache-2.0
"""int8 dp4a acceleration for the SAM 3 / 3.1 Perception-Encoder ViT backbone.

SAM 3.1's compute is dominated by a plain non-causal ViT (Meta Perception Encoder
PE-L: 32 layers, dim 1024, 16 heads -> head_dim 64, ~5184 tokens at 1008^2; global
attention at a few layers, windowed win-24 at the rest, 2D RoPE). That is the exact
non-causal workload fni8's DiT path already accelerates, so **no new kernel is
required** — this module is host-side glue that routes the ViT trunk through the
existing int8 dp4a kernels, SQNR-gated, exactly like `real_model_eval.py` /
`comfyui_superl8.attention` / `comfyui_superl8.ops`.

Two independent, individually-toggleable patches, applied only to the ViT trunk
(`model.backbone.vision_backbone.trunk`) — the mask decoder, prompt/geometry
encoder, DETR detector, and video memory attention stay fp (mask-quality /
tracking-quality load-bearing, per fni8/AGENTS.md):

  * `patch_vit_attention_int8`  — `sam3.model.vitdet`'s `F.scaled_dot_product_attention`
    is swapped (module-scoped proxy, so ONLY the ViT sees it) for a shim that runs
    `superl8.attn_int8_fwd(causal=False)`. The ViT `Block` already does window-partition
    /-unpartition around attention (windows land in the batch dim), so a single SDPA
    shim covers BOTH global (S=5184) and windowed (S=576) layers — no separate varlen
    glue needed. Per-call-site SQNR gate: a site that misses the floor stays fp.
  * `patch_vit_linears_int8` — every `nn.Linear` under the trunk (qkv, proj, mlp.fc1,
    mlp.fc2) is wrapped to run the `superl8.linear_w8a8` dp4a GEMM on an offline per-row
    int8 quant of the weight, with a one-time SQNR gate -> fp fallback per layer.

Both are reversible via the returned handle (`.unpatch()`), so a loader can hold the
fp reference oracle and the int8 path side by side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# fni8 int8-attention head-dim allowlist (must match comfyui_superl8.attention).
SUPPORTED_HEAD_DIMS = (32, 64, 72, 80, 128, 256)
# SQNR floor (dB) for a ViT-trunk call-site / linear to keep the int8 path. 20 dB ==
# quantization-noise power 1% of signal (cos ~0.995) — same bar as attention.py.
DEFAULT_SQNR_FLOOR_DB = 20.0


def _sqnr_db(y_int8: torch.Tensor, y_fp: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio (dB), fp32, per comfyui_superl8.attention."""
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
    # A finite SQNR at/above the floor means int8 genuinely engaged AND is accurate.
    # +inf means fni8's own inner outlier gate already fell back to fp (bit-identical)
    # -> route the site to fp directly so we stop re-dispatching the kernel every call
    # (identical rationale to comfyui_superl8.attention.Int8AttnGate.record).
    return floor_db <= sqnr_db < float("inf")


# --------------------------------------------------------------------------------------
# Attention patch (module-scoped SDPA proxy over sam3.model.vitdet)
# --------------------------------------------------------------------------------------


class _SdpaProxy:
    """Delegates every attribute to the real `torch.nn.functional` EXCEPT
    `scaled_dot_product_attention`, which routes eligible non-causal ViT attention
    through fni8's int8 dp4a kernel (SQNR-gated). Installed as `vitdet.F` so the swap
    is invisible to every other module (mask decoder, text encoder, detector)."""

    def __init__(self, real_F, gate: "AttnGate", floor_db: float, rotate: bool = False):
        self._real = real_F
        self._gate = gate
        self._floor = floor_db
        self._rotate = rotate

    def __getattr__(self, name):
        return getattr(self._real, name)

    def scaled_dot_product_attention(
        self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        scale=None, enable_gqa=False, **kw,
    ):
        orig = self._real.scaled_dot_product_attention
        # 2D-RoPE returns q,k in fp32 (the rotary math upcasts) while v stays fp16/bf16
        # — cast q,k to v's low-precision dtype for the int8 kernel (exactly what fp16
        # SDPA would do), so the RoPE'd global/windowed ViT attention still engages int8.
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
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale, **kw)

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
            q8, k8, v8, causal=False, scale=scale, rotate=self._rotate,
        ).to(out_dtype)
        if cached is True:
            self._gate.n_int8 += 1
            return out8

        # First sighting of this call-site: measure int8 vs fp, cache the verdict.
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
class AttnGate:
    """Per-model SQNR gate + call accounting for the ViT-attention int8 patch."""
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


@dataclass
class AttnPatchHandle:
    module: object
    original_F: object
    gate: AttnGate

    def unpatch(self):
        self.module.F = self.original_F


def patch_vit_attention_int8(
    model, *, floor_db: float = DEFAULT_SQNR_FLOOR_DB, gate: AttnGate | None = None,
    rotate: bool = False,
) -> AttnPatchHandle:
    """Route the SAM ViT trunk's non-causal attention through fni8 int8 dp4a.

    Scoped to `sam3.model.vitdet` only (mask decoder / text encoder / detector keep
    fp). Idempotent-safe: install once per model; call `.unpatch()` to restore fp.

    `rotate=True` applies fni8's Hadamard incoherence rotation to Q/K (logit-invariant,
    issue #74) — the mitigation for image-activation channel outliers that otherwise
    trip fni8's inner SageAttention gate and force a full fp fallback."""
    import sam3.model.vitdet as vitdet

    gate = gate or AttnGate()
    real_F = vitdet.F
    # If already a proxy (double-patch), unwrap to the real module first.
    if isinstance(real_F, _SdpaProxy):
        real_F = real_F._real
    vitdet.F = _SdpaProxy(real_F, gate, floor_db, rotate=rotate)
    return AttnPatchHandle(module=vitdet, original_F=real_F, gate=gate)


# --------------------------------------------------------------------------------------
# Linear patch (W8A8 dp4a on the ViT trunk linears)
# --------------------------------------------------------------------------------------


def _quantize_per_row_i8(w: torch.Tensor):
    """Symmetric per-row (per-output-channel) int8 quant, scales in fp32 (SDNQ recipe,
    fni8/AGENTS.md): scale = max|w| / 127. Returns (int8 [N,K], fp32 scale [N])."""
    w32 = w.detach().float()
    amax = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 127.0)
    q = (w32 / scale).round().clamp_(-127, 127).to(torch.int8)
    return q, scale.squeeze(1).contiguous()


class Int8LinearShim(torch.nn.Module):
    """Drop-in for an `nn.Linear` that runs the fni8 W8A8 dp4a GEMM on an offline
    per-row int8 quant of the weight, with a one-time SQNR gate -> fp fallback.

    dp4a needs K % 4 == 0; a linear that fails that (or the SQNR bar) stays fp — the
    accuracy/eligibility gate decides where int8 is allowed, never a weakened bar."""

    def __init__(self, linear: torch.nn.Linear, floor_db: float, stats: "LinearStats"):
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
            "bias", linear.bias.detach().clone() if linear.bias is not None else None,
            persistent=False,
        )
        # fp reference weight kept for the one-time SQNR check + K%4 fallback.
        self.register_buffer("w_fp", linear.weight.detach().clone(), persistent=False)
        self._dp4a_ok = (self.in_features % 4) == 0

    @classmethod
    def from_resident(cls, w_i8, w_scale, bias, stats: "LinearStats"):
        """Build from PRE-QUANTIZED offline `.fni8` bytes (int8 codes + fp32 scale +
        optional bias) — no runtime quant, no fp reference weight kept, and the SQNR
        decision was made by the offline converter, so the one-time gate is skipped.
        This is the fast-load path (`load_vit_linears_from_fni8`)."""
        self = cls.__new__(cls)
        torch.nn.Module.__init__(self)
        self.out_features, self.in_features = int(w_i8.shape[0]), int(w_i8.shape[1])
        self._floor = 0.0
        self._stats = stats
        self._sqnr_pass = True            # offline-decided int8; no runtime gate
        self.register_buffer("w_i8", w_i8.to(torch.int8), persistent=False)
        self.register_buffer("w_scale", w_scale.to(torch.float32), persistent=False)
        self.register_buffer(
            "bias", bias.detach().clone() if bias is not None else None,
            persistent=False)
        self.register_buffer("w_fp", None, persistent=False)  # reconstructed on demand
        self._dp4a_ok = (self.in_features % 4) == 0
        return self

    def _fp(self, x):
        b = self.bias.to(x.dtype) if self.bias is not None else None
        # Resident shims keep no fp weight — reconstruct from the int8 codes + scale for
        # the (unused-on-the-int8-path) fallback so `_fp` is always safe to call.
        w = self.w_fp if self.w_fp is not None else (self.w_i8.float() * self.w_scale[:, None])
        return torch.nn.functional.linear(x, w.to(x.dtype), b)

    def forward(self, x):
        # Pre-activation linear output, int8-or-fp per the SQNR gate. Used both by the
        # normal nn.Linear call sites (qkv, proj, fc2) and by the patched fused
        # `addmm_act` fc1 path (which applies GELU on top).
        if not self._dp4a_ok or self._sqnr_pass is False:
            self._stats.calls_fp += 1
            return self._fp(x)

        import superl8

        out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
        b = self.bias.to(out_dtype) if self.bias is not None else None
        y = superl8.linear_w8a8(
            x.to(out_dtype), self.w_i8, self.w_scale, bias=b, out_dtype=out_dtype,
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


def _make_int8_addmm_act(orig_addmm_act):
    """A drop-in for `sam3.model.vitdet.addmm_act` that routes an Int8LinearShim `fc1`
    through the int8 dp4a path and applies the fused activation (GELU/ReLU) in fp; any
    non-shim linear falls back to the stock fused kernel."""
    import torch.nn.functional as F

    def int8_addmm_act(activation, linear, mat1):
        if not isinstance(linear, Int8LinearShim):
            return orig_addmm_act(activation, linear, mat1)
        y = linear(mat1)  # gated int8/fp pre-activation linear output
        if activation in (F.relu, torch.nn.ReLU):
            return F.relu(y)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    return int8_addmm_act


@dataclass
class LinearPatchHandle:
    replacements: list  # (parent_module, attr_name, original_linear)
    stats: LinearStats
    vitdet_module: object = None
    orig_addmm_act: object = None

    def unpatch(self):
        for parent, attr, orig in self.replacements:
            setattr(parent, attr, orig)
        if self.vitdet_module is not None and self.orig_addmm_act is not None:
            self.vitdet_module.addmm_act = self.orig_addmm_act


def _trunk(model):
    return model.backbone.vision_backbone.trunk


def patch_vit_linears_int8(
    model, *, floor_db: float = DEFAULT_SQNR_FLOOR_DB,
) -> LinearPatchHandle:
    """Wrap every `nn.Linear` under the ViT trunk with an int8 dp4a W8A8 shim.

    Only the trunk (PE-L QKV/proj/MLP) is touched; conv patch stem, FPN neck convs,
    LayerNorm, GELU, and all of the mask decoder / detector / text-encoder linears
    stay fp."""
    trunk = _trunk(model)
    stats = LinearStats()
    replacements = []
    # Snapshot first (mutating during named_modules() iteration is unsafe).
    linears = [
        (name, mod) for name, mod in trunk.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]
    for name, lin in linears:
        parent = trunk
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        if (lin.in_features % 4) != 0:
            stats.n_skipped_k += 1
        shim = Int8LinearShim(lin, floor_db, stats)
        setattr(parent, attr, shim)
        replacements.append((parent, attr, lin))
        stats.n_wrapped += 1

    # fc1 is called via the module-scoped fused `addmm_act` (GELU folded in), not
    # `nn.Linear.__call__` — route that through the shim too, vitdet-scoped so no
    # other module's fused matmuls are affected.
    import sam3.model.vitdet as vitdet
    orig_addmm = vitdet.addmm_act
    vitdet.addmm_act = _make_int8_addmm_act(orig_addmm)
    return LinearPatchHandle(
        replacements=replacements, stats=stats,
        vitdet_module=vitdet, orig_addmm_act=orig_addmm,
    )


def load_vit_linears_from_fni8(model, reader, *, device: str = "cuda") -> LinearPatchHandle:
    """Install PRE-QUANTIZED (offline `.fni8`) int8 linears onto the ViT trunk — the
    fast-load counterpart to `patch_vit_linears_int8`. No runtime quant, no fp weight
    kept, no per-layer SQNR gate (the offline converter already decided); the loader
    just mmaps the resident int8 codes+scales+bias and places them.

    `reader` is an `superl8.FQReader` over a `.fni8` produced by `sam3_convert`; its
    `__meta__["int8_layers"]` lists the trunk-relative module paths to install."""
    trunk = _trunk(model)
    stats = LinearStats()
    replacements = []
    meta = reader.header.get("__meta__", {})
    int8_layers = meta.get("int8_layers")
    if not int8_layers:
        # Fall back to every per_row_i8 weight entry that is not a bias.
        int8_layers = [n for n in reader.names
                       if reader.info(n).get("scheme") == "per_row_i8"]
    tset = reader.header["tensors"]
    for name in int8_layers:
        qt = reader.get_qtensor(name, device)
        bias = reader.get(name + ".bias", device) if (name + ".bias") in tset else None
        parent = trunk
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        orig = getattr(parent, attr)
        shim = Int8LinearShim.from_resident(qt.data, qt.scale, bias, stats)
        setattr(parent, attr, shim)
        replacements.append((parent, attr, orig))
        stats.n_wrapped += 1

    import sam3.model.vitdet as vitdet
    orig_addmm = vitdet.addmm_act
    vitdet.addmm_act = _make_int8_addmm_act(orig_addmm)
    return LinearPatchHandle(
        replacements=replacements, stats=stats,
        vitdet_module=vitdet, orig_addmm_act=orig_addmm,
    )


@dataclass
class Sam3Int8Handle:
    """Bundle of both patches, so a caller can flip the whole ViT trunk to int8 and
    back with one object."""
    attn: AttnPatchHandle | None = None
    linears: LinearPatchHandle | None = None

    def unpatch(self):
        if self.attn is not None:
            self.attn.unpatch()
        if self.linears is not None:
            self.linears.unpatch()

    def summary(self) -> dict:
        return {
            "attention": self.attn.gate.summary() if self.attn else None,
            "linears": self.linears.stats.summary() if self.linears else None,
        }


def patch_sam3_encoder_int8(
    model, *, attention: bool = True, linears: bool = True,
    floor_db: float = DEFAULT_SQNR_FLOOR_DB, rotate: bool = False,
) -> Sam3Int8Handle:
    """Flip the SAM 3.1 ViT trunk (attention and/or linears) onto the int8 dp4a path.

    Everything outside the trunk stays fp. Returns a handle whose `.unpatch()`
    restores the fp oracle and `.summary()` reports the per-site SQNR gate decisions.
    `rotate=True` enables the Hadamard-rotated int8 attention (outlier mitigation)."""
    h = Sam3Int8Handle()
    if attention:
        h.attn = patch_vit_attention_int8(model, floor_db=floor_db, rotate=rotate)
    if linears:
        h.linears = patch_vit_linears_int8(model, floor_db=floor_db)
    return h


def load_sam3_encoder_int8_from_fni8(
    model, fni8_path: str, *, attention: bool = True, rotate: bool = False,
    floor_db: float = DEFAULT_SQNR_FLOOR_DB, device: str = "cuda",
) -> Sam3Int8Handle:
    """Flip the ViT trunk onto int8 using an OFFLINE-quantized `.fni8` for the linears
    (mmap + place, no runtime quant) and the module-scoped SDPA proxy for attention.

    The int8-vs-fp attention decision is still the SDPA gate's (fni8's inner outlier
    gate keeps the ViT attention fp, as validated), so only the linears come from the
    `.fni8`; `attention=True` still installs the (self-gating) attention proxy."""
    import superl8

    h = Sam3Int8Handle()
    reader = superl8.FQReader(fni8_path)
    try:
        h.linears = load_vit_linears_from_fni8(model, reader, device=device)
    finally:
        reader.close()
    if attention:
        h.attn = patch_vit_attention_int8(model, floor_db=floor_db, rotate=rotate)
    return h
