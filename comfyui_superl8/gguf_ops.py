# SPDX-License-Identifier: Apache-2.0
"""Native GGUF k-quant DiT Linear on the fni8 fused dp4a kernels.

The REAL ComfyUI pipeline loads a GGUF DiT via ComfyUI-GGUF's ``UnetLoaderGGUF``,
which builds every Linear from ``GGMLOps`` and stores each quantized weight as a
``GGMLTensor`` holding the *native GGUF super-block bytes* (``.tensor_type`` = the
ggml quant type, ``.tensor_shape`` = logical ``[out, in]``). Stock ComfyUI-GGUF
**dequantizes those bytes to fp on every forward** and matmuls on the (firmware-
gimped) tensor cores — the 143 s/step path.

Those bytes are byte-identical to what :func:`fni8.gguf.kquant_qtensor` wraps, so a
Q4_K/Q5_K/Q6_K (and Q3_K/Q2_K) DiT Linear can instead run **native-resident** on the
fni8 fused dp4a kernels (``superl8.linear_q{2,3,4,5,6}k``): the k-quant weight stays in
its native layout (VRAM parity, one card) and the dequant is fused into the ``__dp4a``
matmul, never materialized.

This module subclasses ``GGMLOps`` and overrides only the Linear compute; the entire
coherent ComfyUI-GGUF load/dequant/patch machinery is untouched. A per-layer SQNR/cos
gate (checked once, on the first forward) demotes any Linear that doesn't clear the
accuracy bar back to the stock dequant path — the accuracy gate decides where the
native int8 path is allowed, not ideology. bf16-native (F16/F32) DiT weights are not
GGML-quantized and keep the stock path.

Use it two ways:
  * ``enable_fni8_gguf()`` — monkeypatch stock ``GGMLOps.Linear`` in place so the
    existing ``UnetLoaderGGUF`` node transparently takes the fni8 path (real pipeline).
  * ``FNI8GGUFOps`` — the subclass, to pass as ``custom_operations`` directly.
"""
from __future__ import annotations

import logging
import os

_LOG = logging.getLogger("comfyui_superl8.gguf_ops")

# ggml quant type NAME -> (fni8 linear fn attr, native super-block byte size / 256 wts)
_KQUANT = {
    "Q2_K": ("linear_q2k", 84),
    "Q3_K": ("linear_q3k", 110),
    "Q4_K": ("linear_q4k", 144),
    "Q5_K": ("linear_q5k", 176),
    "Q6_K": ("linear_q6k", 210),
}

# cos bar for the per-layer accuracy gate (int8-dp4a vs dequant-fp reference).
_COS_BAR = float(os.environ.get("FNI8_GGUF_COS_BAR", "0.98"))
# force stock dequant path (disable fni8 native) for A/B / debugging.
_DISABLE = os.environ.get("FNI8_GGUF_DISABLE", "0").lower() in {"1", "true", "yes", "on"}


def _fni8_ok():
    try:
        import superl8  # noqa: F401
        import torch

        return torch.cuda.is_available() and hasattr(fni8, "linear_q4k")
    except Exception:
        return False


def _type_name(tt):
    return getattr(tt, "name", None) or str(tt)


def _native_linear(fni8, name, x, w_bytes, bias, out_dtype):
    fn = getattr(fni8, _KQUANT[name][0])
    return fn(x, w_bytes, bias=bias, out_dtype=out_dtype)


def _prepare_standard_lora_residuals(layer, input, out_features, in_features):
    """Return cached additive standard-LoRA tensors or a fallback reason."""
    patch_groups = getattr(getattr(layer, "weight", None), "patches", [])
    if not patch_groups:
        return [], None

    specs = []
    for group in patch_groups:
        if not isinstance(group, (tuple, list)) or len(group) != 2:
            return None, "malformed GGUF patch group"
        patches, _key = group
        if not isinstance(patches, (tuple, list)):
            return None, "malformed Comfy patch list"
        for patch in patches:
            if not isinstance(patch, (tuple, list)) or len(patch) < 5:
                return None, "malformed Comfy patch"
            strength, adapter, strength_model, offset, function = patch[:5]
            if strength_model != 1.0:
                return None, "model-strength patches require stock GGUF"
            if offset is not None or function is not None:
                return None, "offset or weight-function patches require stock GGUF"
            weights = getattr(adapter, "weights", None)
            if not isinstance(weights, (tuple, list)) or len(weights) != 6:
                return None, "non-LoRA adapter requires stock GGUF"
            up, down, alpha, mid, dora_scale, reshape = weights
            if dora_scale is not None:
                return None, "DoRA patches require stock GGUF"
            if mid is not None or reshape is not None:
                return None, "LoCon or reshaped patches require stock GGUF"
            if (
                getattr(up, "ndim", None) != 2
                or getattr(down, "ndim", None) != 2
                or tuple(up.shape) != (out_features, down.shape[0])
                or down.shape[1] != in_features
            ):
                return None, "standard LoRA tensor shape does not match the linear"
            try:
                rank = int(down.shape[0])
                scale = float(strength)
                if alpha is not None:
                    scale *= float(alpha) / rank
            except (TypeError, ValueError):
                return None, "standard LoRA scale is not numeric"
            specs.append((up, down, scale))

    # Full-model adapters can consume the remaining HBM on 16 GiB cards. Keep
    # their source tensors on the patcher's offload device and bound residency
    # to the current linear call instead of accumulating a per-layer GPU cache.
    prepared = [
        (
            up.to(device=input.device, dtype=input.dtype, non_blocking=True),
            down.to(device=input.device, dtype=input.dtype, non_blocking=True),
            scale,
        )
        for up, down, scale in specs
    ]
    return prepared, None


def _apply_standard_lora_residuals(layer, input, output, residuals=None):
    """Add standard LoRA low-rank projections without materializing a full weight."""
    if residuals is None:
        residuals, reason = _prepare_standard_lora_residuals(
            layer, input, output.shape[-1], input.shape[-1]
        )
        if reason is not None:
            raise ValueError(reason)
    for up, down, scale in residuals:
        hidden = input.to(down.dtype) @ down.transpose(0, 1)
        output = output + (hidden @ up.transpose(0, 1)).to(output.dtype) * scale
    return output


def gguf_lora_audit(model):
    """Count native, fallback, and not-yet-executed patched GGUF linears."""
    counts = {"native": 0, "fallback": 0, "unapplied": 0, "none": 0}
    for module in model.modules():
        patched = bool(getattr(getattr(module, "weight", None), "patches", []))
        state = getattr(module, "_fni8_lora_state", None)
        if not patched:
            counts["none"] += 1
        elif state in ("native", "fallback"):
            counts[state] += 1
        else:
            counts["unapplied"] += 1
    return counts


def _mark_lora_fallback(layer, reason):
    if getattr(getattr(layer, "weight", None), "patches", []):
        layer._fni8_lora_state = "fallback"
        layer._fni8_lora_fallback_reason = reason


def _make_forward(base_forward):
    """Build a ``forward_ggml_cast_weights`` that routes native k-quant Linears to
    fni8 and falls back to ``base_forward`` (the stock dequant path) otherwise."""

    def forward_ggml_cast_weights(self, input):
        w = getattr(self, "weight", None)
        name = _type_name(getattr(w, "tensor_type", None))
        # Demoted (gate-failed) or ineligible -> stock dequant path.
        if _DISABLE or getattr(self, "_fni8_demoted", False) or name not in _KQUANT:
            _mark_lora_fallback(self, "native GGUF path is disabled or ineligible")
            return base_forward(self, input)

        shape = getattr(w, "tensor_shape", None)
        if shape is None or len(shape) != 2:
            _mark_lora_fallback(self, "GGUF weight is not a two-dimensional linear")
            return base_forward(self, input)
        out_f, in_f = int(shape[0]), int(shape[1])
        if in_f % 256 != 0 or not _fni8_ok():
            self._fni8_demoted = True
            _mark_lora_fallback(self, "native GGUF kernel is unavailable for this shape")
            return base_forward(self, input)

        residuals, fallback_reason = _prepare_standard_lora_residuals(
            self, input, out_f, in_f
        )
        if fallback_reason is not None:
            self._fni8_lora_state = "fallback"
            self._fni8_lora_fallback_reason = fallback_reason
            self._fni8_lora_patch_count = 0
            self._fni8_demoted = True
            _LOG.info("fni8-gguf: %s; using patched dequant path", fallback_reason)
            return base_forward(self, input)
        self._fni8_lora_patch_count = len(residuals)
        self._fni8_lora_state = "native" if residuals else "none"

        import superl8
        import torch

        dev = input.device
        try:
            row_bytes = (in_f // 256) * _KQUANT[name][1]
            w_bytes = (
                torch.Tensor(w.to(dev)).contiguous().view(torch.uint8).reshape(-1)
                [: out_f * row_bytes].reshape(out_f, row_bytes)
            )
        except Exception as e:  # unexpected byte layout -> stay on stock path
            _LOG.warning("fni8-gguf: byte reshape failed (%s); using dequant path", e)
            self._fni8_demoted = True
            _mark_lora_fallback(self, "native GGUF byte layout is unsupported")
            return base_forward(self, input)

        # dequant the (small) bias through the stock machinery.
        bias = None
        if self.bias is not None:
            bias = self.get_weight(self.bias.to(dev), input.dtype)

        # One-time per-layer accuracy gate: fni8-dp4a vs the stock dequant reference.
        if getattr(self, "_fni8_sqnr_pass", None) is None:
            try:
                from .int8_linear import sqnr_calib_max_rows, sqnr_calib_row_indices

                flat_input = input.reshape(-1, in_f)
                row_indices = sqnr_calib_row_indices(
                    flat_input.shape[0], sqnr_calib_max_rows(), flat_input.device
                )
                calib_input = (
                    flat_input
                    if row_indices is None
                    else flat_input.index_select(0, row_indices)
                )
                ref = base_forward(self, calib_input).float()
                out = _native_linear(fni8, name, input, w_bytes, bias, input.dtype)
                out = _apply_standard_lora_residuals(
                    self, input, out, residuals
                ).float()
                flat_out = out.reshape(-1, out_f)
                calib_out = (
                    flat_out
                    if row_indices is None
                    else flat_out.index_select(0, row_indices)
                )
                cos = torch.nn.functional.cosine_similarity(
                    ref.flatten(), calib_out.flatten(), dim=0
                ).item()
                self._fni8_sqnr_pass = cos >= _COS_BAR
                if not self._fni8_sqnr_pass:
                    _LOG.info(
                        "fni8-gguf: layer demoted (%s cos=%.4f < %.3f) -> dequant path",
                        name, cos, _COS_BAR,
                    )
                    self._fni8_demoted = True
                    _mark_lora_fallback(self, "patched native output failed the SQNR gate")
                    return base_forward(self, input)
                return out.to(input.dtype)
            except Exception as e:
                _LOG.warning("fni8-gguf: gate error (%s); using dequant path", e)
                self._fni8_demoted = True
                _mark_lora_fallback(self, "patched native SQNR gate raised an error")
                return base_forward(self, input)

        output = _native_linear(fni8, name, input, w_bytes, bias, input.dtype)
        return _apply_standard_lora_residuals(self, input, output, residuals)

    return forward_ggml_cast_weights


def _import_gguf_ops():
    """Import ComfyUI-GGUF's ops module (the custom node must be installed)."""
    import importlib

    for mod in ("ComfyUI-GGUF.ops", "ComfyUI_GGUF.ops",
                "custom_nodes.ComfyUI_GGUF.ops",
                "custom_nodes.ComfyUI-GGUF.ops"):
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "GGMLOps") and hasattr(m.GGMLOps, "Linear"):
                return m
        except Exception:
            continue
    # Last resort: find it via the loaded custom node registry.
    import sys

    for name, m in list(sys.modules.items()):
        if name.endswith(".ops") and hasattr(m, "GGMLOps"):
            # Verify it's the real class (not a torch OpNamespace)
            gguf_cls = m.GGMLOps
            if isinstance(gguf_cls, type) and hasattr(gguf_cls, "Linear"):
                return m
    raise ImportError("ComfyUI-GGUF ops module not importable")


def enable_fni8_gguf():
    """Monkeypatch stock ``GGMLOps.Linear`` so ``UnetLoaderGGUF`` takes the fni8
    native k-quant path. Idempotent. Returns True if patched."""
    gguf_ops = _import_gguf_ops()
    Linear = gguf_ops.GGMLOps.Linear
    if getattr(Linear, "_fni8_patched", False):
        return True
    base = Linear.forward_ggml_cast_weights
    Linear.forward_ggml_cast_weights = _make_forward(base)
    Linear._fni8_patched = True
    _LOG.info("fni8-gguf: GGMLOps.Linear patched -> native fused dp4a k-quant")
    return True


def build_ops_class():
    """Return an ``FNI8GGUFOps`` subclass of ``GGMLOps`` with the fni8 Linear."""
    gguf_ops = _import_gguf_ops()

    class FNI8GGUFOps(gguf_ops.GGMLOps):
        class Linear(gguf_ops.GGMLOps.Linear):
            forward_ggml_cast_weights = _make_forward(
                gguf_ops.GGMLOps.Linear.forward_ggml_cast_weights
            )

    return FNI8GGUFOps
