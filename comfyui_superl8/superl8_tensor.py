# SPDX-License-Identifier: Apache-2.0
"""FNI8Tensor — an int8 DiT weight that quacks like a torch.Tensor.

ComfyUI's load/detect/patch pipeline (`comfy.utils.calculate_parameters`,
`model_detection`, `ModelPatcher`, device moves) calls `.nelement()` / `.shape` /
`.to()` on every weight in the state dict, so the quantized weight can't be a plain
dataclass (`superl8.QTensor` has no `.nelement()`). Same problem ComfyUI-GGUF solves
with its `GGMLTensor`: subclass `torch.Tensor` so the int8 [out,in] data carries the
tensor API natively, and attach the fp32 per-row scale as an extra field that
survives device moves / clones. It is NEVER dequantized to fp — `FNI8Ops.Linear`
reads `.q_scale` and runs the dp4a GEMM directly.
"""

from __future__ import annotations

import torch


# The quant-scheme metadata an FNI8Tensor carries alongside its data + scale, so the
# forward path can dispatch (per_row_i8 W8A8 vs per_group_i4 W4A8) without a QTensor.
# `int8_data()` returns whatever codes the scheme stores (int8 for W8A8, uint8-packed
# nibbles for W4A8) — the name is historical.
_FNI8_META = ("q_scheme", "q_group_size", "q_codebook")


class FNI8Tensor(torch.Tensor):
    """Quantized DiT weight as a torch.Tensor subclass: int8/uint8 ``[out, in]`` codes +
    fp32 scale, tagged with its quant scheme (``per_row_i8`` W8A8 or ``per_group_i4``
    W4A8) so ``FNI8Ops.Linear`` routes it to the right dp4a GEMM."""

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor | None = None,
        scheme: str = "per_row_i8",
        group_size: int = 0,
        codebook: str = "",
    ):
        # Share the code data's storage; shape/dtype/device/nelement come free.
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor | None = None,
        scheme: str = "per_row_i8",
        group_size: int = 0,
        codebook: str = "",
    ):
        self.q_scale = scale
        self.q_scheme = scheme
        self.q_group_size = group_size
        self.q_codebook = codebook

    def _meta(self) -> dict:
        """The scheme metadata as constructor kwargs (to re-tag a moved/cloned tensor)."""
        return {
            "scheme": getattr(self, "q_scheme", "per_row_i8"),
            "group_size": getattr(self, "q_group_size", 0),
            "codebook": getattr(self, "q_codebook", ""),
        }

    # -- keep the scale + int8-ness across the ops ComfyUI performs on weights ----
    def to(self, *args, **kwargs):
        # ComfyUI's cast path calls .to(dtype=<compute>, device=<gpu>); we must move
        # DEVICE but stay int8 (never cast the int8 codes to bf16). Move the scale too.
        device = kwargs.get("device")
        if device is None:
            for a in args:
                if isinstance(a, (torch.device, str)):
                    device = a
                    break
        dtype = kwargs.get("dtype")
        if dtype is None:
            for a in args:
                if isinstance(a, torch.dtype):
                    dtype = a
                    break
        # When a dtype is explicitly requested (e.g. eager LoRA patching in
        # ModelPatcher.patch_weight_to_device), dequantize to float so the calling
        # code can operate on a plain tensor without crashing on int codes + float.
        # Scheme-aware: per_group_i4 unpacks nibbles, per_row_i8 multiplies the row scale.
        if dtype is not None and dtype != self.dtype and self.q_scale is not None:
            from .int8_linear import dequantize_qtensor_data

            w = dequantize_qtensor_data(
                self.as_subclass(torch.Tensor), self.q_scale,
                self.q_scheme, self.q_group_size, dtype,
            )
            return w.to(device=device) if device is not None else w
        moved = super().to(device=device) if device is not None else self
        scale = (
            self.q_scale.to(device=device)
            if (self.q_scale is not None and device is not None)
            else self.q_scale
        )
        return FNI8Tensor(torch.as_tensor(moved), scale, **self._meta())

    def cuda(self, *args, **kwargs):
        return self.to(device=torch.device("cuda", *(args[:1])) if args else "cuda")

    def cpu(self, *args, **kwargs):
        return self.to(device="cpu")

    def clone(self, *args, **kwargs):
        # Strip to a plain tensor FIRST (`as_subclass`), else `torch.as_tensor(self)`
        # returns an FNI8Tensor and `.clone()` re-enters this method -> infinite
        # recursion (hit when ComfyUI's `load_state_dict(assign=True)` / ModelPatcher
        # device management clones the weight).
        return FNI8Tensor(
            self.as_subclass(torch.Tensor).clone(*args, **kwargs),
            None if self.q_scale is None else self.q_scale.clone(),
            **self._meta(),
        )

    def detach(self, *args, **kwargs):
        return FNI8Tensor(
            self.as_subclass(torch.Tensor).detach(), self.q_scale, **self._meta()
        )

    def int8_data(self) -> torch.Tensor:
        """The raw code [out,in] as a plain tensor (strip the subclass for the kernel):
        int8 for ``per_row_i8``, uint8-packed nibbles for ``per_group_i4``."""
        return self.as_subclass(torch.Tensor)

    def __repr__(self):  # pragma: no cover - debug aid
        s = None if self.q_scale is None else tuple(self.q_scale.shape)
        return f"FNI8Tensor(shape={tuple(self.shape)}, {self.q_scheme}, scale={s})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # Default torch dispatch, then re-attach the scale + scheme metadata from the
        # first FNI8Tensor argument so incidental ops (.contiguous(), indexing) don't
        # drop them (a per_group_i4 tensor stripped to per_row_i8 would mis-dispatch).
        kwargs = kwargs or {}
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, FNI8Tensor) and getattr(ret, "q_scale", None) is None:
            for a in args:
                if isinstance(a, FNI8Tensor) and a.q_scale is not None:
                    ret.q_scale = a.q_scale
                    ret.q_scheme = getattr(a, "q_scheme", "per_row_i8")
                    ret.q_group_size = getattr(a, "q_group_size", 0)
                    ret.q_codebook = getattr(a, "q_codebook", "")
                    break
        return ret
