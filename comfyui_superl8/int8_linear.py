# SPDX-License-Identifier: Apache-2.0
"""Pure int8 W8A8 dp4a linear — the compute core of the ComfyUI ops patch.

No ComfyUI dependency (imports only `fni8`), so the numerics are unit-testable on
their own. A DiT `Linear` becomes: quantize the activation per-row to int8, keep the
weight int8, run the dp4a GEMM. This is strictly better than the GGUF route (which
dequantizes the weight back to fp and runs an fp matmul on the gimped tensor cores).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from superl8 import QTensor

_log = logging.getLogger(__name__)


def quantize_linear_weight(weight: torch.Tensor) -> "QTensor":
    """fp16/fp32 DiT linear weight [out, in] (in % 4 == 0) -> per_row_i8 QTensor.
    Done once at load; the resident weight stays int8 (half the HBM of fp16)."""
    from superl8 import QTensor
    from superl8.quant.core import quantize_int8_rowwise

    q, s = quantize_int8_rowwise(weight)
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def int8_linear(x: torch.Tensor, w: "QTensor", bias: torch.Tensor | None = None) -> torch.Tensor:
    """W8A8 dp4a linear. x fp16/bf16 [..., K]; w a per_row_i8 QTensor [N, K]. Returns
    the same float dtype as `x`, [..., N]. The activation is quantized per-row inside
    `superl8.linear`; native dtype in, native dtype out — no forced fp16 downcast.

    The dp4a store dtype must be *requested* explicitly: `superl8.linear` defaults to
    fp16, so bf16-native DiTs (Z-Image, Flux) would silently downcast (clipping
    bf16's wider range -> black images) unless we forward `out_dtype`. Only fp16/bf16
    are valid store dtypes; an fp32 activation (the Qwen-Image-Edit DiT runs its
    residual/MLP in fp32, reaching ~1e7) must NOT fall back to fp16 — the MLP
    down-projection sums thousands of terms and its output overflows fp16's 65504 ->
    inf -> NaN (observed at `block0.img_mlp.net.2`, the single-card e2e blocker). Store
    bf16 (fp32's exponent range, no overflow) and let the caller cast back to fp32."""
    import superl8

    out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    return superl8.linear(x, w, bias=bias, out_dtype=out_dtype)


def dp4a_linear_ok(in_features: int) -> bool:
    """The dp4a GEMM needs the contraction dim %4==0. DiT hidden dims always are,
    but small projection tails may not — gate and fp-fall-back if so."""
    return in_features % 4 == 0


def dequantize_weight(
    int8_data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize a per-row int8 weight to fp: w_fp = int8 * scale. Returns in
    `dtype` (fp16/bf16) at the data's device.

    `scale` is co-located onto the int8 data's device first: ComfyUI moves the int8
    codes to the compute device during load but `q_scale` is a plain attribute it
    doesn't know to move, so a resident FNI8Tensor can pair cuda data with a cpu scale
    (e.g. Z-Image's `cap_pad_token` embedding buffer hitting this via
    `FNI8Tensor.to(dtype=...)` inside `pad_zimage`). Without this the multiply raises
    'expected all tensors on the same device'."""
    scale = scale.to(device=int8_data.device, dtype=torch.float32)
    w = int8_data.to(torch.float32) * scale.view(-1, 1)
    return w.to(dtype)


def apply_patches_to_int8_weight(
    int8_data: torch.Tensor,
    q_scale: torch.Tensor,
    weight_dtype: torch.dtype,
    patch_fns: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply float domain patch functions (LoRA) to an int8-weight representation.

    Dequantizes the weight to float, runs each patch function in order, then
    re-quantizes using the *original* per-row scale so the result stays int8 and
    the dp4a GEMM path is preserved.  Returns (new_int8_data, q_scale) where the
    scale is unchanged from the input.
    """
    w_fp = dequantize_weight(int8_data, q_scale, weight_dtype)
    for f in patch_fns:
        w_fp = f(w_fp)
    scale = q_scale.to(w_fp.device, dtype=torch.float32)
    new_int8 = torch.round(w_fp / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return new_int8, q_scale.to(new_int8.device)


def sqnr_gate(y_int8: torch.Tensor, y_fp: torch.Tensor, cos_bar: float = 0.99) -> bool:
    """Returns True when the per-layer int8 linear is accurate enough vs fp."""
    cos = torch.nn.functional.cosine_similarity(
        y_int8.float().flatten(), y_fp.float().flatten(), dim=0
    )
    return cos.item() >= cos_bar


def sqnr_calib_samples(default: int = 1) -> int:
    """Number of forwards to sample before the per-linear int8/fp gate locks its
    decision, from ``$FNI8_SQNR_CALIB_SAMPLES`` (default 1 == legacy first-activation
    gate). Set >1 (e.g. your denoise step count) to calibrate across the trajectory and
    decide on the WORST activation — see `LinearSqnrCalibrator`."""
    import os

    try:
        return max(1, int(os.environ.get("FNI8_SQNR_CALIB_SAMPLES", default)))
    except (TypeError, ValueError):
        return default


def sqnr_calib_max_rows(default: int = 256) -> int:
    """Maximum activation rows used for a linear's fp calibration reference.

    A full DiT activation can contain tens of thousands of rows and make the
    temporary fp output hundreds of MiB. A deterministic spread of rows keeps
    the cosine gate representative while bounding calibration memory.
    """
    import os

    try:
        return max(1, int(os.environ.get("FNI8_SQNR_CALIB_MAX_ROWS", default)))
    except (TypeError, ValueError):
        return default


def sqnr_calib_row_indices(num_rows: int, max_rows: int, device) -> torch.Tensor | None:
    """Return evenly spread deterministic row indices, or ``None`` for all rows."""
    if num_rows <= max_rows:
        return None
    if max_rows == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    return torch.div(
        torch.arange(max_rows, device=device, dtype=torch.long) * (num_rows - 1),
        max_rows - 1,
        rounding_mode="floor",
    )


class LinearSqnrCalibrator:
    """Multi-timestep worst-case SQNR calibration for ONE int8 Linear instance.

    The per-layer gate must NOT lock its int8/fp decision from the first activation it
    sees: a DiT's activation distribution shifts across the denoising trajectory
    (early/mid/late timesteps) and across resolutions, and a one-step calibration has
    MISSED a multi-step collapse in this very repo — a one-step latent cosine of 1.0 hid
    a severe multi-step image collapse (`docs/int8-dit-validation.md`). This gathers up
    to ``calib_samples`` cosine measurements (one per forward, so successive denoise
    steps at possibly different resolutions are all sampled) and finalizes the decision
    CONSERVATIVELY on the WORST cosine seen. ``calib_samples == 1`` reproduces the legacy
    first-activation behavior, so enabling this is opt-in and backward-compatible.

    Pure/torch-only (no ComfyUI) so it is unit-testable independent of `ops.py`."""

    def __init__(self, calib_samples: int = 1, cos_bar: float = 0.99):
        self.calib_samples = max(1, int(calib_samples))
        self.cos_bar = cos_bar
        self.n = 0
        self.worst_cos = float("inf")
        self.passed: bool | None = None  # finalized decision (None while calibrating)

    def needs_measurement(self) -> bool:
        """True while the fp reference must still be computed (calibration not done)."""
        return self.passed is None

    def observe(self, y_int8: torch.Tensor, y_fp: torch.Tensor) -> bool:
        """Measure one activation's int8-vs-fp cosine, fold it into the running worst,
        and finalize once ``calib_samples`` are gathered. Returns the effective decision
        to use for THIS forward (worst-so-far while calibrating, the locked verdict once
        finalized)."""
        cos = torch.nn.functional.cosine_similarity(
            y_int8.float().flatten(), y_fp.float().flatten(), dim=0
        ).item()
        self.worst_cos = min(self.worst_cos, cos)
        self.n += 1
        verdict = self.worst_cos >= self.cos_bar
        if self.n >= self.calib_samples:
            self.passed = verdict
        return verdict


# ---------- W4A8 (int4 weight, int8 activation) ----------
#
# Per-group int4 weight quantization: each group of `W4_GROUP_SIZE` columns shares a
# scale. Two int4 values are packed per byte (low nibble = even col, high nibble = odd).
# fni8's W4A8 dp4a GEMM unpacks the nibbles in-register (sm_70 has no int4 matmul), so no
# full-weight fp temporary is allocated and compute stays at the W8A8 dp4a rate while the
# resident weight is half the HBM of int8.

W4_GROUP_SIZE = 128


def quantize_weight_int4(
    weight: torch.Tensor, group_size: int = W4_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] fp weight -> per-group int4, packed.

    Returns ``(packed_i4 [N, K//2], scale_f32 [N, K//group_size])``.
    Each byte stores two int4 values (low nibble = col i, high nibble = col i+1).
    Symmetric int4 range [-7, 7] with scale = max_abs / 7 (matches the resident
    ``.fni8`` ``per_group_i4`` blob fni8's ``gemm_w4a8`` kernel consumes)."""
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be a multiple of group_size={group_size}"
    assert K % 2 == 0, f"K={K} must be even for byte packing"
    G = K // group_size

    w = weight.float()
    w_g = w.view(N, G, group_size)
    amax = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0
    q = (w_g / scale).round().clamp(-7, 7).to(torch.int8)

    q_flat = q.view(N, K)
    lo = (q_flat[:, 0::2] & 0x0F).to(torch.uint8)
    hi = ((q_flat[:, 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = lo | hi

    return packed.contiguous(), scale.squeeze(-1).float().contiguous()


def dequantize_int4_weight(
    packed_i4: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    N: int,
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack + dequantize an int4 weight to fp.

    ``packed_i4 [N, K//2]`` uint8, ``scale [N, K//group_size]`` -> fp weight [N, K]."""
    lo = (packed_i4 & 0x0F).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = ((packed_i4 >> 4) & 0x0F).to(torch.int8)
    hi = torch.where(hi > 7, hi - 16, hi)

    q_flat = torch.stack([lo, hi], dim=-1).view(N, K).float()
    scale_exp = scale.unsqueeze(-1).expand(-1, -1, group_size).reshape(N, K)
    w_fp = q_flat * scale_exp
    return w_fp.to(dtype)


def w4a8_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
    w_shape: tuple[int, int],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W4A8 dp4a linear via fni8's in-register int4->int8 unpack + GEMM.

    x fp16/bf16 [..., K]; w_packed uint8 [N, K//2]; w_scale fp32 [N, K//group_size].

    Falls back to a Python int4->int8 re-quant + W8A8 GEMM ONLY when the installed fni8
    predates the native ``gemm_w4a8`` kernel (backward compat with W8A8-only builds).
    The ``per_group_i4`` QTensor carries ``codebook="int4"`` — without it superl8.linear
    rejects the weight (nf4 is a non-integer codebook, not dp4a-compatible)."""
    import superl8
    from superl8 import QTensor

    out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16

    # Native W4A8: route the packed int4 weight straight to fni8's gemm_w4a8 dp4a kernel
    # (verified on V100: 24/24, ~41 dB). Any dispatch/config error from a W4A8-capable
    # fni8 now propagates instead of being silently demoted to the slow fallback.
    if hasattr(getattr(fni8, "_C", None), "gemm_w4a8"):
        qt = QTensor(
            w_packed,
            w_scale,
            scheme="per_group_i4",
            group_size=group_size,
            codebook="int4",
        )
        return superl8.linear(x, qt, bias=bias, out_dtype=out_dtype)

    # Fallback: this fni8 build has no int4 GEMM. Unpack int4 -> int8 per-row and run the
    # W8A8 dp4a path — lossless re-quant but ~2x slower (no in-kernel int4 unpack, and a
    # temporary full-width int8 weight). Loud so it never masquerades as the fast path.
    _log.warning(
        "fni8 build lacks the gemm_w4a8 W4A8 kernel; using the slower int4->int8 "
        "W8A8 fallback for this linear."
    )
    N, K = w_shape
    w_i8, s_i8 = _int4_packed_to_int8_rowwise(w_packed, w_scale, group_size, N, K)
    qt_i8 = QTensor(w_i8, s_i8, scheme="per_row_i8")
    return superl8.linear(x, qt_i8, bias=bias, out_dtype=out_dtype)


def _int4_packed_to_int8_rowwise(
    packed_i4: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    N: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack int4 -> dequantize -> re-quantize int8 per-row. Returns
    ``(int8_data [N, K], per_row_scale [N])`` for scheme='per_row_i8'. Used only as the
    fallback when the installed fni8 predates the native W4A8 kernel."""
    w_fp = dequantize_int4_weight(packed_i4, scale, group_size, N, K, torch.float32)
    row_max = w_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    row_scale = row_max / 127.0
    w_i8 = (w_fp / row_scale).round().clamp(-128, 127).to(torch.int8)
    return w_i8.contiguous(), row_scale.squeeze(-1).float().contiguous()


# ---------- scheme-aware helpers (per_row_i8 + per_group_i4 in ONE dispatch) ----------
#
# The resident weight is an `FNI8Tensor` carrying (data, scale, scheme, group_size).
# `per_row_i8` keeps int8 codes + one fp32 scale/row; `per_group_i4` keeps uint8-packed
# nibbles + an fp32 scale/(row, group). The DiT forward (`FNI8Ops.Linear`), the SQNR
# gate and `FNI8Tensor.to(dtype=...)` all need the SAME dequant/reconstruct per scheme —
# these two helpers are that single dispatch so no call site hard-codes per_row_i8.

_INT8_IN_FEATURES = {"per_row_i8": 1, "per_group_i4": 2}  # cols -> real K multiplier


def in_features_for_scheme(cols: int, scheme: str) -> int:
    """Real contraction dim K from a resident weight's stored column count. `per_group_i4`
    packs two 4-bit weights per byte, so K = stored_cols * 2; `per_row_i8` stores K."""
    return cols * _INT8_IN_FEATURES.get(scheme, 1)


def dequantize_qtensor_data(
    data: torch.Tensor,
    scale: torch.Tensor,
    scheme: str,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Scheme-aware dequant of a resident ``(data, scale)`` weight to fp ``[N, K]``.

    ``per_row_i8`` -> :func:`dequantize_weight` (int8 * per-row scale); ``per_group_i4``
    -> unpack the packed nibbles + apply the per-group scale
    (:func:`dequantize_int4_weight`). One dequant for the SQNR gate, the fp fallback and
    ``FNI8Tensor.to(dtype=...)``."""
    if scheme == "per_group_i4":
        N, cols = data.shape
        return dequantize_int4_weight(
            data,
            scale.to(device=data.device, dtype=torch.float32),
            group_size,
            N,
            cols * 2,
            dtype,
        )
    return dequantize_weight(data, scale, dtype)


def resident_qtensor(
    data: torch.Tensor, scale: torch.Tensor, scheme: str, group_size: int, codebook: str
) -> "QTensor":
    """Rebuild the fni8 :class:`QTensor` a resident weight represents, for ``superl8.linear``.

    ``per_group_i4`` MUST carry ``codebook='int4'`` (fni8 rejects nf4 for dp4a); default
    to it when the loader did not record one. All other schemes reconstruct as
    ``per_row_i8``."""
    from superl8 import QTensor

    if scheme == "per_group_i4":
        return QTensor(
            data,
            scale,
            scheme="per_group_i4",
            group_size=group_size,
            codebook=codebook or "int4",
        )
    return QTensor(data, scale, scheme="per_row_i8")
