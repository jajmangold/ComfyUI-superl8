# SPDX-License-Identifier: Apache-2.0
"""FNI8Ops — the `custom_operations` class ComfyUI builds every DiT Linear from.

Follows the ComfyUI-GGUF template (subclass `comfy.ops.manual_cast`, override the
Linear's cast-weight forward), but instead of dequantizing the weight to fp and
matmul'ing on the (gimped) tensor cores, it keeps the weight int8 and runs the fni8
W8A8 dp4a GEMM. A per-layer accuracy gate falls back to the fp CUDA-core path.

Only defined when ComfyUI is importable; the numerics live in `int8_linear` and are
tested independently.
"""

from __future__ import annotations

import torch

from ._comfy_compat import COMFY, MANUAL_CAST
from .superl8_tensor import FNI8Tensor
from .int8_linear import (
    LinearSqnrCalibrator,
    apply_patches_to_int8_weight,
    dequantize_qtensor_data,
    dp4a_linear_ok,
    in_features_for_scheme,
    int8_linear,
    resident_qtensor,
    sqnr_calib_max_rows,
    sqnr_calib_row_indices,
    sqnr_calib_samples,
)

if COMFY:

    class FNI8Ops(MANUAL_CAST):
        """int8 dp4a ops. Pass as `model_options={'custom_operations': FNI8Ops}`."""

        class Linear(MANUAL_CAST.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._fni8_sqnr_pass = None  # None=calibrating, True=pass, False=fail
                self._fni8_calib = None  # LinearSqnrCalibrator, lazily created

            def forward_comfy_cast_weights(self, x, *args, **kwargs):
                w = self.weight
                # Skip-listed linears (adaLN/modulation/head) load as plain fp
                # tensors -> ComfyUI's normal cast+matmul path. Only the quantized
                # FNI8Tensor linears take the dp4a path.
                if not isinstance(w, FNI8Tensor):
                    return super().forward_comfy_cast_weights(x, *args, **kwargs)

                # Resident weight is quantized offline into the .fni8 as one of two
                # dp4a-compatible schemes: `per_row_i8` (W8A8, int8 codes) or
                # `per_group_i4` (W4A8, uint8-packed nibbles, per-group scale). Both run
                # the SAME dp4a GEMM (sm_70 has no int4 matmul, so W4A8 unpacks nibbles
                # -> int8 in-register); `superl8.linear` dispatches on the scheme.
                scheme = getattr(w, "q_scheme", "per_row_i8")
                group_size = getattr(w, "q_group_size", 0)
                codebook = getattr(w, "q_codebook", "")

                # Gate 1: contraction dim must be %4 == 0 for dp4a. `per_group_i4` stores
                # K//2 bytes/row, so the real K is 2x the column count.
                if not dp4a_linear_ok(in_features_for_scheme(w.shape[1], scheme)):
                    return super().forward_comfy_cast_weights(x, *args, **kwargs)

                # Feed the resident codes + scale straight to dp4a; never dequantize on
                # the fast path. Normalize BOTH data and scale onto the activation's
                # device up front. ComfyUI's ModelPatcher moves the weight's code storage
                # to the compute device during load, but `q_scale` is a plain Python
                # attribute it doesn't know to move — so a resident FNI8Tensor can have
                # cuda data with a cpu scale. Forcing both here keeps every downstream use
                # (dp4a GEMM + the SQNR-gate dequant) on one device.
                data = w.int8_data().to(x.device)
                scale = w.q_scale.to(x.device)
                b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
                orig = x.shape[:-1]
                x_flat = x.reshape(-1, x.shape[-1])

                patches = self.weight_function
                if len(patches) > 0 and scheme != "per_row_i8":
                    # LoRA on a low-bit (per_group_i4) weight: the int8-domain accumulate
                    # only round-trips per_row_i8 cleanly. Fold the patches in the float
                    # domain and run the fp path — correct, if not the int4 fast path
                    # (LoRA on a sub-8-bit DiT is a rare combination).
                    w_fp = dequantize_qtensor_data(data, scale, scheme, group_size, torch.float32)
                    for f in patches:
                        w_fp = f(w_fp)
                    b32 = b.float() if b is not None else None
                    y = torch.nn.functional.linear(x_flat.float(), w_fp, b32)
                    return y.reshape(*orig, -1).to(x.dtype)
                if len(patches) > 0:
                    # per_row_i8 LoRA: merge via int8 accumulation, keep the dp4a path.
                    data, scale = apply_patches_to_int8_weight(data, scale, x.dtype, patches)

                # Already demoted to the fp fallback by the SQNR gate: run ONLY the fp32
                # path — do NOT execute the dp4a GEMM whose result the fallback would just
                # discard. On the Z-Image DiT the demoted layers are the large FFN `w2`
                # down-projections (K=10240; SQNR-gate failures under the published,
                # un-K-smoothed weight), so this skipped GEMM is a measurable slice of
                # every denoise step. Numerically identical to before — the emitted value
                # is exactly the fp32 fallback either way; only the wasted work is gone.
                if self._fni8_sqnr_pass is False:
                    y = self._fni8_fp_fallback(x_flat, data, scale, scheme, group_size, b)
                    return y.reshape(*orig, -1).to(x.dtype)

                qt = resident_qtensor(data, scale, scheme, group_size, codebook)
                # int8_linear stores in x's native dtype (fp16/bf16) — bf16-native DiTs
                # (Z-Image, Flux) must stay bf16 end to end or fp16 overflow -> black.
                y = int8_linear(x_flat, qt, b)

                # Gate 2: per-layer SQNR check (cos ≈ 1.0 bar, rel-L1 ≲ 0.02, per the
                # AGENTS.md numerics contract). Calibrated across MULTIPLE forwards
                # (denoise timesteps / resolutions) and finalized on the WORST cosine
                # seen — a DiT's activations shift along the denoise trajectory, and a
                # decision locked from the FIRST activation has missed a multi-step
                # collapse in this repo (a one-step latent cos of 1.0 hid a severe
                # multi-step image collapse; see docs/int8-dit-validation.md).
                # calib_samples=1 (default) keeps the legacy first-activation behavior;
                # set $FNI8_SQNR_CALIB_SAMPLES>1 to opt in. When LoRA patches have been
                # applied, the check uses the patched weight for a fair comparison.
                #
                # `_fni8_sqnr_pass` stays None until the calibrator LOCKS a verdict, so
                # this block keeps measuring while calibrating and is skipped once locked:
                # a locked False is served by the fast fp-skip above; a locked True just
                # runs the int8 GEMM already computed.
                if self._fni8_sqnr_pass is None:
                    if self._fni8_calib is None:
                        self._fni8_calib = LinearSqnrCalibrator(calib_samples=sqnr_calib_samples())
                    row_idx = sqnr_calib_row_indices(
                        x_flat.shape[0], sqnr_calib_max_rows(), x_flat.device
                    )
                    x_calib = x_flat if row_idx is None else x_flat.index_select(0, row_idx)
                    y_calib = y if row_idx is None else y.index_select(0, row_idx)
                    w_deq = dequantize_qtensor_data(data, scale, scheme, group_size, x.dtype)
                    y_fp = torch.nn.functional.linear(x_calib, w_deq, b)
                    use_int8 = self._fni8_calib.observe(y_calib, y_fp)
                    self._fni8_sqnr_pass = self._fni8_calib.passed
                    if not use_int8:
                        del x_calib, y_calib, y_fp, w_deq
                        y = self._fni8_fp_fallback(x_flat, data, scale, scheme, group_size, b)

                return y.reshape(*orig, -1).to(x.dtype)

            def _fni8_fp_fallback(self, x_flat, data, scale, scheme, group_size, b):
                # fp fallback runs on the fp32 CUDA cores — never the firmware-gimped
                # fp16 tensor cores, and NOT bf16. Measured on this fleet (torch
                # 2.10+cu129, real V100): on the DiT Linear shapes an fp32 matmul is
                # ~1.2-1.3x faster than an fp16/bf16 one (bf16 is 2-3x slower at small
                # batch and never wins here), and fp32 is the numerically-correct
                # fallback dtype. See the #146 fp16-matmul audit.
                w_deq = dequantize_qtensor_data(data, scale, scheme, group_size, torch.float32)
                b32 = b.float() if b is not None else None
                return torch.nn.functional.linear(x_flat.float(), w_deq, b32)
else:  # pragma: no cover

    class FNI8Ops:  # placeholder so imports don't explode outside ComfyUI
        def __init__(self, *a, **k):
            raise RuntimeError("FNI8Ops requires ComfyUI (comfy.ops).")
