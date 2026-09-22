# SPDX-License-Identifier: Apache-2.0
"""Runtime GPU-arch gate. The fni8 kernels are compiled sm_70 (Volta) only — dp4a on
the CUDA cores, no int8 tensor cores. On other archs we must fall back (and never onto
the CMP fleet's firmware-crippled fp16 tensor cores). Borrowed from the nunchaku
runtime-probe pattern."""
from __future__ import annotations


def is_sm70(device=0) -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) == (7, 0)


def require_sm70(device=0):
    if not is_sm70(device):
        import torch

        cap = torch.cuda.get_device_capability(device) if torch.cuda.is_available() else None
        raise RuntimeError(
            f"ComfyUI-superl8 kernels are sm_70 (Volta) only; this device is {cap}. "
            "The int8 dp4a path won't run here — use a Volta/CMP-100-210 GPU or the fp path."
        )
