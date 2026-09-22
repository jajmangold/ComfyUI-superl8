# SPDX-License-Identifier: Apache-2.0
"""Forward-runs proof: the REAL LTX-2.3 b4 (per_group_i4) DiT linears execute through
the ComfyUI-superl8 W4A8 dp4a path.

This is the critical enabler for the single-card LTX shrink (#80): before this, the
comfy DiT `Linear` only accepted `per_row_i8` (W8A8) and the shipped `per_group_i4`
(4-bit) b4 checkpoint could not run at all. The full 4-bit DiT is still large, so this
does NOT load the whole model — it pulls the real per_group_i4 weights of ONE
transformer block straight out of `Lightricks__LTX-2.3.dit.b4.fni8` and drives each one
through `FNI8Ops.Linear`, proving the shipped 4-bit weights FORWARD on the real dp4a
kernel and match the fp reference at the comfy latent gate (cos >= 0.985).

Needs a Volta/CMP (sm_70) GPU, a W4A8-capable fni8, and the mounted b4 `.fni8`.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("comfy")
pytest.importorskip("superl8")

import torch  # noqa: E402

from comfyui_superl8.superl8_tensor import FNI8Tensor  # noqa: E402
from comfyui_superl8.gate import is_sm70  # noqa: E402
from comfyui_superl8.int8_linear import dequantize_int4_weight  # noqa: E402
from comfyui_superl8.ops import FNI8Ops  # noqa: E402

pytestmark = pytest.mark.comfy_e2e

# The b4 (per_group_i4) LTX-2.3 DiT, mounted at the fleet-runner weights path.
B4_PATH = os.path.join(
    os.environ.get("FNI8_WEIGHTS_DIR", ""),
    "Lightricks__LTX-2.3.dit.b4.fni8",
)
BLOCK = "model.diffusion_model.transformer_blocks.0."


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.fixture(scope="module")
def block0_i4_linears():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    from superl8 import _C

    if not hasattr(_C, "gemm_w4a8"):
        pytest.skip("installed fni8 lacks the gemm_w4a8 W4A8 kernel")
    if not os.path.exists(B4_PATH):
        pytest.skip(f"{B4_PATH} not mounted")

    from superl8 import FQReader

    out = {}
    with FQReader(B4_PATH) as r:
        for name in r.names:
            if not name.startswith(BLOCK) or not name.endswith(".weight"):
                continue
            e = r.header["tensors"][name]
            if e.get("scheme") != "per_group_i4":
                continue
            out[name[len(BLOCK):]] = r.get_qtensor(name, device="cpu")
    if not out:
        pytest.skip("no per_group_i4 linears found in LTX-2.3 b4 block 0")
    return out


def test_ltx23_b4_block0_linears_forward_and_match_fp(block0_i4_linears):
    """Every real per_group_i4 linear in LTX-2.3 b4 block 0 forwards through the W4A8
    dp4a path with finite output and cos >= 0.985 vs its fp-dequant reference."""
    dev = torch.device("cuda")
    dtype = torch.bfloat16  # LTX-2.3 native compute dtype
    g = torch.Generator().manual_seed(0)

    checked = 0
    worst_cos = 1.0
    for name, qt in sorted(block0_i4_linears.items()):
        N, cols = qt.data.shape
        K = cols * 2  # per_group_i4 packs two nibbles per byte
        assert K % 4 == 0, f"{name}: contraction dim {K} not dp4a-eligible"

        lin = FNI8Ops.Linear(K, N, bias=False, dtype=dtype, device=dev)
        lin._parameters["weight"] = FNI8Tensor(
            qt.data.to(dev), qt.scale.to(dev),
            scheme="per_group_i4", group_size=qt.group_size, codebook=qt.codebook,
        )

        x = (torch.randn(32, K, generator=g) * 0.5).to(dev, dtype)
        y = lin.forward_comfy_cast_weights(x)
        assert torch.isfinite(y).all(), f"{name}: non-finite W4A8 output"
        assert y.shape == (32, N), f"{name}: bad output shape {tuple(y.shape)}"

        w_deq = dequantize_int4_weight(
            qt.data.to(dev), qt.scale.to(dev), qt.group_size, N, K, dtype
        )
        y_ref = torch.nn.functional.linear(x, w_deq, None)
        cos = _cos(y, y_ref)
        assert cos >= 0.985, f"{name}: W4A8 diverged from fp, cos {cos:.4f}"
        assert lin._fni8_sqnr_pass is True, f"{name}: demoted off the dp4a path"
        worst_cos = min(worst_cos, cos)
        checked += 1

    assert checked >= 5, f"expected several i4 linears, only exercised {checked}"
    print(f"\nLTX-2.3 b4 block0: {checked} per_group_i4 linears forwarded "
          f"through W4A8 dp4a; worst cos {worst_cos:.4f}")
