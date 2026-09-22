# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the SAM 3D lane (issues: SAM 3D Objects, SAM 3D Body).

No CUDA / no gated checkpoint. Exercises the weight-independent pieces:
- `comfyui_superl8.sam3d_metrics` — MPJPE / PA-MPJPE / per-vertex / Chamfer / F-score.
- `comfyui_superl8.sam3d_encoder` — per-row int8 quant, SQNR gate, and the generalized
  backbone-linear patch's reversible swap + K%4 / SQNR fp-fallback routing (a fake `fni8`
  module is injected so the int8 branch runs on CPU).

The end-to-end fp-vs-int8 geometric proof on the real models (posed MHR mesh, 3D asset)
lives behind the HF gate + a Volta GPU — see `docs/sam3d-fni8-design.md`."""

from __future__ import annotations

import sys
import types

import torch

import comfyui_superl8.sam3d_encoder as se
import comfyui_superl8.sam3d_metrics as m


# ---------------- geometric metrics ----------------


def test_mpjpe_zero_on_identical():
    j = torch.randn(70, 3)
    assert m.mpjpe(j, j.clone()) == 0.0


def test_mpjpe_known_offset():
    j = torch.randn(70, 3)
    off = j + torch.tensor([3.0, 4.0, 0.0])  # each joint moved 5 units
    assert abs(m.mpjpe(off, j) - 5.0) < 1e-4


def test_pa_mpjpe_invariant_to_similarity_transform():
    torch.manual_seed(0)
    j = torch.randn(70, 3)
    # apply a rotation + translation + scale; PA-MPJPE must be ~0.
    theta = 0.7
    r = torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0],
            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta)), 0],
            [0, 0, 1],
        ]
    )
    moved = 2.5 * (j @ r.t()) + torch.tensor([10.0, -3.0, 4.0])
    assert m.pa_mpjpe(moved, j) < 1e-3
    # raw MPJPE is large — proving PA removed a real global transform.
    assert m.mpjpe(moved, j) > 1.0


def test_per_vertex_error_fields_and_hausdorff():
    a = torch.zeros(100, 3)
    b = torch.zeros(100, 3)
    b[0] = torch.tensor([0.0, 0.0, 7.0])  # one vertex off by 7
    e = m.per_vertex_error(a, b)
    assert abs(e["max_mm"] - 7.0) < 1e-4
    assert e["mean_mm"] < e["max_mm"]
    assert e["rms_mm"] > e["mean_mm"]  # rms >= mean, strict here (one big outlier)


def test_chamfer_zero_on_identical_and_symmetric():
    p = torch.randn(50, 3)
    assert m.chamfer_distance(p, p.clone()) < 1e-6
    q = torch.randn(40, 3)
    d_pq = m.chamfer_distance(p, q)
    d_qp = m.chamfer_distance(q, p)
    assert abs(d_pq - d_qp) < 1e-5  # symmetric


def test_chamfer_fscore_perfect_and_threshold():
    p = torch.randn(64, 3)
    # 1e-2 (not 1e-3): torch.cdist is matmul-based and leaves ~1e-3 float32 noise even
    # for identical point sets; 1e-2 is still a tiny threshold and gives a perfect score.
    assert m.chamfer_fscore(p, p.clone(), threshold=1e-2) == 1.0
    far = p + 100.0
    assert m.chamfer_fscore(p, far, threshold=1.0) == 0.0


# ---------------- int8 quant + SQNR gate ----------------


def test_quantize_per_row_recipe():
    torch.manual_seed(1)
    w = torch.randn(16, 64)
    q, scale = se._quantize_per_row_i8(w)
    assert q.dtype == torch.int8 and q.shape == w.shape
    assert scale.dtype == torch.float32 and scale.shape == (16,)
    expect = w.abs().amax(dim=1) / 127.0
    assert torch.allclose(scale, expect, atol=1e-6)
    # dequant round-trip error is bounded by half an lsb per element.
    deq = q.float() * scale.unsqueeze(1)
    assert (deq - w).abs().max() <= (scale.max() * 0.5 + 1e-6)


def test_sqnr_gate_semantics():
    y = torch.randn(8, 8)
    assert se._sqnr_db(y, y.clone()) == float("inf")
    assert se._passes(float("inf"), 20.0) is False  # self-fell-back -> route fp
    assert se._passes(float("-inf"), 20.0) is False


# ---------------- generalized backbone linear patch ----------------


class _TinyViT(torch.nn.Module):
    """Stand-in backbone: two blocks each with qkv/proj + MLP; plus a decoder head that
    must NOT be touched when only the backbone root is patched."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "qkv": torch.nn.Linear(64, 192),
                        "proj": torch.nn.Linear(64, 64),
                        "fc1": torch.nn.Linear(64, 256),
                        "fc2": torch.nn.Linear(256, 64),
                    }
                )
                for _ in range(2)
            ]
        )


def _install_fake_fni8(monkeypatch):
    """Inject a minimal fake `fni8` (just `linear_w8a8`) so the int8 branch runs on CPU.

    Uses `monkeypatch.setitem` so `sys.modules["fni8"]` is AUTO-RESTORED at test teardown
    — a bare `sys.modules["fni8"] = fake` would leak a QTensor-less module into every later
    test in the process (e.g. test_te_int8's `from superl8 import QTensor`)."""
    fake = types.ModuleType("superl8")

    def linear_w8a8(x, w_i8, w_scale, bias=None, out_dtype=torch.float16):
        y = x.to(torch.float32) @ (w_i8.float() * w_scale.unsqueeze(1)).t()
        if bias is not None:
            y = y + bias.to(torch.float32)
        return y.to(out_dtype)

    fake.linear_w8a8 = linear_w8a8
    monkeypatch.setitem(sys.modules, "superl8", fake)
    return fake


def test_backbone_linear_patch_wraps_only_root_and_is_reversible():
    torch.manual_seed(2)
    vit = _TinyViT()
    n_lin = sum(isinstance(mm, torch.nn.Linear) for mm in vit.modules())
    assert n_lin == 8  # 4 linears x 2 blocks
    h = se.patch_backbone_linears_int8(vit)
    assert h.stats.n_wrapped == 8
    assert all(
        isinstance(getattr(vit.blocks[i], k), se.Int8LinearShim)
        for i in range(2)
        for k in ("qkv", "proj", "fc1", "fc2")
    )
    h.unpatch()
    assert all(
        isinstance(getattr(vit.blocks[i], k), torch.nn.Linear)
        for i in range(2)
        for k in ("qkv", "proj", "fc1", "fc2")
    )


def test_int8_shim_forward_matches_fp_within_sqnr_and_routes(monkeypatch):
    torch.manual_seed(3)
    _install_fake_fni8(monkeypatch)
    lin = torch.nn.Linear(64, 128)  # K=64, divisible by 4
    stats = se.LinearStats()
    shim = se.Int8LinearShim(lin, floor_db=20.0, stats=stats)
    x = torch.randn(4, 64)
    ref = torch.nn.functional.linear(x, lin.weight, lin.bias)
    out = shim(x)
    # int8 GEMM output close to fp; per-row int8 of a random weight clears 20 dB.
    assert se._sqnr_db(out, ref) > 20.0
    assert stats.calls_int8 == 1 and stats.calls_fp == 0
    assert shim._sqnr_pass is True


def test_int8_shim_k_not_mult4_stays_fp(monkeypatch):
    _install_fake_fni8(monkeypatch)
    lin = torch.nn.Linear(63, 32)  # K=63, NOT divisible by 4 -> dp4a ineligible
    stats = se.LinearStats()
    shim = se.Int8LinearShim(lin, floor_db=20.0, stats=stats)
    x = torch.randn(2, 63)
    out = shim(x)
    ref = torch.nn.functional.linear(x, lin.weight, lin.bias)
    assert torch.allclose(out, ref, atol=1e-5)  # bit-for-bit fp path
    assert stats.calls_fp == 1 and stats.calls_int8 == 0


# ---------------- global SDPA patch (fully-qualified F.sdpa call sites) ----------------


def test_global_sdpa_patch_delegates_and_restores():
    """The global patch handles backbones (e.g. DINOv3) that call
    `torch.nn.functional.scaled_dot_product_attention` fully-qualified. On CPU tensors the
    int8 branch is ineligible (not cuda) so it must delegate to the real SDPA bit-for-bit,
    and `.unpatch()` must restore the exact original function object."""
    import torch.nn.functional as F

    orig = F.scaled_dot_product_attention
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    ref = orig(q, k, v)

    h = se.patch_global_sdpa_int8()
    assert F.scaled_dot_product_attention is not orig  # actually swapped
    out = F.scaled_dot_product_attention(q, k, v)  # CPU -> ineligible -> delegates
    assert torch.allclose(out, ref, atol=1e-6)
    assert h.gate.n_fallback == 1 and h.gate.n_int8 == 0

    h.unpatch()
    assert F.scaled_dot_product_attention is orig  # exact restore
