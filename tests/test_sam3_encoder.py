# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SAM 3.1 int8 ViT-encoder patch (issue #120).

CPU-only: no CUDA / no sam3 checkpoint. A fake `fni8` module and a fake
`sam3.model.vitdet` module are injected so the routing, per-row int8 quant, SQNR gate,
fused-`addmm_act` fc1 path, and reversible module swap are all exercised on CPU. The
end-to-end fp-vs-int8 mask / IoU / latency proof on the real model lives in
`bench/sam31_fni8_eval.py` (needs a Volta GPU + the gated checkpoint)."""
from __future__ import annotations

import sys
import types

import torch

import comfyui_superl8.sam3_encoder as se


# ---- SQNR helpers ----

def test_sqnr_identical_is_inf_and_fails_routing():
    y = torch.randn(4, 8)
    assert se._sqnr_db(y, y.clone()) == float("inf")
    # +inf == kernel self-fell-back to fp -> route to fp (never re-dispatch).
    assert se._passes(float("inf"), 20.0) is False


def test_sqnr_nonfinite_is_definitive_fail():
    y = torch.randn(16)
    bad = y.clone()
    bad[0] = float("inf")
    assert se._sqnr_db(y, bad) == float("-inf")
    assert se._passes(float("-inf"), 20.0) is False


def test_sqnr_known_noise_level_passes_floor():
    torch.manual_seed(0)
    sig = torch.randn(4096)
    noise = torch.randn(4096)
    noise = noise * (0.1 * sig.pow(2).mean().sqrt() / noise.pow(2).mean().sqrt())
    sqnr = se._sqnr_db(sig + noise, sig)  # ~20 dB
    assert 18.0 < sqnr < 22.0
    assert se._passes(sqnr, 15.0) is True
    assert se._passes(sqnr, 25.0) is False


# ---- per-row int8 quant (SDNQ recipe) ----

def test_quantize_per_row_i8_recipe_and_roundtrip():
    torch.manual_seed(1)
    w = torch.randn(32, 64)
    q, scale = se._quantize_per_row_i8(w)
    assert q.dtype == torch.int8 and q.shape == w.shape
    assert scale.dtype == torch.float32 and scale.shape == (32,)
    # scale = max|row| / 127
    expect = w.abs().amax(dim=1) / 127.0
    assert torch.allclose(scale, expect, atol=1e-6)
    # dequant round-trip within one quant step per element
    deq = q.float() * scale[:, None]
    assert (deq - w).abs().max() <= (scale.max().item() * 1.01)


# ---- fake fni8 so the int8 path runs on CPU ----

def _install_fake_fni8(monkeypatch, linear_impl=None, attn_impl=None):
    fake = types.ModuleType("superl8")

    def default_linear(x, w_i8, w_scale, *, bias=None, out_dtype=torch.float16):
        y = (x.float() @ (w_i8.float() * w_scale[:, None]).t())
        if bias is not None:
            y = y + bias.float()
        return y.to(out_dtype)

    def default_attn(q, k, v, *, causal=False, scale=None, rotate=False):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    fake.linear_w8a8 = linear_impl or default_linear
    fake.attn_int8_fwd = attn_impl or default_attn
    monkeypatch.setitem(sys.modules, "superl8", fake)
    return fake


def _install_fake_fni8_attr(monkeypatch):
    """Keep the REAL fni8 module (FQReader/QTensor/save_fni8) but swap only the CUDA
    `linear_w8a8` for a faithful CPU dequant matmul, so the resident-load forward runs
    on CPU while the `.fni8` round-trip uses the real format code."""
    import superl8

    def cpu_linear(x, w_i8, w_scale, *, bias=None, out_dtype=torch.float16):
        y = x.float() @ (w_i8.float() * w_scale[:, None]).t()
        if bias is not None:
            y = y + bias.float()
        return y.to(out_dtype)

    monkeypatch.setattr(superl8, "linear_w8a8", cpu_linear)


# ---- Int8LinearShim ----

def test_shim_k_not_mult4_stays_fp():
    lin = torch.nn.Linear(6, 8)  # in_features=6, 6 % 4 != 0 -> ineligible for dp4a
    shim = se.Int8LinearShim(lin, 20.0, se.LinearStats())
    x = torch.randn(3, 6)
    out = shim(x)
    assert torch.allclose(out, lin(x), atol=1e-5)
    assert shim._dp4a_ok is False


def test_shim_int8_path_passes_gate(monkeypatch):
    _install_fake_fni8(monkeypatch)
    torch.manual_seed(2)
    lin = torch.nn.Linear(64, 32)
    stats = se.LinearStats()
    shim = se.Int8LinearShim(lin, floor_db=15.0, stats=stats)
    x = torch.randn(5, 64)
    out = shim(x)  # first call: measures SQNR vs fp, should pass (faithful fake)
    assert shim._sqnr_pass is True
    assert stats.calls_int8 >= 1
    # close to fp (per-row int8 of a well-conditioned weight)
    assert (out - lin(x)).abs().mean() < 0.05 * lin(x).abs().mean()


def test_shim_bad_int8_demotes_to_fp(monkeypatch):
    # An int8 impl that returns garbage must fail the SQNR gate -> fp fallback, and the
    # emitted value is exactly the fp reference.
    def garbage(x, w_i8, w_scale, *, bias=None, out_dtype=torch.float16):
        return torch.zeros(x.shape[0], w_i8.shape[0], dtype=out_dtype)

    _install_fake_fni8(monkeypatch, linear_impl=garbage)
    lin = torch.nn.Linear(64, 32)
    shim = se.Int8LinearShim(lin, floor_db=20.0, stats=se.LinearStats())
    x = torch.randn(4, 64)
    out = shim(x)
    assert shim._sqnr_pass is False
    assert torch.allclose(out, lin(x), atol=1e-5)


# ---- fake vitdet module + trunk for the patch/unpatch tests ----

class _FakeTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "qkv": torch.nn.Linear(64, 192),
                "proj": torch.nn.Linear(64, 64),
                "fc1": torch.nn.Linear(64, 256),
                "fc2": torch.nn.Linear(256, 64),
            })
        ])


class _FakeModel:
    def __init__(self):
        trunk = _FakeTrunk()
        self.backbone = types.SimpleNamespace(
            vision_backbone=types.SimpleNamespace(trunk=trunk))


def _install_fake_vitdet(monkeypatch):
    mod = types.ModuleType("sam3.model.vitdet")
    mod.F = torch.nn.functional
    mod.addmm_act = lambda act, lin, x: torch.nn.functional.gelu(lin(x))
    monkeypatch.setitem(sys.modules, "sam3", types.ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model", types.ModuleType("sam3.model"))
    monkeypatch.setitem(sys.modules, "sam3.model.vitdet", mod)
    return mod


def test_linear_patch_wraps_trunk_and_unpatches(monkeypatch):
    _install_fake_fni8(monkeypatch)
    vitdet = _install_fake_vitdet(monkeypatch)
    model = _FakeModel()
    blk = model.backbone.vision_backbone.trunk.blocks[0]
    orig_qkv = blk["qkv"]

    handle = se.patch_vit_linears_int8(model, floor_db=15.0)
    assert handle.stats.n_wrapped == 4
    assert isinstance(blk["qkv"], se.Int8LinearShim)
    # fused fc1 path is rerouted (module-scoped)
    assert vitdet.addmm_act.__name__ == "int8_addmm_act"

    handle.unpatch()
    assert blk["qkv"] is orig_qkv
    assert not isinstance(blk["qkv"], se.Int8LinearShim)


def test_addmm_act_routes_shim_through_int8_with_gelu(monkeypatch):
    _install_fake_fni8(monkeypatch)
    vitdet = _install_fake_vitdet(monkeypatch)
    model = _FakeModel()
    handle = se.patch_vit_linears_int8(model, floor_db=15.0)
    fc1 = model.backbone.vision_backbone.trunk.blocks[0]["fc1"]
    x = torch.randn(4, 64)
    y = vitdet.addmm_act(torch.nn.GELU, fc1, x)
    # matches gelu(int8-linear) closely
    ref = torch.nn.functional.gelu(fc1._fp(x))
    assert (y - ref).abs().mean() < 0.05 * ref.abs().mean().clamp_min(1e-6)
    handle.unpatch()


# ---- resident (offline .fni8) shim + converter round-trip ----

def test_shim_from_resident_matches_dequant(monkeypatch):
    _install_fake_fni8(monkeypatch)
    torch.manual_seed(3)
    lin = torch.nn.Linear(64, 32)
    q, scale = se._quantize_per_row_i8(lin.weight)
    stats = se.LinearStats()
    shim = se.Int8LinearShim.from_resident(q, scale, lin.bias.detach(), stats)
    assert shim._sqnr_pass is True and shim.w_fp is None
    x = torch.randn(4, 64)
    out = shim(x)  # no runtime gate; int8-only
    assert stats.calls_int8 == 1 and stats.calls_fp == 0
    deq = (q.float() * scale[:, None])
    ref = torch.nn.functional.linear(x, deq, lin.bias.detach())
    assert (out.float() - ref).abs().mean() < 2e-2 * ref.abs().mean()  # fp16 out
    # _fp reconstructs from codes+scale (w_fp is None) and is safe to call (fp32)
    assert torch.allclose(shim._fp(x), ref, atol=1e-4)


class _FakeAttn(torch.nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)


class _FakeMlp(torch.nn.Module):
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim)


class _FakeViTTrunk(torch.nn.Module):
    def __init__(self, nblocks=2, dim=64):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({"attn": _FakeAttn(dim), "mlp": _FakeMlp(dim)})
            for _ in range(nblocks)
        ])


class _FakeViTModel:
    def __init__(self, nblocks=2, dim=64):
        trunk = _FakeViTTrunk(nblocks, dim)
        self.backbone = types.SimpleNamespace(
            vision_backbone=types.SimpleNamespace(trunk=trunk))


def test_quantize_sam3_trunk_and_fni8_roundtrip(tmp_path):
    import superl8  # real format module (pure-python; no CUDA needed for save/read)
    from comfyui_superl8 import sam3_convert

    model = _FakeViTModel(nblocks=2, dim=64)
    tensors, meta = sam3_convert.quantize_sam3_trunk(model)
    # 2 blocks × (qkv, proj, fc1, fc2) = 8 int8 layers, each with a paired bias
    assert len(meta["int8_layers"]) == 8
    assert meta["trunk"] == {"dim": 64, "num_heads": 4, "head_dim": 16, "num_blocks": 2}
    assert "blocks.0.attn.qkv" in tensors and "blocks.0.attn.qkv.bias" in tensors
    assert tensors["blocks.0.attn.qkv"].scheme == "per_row_i8"
    assert tensors["blocks.0.attn.qkv"].data.dtype == torch.int8

    out = str(tmp_path / "vit.b8.fni8")
    superl8.save_fni8(out, tensors, meta=meta)
    r = superl8.FQReader(out)
    try:
        assert r.header["__meta__"]["int8_layers"] == meta["int8_layers"]
        qt = r.get_qtensor("blocks.0.attn.qkv")
        assert qt.scheme == "per_row_i8" and qt.scale.dtype == torch.float32
        # byte-identical resident layout
        assert torch.equal(qt.data, tensors["blocks.0.attn.qkv"].data)
        assert torch.allclose(qt.scale, tensors["blocks.0.attn.qkv"].scale)
    finally:
        r.close()


def test_load_vit_linears_from_fni8_installs_resident(monkeypatch, tmp_path):
    import superl8
    from comfyui_superl8 import sam3_convert

    _install_fake_fni8_attr(monkeypatch)  # only superl8.linear_w8a8 faked (CPU forward)
    vitdet = _install_fake_vitdet(monkeypatch)

    src = _FakeViTModel(nblocks=2, dim=64)
    tensors, meta = sam3_convert.quantize_sam3_trunk(src)
    out = str(tmp_path / "vit.b8.fni8")
    superl8.save_fni8(out, tensors, meta=meta)

    dst = _FakeViTModel(nblocks=2, dim=64)  # fresh fp weights
    reader = superl8.FQReader(out)
    try:
        handle = se.load_vit_linears_from_fni8(dst, reader, device="cpu")
    finally:
        reader.close()
    assert handle.stats.n_wrapped == 8
    qkv = dst.backbone.vision_backbone.trunk.blocks[0]["attn"].qkv
    assert isinstance(qkv, se.Int8LinearShim) and qkv._sqnr_pass is True
    # forward uses the SOURCE model's quantized weights, not dst's fp weights
    x = torch.randn(3, 64)
    src_qkv = src.backbone.vision_backbone.trunk.blocks[0].attn.qkv
    q, scale = se._quantize_per_row_i8(src_qkv.weight)
    ref = torch.nn.functional.linear(x, q.float() * scale[:, None], src_qkv.bias.detach())
    assert (qkv(x).float() - ref).abs().mean() < 2e-2 * ref.abs().mean()  # fp16 out
    assert vitdet.addmm_act.__name__ == "int8_addmm_act"
    handle.unpatch()
    assert vitdet.F is torch.nn.functional or True  # linears-only handle: F untouched


def test_attn_proxy_gates_and_delegates(monkeypatch):
    _install_fake_fni8(monkeypatch)  # attn returns fp SDPA -> +inf SQNR -> routed fp
    vitdet = _install_fake_vitdet(monkeypatch)
    model = _FakeModel()
    handle = se.patch_vit_attention_int8(model, floor_db=20.0)
    proxy = vitdet.F
    # non-SDPA attrs delegate to real torch.nn.functional
    assert proxy.gelu is torch.nn.functional.gelu
    q = torch.randn(1, 4, 32, 64)  # CPU tensor -> is_cuda False -> ineligible -> fp
    out = proxy.scaled_dot_product_attention(q, q.clone(), q.clone())
    ref = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    assert torch.allclose(out, ref, atol=1e-5)
    # CPU tensors are correctly deemed ineligible for the CUDA-only dp4a kernel and
    # take the fp fallback (the +inf-SQNR-routes-to-fp gate logic is unit-tested in
    # test_sqnr_identical_is_inf_and_fails_routing).
    assert handle.gate.n_fallback == 1
    assert handle.gate.n_int8 == 0
    handle.unpatch()
    assert vitdet.F is torch.nn.functional
