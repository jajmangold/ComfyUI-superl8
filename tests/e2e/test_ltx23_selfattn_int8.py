# SPDX-License-Identifier: Apache-2.0
"""Regression guard: LTX-2.3 VIDEO self-attention must stay on the int8 dp4a path.

The full 26 GB LTX-2.3 DiT does not fit one 16 GB card, so this builds ONE
`BasicAVTransformerBlock` at the exact shipped LTX-2.3 (22b) config (v_dim 4096, 32
video heads, head-dim 128, bf16, gated attn, cross-attn adaLN) — the block whose 48×
repetition dominates a denoise step — and drives it through the production
`optimized_attention_override`. No `.fni8` file needed (config-built + random int8
weights), so it runs anywhere a Volta/CMP GPU + fni8 are present.

It asserts the thing that keeps LTX-2.3 fast on this fleet:
  1. the bidirectional int8 dp4a FA2 kernel (`superl8.attn_int8_fwd`) is actually called
     for the video self-attention (head-dim 128, mask=None), NOT fp SDPA;
  2. the per-call-site SQNR gate ADMITS int8 (does not demote to fp);
  3. int8 block output matches the fp block at cosine > 0.99 (quality gate);
  4. the int8 block beats the fp block in latency (the fleet-specific dp4a win holds).

A silent regression to fp SDPA — a wiring break or an over-strict gate — would cost ~30%
on the dominant op; this test fails loudly if that happens. See
docs/ltx23-block-profile.md.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("comfy.ldm.lightricks.av_model")
pytest.importorskip("superl8")

import torch

import comfyui_superl8.attention as A
from comfyui_superl8.attention import Int8AttnGate, make_fni8_attn_override
from comfyui_superl8.gate import is_sm70

pytestmark = pytest.mark.comfy_e2e

# Real LTX-2.3 (22b) block config, read off Lightricks__LTX-2.3.dit.b8.fni8.
CFG = dict(v_dim=4096, a_dim=2048, v_heads=32, a_heads=32, vd_head=128, ad_head=64,
           v_context_dim=4096, a_context_dim=2048,
           apply_gated_attention=True, cross_attention_adaln=True)
VLEN, CTX = 2048, 256  # short-video token count; small enough to be quick


def _build_block(operations, dtype, dev):
    from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock

    return BasicAVTransformerBlock(
        v_dim=CFG["v_dim"], a_dim=CFG["a_dim"], v_heads=CFG["v_heads"],
        a_heads=CFG["a_heads"], vd_head=CFG["vd_head"], ad_head=CFG["ad_head"],
        v_context_dim=CFG["v_context_dim"], a_context_dim=CFG["a_context_dim"],
        apply_gated_attention=CFG["apply_gated_attention"],
        cross_attention_adaln=CFG["cross_attention_adaln"],
        dtype=dtype, device=dev, operations=operations,
    ).eval()


def _init_and_quantize(fp_block, int8_block):
    import torch.nn as nn

    from comfyui_superl8.superl8_tensor import FNI8Tensor
    from comfyui_superl8.int8_linear import quantize_linear_weight

    g = torch.Generator().manual_seed(1234)
    dev = int8_block.attn1.to_q.weight.device
    fp_params = dict(fp_block.named_parameters())
    i8_params = dict(int8_block.named_parameters())
    i8_mods = dict(int8_block.named_modules())
    with torch.no_grad():
        for name, p in fp_params.items():
            v = (torch.randn(p.shape, generator=g) * 0.02).to(p.dtype)
            p.copy_(v.to(p.device))
            if name in i8_params:
                i8_params[name].copy_(v.to(i8_params[name].device))
        for name, m in fp_block.named_modules():
            if isinstance(m, nn.Linear):
                qt = quantize_linear_weight(m.weight.data.float())
                i8_mods[name]._parameters["weight"] = FNI8Tensor(
                    qt.data.to(dev), qt.scale.to(dev))
                if m.bias is not None:
                    i8_mods[name]._parameters["bias"].data.copy_(m.bias.data.to(dev))


@pytest.fixture(scope="module")
def blocks():
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    import comfy.ops

    dev = torch.device("cuda:0")
    dt = torch.bfloat16
    fp_block = _build_block(comfy.ops.manual_cast, dt, dev)
    i8_block = _build_block(__import__("comfyui_superl8.ops", fromlist=["FNI8Ops"]).FNI8Ops, dt, dev)
    _init_and_quantize(fp_block, i8_block)
    return fp_block, i8_block, dev, dt


def _inputs(dev, dt):
    g = torch.Generator().manual_seed(0)

    def mk(*s):
        return (torch.randn(*s, generator=g) * 0.5).to(dev, dt)

    ax = torch.empty(1, 0, CFG["a_dim"], device=dev, dtype=dt)  # disables audio + AV-cross
    return dict(
        vx=mk(1, VLEN, CFG["v_dim"]), ax=ax,
        v_context=mk(1, CTX, CFG["v_context_dim"]),
        v_timestep=mk(1, 1, 9 * CFG["v_dim"]),
        v_prompt_timestep=mk(1, 1, 2 * CFG["v_dim"]),
    )


def _blk_kwargs(inp):
    return dict(v_context=inp["v_context"], a_context=None, attention_mask=None,
                v_timestep=inp["v_timestep"], a_timestep=None, v_pe=None, a_pe=None,
                self_attention_mask=None, v_prompt_timestep=inp["v_prompt_timestep"])


def test_ltx23_video_selfattn_engages_int8(blocks):
    fp_block, i8_block, dev, dt = blocks
    inp = _inputs(dev, dt)

    gate = Int8AttnGate()
    override = make_fni8_attn_override(gate)
    topts_i8 = {"optimized_attention_override": override, "run_vx": True, "run_ax": False,
                "a2v_cross_attn": False, "v2a_cross_attn": False}
    topts_fp = {"run_vx": True, "run_ax": False, "a2v_cross_attn": False, "v2a_cross_attn": False}

    # Spy on the int8 kernel to prove it is the path actually taken for self-attention.
    calls = {"n": 0}
    orig = A._int8_dp4a

    def spy(q, k, v):
        calls["n"] += 1
        return orig(q, k, v)

    A._int8_dp4a = spy
    try:
        with torch.no_grad():
            y_i8 = i8_block((inp["vx"].clone(), inp["ax"]), transformer_options=topts_i8,
                            **_blk_kwargs(inp))[0].float()
            y_fp = fp_block((inp["vx"].clone(), inp["ax"]), transformer_options=topts_fp,
                            **_blk_kwargs(inp))[0].float()
    finally:
        A._int8_dp4a = orig

    # 1. the int8 dp4a FA2 kernel was actually called (self-attn, and unmasked cross-attn).
    assert calls["n"] >= 1, "int8 dp4a attention kernel was never called — self-attn fell to fp"

    # 2. the video self-attention signature was ADMITTED by the SQNR gate (not demoted).
    sig = (1, CFG["v_heads"], VLEN, CFG["vd_head"], "torch.bfloat16")
    assert sig in gate.decisions, f"no gate decision for the video self-attn signature {sig}"
    passed, sqnr_db = gate.decisions[sig]
    assert passed, f"video self-attn was DEMOTED to fp SDPA (SQNR {sqnr_db:.1f} dB below floor)"

    # 3. quality: int8 block output matches fp block.
    cos = torch.nn.functional.cosine_similarity(y_i8.flatten(), y_fp.flatten(), dim=0).item()
    assert cos > 0.99, f"int8 block output diverged from fp: cosine {cos:.4f}"
    assert torch.isfinite(y_i8).all(), "int8 block produced non-finite output"


def test_ltx23_block_int8_beats_fp_latency(blocks):
    fp_block, i8_block, dev, dt = blocks
    inp = _inputs(dev, dt)
    gate = Int8AttnGate()
    override = make_fni8_attn_override(gate)
    topts_i8 = {"optimized_attention_override": override, "run_vx": True, "run_ax": False,
                "a2v_cross_attn": False, "v2a_cross_attn": False}
    topts_fp = {"run_vx": True, "run_ax": False, "a2v_cross_attn": False, "v2a_cross_attn": False}

    def run(blk, topts):
        return blk((inp["vx"].clone(), inp["ax"]), transformer_options=topts, **_blk_kwargs(inp))

    def timed(blk, topts, it=8):
        with torch.no_grad():
            for _ in range(3):
                run(blk, topts)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(it):
                run(blk, topts)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / it * 1e3

    ms_i8 = timed(i8_block, topts_i8)
    ms_fp = timed(fp_block, topts_fp)
    # On the CMP fleet (dead tensor cores) int8 dp4a must beat fp; allow a margin for noise.
    assert ms_i8 < ms_fp, f"int8 block ({ms_i8:.1f} ms) not faster than fp ({ms_fp:.1f} ms)"
