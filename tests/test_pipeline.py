# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pipeline-parallel primitives (no ComfyUI, CPU-only): the transport
codec round-trip + metrics, the block-split planner, DiT block counting, and the
subclass-preserving cross-device parameter move. The real 2-GPU LTX-2.3 split is
exercised by `tests/e2e/test_pipeline_ltx.py`."""
import math

import pytest
import torch

pytest.importorskip("superl8")

from comfyui_superl8 import multigpu
from comfyui_superl8.superl8_tensor import FNI8Tensor


# ---- transport codec (superl8.transport) ----------------------------------------

def test_transport_codec_int8_roundtrip_and_metrics():
    """int8 transport of a controlled activation reconstructs it within the codec bar
    (cos ~ 1) and reports a ~2x-vs-fp16 wire saving. This is the same round-trip the
    LTX split does at each stage boundary, run on one device (the cross-device move is
    the same call with a different dst — covered in the e2e)."""
    codec = multigpu.TransportCodec(scheme="int8")
    if not codec.available():
        pytest.skip("superl8.transport not importable")
    x = torch.randn(2, 512, 4096, dtype=torch.float16)  # LTX video-hidden shape/dtype
    y, stats = codec.transport(x, "cpu")

    assert y.shape == x.shape and y.dtype == x.dtype
    assert stats.cos > 0.999, f"int8 transport cos too low: {stats.cos}"
    # fp16 input, int8 codes + fp32 per-row scale ⇒ ~2x on-wire reduction.
    assert 1.8 <= stats.ratio <= 2.1, f"unexpected int8 wire ratio {stats.ratio}"
    assert stats.raw_bytes == x.numel() * x.element_size()
    assert stats.wire_bytes < stats.raw_bytes
    assert stats.link_ms > 0
    d = stats.as_dict()
    assert d["scheme"] == "int8" and d["src"] == "cpu" and d["dst"] == "cpu"


def test_transport_codec_int4_is_smaller_and_lossier_than_int8():
    codec8 = multigpu.TransportCodec(scheme="int8")
    codec4 = multigpu.TransportCodec(scheme="int4")
    if not codec8.available():
        pytest.skip("superl8.transport not importable")
    x = torch.randn(1, 256, 2048, dtype=torch.float16)
    _, s8 = codec8.transport(x, "cpu")
    _, s4 = codec4.transport(x, "cpu")
    assert s4.wire_bytes < s8.wire_bytes, "int4 must move fewer bytes than int8"
    assert s4.ratio > s8.ratio
    assert s4.cos <= s8.cos + 1e-6, "int4 should be no more accurate than int8"


def test_transport_fp16_scheme_is_lossless():
    codec = multigpu.TransportCodec(scheme="fp16")
    if not codec.available():
        pytest.skip("superl8.transport not importable")
    x = torch.randn(4, 64, 128, dtype=torch.float16)
    y, stats = codec.transport(x, "cpu")
    assert torch.equal(y, x), "fp16 transport of an fp16 tensor must be exact"
    assert stats.cos >= 0.9999  # exact tensor; cos in fp32 carries tiny rounding noise


def test_effective_transfer_ms_matches_link_bandwidth():
    # 250 MB/s ⇒ 250e6 bytes takes ~1000 ms.
    ms = multigpu.effective_transfer_ms(250_000_000)
    assert math.isclose(ms, 1000.0, rel_tol=0.05)


# ---- block-split planner -----------------------------------------------------

def test_plan_pipeline_even_split():
    plan = multigpu.plan_pipeline(48, devices=["cuda:0", "cuda:1"])
    assert plan.stage_ranges == [(0, 24), (24, 24)]
    assert plan.num_stages == 2
    assert plan.stage_of_block(0) == 0
    assert plan.stage_of_block(24) == 1
    assert plan.stage_of_block(47) == 1


def test_plan_pipeline_head_weight_bias_shrinks_stage0():
    """Stage 0 also carries the prologue/connectors/output, so head_weight_blocks moves
    blocks off it to balance physical VRAM. 10 block-equivalents ⇒ shift 5 blocks."""
    plan = multigpu.plan_pipeline(48, devices=["cuda:0", "cuda:1"], head_weight_blocks=10)
    assert plan.stage_ranges == [(0, 19), (19, 29)]
    # every block still maps to exactly one stage, contiguous, covering all 48
    covered = [plan.stage_of_block(i) for i in range(48)]
    assert covered == [0] * 19 + [1] * 29


def test_plan_pipeline_three_way():
    plan = multigpu.plan_pipeline(48, devices=["cuda:0", "cuda:1", "cuda:2"])
    assert plan.stage_ranges == [(0, 16), (16, 16), (32, 16)]


def test_plan_pipeline_falls_back_to_two_devices():
    plan = multigpu.plan_pipeline(10, devices=["cuda:0"])
    assert len(plan.devices) == 2


# ---- DiT block counting ------------------------------------------------------

def test_count_dit_blocks_transformer_blocks_modulelist():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(48)])

    assert multigpu.count_dit_blocks(M()) == 48


def test_count_dit_blocks_from_state_dict_keys():
    class Fake:
        def state_dict(self):
            return {f"transformer_blocks.{i}.attn.to_q.weight": None for i in range(12)}

    assert multigpu.count_dit_blocks(Fake()) == 12


def test_count_dit_blocks_mmdit_double_single():
    class Fake:
        def state_dict(self):
            d = {f"double_blocks.{i}.x": None for i in range(19)}
            d.update({f"single_blocks.{i}.x": None for i in range(38)})
            return d

    assert multigpu.count_dit_blocks(Fake()) == 57


# ---- subclass-preserving cross-device move -----------------------------------

def test_move_module_preserves_fni8tensor_subclass():
    """The physical block move must keep int8 FNI8Tensor weights int8 (with their scale)
    — a plain `.to()` through nn.Module._apply would strip the subclass."""
    lin = torch.nn.Linear(8, 8, bias=True)
    q = torch.randint(-127, 127, (8, 8), dtype=torch.int8)
    scale = torch.rand(8, dtype=torch.float32)
    lin._parameters["weight"] = FNI8Tensor(q, scale)
    multigpu.move_module_to_device(lin, "cpu")  # cpu->cpu no-op device, but exercises path
    w = lin._parameters["weight"]
    assert isinstance(w, FNI8Tensor), "move stripped the FNI8Tensor subclass"
    assert w.q_scale is not None and w.dtype == torch.int8


def test_module_vram_bytes_counts_int8_plus_scale():
    lin = torch.nn.Linear(4, 4, bias=False)
    q = torch.zeros(4, 4, dtype=torch.int8)
    lin._parameters["weight"] = FNI8Tensor(q, torch.zeros(4, dtype=torch.float32))
    # 16 int8 codes (1 B each) + 4 fp32 scales (4 B each) = 16 + 16 = 32
    assert multigpu.module_vram_bytes(lin) == 32


# ---- controlled split-in-a-loop equivalence (the mechanism, CPU) -------------

def test_split_block_loop_matches_single_device_within_codec_bar():
    """Run a stack of blocks straight through vs. split into two 'stages' with a codec
    handoff at the boundary. With int8 transport the split output matches the single
    run within the codec's accuracy bar (cos ~ 1) — the deterministic proof of the
    pipeline mechanism, independent of any GPU."""
    torch.manual_seed(0)
    blocks = [torch.nn.Linear(64, 64) for _ in range(8)]
    for b in blocks:
        b.eval()
    x = torch.randn(2, 16, 64)

    # single run
    h = x
    with torch.no_grad():
        for b in blocks:
            h = h + 0.1 * torch.tanh(b(h))
    ref = h

    # split run: boundary after block 4, hidden crosses the codec
    codec = multigpu.TransportCodec(scheme="int8")
    if not codec.available():
        pytest.skip("superl8.transport not importable")
    h = x
    with torch.no_grad():
        for i, b in enumerate(blocks):
            if i == 4:
                h, stats = codec.transport(h.contiguous(), "cpu")
                assert stats.cos > 0.999
            h = h + 0.1 * torch.tanh(b(h))
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten(), h.flatten(), dim=0).item()
    assert cos > 0.999, f"split-vs-single diverged past the codec bar: cos={cos}"
