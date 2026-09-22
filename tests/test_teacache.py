# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TeaCache cross-step DiT cache controller (`comfyui_superl8.teacache`).

Pure controller logic — no ComfyUI, no GPU. Verifies the skip signal, the cumulative
threshold, warmup, the consecutive-skip cap, per-stream isolation (cond/uncond),
generation reset from the sigma schedule, the tea/taylor strategies, the SQNR spot-check
gate, and that the DIFFUSION_MODEL wrapper actually short-circuits `executor` on a skip.
"""
from __future__ import annotations

import pytest
import torch

import torch.nn as nn

from comfyui_superl8.teacache import (
    TeaCacheController,
    _poly_rescale,
    _rel_l1,
    cache_sqnr,
    find_block_list,
    install_residual_block_cache,
    make_teacache_wrapper,
)


def _to(sigma, sched, cond_or_uncond=(0,)):
    return {
        "sigmas": torch.tensor([float(sigma)]),
        "sample_sigmas": torch.tensor([float(s) for s in sched]),
        "cond_or_uncond": list(cond_or_uncond),
    }


def test_rel_l1_basic():
    a = torch.ones(4, 4)
    assert _rel_l1(a, a) == 0.0
    assert _rel_l1(2 * a, a) == 1.0


def test_poly_rescale_identity_and_horner():
    assert _poly_rescale(0.5, None) == 0.5
    assert _poly_rescale(2.0, [1.0, 2.0, 3.0]) == 17.0  # 1 + 2*2 + 3*4


def test_cache_sqnr_metric():
    a = torch.randn(8, 8)
    assert cache_sqnr(a, a) == float("inf")
    assert cache_sqnr(torch.full((4,), float("nan")), torch.ones(4)) == float("-inf")
    x = torch.ones(1000)
    noise = torch.ones(1000) * 0.1
    db = cache_sqnr(x + noise, x)  # noise power 0.01 => ~20 dB
    assert 19.0 < db < 21.0


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        TeaCacheController(strategy="bogus")


def test_first_step_never_skips():
    c = TeaCacheController(rel_l1_thresh=10.0, warmup_steps=0, strategy="tea")
    x = torch.randn(1, 4, 8, 8)
    skip, st = c.should_skip(x, _to(1.0, [1.0, 0.5, 0.0]))
    assert skip is False  # no cached output yet
    c.record_output(st, x, x)
    c.note_step(False, st)


def test_identical_input_skips_after_warmup():
    c = TeaCacheController(rel_l1_thresh=0.05, warmup_steps=1, max_consecutive_skips=10,
                           strategy="tea")
    sched = [1.0, 0.9, 0.8, 0.7]
    x = torch.randn(1, 4, 8, 8)
    skip, st = c.should_skip(x, _to(1.0, sched))
    assert not skip
    c.record_output(st, x.clone(), x)
    c.note_step(False, st)
    c.step_index += 1
    skip, st = c.should_skip(x, _to(0.9, sched))  # identical input -> rel_l1 ~ 0 -> skip
    assert skip is True
    c.note_step(True, st)


def test_large_change_forces_compute():
    c = TeaCacheController(rel_l1_thresh=0.05, warmup_steps=1, strategy="tea")
    sched = [1.0, 0.9, 0.8]
    x0 = torch.zeros(1, 4, 8, 8)
    skip, st = c.should_skip(x0, _to(1.0, sched))
    c.record_output(st, x0.clone(), x0)
    c.note_step(False, st)
    c.step_index += 1
    x1 = torch.ones(1, 4, 8, 8) * 100.0
    skip, st = c.should_skip(x1, _to(0.9, sched))
    assert skip is False


def test_max_consecutive_skips_cap():
    c = TeaCacheController(rel_l1_thresh=100.0, warmup_steps=0, max_consecutive_skips=2,
                           strategy="tea")
    sched = [1.0, 0.9, 0.8, 0.7, 0.6]
    x = torch.randn(1, 4, 8, 8)
    skip, st = c.should_skip(x, _to(1.0, sched))
    c.record_output(st, x.clone(), x)
    c.note_step(False, st)
    c.step_index += 1
    skips = []
    for s in sched[1:]:
        skip, st = c.should_skip(x, _to(s, sched))
        skips.append(skip)
        c.note_step(skip, st)
        if not skip:
            c.record_output(st, x.clone(), x)
        c.step_index += 1
    run = 0
    for s in skips:
        run = run + 1 if s else 0
        assert run <= 2


def test_cond_uncond_isolation():
    c = TeaCacheController(rel_l1_thresh=0.05, warmup_steps=0, strategy="tea")
    sched = [1.0, 0.9]
    xc = torch.zeros(1, 4, 8, 8)
    xu = torch.ones(1, 4, 8, 8) * 50.0
    skip, stc = c.should_skip(xc, _to(1.0, sched, cond_or_uncond=(0,)))
    c.record_output(stc, xc.clone(), xc)
    skip, stu = c.should_skip(xu, _to(1.0, sched, cond_or_uncond=(1,)))
    assert stc is not stu
    assert skip is False  # uncond has no cache yet


def test_generation_reset():
    c = TeaCacheController(rel_l1_thresh=0.05, warmup_steps=0, strategy="tea")
    sched = [1.0, 0.9, 0.8]
    x = torch.randn(1, 4, 8, 8)
    skip, st = c.should_skip(x, _to(1.0, sched))
    c.record_output(st, x.clone(), x)
    c.note_step(False, st)
    c.step_index += 1
    skip, st = c.should_skip(x, _to(1.0, sched))  # new gen: sigma back at schedule[0]
    assert skip is False
    assert st.cached_output is None


def test_sqnr_gate_disables_caching_on_divergent_output():
    # Outputs that jump wildly between computed steps -> low SQNR -> caching disabled.
    c = TeaCacheController(rel_l1_thresh=100.0, warmup_steps=0, sqnr_floor_db=20.0,
                           strategy="tea")
    sched = [1.0, 0.9, 0.8]
    x = torch.randn(1, 4, 8, 8)
    skip, st = c.should_skip(x, _to(1.0, sched))
    c.record_output(st, torch.zeros(1, 4, 8, 8), x)   # first output
    c.note_step(False, st)
    c.step_index += 1
    # Force a second compute (identical input but pretend metric said compute): a wildly
    # different output should trip the gate.
    skip, st = c.should_skip(x, _to(0.9, sched))
    c.record_output(st, torch.ones(1, 4, 8, 8) * 100.0, x)
    assert c.caching_disabled is True
    assert c.gate_sqnr_db is not None and c.gate_sqnr_db < 20.0


def test_taylor_sensitivity_tracks_dy_dx():
    c = TeaCacheController(rel_l1_thresh=0.05, warmup_steps=0, strategy="taylor")
    sched = [1.0, 0.9, 0.8, 0.7]
    # step0 compute
    x0 = torch.ones(1, 4, 4, 4)
    skip, st = c.should_skip(x0, _to(1.0, sched))
    c.record_output(st, x0.clone(), x0)
    c.step_index += 1
    # step1 compute: input doubles (dx=1), output x4 (dy=3) => sensitivity moves toward 3
    x1 = x0 * 2
    skip, st = c.should_skip(x1, _to(0.9, sched))
    c.record_output(st, x0 * 4, x1)
    assert st.sensitivity > 1.0  # amplifying model -> more conservative metric


def test_wrapper_skips_executor_and_reuses_output():
    c = TeaCacheController(rel_l1_thresh=0.05, warmup_steps=1, max_consecutive_skips=10,
                           strategy="tea")
    wrapper = make_teacache_wrapper(c)
    sched = [1.0, 0.9, 0.8]
    calls = {"n": 0}

    def executor(x, timesteps, **kwargs):
        calls["n"] += 1
        return x * 2.0

    x = torch.randn(1, 4, 8, 8)
    out0 = wrapper(executor, x, torch.tensor([1.0]), transformer_options=_to(1.0, sched))
    assert calls["n"] == 1
    assert torch.allclose(out0, x * 2.0)
    out1 = wrapper(executor, x, torch.tensor([0.9]), transformer_options=_to(0.9, sched))
    assert calls["n"] == 1  # skipped -> executor not called again
    assert torch.allclose(out1, x * 2.0)
    assert c.skipped_steps == 1


class _AddBlock(nn.Module):
    """A stand-in single-stream transformer block: h -> h + delta (delta is the block's
    'contribution'). Signature mirrors Lumina's `layer(img, *rest)` returning one tensor."""

    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, h, *args, **kwargs):
        return h + self.delta


class _DiT(nn.Module):
    def __init__(self, deltas):
        super().__init__()
        self.layers = nn.ModuleList([_AddBlock(d) for d in deltas])


def test_find_block_list_single_stream():
    dit = _DiT([1.0, 2.0])
    assert find_block_list(dit) is dit.layers
    assert find_block_list(nn.Linear(4, 4)) is None


def test_residual_block_cache_skip_reuses_delta_and_restores():
    dit = _DiT([1.0, 2.0, 3.0])  # stack adds +6 to h
    ctrl = TeaCacheController(mode="residual")

    def run_blocks(h):
        for blk in dit.layers:
            h = blk.forward(h)
        return h

    restore = install_residual_block_cache(dit.layers, ctrl)
    # computed step: run the real stack, capture residual (=+6)
    ctrl.skip_active = False
    h0 = torch.zeros(1, 4)
    out0 = run_blocks(h0)
    assert torch.allclose(out0, h0 + 6.0)
    assert ctrl._residual is not None and torch.allclose(ctrl._residual, torch.full((1, 4), 6.0))
    # skipped step: different input, blocks must NOT run — output = h_in + cached residual
    ctrl.skip_active = True
    h1 = torch.ones(1, 4) * 5.0
    out1 = run_blocks(h1)
    assert torch.allclose(out1, h1 + 6.0)  # +6 reused, blocks skipped
    restore()
    # after restore the blocks run normally again
    assert torch.allclose(run_blocks(h1), h1 + 6.0)


def test_residual_mode_have_cache_gates_skip():
    c = TeaCacheController(rel_l1_thresh=0.05, warmup_steps=0, strategy="tea", mode="residual")
    sched = [1.0, 0.9]
    x = torch.randn(1, 4, 8, 8)
    # no residual cached yet -> cannot skip even with tiny change
    skip, st = c.should_skip(x, _to(1.0, sched))
    assert skip is False
    c._residual = torch.zeros(1, 4)  # pretend a residual was captured
    skip, st = c.should_skip(x, _to(0.9, sched))  # identical input -> now skippable
    assert skip is True


def test_thresh_zero_never_skips():
    c = TeaCacheController(rel_l1_thresh=0.0, warmup_steps=0, strategy="tea")
    sched = [1.0, 0.9]
    x = torch.randn(1, 4, 8, 8)
    skip, st = c.should_skip(x, _to(1.0, sched))
    c.record_output(st, x.clone(), x)
    c.note_step(False, st)
    c.step_index += 1
    skip, st = c.should_skip(x, _to(0.9, sched))
    assert skip is False
