# SPDX-License-Identifier: Apache-2.0
"""Hadamard-rotation SQNR validation (issue #74): measure the int8 attention
SQNR improvement when ``rotate=True`` vs ``rotate=False``, and confirm
rotation preserves correctness (cosine >= 0.99 vs fp SDPA) with no measurable
latency cost.

The dominant int8 error source is channel outliers in K (AGENTS.md HARD RULE:
"always apply K-smoothing"). Rotation spreads those outliers uniformly across
channels so the int8 quantizer's uniform grid captures them with lower rel-err
(measured 1.4x improvement on real DiT K activations, see
docs/int8-dit-validation.md#task-2). This test injects synthetic per-channel
K-outliers (mimicking real DiT activations) and asserts the SQNR delta is
real: rotation must not regress and typically improves SQNR on outlier-heavy K.

Unlike the bench script (``bench/profile_ltx23_rotate.py``) this is a gated
pytest so CI runs it on every sm_70 node and flags a silent kernel-side
regression in the rotation path.
"""
from __future__ import annotations

import pytest
import statistics
import time

pytest.importorskip("superl8")

import torch

from comfyui_superl8.gate import is_sm70

pytestmark = pytest.mark.comfy_e2e


def _sync():
    torch.cuda.synchronize()


def _sqnr_cosine(a, b):
    """Return (SQNR in dB, cosine similarity). Both computed in fp32.
    SQNR = 10*log10(mean(b^2) / mean((a-b)^2)), +inf when bit-identical."""
    a = a.detach().float()
    b = b.detach().float()
    signal = b.pow(2).mean()
    noise = (a - b).pow(2).mean()
    eps = torch.tensor(1e-30, device=a.device)
    sqnr = (10.0 * torch.log10(signal / torch.max(noise, eps))).item()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item()
    return float("inf") if noise.item() == 0.0 else sqnr, cos


def _timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


@pytest.mark.parametrize("S", [1024, 2048, 4096])
def test_rotate_sqnr_never_regresses(S):
    """Hadamard rotation must not regress SQNR on outlier-injected K and
    typically improves it — measured SQNRdelta must be >= 0 dB."""
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    import superl8

    dev = torch.device("cuda:0")
    dt = torch.bfloat16
    B, H, D = 1, 16, 128
    g = torch.Generator(device=dev).manual_seed(1)

    q = torch.randn(B, H, S, D, generator=g, device=dev, dtype=dt) * 0.5
    k = torch.randn(B, H, S, D, generator=g, device=dev, dtype=dt) * 0.5
    v = torch.randn(B, H, S, D, generator=g, device=dev, dtype=dt) * 0.5

    # Inject strong per-channel outliers into K (the dominant int8 error source
    # per AGENTS.md), mimicking real DiT activations.  Also add a few token
    # outliers.
    k[..., ::17] *= 12.0
    q[..., ::23] *= 8.0
    k[:, :, ::128, :] *= 6.0

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()
    )

    out_f = superl8.attn_int8_fwd(q, k, v, causal=False, rotate=False)
    out_t = superl8.attn_int8_fwd(q, k, v, causal=False, rotate=True)

    sqnr_f, cos_f = _sqnr_cosine(out_f, ref)
    sqnr_t, cos_t = _sqnr_cosine(out_t, ref)

    # 1. rotation must not regress SQNR below 20 dB floor.
    assert sqnr_t >= 20.0, (
        f"rotate=True SQNR {sqnr_t:.1f} dB below 20 dB gate floor at S={S}"
    )

    # 2. rotation must improve or at worst not regress SQNR vs no-rotation.
    delta_db = sqnr_t - sqnr_f
    assert delta_db >= -0.1, (
        f"rotate=True SQNR ({sqnr_t:.1f} dB) regressed below rotate=False "
        f"({sqnr_f:.1f} dB) by Δ={delta_db:.2f} dB at S={S}"
    )

    # 3. correctness: rotate=True must stay above cosine 0.99 vs fp reference.
    assert cos_t >= 0.99, (
        f"rotate=True cosine {cos_t:.6f} < 0.99 vs fp SDPA at S={S}"
    )

    # 4. rotation has no measurable latency cost.
    ms_f = _timed(lambda: superl8.attn_int8_fwd(q, k, v, causal=False, rotate=False))
    ms_t = _timed(lambda: superl8.attn_int8_fwd(q, k, v, causal=False, rotate=True))
    overhead_pct = (ms_t - ms_f) / max(ms_f, 1e-6) * 100
    assert overhead_pct < 10, (
        f"rotate=True latency ({ms_t:.3f} ms) >10% slower than rotate=False "
        f"({ms_f:.3f} ms, {overhead_pct:+.1f}%) at S={S}"
    )
