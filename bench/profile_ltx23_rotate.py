# SPDX-License-Identifier: Apache-2.0
"""Quantify the Hadamard-rotation (`rotate=True`) lever for int8 self-attention (issue
#74) on outlier-heavy activations like a real DiT's: perf cost (should be ~free,
logit-invariant) and the int8-attention SQNR improvement (reduced quant error -> more
call-sites clear the SQNR gate and STAY on the fast int8 path instead of demoting to fp
SDPA, which is SLOWER than int8 dp4a on this fleet).

    python3 bench/profile_ltx23_rotate.py
"""
from __future__ import annotations

import statistics
import time

import torch


def _sync():
    torch.cuda.synchronize()


def _t(fn, iters=20, warmup=5):
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


def _sqnr(a, b):
    a = a.float()
    b = b.float()
    n = (a - b).pow(2).mean().item()
    s = b.pow(2).mean().item()
    return 10.0 * torch.log10(torch.tensor(s / max(n, 1e-30))).item()


def main():
    from comfyui_superl8.gate import is_sm70
    if not is_sm70():
        print("SKIP: needs sm_70")
        return
    import superl8

    dev = torch.device("cuda:0")
    B, H, D = 1, 32, 128
    g = torch.Generator().manual_seed(0)

    print(f"{'seq':>6} {'rot=F ms':>9} {'rot=T ms':>9} {'SQNR_F dB':>10} {'SQNR_T dB':>10} "
          f"{'d_err':>7}   (bf16, channel-outlier K)")
    for S in (2048, 4096, 8192):
        for dt in (torch.bfloat16,):
            q = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dev, dt)
            k = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dev, dt)
            v = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dev, dt)
            # Inject strong per-channel outliers into K (the dominant int8 error source
            # per AGENTS.md) + a few token outliers, mimicking real DiT activations.
            k[..., ::17] *= 12.0
            q[..., ::23] *= 8.0
            k[:, :, ::128, :] *= 6.0

            ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())

            out_f = superl8.attn_int8_fwd(q, k, v, causal=False, rotate=False)
            out_t = superl8.attn_int8_fwd(q, k, v, causal=False, rotate=True)
            sq_f = _sqnr(out_f, ref)
            sq_t = _sqnr(out_t, ref)

            t_f = _t(lambda: superl8.attn_int8_fwd(q, k, v, causal=False, rotate=False))
            t_t = _t(lambda: superl8.attn_int8_fwd(q, k, v, causal=False, rotate=True))
            print(f"{S:6d} {t_f:9.3f} {t_t:9.3f} {sq_f:10.2f} {sq_t:10.2f} {sq_t - sq_f:7.2f}")


if __name__ == "__main__":
    main()
