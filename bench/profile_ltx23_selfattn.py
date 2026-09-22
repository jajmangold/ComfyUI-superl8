# SPDX-License-Identifier: Apache-2.0
"""Micro-profile the LTX-2.3 VIDEO self-attention internals to locate the residual cost
after int8 wiring: projections (to_q/k/v, int8) vs the dp4a FA2 attention kernel vs the
gated-attention epilogue vs q/k RMSNorm vs to_out. Sweeps sequence length so the O(S^2)
attention kernel vs O(S) projection/FFN scaling is visible.

    python3 bench/profile_ltx23_selfattn.py
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


def main():
    from comfyui_superl8.gate import is_sm70
    if not is_sm70():
        print("SKIP: needs sm_70")
        return
    import superl8
    from superl8 import QTensor
    from comfyui_superl8.int8_linear import int8_linear, quantize_linear_weight

    dev = torch.device("cuda:0")
    dt = torch.bfloat16
    dim, heads, hd = 4096, 32, 128
    g = torch.Generator().manual_seed(0)

    def qlin(out_f, in_f):
        w = (torch.randn(out_f, in_f, generator=g) * 0.02).to(dt)
        qt = quantize_linear_weight(w.float())
        return QTensor(qt.data.to(dev), qt.scale.to(dev), scheme="per_row_i8")

    wq, wk, wv, wo = (qlin(dim, dim) for _ in range(4))
    wgate = qlin(heads, dim)

    print(f"{'seq':>6} {'proj_qkv':>9} {'attn_i8':>9} {'attn_sdpa':>10} {'to_out':>8} "
          f"{'gate_ep':>8} {'rmsnorm':>8}   (ms, bf16)")
    for S in (1024, 2048, 4096, 8192):
        x = (torch.randn(1, S, dim, generator=g) * 0.5).to(dev, dt)
        rms = torch.nn.RMSNorm(dim, eps=1e-5, dtype=dt, device=dev)

        def proj():
            return int8_linear(x.reshape(-1, dim), wq), int8_linear(x.reshape(-1, dim), wk), \
                   int8_linear(x.reshape(-1, dim), wv)

        q = int8_linear(x.reshape(-1, dim), wq).view(1, S, heads, hd).transpose(1, 2).contiguous()
        k = q.clone()
        v = q.clone()

        def attn_i8():
            return superl8.attn_int8_fwd(q, k, v, causal=False).to(dt)

        def attn_sdpa():
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        oin = torch.randn(S, dim, generator=g).to(dev, dt)

        def to_out():
            return int8_linear(oin, wo)

        def gate_ep():
            gl = int8_linear(x.reshape(-1, dim), wgate).view(1, S, heads)
            out = oin.view(1, S, heads, hd)
            gates = 2.0 * torch.sigmoid(gl)
            return (out * gates.unsqueeze(-1)).view(1, S, dim)

        def rmsnorm():
            return rms(x)

        r = [_t(proj), _t(attn_i8), _t(attn_sdpa), _t(to_out), _t(gate_ep), _t(rmsnorm)]
        print(f"{S:6d} {r[0]:9.3f} {r[1]:9.3f} {r[2]:10.3f} {r[3]:8.3f} {r[4]:8.3f} {r[5]:8.3f}")


if __name__ == "__main__":
    main()
