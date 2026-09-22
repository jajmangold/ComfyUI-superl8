# Qwen-Image / Qwen-Image-Edit DiT — int8 profiling findings

Profiled the `QwenImageTransformer2DModel` (base Qwen-Image and Qwen-Image-Edit-2509,
identical `unet_config`) on the int8 dp4a path, on a **real Tesla V100 (GPU 9), bf16**,
using the published `Qwen__Qwen-Image-Edit-2509.dit.b8.fni8`. Harness:
[`bench/profile_qwen_image.py`](../bench/profile_qwen_image.py).

**Why block-level, not whole-model:** the b8 Qwen DiT is 20.5 GB int8 and does not fit a
16 GB Volta/CMP card — a whole-model single-card forward thrashes CPU-offload PCIe and
profiles the offloader, not the kernels. All 60 blocks are identical
`QwenImageTransformerBlock`s and dominate the FLOPs (prologue / embeddings / output head
<1%), so per-block latency × 60 is an honest, in-budget DiT-step estimate with real int8
weights (block 0, loaded from the `.fni8`). Sequence: 4096 image tokens (~1024 px) + 256
text tokens, dim 3072, 24 heads, head-dim 128.

## Headline: Qwen-Image-Edit is already well-optimized on the int8 path

The int8 acceleration is fully engaged where it is both correct and profitable:

| finding | evidence |
|---|---|
| **All 14 block Linear call-sites run int8 dp4a** — 6 QKV projections (`to_q/k/v`, `add_q/k/v_proj`), `to_out`/`to_add_out`, both `img_mlp`/`txt_mlp` GELU FFNs, and the two `img_mod`/`txt_mod` modulation projections. **0 fp fallbacks.** | int8-engagement audit; torch.profiler shows `fni8::gemm_w8a8_kernel` = **45.6 %** of block CUDA time (the FLOP bulk) |
| **int8 attention engages for un-padded prompts.** ComfyUI itself drops an all-ones text mask (`comfy/text_encoders/qwen_image.py`), so a single prompt reaches the DiT mask-free and the bidirectional int8 FA override fires. | `optimized_attention_masked` is `@wrap_attn`-decorated → the `optimized_attention_override` hook reaches it |

Per-op CUDA breakdown (torch.profiler, 20 block iters, no mask):

```
fni8::gemm_w8a8_kernel<bf16>   45.6%   int8 dp4a linears (FFN + all projections)  <- the int8 win
scaled_dot_product_attention   ~46%    attention, running on the fp SDPA MATH backend
  aten::bmm (magma bf16 sgemm)  28.6%    QK^T and PV
  aten::_safe_softmax + softmax 18.2%    softmax (fp32)
```

## Why attention runs fp — and why that is CORRECT, not a missed lever

Attention (~46 % of the block) does **not** run int8, even mask-free. `fni8.attn_int8_fwd`
has its own **SageAttention accuracy gate** (`detect_q_outlier_domination` — the
`aten::median` visible in the profile): when a Q row is outlier-dominated it quantizes
nothing and returns plain fp SDPA. Qwen-Image's post-projection/RMSNorm/RoPE Q is
**pathologically outlier-heavy**:

```
Q outlier max|q|/median|q|:  max = 57,600   mean = 672    (gate fires above 12)
```

So the kernel correctly rejects int8 QK. Measured on the captured real Q/K/V (4096+256):

| path | SQNR vs fp | attention time |
|---|---|---|
| fp SDPA (what actually runs) | — (reference) | 50.4 ms |
| forced int8, no rotate (bypass gate) | **20.3 dB** | — |
| forced int8, Hadamard rotate (bypass gate) | **23.5 dB** | 33.0 ms (**1.53×**) |

There **is** a latent 1.53× attention win (~16 % of the block) from forcing int8, but it
sits at/just above the 20 dB attention floor **with a 5.7e4× outlier**, and taking it means
**bypassing fni8's SageAttention gate** — i.e. "weakening a gate to claim a win," which the
numerics contract forbids. The documented multi-step int8 image-collapse failure mode
(Z-Image, `docs/int8-dit-validation.md`: single-step SQNR looked fine, the 8-step image
collapsed) is exactly the risk here, and it can't be validated on the 16 GB card (the 20.5 GB
model doesn't fit). **Verdict: attention staying fp is a quality-correct decision.** Making it
int8-safe is a *kernel-side* change (make the SageAttention gate rotation-aware, since the
Hadamard rotation is logit-invariant and lifts SQNR 20.3→23.5 dB) belonging in the `fni8`
repo behind full multi-step image validation — not a ComfyUI-superl8 patch.

## The one safe, shipped optimization: pin self-falling-back attention sites to fp

Because the kernel falls back internally, the outer `Int8AttnGate` measured **SQNR = +inf**
(int8 output bit-identical to fp) and cached the site as *"int8 passed"*. Every subsequent
denoise step then re-dispatched `attn_int8_fwd`, re-paid its **per-call outlier median over
the whole Q (~13 M elements)**, and fell back to the same fp SDPA again — pure wasted work.

The fix (`comfyui_superl8/attention.py`): a **+inf SQNR is treated as a routing FAIL** — it means
the kernel self-fell-back, so the site is pinned to fp SDPA and skips the redundant kernel
round-trip thereafter. A genuinely-engaged int8 site has a *finite* SQNR (Z-Image self-attn
~47 dB) and is unaffected.

Measured on the real Qwen-Image-Edit block (V100, bf16, 4096+256, single-sync):

```
block step:  OLD (re-dispatch kernel each step) = 120.1 ms
             NEW (pin fp after 1st sighting)     = 116.9 ms   -> 2.6% faster
est DiT step (60 blocks):  7.21 s -> 7.02 s   (~0.19 s/step)
output parity NEW vs OLD:  max|Δ| = 0   (bit-identical)
```

Zero quality risk (output is bit-identical — fp SDPA is what the kernel already ran), and it
makes the SQNR-gate accounting honest (`bench/validate_int8_attn_sqnr.py` now reports these
sites as fp, not as inf-dB "passes"). Fleet-specific caveat: measured on this CMP/V100 fleet
(fp16 tensor cores firmware-dead); the number does not transfer to a real V100.

Tests: `tests/test_attn_gate.py::test_gate_routes_selffallback_to_fp` (+ the accurate-int8
case rewritten to use a finite-SQNR kernel).
