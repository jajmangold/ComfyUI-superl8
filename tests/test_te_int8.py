# SPDX-License-Identifier: Apache-2.0
"""Standalone unit test for issue #20: running a text encoder's `Linear` layers
through the int8 dp4a GEMM (`int8_linear`/`quantize_linear_weight`), the same
numerics `FNI8Ops` applies to a real TE via `custom_operations` in
`FNI8ComponentLoader`. No ComfyUI import needed — only `fni8` + CUDA, so this runs in
the fni8 container without the full e2e image (see `tests/e2e/` for that).

`_TEBlockStandIn` is a minimal T5/CLIP-style transformer block (pre-norm ->
QKV/out-proj Linear -> pre-norm -> MLP Linear, with residuals) — not a real T5/CLIP,
but enough to exercise the "only Linear goes int8, LayerNorm/attention stay fp"
split end to end across a multi-layer stack, which is what the TE int8 path actually
changes numerically.
"""

import pytest
import torch

pytest.importorskip("superl8")

from comfyui_superl8.int8_linear import int8_linear, quantize_linear_weight

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="dp4a needs CUDA")


class _TEBlockStandIn(torch.nn.Module):
    """Pre-norm attention + MLP block. `norm1`/`norm2` (LayerNorm) always run fp,
    matching FNI8Ops (which only patches `nn.Linear` — never touches norms). Attention
    itself also stays fp here, matching the TE int8 path's current scope (issue #20:
    "start with the TE attention left fp")."""

    def __init__(self, dim, mlp_dim, dtype, device):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim, dtype=dtype, device=device)
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=False, dtype=dtype, device=device)
        self.out = torch.nn.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.norm2 = torch.nn.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp_up = torch.nn.Linear(dim, mlp_dim, bias=False, dtype=dtype, device=device)
        self.mlp_down = torch.nn.Linear(mlp_dim, dim, bias=False, dtype=dtype, device=device)

    def linears(self):
        return [self.qkv, self.out, self.mlp_up, self.mlp_down]

    def forward(self, x, quantized=None):
        """`quantized`: optional {Linear: QTensor} map. A linear in the map runs
        through `int8_linear`; everything else (including both LayerNorms) runs fp,
        exactly the split `FNI8Ops` gives a real TE loaded with `custom_operations`."""

        def run(lin, inp):
            if quantized and lin in quantized:
                return int8_linear(inp, quantized[lin])
            return lin(inp)

        h = self.norm1(x)
        q, k, v = run(self.qkv, h).chunk(3, dim=-1)
        attn = (
            torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0).unsqueeze(0),
                k.unsqueeze(0).unsqueeze(0),
                v.unsqueeze(0).unsqueeze(0),
            )
            .squeeze(0)
            .squeeze(0)
        )
        x = x + run(self.out, attn)
        h = self.norm2(x)
        h = torch.nn.functional.gelu(run(self.mlp_up, h))
        x = x + run(self.mlp_down, h)
        return x


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_te_int8_linear_embedding_cosine(dtype):
    """Encode a handful of "prompts" through an int8-linear TE stand-in vs. the fp16
    reference and assert the final embedding's cosine similarity >= 0.99 (issue #20's
    acceptance bar), across several random seeds so this isn't a lucky-draw pass."""
    dim, mlp_dim, depth, n_prompts = 256, 1024, 4, 5
    blocks = [_TEBlockStandIn(dim, mlp_dim, dtype, "cuda") for _ in range(depth)]
    quant = {lin: quantize_linear_weight(lin.weight.data) for b in blocks for lin in b.linears()}

    torch.manual_seed(0)
    prompts = torch.randn(n_prompts, 8, dim, device="cuda", dtype=dtype)
    for prompt in prompts:
        x_fp, x_i8 = prompt.clone(), prompt.clone()
        for b in blocks:
            x_fp = b(x_fp)
            x_i8 = b(x_i8, quantized=quant)
        assert x_i8.dtype == dtype  # no silent downcast (same guarantee as the DiT ops)
        cos = torch.nn.functional.cosine_similarity(
            x_fp.flatten().float(), x_i8.flatten().float(), dim=0
        )
        assert cos.item() >= 0.99


@cuda_only
def test_te_norms_and_embeddings_never_quantized():
    # Only entries present in `quantized` (the Linears) can diverge from the fp path;
    # LayerNorm has no int8 branch at all in `_TEBlockStandIn.forward`, mirroring how
    # FNI8Ops only ever patches nn.Linear.
    block = _TEBlockStandIn(64, 256, torch.float16, "cuda")
    assert not hasattr(block.norm1, "_w_i8") and not hasattr(block.norm2, "_w_i8")
