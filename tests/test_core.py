# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure logic (no ComfyUI): the int8 dp4a linear, bidirectional
int8 attention, the .fni8 DiT round-trip, and the arch registry. Run in the fni8
test container; the dp4a paths need CUDA."""

import pytest
import torch

pytest.importorskip("superl8")

from comfyui_superl8 import arch, multigpu
from comfyui_superl8.attention import SUPPORTED_HEAD_DIMS, fni8_attention
from comfyui_superl8.int8_linear import (
    dequantize_weight,
    dp4a_linear_ok,
    int8_linear,
    quantize_linear_weight,
    sqnr_gate,
)
from comfyui_superl8.loader import (
    ltx2_detection_metadata,
    load_fni8_dit,
    quantize_dit_state_dict,
    save_dit_fni8,
)

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="dp4a needs CUDA")


# ---- arch registry (pure) ----


def test_arch_registry_has_popular_families():
    fams = set(arch.families())
    for f in (
        "flux1",
        "flux2",
        "sd3",
        "sd35",
        "qwen_image",
        "zimage",
        "wan21",
        "wan22",
        "ltx_video",
        "pixart",
        "sana",
        "hunyuan_video",
        "mochi",
        "cogvideox",
    ):
        assert f in fams, f
    assert arch.get("flux.2").name == "flux2"  # alias resolution
    assert arch.get("z-image").name == "zimage"
    assert arch.get("ideogram").open_weights is False  # closed weights flagged


def test_arch_entries_well_formed():
    for f in arch.families():
        a = arch.get(f)
        assert a.attn_seam in ("attn1_replace", "object_patch")
        assert a.kind in ("image", "video", "audio")
        assert a.head_dim > 0


def test_flux2_arch_documents_klein_stack():
    # Klein 9B (Q6_K GGUF) is the edit engine; its arch entry must document the real
    # TE/VAE the working ComfyUI graph uses (Qwen3-8B + Flux.2 VAE), not the earlier
    # unconfirmed Mistral guess. head_dim 128 must stay in the int8-attention allowlist
    # (SUPPORTED_HEAD_DIMS) — that is what makes Klein's dp4a attention engage.
    a = arch.get("flux2")
    assert a.head_dim == 128 and a.head_dim in SUPPORTED_HEAD_DIMS
    assert a.attn_seam == "object_patch"  # Flux-family -> optimized_attention wrap
    assert "qwen3" in a.text_encoder.lower().replace("-", "").replace(" ", "")
    assert a.clip_type == "flux2"


def test_gguf_loader_node_registered():
    # UnetLoaderFNI8GGUF (the fast Klein path: int8 dp4a linear + auto attention patch)
    # must be exposed to ComfyUI and expose the loader interface. Pure check — no GPU.
    from comfyui_superl8.nodes import NODE_CLASS_MAPPINGS

    assert "UnetLoaderFNI8GGUF" in NODE_CLASS_MAPPINGS
    node = NODE_CLASS_MAPPINGS["UnetLoaderFNI8GGUF"]
    assert node.RETURN_TYPES == ("MODEL",)
    assert node.FUNCTION == "load" and hasattr(node, "load")
    assert "flux2" in arch.families()


def test_arch_entries_document_text_encoder_and_vae():
    # Every open-weights arch must document which fp TE + fp VAE ComfyUI expects
    # (issue #13) — a DiT graph is unrunnable without them, and only the DiT
    # (`text_encoder`/`vae` are never in `_SKIP_HINTS`'s inverse: they're not even
    # DiT state-dict keys) goes int8.
    for f in arch.families():
        a = arch.get(f)
        if a.open_weights:
            assert a.text_encoder, f"{f} missing text_encoder doc"
            assert a.vae, f"{f} missing vae doc"
        # clip_type is best-effort and may be "" (unconfirmed); when set, it must be
        # a plain identifier so `comfy.sd.CLIPType[a.clip_type.upper()]` is a valid
        # lookup shape (actual membership needs a running ComfyUI import).
        assert a.clip_type == "" or a.clip_type.replace("_", "").isalnum()


# ---- multi-GPU policy (pure) ----


def test_multigpu_policy_matches_fleet_verdict():
    assert multigpu.is_link_tolerant("cfg_parallel")
    assert multigpu.is_link_tolerant("component_parallel")
    assert multigpu.is_link_tolerant("pipefusion")
    assert not multigpu.is_link_tolerant("tensor_parallel")  # dead like LLM TP
    assert not multigpu.is_link_tolerant("usp")
    assert multigpu.STRATEGIES["block_offload"] == "fit_only"


def test_component_placement_spreads_devices():
    p = multigpu.plan_components(["cuda:0", "cuda:1", "cuda:2"])
    assert p.dit == "cuda:0" and p.text_encoder == "cuda:1" and p.vae == "cuda:2"
    p1 = multigpu.plan_components(["cuda:0"])
    assert p1.dit == p1.text_encoder == p1.vae == "cuda:0"  # 1 GPU -> all colocated


def test_ltx_component_staging_assigns_resident_pipeline_roles():
    p = multigpu.plan_ltx_components(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    assert p.dit == "cuda:0"
    assert p.upscaler == "cuda:1"
    assert p.video_vae == "cuda:2"
    assert p.audio_vae == "cuda:3"

    p2 = multigpu.plan_ltx_components(["cuda:0", "cuda:1"])
    assert p2.dit == "cuda:0"
    assert p2.upscaler == "cuda:1"
    assert p2.video_vae == "cuda:1"
    assert p2.audio_vae == "cuda:1"


def test_cfg_parallel_plan_spreads_devices():
    p = multigpu.plan_cfg_parallel(["cuda:0", "cuda:1"])
    assert p.cond_device == "cuda:0" and p.uncond_device == "cuda:1"
    p1 = multigpu.plan_cfg_parallel(["cuda:0"])
    assert p1.cond_device == p1.uncond_device == "cuda:0"  # 1 GPU -> condensed
    p0 = multigpu.plan_cfg_parallel([])
    assert p0.cond_device == "cuda:0" and p0.uncond_device == "cuda:0"


def test_available_devices_empty():
    # available_devices returns real or empty depending on CUDA — test the contract
    devs = multigpu.available_devices()
    assert isinstance(devs, list)
    if devs:
        assert devs[0].startswith("cuda:")


def test_cfg_parallel_wrapper_single_branch_passthrough():
    """When there's only cond or only uncond, the wrapper passes through to the
    primary model's apply_model directly (no threading overhead)."""
    plan = multigpu.CFGParallelPlan("cuda:0", "cuda:1")
    call_log = {"n": 0}

    class FakeModel:
        def apply_model(self, x, t, c, cond_or_uncond=None):
            call_log["n"] += 1
            return x * 2.0

    wrapper = multigpu.make_cfg_parallel_wrapper(FakeModel(), None, plan)
    kwargs = {"input": torch.randn(1, 4, 8, 8), "timestep": 1.0, "c": [{}], "cond_or_uncond": [0]}
    out = wrapper(lambda **kw: kw["input"] * 2.0, kwargs)
    assert call_log["n"] == 1
    assert torch.allclose(out, kwargs["input"] * 2.0)


def test_cfg_parallel_wrapper_dispatch_and_gather():
    """Simulate a 2-batch (cond+uncond) forward: the wrapper splits the rows,
    sends each to a separate device, and gathers the results back in order."""
    plan = multigpu.CFGParallelPlan("cpu", "cpu")

    class FakeModel:
        def apply_model(self, x, t, c, cond_or_uncond=None):
            return x + 10.0

    # The wrapper calls replica_dit(x, t, c_moved) for the uncond branch
    class FakeReplica(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_called = False

        def forward(self, x, t, c):
            self.forward_called = True
            return x + 20.0

    replica = FakeReplica()
    wrapper = multigpu.make_cfg_parallel_wrapper(FakeModel(), replica, plan)
    batch_x = torch.randn(2, 4, 8, 8)
    kwargs = {
        "input": batch_x,
        "timestep": 1.0,
        "c": [{"pooled": torch.randn(1, 768)}, {"pooled": torch.randn(1, 768)}],
        "cond_or_uncond": [0, 1],
    }
    out = wrapper(lambda **kw: kw["input"] + 10.0, kwargs)
    assert out.shape == (2, 4, 8, 8)
    assert replica.forward_called
    assert torch.allclose(out[0], batch_x[0] + 10.0)
    assert torch.allclose(out[1], batch_x[1] + 20.0)


def test_cfg_parallel_plan_is_link_tolerant():
    assert multigpu.is_link_tolerant("cfg_parallel")
    assert multigpu.is_link_tolerant("component_parallel")
    assert not multigpu.is_link_tolerant("tensor_parallel")


def test_build_cfg_parallel_replica():
    """A plain torch module can be deep-copied and moved to a target device
    (CPU in this test, since no GPU in CI)."""
    import torch.nn as nn

    class TinyDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 64)

        def forward(self, x, t, c):
            return self.proj(x)

    dit = TinyDiT()
    replica = multigpu.build_cfg_parallel_replica(dit, "cpu")
    assert replica is not dit
    # Both should produce the same output for the same input
    x = torch.randn(2, 64)
    out1 = dit(x, torch.tensor(1.0), {})
    out2 = replica(x, torch.tensor(1.0), {})
    assert torch.allclose(out1, out2)


def test_component_parallel_apply_updates_device():
    """component_parallel_apply should move the diffusion model and update
    load_device to the placement's DiT device. Uses a plain container holding a
    real sub-module on CPU (no CUDA, no self-reference) so the move is exercised
    without a GPU."""
    import torch.nn as nn

    class FakePatcher:
        """Stand-in for ComfyUI's ModelPatcher: a plain object (not an nn.Module)
        carrying a diffusion_model submodule and a load_device attribute."""

        def __init__(self):
            self.diffusion_model = nn.Linear(8, 8)
            self.load_device = torch.device("meta")

    model = FakePatcher()
    placement = multigpu.Placement(dit="cpu")
    multigpu.component_parallel_apply(model, placement)
    # DiT moved to the placement device and load_device updated to match.
    assert next(model.diffusion_model.parameters()).device == torch.device("cpu")
    assert model.load_device == torch.device("cpu")


# ---- int8 linear (dp4a) ----


def test_dp4a_linear_ok():
    assert dp4a_linear_ok(4096) and not dp4a_linear_ok(66)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_int8_linear_matches_fp16(dtype):
    x = torch.randn(16, 1024, device="cuda", dtype=dtype)
    w = torch.randn(2048, 1024, device="cuda", dtype=dtype) * 0.1
    qt = quantize_linear_weight(w)
    y = int8_linear(x, qt)
    ref = torch.nn.functional.linear(x, w)
    cos = torch.nn.functional.cosine_similarity(y.flatten().float(), ref.flatten().float(), dim=0)
    # bf16-native DiTs must not get silently downcast to fp16 mid-graph (that clips
    # bf16's wider dynamic range and is what produced black images).
    assert y.shape == (16, 2048) and y.dtype == dtype and cos.item() >= 0.99


@cuda_only
def test_dp4a_linear_fp32_massive_activation_stays_finite():
    """Regression (Qwen-Image-Edit single-card e2e blocker): the w4a8/int8 DiT runs its
    residual + MLP in **fp32** with MASSIVE activations (this DiT reaches ~1e7). fni8's
    dp4a GEMM can only STORE fp16 or bf16; when the activation is fp32 the store dtype
    must NOT fall back to fp16 (max 65504). The FeedForward down-projection sums
    thousands of terms so its output overflows fp16 -> inf -> NaN -- observed at
    `block0.img_mlp.net.2` and the whole reason the single-card e2e latent went
    non-finite. The fix stores bf16 (fp32's exponent range) and returns finite.

    The reference fp32 output DELIBERATELY exceeds 65504 so an fp16 store would inf."""
    torch.manual_seed(0)
    # FeedForward down-proj geometry: a wide contraction makes the summed output large.
    x = torch.randn(64, 12288, device="cuda", dtype=torch.float32) * 5.0e3
    w = torch.randn(3072, 12288, device="cuda", dtype=torch.float32) * 0.05
    qt = quantize_linear_weight(w)  # per_row_i8
    y = int8_linear(x, qt)
    ref = torch.nn.functional.linear(x, w)
    assert ref.abs().max().item() > 65504.0, "test must drive the output past fp16 max"
    assert torch.isfinite(y).all(), "fp32 activation overflowed the fp16 store -> inf/NaN"
    assert y.dtype == torch.bfloat16, "fp32 activation must store bf16, never fp16"
    cos = torch.nn.functional.cosine_similarity(y.flatten().float(), ref.flatten().float(), dim=0)
    assert cos.item() >= 0.99


def test_dequantize_weight():
    w = torch.randn(256, 128, dtype=torch.float32)
    qt = quantize_linear_weight(w)
    w_deq = dequantize_weight(qt.data, qt.scale, torch.float32)
    cos = torch.nn.functional.cosine_similarity(w.flatten(), w_deq.flatten(), dim=0)
    assert cos >= 0.99


@cuda_only
def test_dequantize_weight_cross_device_scale():
    # Regression (issue #82): a resident FNI8Tensor can pair cuda int8 data with a cpu
    # q_scale (ComfyUI moves the codes but not the plain scale attribute), which crashed
    # Z-Image's cap_pad_token dequant in pad_zimage. dequantize_weight must co-locate the
    # scale onto the data's device instead of raising a device-mismatch error.
    w = torch.randn(256, 128, dtype=torch.float32)
    qt = quantize_linear_weight(w)
    int8_cuda = qt.data.cuda()
    scale_cpu = qt.scale.cpu()
    w_deq = dequantize_weight(int8_cuda, scale_cpu, torch.float16)
    assert w_deq.device.type == "cuda"
    cos = torch.nn.functional.cosine_similarity(w.flatten().cuda(), w_deq.flatten().float(), dim=0)
    assert cos >= 0.99


def test_sqnr_gate_pass():
    y = torch.randn(1000)
    assert sqnr_gate(y, y)  # identical -> cos = 1.0 >= 0.99


def test_sqnr_gate_fail():
    y1 = torch.randn(1000)
    y2 = torch.randn(1000)  # independent -> cos ~ 0
    assert not sqnr_gate(y1, y2)


@cuda_only
def test_int8_linear_sqnr_gate_fail():
    """Adversarial weight where most elements quantize below the int8 granularity,
    so the dp4a path loses information and the per-layer SQNR gate blocks it."""
    out_dim, in_dim = 256, 8192
    # Each row: col 0 = 127 (sets per-row scale = 1.0), rest ~ 0.49 (< 0.5 = below
    # int8 step) — so the int8 weight earns ~zero contribution from cols 1..in_dim-1.
    # With 8191 lost columns, the aggregate error reliably drops cos below 0.99.
    rng = torch.Generator(device="cuda").manual_seed(42)
    w = torch.randn(out_dim, in_dim, device="cuda", dtype=torch.float16, generator=rng) * 0.49
    w[:, 0] = 127.0

    x = torch.randn(128, in_dim, device="cuda", dtype=torch.float16)
    qt = quantize_linear_weight(w)
    y_int8 = int8_linear(x, qt)
    y_fp = torch.nn.functional.linear(x, w)
    cos = torch.nn.functional.cosine_similarity(
        y_int8.float().flatten(), y_fp.float().flatten(), dim=0
    )
    assert not sqnr_gate(y_int8, y_fp)
    assert cos.item() < 0.99


# ---- bidirectional attention ----


@cuda_only
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bidirectional_attention_matches_fp(D, dtype):
    B, S, heads = 1, 32, 4
    inner = heads * D
    q = torch.randn(B, S, inner, device="cuda", dtype=dtype)
    k = torch.randn(B, S, inner, device="cuda", dtype=dtype)
    v = torch.randn(B, S, inner, device="cuda", dtype=dtype)
    out = fni8_attention(q, k, v, heads)
    assert out.shape == (B, S, inner)
    # bf16 in -> bf16 out: no silent downcast to fp16 (that's what clipped bf16-native
    # DiTs' dynamic range and produced black images).
    assert out.dtype == dtype
    # fp reference, same dtype as input
    qh, kh, vh = (t.view(B, S, heads, D).transpose(1, 2) for t in (q, k, v))
    ref = (
        torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
        .transpose(1, 2)
        .reshape(B, S, inner)
    )
    cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos.item() >= 0.99


def test_attention_unsupported_dim_falls_back_to_fp():
    # D=48 not in the fni8 allowlist -> must still work (fp fallback), no fni8 call
    B, S, heads, D = 1, 8, 2, 48
    assert D not in SUPPORTED_HEAD_DIMS
    q = torch.randn(B, S, heads * D)
    out = fni8_attention(q, q.clone(), q.clone(), heads)
    assert out.shape == (B, S, heads * D) and torch.isfinite(out).all()


# ---- .fni8 DiT round-trip ----


def test_dit_quantize_selects_linears_not_norms():
    sd = {
        "double_blocks.0.img_attn.qkv.weight": torch.randn(3072, 1024),
        "double_blocks.0.img_mlp.0.weight": torch.randn(4096, 1024),
        "double_blocks.0.img_norm1.weight": torch.randn(1024),  # norm -> raw
        "double_blocks.0.modulation.lin.weight": torch.randn(6144, 1024),  # modulation -> raw
        "final_layer.linear.weight": torch.randn(64, 1024),  # now int8 via override
    }
    q = quantize_dit_state_dict(sd)
    assert q["double_blocks.0.img_attn.qkv.weight"].scheme == "per_row_i8"
    assert q["double_blocks.0.img_mlp.0.weight"].scheme == "per_row_i8"
    assert q["double_blocks.0.img_norm1.weight"].scheme == "raw"
    assert q["double_blocks.0.modulation.lin.weight"].scheme == "raw"  # excluded by _SKIP_HINTS
    # final_layer was previously raw but now int8 via _LOADER_SKIP_OVERRIDE
    assert q["final_layer.linear.weight"].scheme == "per_row_i8"


def test_dit_fni8_roundtrip(tmp_path):
    sd = {
        "blocks.0.attn.to_q.weight": torch.randn(512, 256),
        "blocks.0.norm.weight": torch.randn(256),
    }
    path = str(tmp_path / "dit.fni8")
    save_dit_fni8(path, sd, arch="flux1")
    loaded = load_fni8_dit(path, device="cpu")
    from comfyui_superl8.superl8_tensor import FNI8Tensor

    # Linear -> FNI8Tensor: a torch.Tensor subclass (so ComfyUI's load pipeline works)
    # carrying the int8 data + fp32 per-row scale.
    w = loaded["blocks.0.attn.to_q.weight"]
    assert isinstance(w, FNI8Tensor) and w.dtype == torch.int8 and w.shape == (512, 256)
    assert w.q_scale is not None and w.q_scale.shape == (512,)
    # Norm -> plain tensor in the model's native dtype (not a QTensor, not forced fp16).
    norm = loaded["blocks.0.norm.weight"]
    assert isinstance(norm, torch.Tensor) and not isinstance(norm, FNI8Tensor)
    assert norm.shape == (256,)


# ---- LTX-2 (ltxav) bundle: shape-inferred detection metadata (no CUDA / weights) ----


def _synthetic_ltx2_dit_sd():
    """A minimal LTX-2.3-shaped extracted DiT state dict (post `model.diffusion_model.`
    strip): the markers `ltx2_detection_metadata` reads, at the real checkpoint's dims.
    Video connector 4096 = 32*128, audio connector 2048 = 32*64 (head_dim 64), gated."""
    import torch as _t

    sd = {
        "audio_adaln_single.linear.weight": _t.empty(18432, 2048),  # marks ltxav
        "transformer_blocks.0.scale_shift_table": _t.empty(9, 4096),  # prompt-adaLN -> True
        "video_embeddings_connector.learnable_registers": _t.empty(128, 4096),
        "audio_embeddings_connector.learnable_registers": _t.empty(128, 2048),
        # Caption/text projections, already renamed from the top-level
        # `text_embedding_projection.*_aggregate_embed.*` by `_LTX2_CAPTION_REMAP`.
        # in=188160 (caption_channels), out=inner_dim (4096 video / 2048 audio).
        "caption_projection.linear_1.weight": _t.empty(4096, 188160),
        "audio_caption_projection.linear_1.weight": _t.empty(2048, 188160),
    }
    for i in (0, 1):  # two connector layers, gated (to_gate_logits present)
        sd[f"video_embeddings_connector.transformer_1d_blocks.{i}.attn1.to_gate_logits.weight"] = (
            _t.empty(32, 4096)
        )
        sd[f"audio_embeddings_connector.transformer_1d_blocks.{i}.attn1.to_gate_logits.weight"] = (
            _t.empty(32, 2048)
        )
    return sd


def test_ltx2_detection_metadata_recovers_shape_config():
    import json

    meta = ltx2_detection_metadata(_synthetic_ltx2_dit_sd())
    assert meta is not None and "config" in meta
    tf = json.loads(meta["config"])["transformer"]
    assert tf["cross_attention_adaln"] is True
    assert tf["connector_num_attention_heads"] == 32  # 4096 / 128
    assert tf["connector_attention_head_dim"] == 128
    assert tf["audio_connector_num_attention_heads"] == 32  # to_gate_logits rows
    assert tf["audio_connector_attention_head_dim"] == 64  # 2048 / 32 (NOT 16*128)
    assert tf["connector_num_layers"] == 2
    assert tf["connector_apply_gated_attention"] is True
    # LTX-2.3 caption projection: single-linear NormSingleLinearTextProjection needs
    # caption_proj_before_connector=True and the non-default caption_channels injected
    # from the projection's input width (else comfy builds it at its 3840 default and the
    # 188160-wide weight fails to load -> random-init conditioning -> NaN latent).
    assert tf["caption_proj_before_connector"] is True
    assert tf["caption_channels"] == 188160


def test_ltx2_caption_projection_remap_targets_model_keys():
    """The archive's top-level `text_embedding_projection.*_aggregate_embed.*` must map
    onto the exact keys `comfy.ldm.lightricks.av_model` builds — video -> caption_projection,
    audio -> audio_caption_projection, each a single `linear_1`. A pure rename (no
    transpose/split): comfy stores these as raw fp nn.Linear [out, in]."""
    from comfyui_superl8.loader import _LTX2_CAPTION_REMAP

    assert _LTX2_CAPTION_REMAP == {
        "text_embedding_projection.video_aggregate_embed.weight": "caption_projection.linear_1.weight",
        "text_embedding_projection.video_aggregate_embed.bias": "caption_projection.linear_1.bias",
        "text_embedding_projection.audio_aggregate_embed.weight": "audio_caption_projection.linear_1.weight",
        "text_embedding_projection.audio_aggregate_embed.bias": "audio_caption_projection.linear_1.bias",
    }


def test_ltx2_detection_metadata_none_for_non_ltxav():
    # A plain (non-LTX-2) DiT has no audio_adaln_single -> no override, no metadata.
    assert ltx2_detection_metadata({"blocks.0.attn.to_q.weight": torch.empty(64, 64)}) is None
    # LTX-Video v1 (native ltxv, no audio branch) is also left untouched.
    assert (
        ltx2_detection_metadata({"transformer_blocks.0.scale_shift_table": torch.empty(6, 2048)})
        is None
    )
