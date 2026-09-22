# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from comfyui_superl8 import zimage_profile as profile


def test_canonical_profile_is_exact_fni8_nvme_contract():
    assert profile.DIT_RELATIVE_PATH.endswith(".fni8")
    assert profile.DIT_SHA256 == (
        "1051c94040c0b831c5f1a307d66d51d19a812da45a7742e814782b2027a0355b"
    )
    assert profile.GGUF_CANDIDATE_RELATIVE_PATH.endswith(".gguf")
    assert profile.STEPS == 8
    assert profile.CFG == 1.0
    assert profile.FLOW_SHIFT == 3.0
    assert profile.SAMPLER == "res_multistep"
    assert profile.SCHEDULER == "simple"


def test_resolve_files_rejects_noncanonical_root(tmp_path):
    with pytest.raises(ValueError, match="must be NVMe"):
        profile.resolve_files(str(tmp_path), verify=False)


def test_resolve_files_rejects_symlink(monkeypatch, tmp_path):
    root = tmp_path / "models"
    target = tmp_path / "target"
    target.write_bytes(b"model")
    for relative in (
        profile.DIT_RELATIVE_PATH,
        profile.VAE_RELATIVE_PATH,
        profile.TEXT_ENCODER_RELATIVE_PATH,
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    monkeypatch.setattr(profile, "CANONICAL_MODEL_ROOT", str(root))
    with pytest.raises(ValueError, match="must not be a symlink"):
        profile.resolve_files(verify=False)


def test_resolve_files_rejects_checksum_drift(monkeypatch, tmp_path):
    root = tmp_path / "models"
    for relative in (
        profile.DIT_RELATIVE_PATH,
        profile.VAE_RELATIVE_PATH,
        profile.TEXT_ENCODER_RELATIVE_PATH,
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wrong")
    monkeypatch.setattr(profile, "CANONICAL_MODEL_ROOT", str(root))
    with pytest.raises(ValueError, match="checksum mismatch"):
        profile.resolve_files()


@pytest.mark.parametrize("width,height", [(513, 512), (512, 513), (895, 1152)])
def test_geometry_rejects_fractional_latents(width, height):
    with pytest.raises(ValueError, match="multiples of 8"):
        profile.validate_geometry(width, height)


def test_geometry_returns_explicit_rectangular_latent():
    assert profile.validate_geometry(896, 1152) == (144, 112)


def _synthetic_zimage(pad_names=("x_pad_token", "cap_pad_token")):
    import torch

    from comfyui_superl8.superl8_tensor import FNI8Tensor

    class QuantLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._parameters["weight"] = FNI8Tensor(
                torch.zeros((1, 4), dtype=torch.int8), torch.ones(1)
            )
            self._fni8_sqnr_pass = True

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = QuantLinear()
            self.out = QuantLinear()

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([Block() for _ in range(34)])
    model.ffn = torch.nn.ModuleList([QuantLinear() for _ in range(102)])
    for name in pad_names:
        model._parameters[name] = FNI8Tensor(
            torch.zeros((1, 4), dtype=torch.int8), torch.ones(1)
        )
    return model


def test_inventory_distinguishes_170_linears_from_two_pad_tokens():
    model = _synthetic_zimage()
    inventory = profile.assert_int8_inventory(model, require_cuda=False)
    assert inventory == {
        "tensors_total": 172,
        "cuda_tensors": 0,
        "linears": 170,
        "cuda_linears": 0,
        "qkv": 34,
        "out": 34,
        "pad_tokens": ["cap_pad_token", "x_pad_token"],
        "cuda_pad_tokens": [],
    }


def test_runtime_inventory_counts_only_linear_modules():
    runtime = profile.runtime_int8_inventory(_synthetic_zimage())
    assert runtime == {
        "total": 170,
        "int8": 170,
        "fallback": 0,
        "pending": 0,
        "fallback_layers": [],
        "invalid_fallback_layers": [],
        "pending_layers": [],
    }


def test_runtime_engagement_accepts_only_evidenced_quality_fallback():
    model = _synthetic_zimage()
    fallback = model.ffn[0]
    fallback._fni8_sqnr_pass = False
    fallback._fni8_calib = SimpleNamespace(
        n=8,
        calib_samples=8,
        worst_cos=0.985,
        cos_bar=0.99,
        passed=False,
    )
    runtime = profile.assert_runtime_int8_engagement(model)
    assert runtime["int8"] == 169
    assert runtime["fallback"] == 1
    assert runtime["pending"] == 0
    assert runtime["invalid_fallback_layers"] == []
    assert runtime["fallback_layers"] == [
        {
            "name": "ffn.0",
            "samples": 8,
            "required_samples": 8,
            "worst_cos": 0.985,
            "cos_bar": 0.99,
            "quality_gate_valid": True,
        }
    ]


def test_runtime_engagement_rejects_unproven_fallback():
    model = _synthetic_zimage()
    model.ffn[0]._fni8_sqnr_pass = False
    with pytest.raises(RuntimeError, match="incomplete or unproven"):
        profile.assert_runtime_int8_engagement(model)


def test_runtime_engagement_rejects_pending_linear():
    model = _synthetic_zimage()
    model.ffn[0]._fni8_sqnr_pass = None
    with pytest.raises(RuntimeError, match="incomplete or unproven"):
        profile.assert_runtime_int8_engagement(model)


def test_inventory_rejects_same_count_with_wrong_pad_token_set():
    model = _synthetic_zimage(("x_pad_token", "unexpected_pad_token"))
    with pytest.raises(RuntimeError, match="non-Linear FNI8Tensor set mismatch"):
        profile.assert_int8_inventory(model, require_cuda=False)


def test_explicit_residency_move_covers_linears_and_pad_tokens_and_preserves_metadata():
    import torch

    model = _synthetic_zimage()
    before = {
        name: (param.q_scheme, param.q_group_size, param.q_codebook)
        for name, param in model.named_parameters()
    }
    inventory = profile.move_superl8_tensors_(model, torch.device("cpu"))
    after = {
        name: (param.q_scheme, param.q_group_size, param.q_codebook)
        for name, param in model.named_parameters()
    }
    assert inventory["tensors_total"] == 172
    assert inventory["linears"] == 170
    assert inventory["pad_tokens"] == ["cap_pad_token", "x_pad_token"]
    assert after == before
    assert all(param.q_scale.device.type == "cpu" for param in model.parameters())


def test_explicit_residency_move_places_code_and_scale_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires the safely elected test GPU")
    model = _synthetic_zimage()
    inventory = profile.move_superl8_tensors_(model, torch.device("cuda"))
    assert inventory["cuda_tensors"] == 172
    assert inventory["cuda_linears"] == 170
    assert inventory["cuda_pad_tokens"] == ["cap_pad_token", "x_pad_token"]
    for parameter in model.parameters():
        assert parameter.device.type == "cuda"
        assert parameter.q_scale.device.type == "cuda"
