# SPDX-License-Identifier: Apache-2.0
"""Native GGUF activation contract."""

from pathlib import Path

import torch

from bench.full_pipeline_ltx_gguf import tensor_sha256


def test_package_records_native_gguf_patch_attempt():
    """Installing this node pack must not require a separate enable-only custom node."""
    package_init = Path(__file__).parents[1] / "comfyui_superl8" / "__init__.py"
    source = package_init.read_text()
    assert "GGUF_NATIVE_ENABLED" in source
    assert "enable_fni8_gguf" in source


def test_native_gguf_gate_bounds_reference_rows(monkeypatch):
    import comfyui_superl8.gguf_ops as gguf_ops

    class _Weight:
        tensor_type = "Q4_K"
        tensor_shape = (2, 256)

        def to(self, device):
            return torch.zeros(2, 144, dtype=torch.uint8, device=device)

    class _Layer:
        weight = _Weight()
        bias = None

    reference_rows = []

    def base_forward(layer, value):
        reference_rows.append(value.reshape(-1, value.shape[-1]).shape[0])
        return value.sum(dim=-1, keepdim=True).expand(*value.shape[:-1], 2)

    def native_linear(fni8, name, value, weight, bias, out_dtype):
        return value.sum(dim=-1, keepdim=True).expand(*value.shape[:-1], 2)

    monkeypatch.setenv("FNI8_SQNR_CALIB_MAX_ROWS", "4")
    monkeypatch.setattr(gguf_ops, "_fni8_ok", lambda: True)
    monkeypatch.setattr(gguf_ops, "_native_linear", native_linear)
    value = torch.randn(2, 8, 256)

    output = gguf_ops._make_forward(base_forward)(_Layer(), value)

    assert output.shape == (2, 8, 2)
    assert reference_rows == [4]


def test_tensor_sha256_is_deterministic_and_content_sensitive():
    value = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value + 1)
