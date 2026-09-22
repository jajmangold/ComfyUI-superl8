# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for Comfy LoRA patches on the native GGUF path."""

from __future__ import annotations

import sys
import types

import torch

from comfyui_superl8 import gguf_ops


class _StandardLoRA:
    def __init__(self, up, down, *, alpha=None, mid=None, dora=None, reshape=None):
        self.weights = (up, down, alpha, mid, dora, reshape)


class _Weight:
    tensor_type = types.SimpleNamespace(name="Q4_K")

    def __init__(self, patches):
        self.tensor_shape = (3, 256)
        self.patches = (
            [(patches, "transformer.test.weight")]
            if patches
            else []
        )
        self._bytes = torch.zeros(3, 144, dtype=torch.uint8)

    def to(self, _device):
        return self._bytes


class _Layer:
    def __init__(self, patches):
        self.weight = _Weight(patches)
        self.bias = None
        self._fni8_sqnr_pass = True


def _patch(adapter, *, strength=1.0, strength_model=1.0, offset=None, function=None):
    return (strength, adapter, strength_model, offset, function)


def _native_forward(monkeypatch, base_weight):
    fake_fni8 = types.SimpleNamespace(
        linear_q4k=lambda x, _weight, bias, out_dtype: torch.nn.functional.linear(
            x, base_weight, bias
        ).to(out_dtype)
    )
    monkeypatch.setitem(sys.modules, "superl8", fake_fni8)
    monkeypatch.setattr(gguf_ops, "_fni8_ok", lambda: True)

    def base_forward(layer, value):
        output = torch.nn.functional.linear(value, base_weight)
        return gguf_ops._apply_standard_lora_residuals(layer, value, output)

    return gguf_ops._make_forward(base_forward)


def test_native_gguf_path_applies_stacked_standard_lora(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    value = torch.randn(2, 4, 256, generator=generator)
    base = torch.randn(3, 256, generator=generator)
    down_a = torch.randn(8, 256, generator=generator)
    up_a = torch.randn(3, 8, generator=generator)
    down_b = torch.randn(4, 256, generator=generator)
    up_b = torch.randn(3, 4, generator=generator)
    patches = [
        _patch(_StandardLoRA(up_a, down_a, alpha=4.0), strength=0.5),
        _patch(_StandardLoRA(up_b, down_b), strength=1.25),
    ]
    patches[0][1].multiplier = 9.0
    layer = _Layer(patches)

    output = _native_forward(monkeypatch, base)(layer, value)
    expected = torch.nn.functional.linear(value, base)
    expected += 0.25 * torch.nn.functional.linear(
        torch.nn.functional.linear(value, down_a), up_a
    )
    expected += 1.25 * torch.nn.functional.linear(
        torch.nn.functional.linear(value, down_b), up_b
    )

    torch.testing.assert_close(output, expected)
    assert layer._fni8_lora_state == "native"
    assert layer._fni8_lora_patch_count == 2


def test_standard_lora_preparation_does_not_persist_device_tensors():
    adapter = _StandardLoRA(torch.ones(3, 2), torch.ones(2, 256))
    layer = _Layer([_patch(adapter)])
    value = torch.ones(1, 256)

    first, reason = gguf_ops._prepare_standard_lora_residuals(layer, value, 3, 256)
    second, repeated_reason = gguf_ops._prepare_standard_lora_residuals(
        layer, value, 3, 256
    )

    assert reason is None
    assert repeated_reason is None
    assert len(first) == len(second) == 1
    assert not hasattr(layer, "_fni8_lora_cache")


def test_no_patch_preserves_native_output(monkeypatch):
    generator = torch.Generator().manual_seed(11)
    value = torch.randn(3, 256, generator=generator)
    base = torch.randn(3, 256, generator=generator)
    layer = _Layer([])

    output = _native_forward(monkeypatch, base)(layer, value)

    torch.testing.assert_close(output, torch.nn.functional.linear(value, base))
    assert layer._fni8_lora_state == "none"
    assert layer._fni8_lora_patch_count == 0


def test_audit_reports_patches_that_have_not_executed():
    adapter = _StandardLoRA(torch.ones(3, 2), torch.ones(2, 256))
    patched = _Layer([_patch(adapter)])
    plain = _Layer([])

    assert gguf_ops.gguf_lora_audit(
        types.SimpleNamespace(modules=lambda: [patched, plain])
    ) == {"native": 0, "fallback": 0, "unapplied": 1, "none": 1}


def test_unsupported_patch_falls_back_instead_of_being_ignored(monkeypatch):
    value = torch.ones(1, 256)
    adapter = _StandardLoRA(torch.ones(3, 2), torch.ones(2, 256), dora=torch.ones(3))
    layer = _Layer([_patch(adapter)])
    calls = []

    def base_forward(_layer, input_value):
        calls.append(input_value)
        return torch.full((1, 3), 17.0)

    fake_fni8 = types.SimpleNamespace(
        linear_q4k=lambda *_args, **_kwargs: torch.full((1, 3), -1.0)
    )
    monkeypatch.setitem(sys.modules, "superl8", fake_fni8)
    monkeypatch.setattr(gguf_ops, "_fni8_ok", lambda: True)

    output = gguf_ops._make_forward(base_forward)(layer, value)

    torch.testing.assert_close(output, torch.full((1, 3), 17.0))
    assert calls == [value]
    assert layer._fni8_lora_state == "fallback"
    assert "DoRA" in layer._fni8_lora_fallback_reason


def test_invalid_standard_lora_shape_falls_back(monkeypatch):
    value = torch.ones(1, 256)
    adapter = _StandardLoRA(torch.ones(4, 2), torch.ones(2, 256))
    layer = _Layer([_patch(adapter)])

    monkeypatch.setitem(
        sys.modules,
        "fni8",
        types.SimpleNamespace(linear_q4k=lambda *_args, **_kwargs: torch.zeros(1, 3)),
    )
    monkeypatch.setattr(gguf_ops, "_fni8_ok", lambda: True)
    output = gguf_ops._make_forward(
        lambda _layer, _value: torch.full((1, 3), 23.0)
    )(layer, value)

    torch.testing.assert_close(output, torch.full((1, 3), 23.0))
    assert layer._fni8_lora_state == "fallback"
    assert "shape" in layer._fni8_lora_fallback_reason


def test_kernel_ineligible_patched_layer_is_audited_as_fallback(monkeypatch):
    value = torch.ones(1, 256)
    adapter = _StandardLoRA(torch.ones(3, 2), torch.ones(2, 256))
    layer = _Layer([_patch(adapter)])
    monkeypatch.setattr(gguf_ops, "_fni8_ok", lambda: False)

    output = gguf_ops._make_forward(
        lambda _layer, _value: torch.full((1, 3), 29.0)
    )(layer, value)

    torch.testing.assert_close(output, torch.full((1, 3), 29.0))
    assert layer._fni8_lora_state == "fallback"
    assert gguf_ops.gguf_lora_audit(
        types.SimpleNamespace(modules=lambda: [layer])
    ) == {"native": 0, "fallback": 1, "unapplied": 0, "none": 0}
