# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 forward configuration must survive the `.fni8` load path."""

from __future__ import annotations

import json
import sys
import types

# Keep this pure config test runnable in the lightweight CPU unit-test lane.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float16 = "float16"
    sys.modules["torch"] = torch_stub

from comfyui_superl8.loader import ltx2_detection_metadata


class _FakeTensor:
    def __init__(self, *shape: int):
        self.shape = shape

    @property
    def ndim(self):
        return len(self.shape)


def _ltx2_state_dict():
    return {
        "audio_adaln_single.linear.weight": _FakeTensor(3072, 1024),
        "transformer_blocks.0.scale_shift_table": _FakeTensor(9, 2048),
    }


def test_ltx23_detection_restores_forward_geometry():
    metadata = ltx2_detection_metadata(_ltx2_state_dict())
    config = json.loads(metadata["config"])["transformer"]

    assert config["rope_type"] == "split"
    assert config["frequencies_precision"] == "float64"
    assert config["causal_temporal_positioning"] is True
    assert config["use_middle_indices_grid"] is True


def test_persisted_transformer_config_is_authoritative():
    persisted = {
        "rope_type": "default",
        "frequencies_precision": "float32",
        "causal_temporal_positioning": False,
        "use_middle_indices_grid": False,
        "checkpoint_marker": "preserved",
    }
    metadata = ltx2_detection_metadata(_ltx2_state_dict(), persisted=persisted)
    config = json.loads(metadata["config"])["transformer"]

    assert config["cross_attention_adaln"] is True
    for key, value in persisted.items():
        assert config[key] == value


def test_persisted_transformer_config_reads_only_header(monkeypatch):
    class _Reader:
        def __init__(self, path):
            assert path == "/weights/ltx23.fni8"
            self.header = {"__meta__": {"config": {"transformer": {"rope_type": "split"}}}}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setitem(sys.modules, "superl8", types.SimpleNamespace(FQReader=_Reader))
    from comfyui_superl8.loader import ltx2_persisted_transformer_config

    assert ltx2_persisted_transformer_config("/weights/ltx23.fni8") == {"rope_type": "split"}


def test_non_ltx_state_dict_has_no_detection_metadata():
    assert ltx2_detection_metadata({"linear.weight": _FakeTensor(64, 64)}) is None
