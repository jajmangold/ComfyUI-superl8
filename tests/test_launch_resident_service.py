# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/launch_resident_service.py (issue #181 root-review
blocker #1: the resident service cannot safely elect its own GPU -- election
and CUDA_VISIBLE_DEVICES binding must happen externally, before this process
or its torch/comfy imports exist at all).

No GPU or real weights required: nvidia-smi and the child process are mocked;
the property that matters here -- that this script itself never imports
torch/comfy, and that it sets CUDA_VISIBLE_DEVICES in the CHILD's env before
spawning it -- doesn't need real hardware to verify.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LAUNCHER_PATH = os.path.join(_REPO_ROOT, "scripts", "launch_resident_service.py")


def _load_launcher():
    spec = importlib.util.spec_from_file_location("fni8_launch_resident_service_test", _LAUNCHER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_source_never_imports_torch_or_comfy():
    """The launcher must be provably torch/comfy-free -- it is what elects and
    binds CUDA_VISIBLE_DEVICES before ANY process in this tree could have
    touched CUDA, so it can never itself be the thing that touches CUDA first.
    Source-scanning is the right check here: the property being proven is
    about what the module does NOT do, not about its runtime behavior."""
    src = open(_LAUNCHER_PATH, encoding="utf-8").read()
    assert "import torch" not in src
    assert "import comfy" not in src
    assert "from comfyui_superl8 import" not in src
    assert "import comfyui_superl8" not in src
    # It must load gpu_election.py by file path, not via the package import
    # that would trigger comfyui_superl8/__init__.py (which DOES import torch/comfy).
    assert "from comfyui_superl8 import gpu_election" not in src
    assert "spec_from_file_location" in src


def test_loading_launcher_does_not_import_torch_or_comfy():
    """Loading the module (not just reading its text) must not have the side
    effect of importing torch/comfy either."""
    before = set(sys.modules)
    _load_launcher()
    after = set(sys.modules)
    new_modules = after - before
    assert not any(m == "torch" or m.startswith("torch.") for m in new_modules)
    assert not any(m == "comfy" or m.startswith("comfy.") for m in new_modules)


def test_main_binds_cuda_visible_devices_before_spawning_child():
    """The elected UUID must land in CUDA_VISIBLE_DEVICES in the CHILD's
    environment, and the child must be spawned as `python -m
    comfyui_superl8.resident_service` (a fresh interpreter, so torch/CUDA in that
    process initializes only after the binding is already in its env)."""
    mod = _load_launcher()
    fake_gpu = SimpleNamespace(index=3, uuid="GPU-deadbeef-0000", name="Tesla V100-PCIE-16GB")
    fake_lock = MagicMock()

    captured = {}

    def fake_popen(cmd, env=None, cwd=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
        proc = MagicMock()
        proc.wait.return_value = 0
        return proc

    with patch.object(mod, "elect", return_value=(fake_gpu, fake_lock)), \
         patch.object(subprocess, "Popen", side_effect=fake_popen), \
         patch("signal.signal"):
        rc = mod.main(["--weights-dir", "/mnt/weights"])

    assert rc == 0
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-deadbeef-0000"
    assert captured["env"]["FNI8_GPU_UUID"] == "GPU-deadbeef-0000"
    assert captured["env"]["FNI8_GPU_ELECTED"] == "1"
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1:3] == ["-m", "comfyui_superl8.resident_service"]
    assert "--weights-dir" in captured["cmd"] and "/mnt/weights" in captured["cmd"]
    fake_lock.close.assert_called_once()


def test_main_fails_closed_on_election_failure_and_spawns_nothing():
    mod = _load_launcher()
    with patch.object(mod, "elect", side_effect=RuntimeError("no eligible GPU")), \
         patch.object(subprocess, "Popen") as popen:
        rc = mod.main(["--weights-dir", "/mnt/weights"])

    assert rc == 1
    popen.assert_not_called()


def test_main_releases_lock_even_if_spawn_fails():
    mod = _load_launcher()
    fake_gpu = SimpleNamespace(index=1, uuid="GPU-aaaa", name="CMP 100-210")
    fake_lock = MagicMock()

    with patch.object(mod, "elect", return_value=(fake_gpu, fake_lock)), \
         patch.object(subprocess, "Popen", side_effect=OSError("spawn failed")):
        rc = mod.main(["--weights-dir", "/mnt/weights"])

    assert rc == 1
    fake_lock.close.assert_called_once()


# ---------------------------------------------------------------------------
# Retry logic (#199, #200)
# ---------------------------------------------------------------------------


def test_retry_succeeds_on_second_attempt():
    """_retry_elect retries on failure and succeeds."""
    mod = _load_launcher()
    fake_gpu = SimpleNamespace(index=1, uuid="GPU-retry-ok", name="Tesla V100")
    fake_lock = MagicMock()

    call_count = [0]

    def flaky_elect(lock_dir):
        call_count[0] += 1
        if call_count[0] < 2:
            raise RuntimeError("transient election failure")
        return fake_gpu, fake_lock

    with patch.object(mod, "elect", side_effect=flaky_elect), \
         patch("time.sleep"):
        gpu, lock = mod._retry_elect("/tmp/test-locks")

    assert gpu.uuid == "GPU-retry-ok"
    assert call_count[0] == 2


def test_retry_exhausts_budget():
    """_retry_elect raises after exhausting retries."""
    mod = _load_launcher()

    with patch.object(mod, "elect", side_effect=RuntimeError("no GPU")), \
         patch("time.sleep"), \
         patch.object(mod, "_RETRY_TOTAL_BUDGET_S", 0.0):
        with pytest.raises(RuntimeError, match="no GPU"):
            mod._retry_elect("/tmp/test-locks")


def test_retry_respects_total_budget():
    """_retry_elect stops retrying when time budget is exceeded."""
    mod = _load_launcher()

    sleep_calls = []

    def track_sleep(s):
        sleep_calls.append(s)

    with patch.object(mod, "elect", side_effect=RuntimeError("no GPU")), \
         patch("time.sleep", side_effect=track_sleep), \
         patch.object(mod, "_RETRY_TOTAL_BUDGET_S", 15.0), \
         patch.object(mod, "_RETRY_BASE_DELAY_S", 10.0):
        with pytest.raises(RuntimeError, match="no GPU"):
            mod._retry_elect("/tmp/test-locks")

    # Should have retried at least once with sleep
    assert len(sleep_calls) >= 1


def test_main_uses_retry():
    """main() calls _retry_elect instead of elect directly."""
    mod = _load_launcher()
    fake_gpu = SimpleNamespace(index=1, uuid="GPU-aaaa", name="CMP 100-210")
    fake_lock = MagicMock()

    with patch.object(mod, "_retry_elect", return_value=(fake_gpu, fake_lock)) as mock_retry, \
         patch.object(subprocess, "Popen", side_effect=OSError("spawn failed")), \
         patch("signal.signal"):
        mod.main(["--weights-dir", "/mnt/weights"])

    mock_retry.assert_called_once()
