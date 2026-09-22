# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / ".github" / "scripts" / "elect_gpu.py"
SPEC = importlib.util.spec_from_file_location("fni8_elect_gpu_script", SCRIPT)
elect_gpu = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = elect_gpu
SPEC.loader.exec_module(elect_gpu)


def _result(stdout: str = "", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, returncode=returncode)


def test_host_lock_is_owned_by_task_container_and_auto_removed(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "task-container")
    task_id = "a" * 64
    lock_id = "b" * 64

    with patch.object(
        elect_gpu.subprocess,
        "run",
        side_effect=[
            _result(f"{task_id}\n"),
            _result("994:972\n"),
            _result(f"{lock_id}\n"),
            _result("true\n"),
        ],
    ) as run:
        handle = elect_gpu._acquire_host_lock("/locks", "GPU-dead-beef")

    assert handle.container_id == lock_id
    assert handle.owner_container == task_id
    command = run.call_args_list[2].args[0]
    assert command[0:3] == ["docker", "run", "-d"]
    assert "--rm" in command
    assert command[command.index("--pid") + 1] == f"container:{task_id}"
    assert command[command.index("--name") + 1].endswith(task_id[:12])
    assert "/proc/1/stat" in command[-1]


def test_host_lock_rejects_invalid_uuid_without_docker():
    with patch.object(elect_gpu.subprocess, "run") as run:
        assert elect_gpu._acquire_host_lock("/locks", "../../escape") is None
    run.assert_not_called()


def test_owned_workload_cleanup_uses_all_exact_labels():
    task_id = "a" * 64
    child_id = "c" * 64

    with patch.object(
        elect_gpu.subprocess,
        "run",
        side_effect=[_result(f"{child_id}\n"), _result()],
    ) as run:
        elect_gpu._remove_owned_workload(task_id, "GPU-dead-beef")

    query = run.call_args_list[0].args[0]
    assert query[:3] == ["docker", "ps", "-aq"]
    assert "label=content-factory.role=fni8-ci-e2e" in query
    assert f"label=content-factory.owner-container={task_id}" in query
    assert "label=content-factory.gpu-uuid=GPU-dead-beef" in query
    assert run.call_args_list[1].args[0] == ["docker", "rm", "-f", child_id]


def test_owned_workload_cleanup_timeout_is_nonfatal():
    with patch.object(
        elect_gpu.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(["docker", "ps"], 60),
    ):
        elect_gpu._remove_owned_workload("a" * 64, "GPU-dead-beef")


def test_main_rejects_lock_dir_without_value(capsys):
    assert elect_gpu.main(["--lock-dir", "--", "/bin/true"]) == 2
    assert "--lock-dir requires a value" in capsys.readouterr().err
