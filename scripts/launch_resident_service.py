#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""External, pre-Torch launcher for the Z-Image resident service (issue #181,
root-review blocker #1).

`comfyui_superl8.resident_service` cannot safely elect its own GPU: by the time its
`_load_pipeline` runs, `comfyui_superl8/__init__.py` (imported transitively the moment
`python -m comfyui_superl8.resident_service` starts -- Python imports the parent package
before the submodule) has already pulled in `.nodes`, `.gguf_ops`, and friends, which
import `comfy`/`torch`. `comfy.model_management` and this pack's own `gate.is_sm70()`
touch `torch.cuda` at that point, which lazily initializes a CUDA context against
whatever the driver enumerates as device 0 -- the *unfiltered* fleet, potentially
including the pinned live-server card. Setting `os.environ["FNI8_GPU"]` afterward,
from inside that same process, binds nothing: the CUDA context is already live.

This script is the fix: it is stdlib-only (no `torch`, no `comfy`, no `comfyui_superl8`
package import -- it loads `gpu_election.py` by file path, exactly like
`.github/scripts/elect_gpu.py`, specifically to avoid triggering
`comfyui_superl8/__init__.py`), so it can never itself touch a CUDA context. It probes
the fleet, elects a card by UUID, takes the fleet's advisory UUID lock, and only then
sets `CUDA_VISIBLE_DEVICES` to that UUID before spawning the resident service as a
*separate child process* that inherits the restricted environment. Torch never
initializes CUDA in this process, and it only ever sees one device -- the elected
card -- in the child. The lock is held by this (parent) process for the child's
entire lifetime and released when the child exits.

Retry logic (#199, #200): on GPU election failure (NoEligibleGpuError, nvidia-smi
issues), retries with bounded exponential backoff (total <= ~3 min) before giving up.
This handles transient fleet-wide lock contention during container restarts.

Usage:
    python3 scripts/launch_resident_service.py -- \\
        --weights-dir /mnt/24tb/fni8-forge/weights --port 8410

Everything after `--` is passed straight to `python3 -m comfyui_superl8.resident_service`.
Fails closed (nonzero exit, no child spawned) if no card can be elected after all
retries, exactly like `.github/scripts/elect_gpu.py`. Never use the pinned live-server
GPU (AGENTS.md); `FNI8_PINNED_GPU_UUIDS` must name it.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Retry configuration (#199): bounded exponential backoff.
# Total retry budget: ~180 s (3 min).  Attempts: 1 initial + 5 retries.
# Backoff: 5, 10, 20, 40, 80, 125 s (sum = 280, but we cap at 3 min).
_MAX_RETRIES = 5
_RETRY_BASE_DELAY_S = 5.0
_RETRY_MAX_DELAY_S = 80.0
_RETRY_TOTAL_BUDGET_S = 180.0


def _load_gpu_election():
    """Load `comfyui_superl8/gpu_election.py` by file path — NOT `from comfyui_superl8
    import gpu_election` — so this process never imports the `comfyui_superl8` package
    (and therefore never imports torch/comfy). See module docstring."""
    path = os.path.join(_REPO_ROOT, "comfyui_superl8", "gpu_election.py")
    spec = importlib.util.spec_from_file_location("fni8_gpu_election", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass() needs the module registered
    spec.loader.exec_module(mod)
    return mod


def _nvidia_smi(query: str) -> str:
    query_flag = "--query-compute-apps" if "gpu_uuid" in query else "--query-gpu"
    result = subprocess.run(
        ["nvidia-smi", f"{query_flag}={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout


def elect(lock_dir: str):
    """Probe the fleet and elect+lock a card. Returns `(gpu, lock_handle)` from
    `gpu_election.elect_and_lock` — pure stdlib, no torch/CUDA touched here."""
    gpu_election = _load_gpu_election()

    pinned_raw = os.environ.get("FNI8_PINNED_GPU_UUIDS", "")
    pinned_uuids = {u.strip() for u in pinned_raw.split(",") if u.strip()}

    gpus = gpu_election.parse_gpu_csv(
        _nvidia_smi("index,uuid,name,compute_cap,memory.total,memory.free")
    )
    resident = gpu_election.parse_compute_apps_csv(_nvidia_smi("gpu_uuid,pid"))

    gpu, lock_handle = gpu_election.elect_and_lock(gpus, resident, pinned_uuids, lock_dir)
    return gpu, lock_handle


def _retry_elect(lock_dir: str) -> tuple:
    """Elect a GPU with bounded retry and exponential backoff (#199).

    Returns (gpu, lock_handle) on success.  Raises the last exception
    after all retries are exhausted or the total budget is exceeded.
    """
    last_error: Exception | None = None
    total_elapsed = 0.0

    for attempt in range(1 + _MAX_RETRIES):
        try:
            return elect(lock_dir)
        except Exception as e:
            last_error = e
            if attempt >= _MAX_RETRIES:
                break
            delay = min(_RETRY_BASE_DELAY_S * (2**attempt), _RETRY_MAX_DELAY_S)
            if total_elapsed + delay > _RETRY_TOTAL_BUDGET_S:
                delay = max(0, _RETRY_TOTAL_BUDGET_S - total_elapsed)
            if delay <= 0:
                break
            print(
                f"launch_resident_service: election attempt {attempt + 1} failed: {e}; "
                f"retrying in {delay:.0f}s (budget remaining: "
                f"{_RETRY_TOTAL_BUDGET_S - total_elapsed:.0f}s)",
                file=sys.stderr,
            )
            time.sleep(delay)
            total_elapsed += delay

    raise last_error  # type: ignore[misc]


def main(argv: list[str]) -> int:
    lock_dir = os.environ.get("FNI8_GPU_LOCK_DIR", "/tmp/fni8-locks")
    args = list(argv)
    if "--lock-dir" in args:
        i = args.index("--lock-dir")
        if i + 1 >= len(args) or args[i + 1] == "--":
            print("launch_resident_service.py: --lock-dir requires a value", file=sys.stderr)
            return 2
        lock_dir = args[i + 1]
        del args[i : i + 2]
    if args and args[0] == "--":
        args = args[1:]

    try:
        gpu, lock_handle = _retry_elect(lock_dir)
    except Exception as e:  # gpu_election.NoEligibleGpuError, nvidia-smi failures, etc.
        print(f"::error::GPU election failed after retries: {e}", file=sys.stderr)
        return 1

    print(
        f"launch_resident_service: elected GPU {gpu.index} ({gpu.uuid}, {gpu.name}) — "
        "lock acquired, binding CUDA_VISIBLE_DEVICES before spawning the service",
        file=sys.stderr,
    )

    child_env = dict(os.environ)
    # UUID, not index: stable across index churn/reassignment (same reasoning as
    # gpu_election.py's own election-by-UUID). This is read by the CUDA driver in
    # the CHILD process the first time it initializes a CUDA context — well before
    # that, the child will not yet have touched torch.
    child_env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    child_env["FNI8_GPU_UUID"] = gpu.uuid
    child_env["FNI8_GPU_ELECTED"] = "1"
    # FNI8_GPU is *not* a binding mechanism inside this process tree (nothing here
    # reads it to select a device) — kept only for log/provenance parity with the
    # rest of the repo's tooling.
    child_env["FNI8_GPU"] = str(gpu.index)

    cmd = [sys.executable, "-m", "comfyui_superl8.resident_service", *args]
    try:
        proc = subprocess.Popen(cmd, env=child_env, cwd=_REPO_ROOT)
    except Exception as e:
        print(f"::error::failed to spawn resident service: {e}", file=sys.stderr)
        lock_handle.close()
        return 1

    def _forward(signum, _frame):
        proc.send_signal(signum)

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    try:
        return proc.wait()
    finally:
        # Held for the child's entire lifetime; releasing here (after the child has
        # actually exited) is what makes the UUID lock "external" rather than
        # advisory-only within a single process.
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
