# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fail-closed real-weight GPU election (issue #173).

Pure logic, no CUDA/torch/model weights: synthetic `nvidia-smi`-shaped CSV and
fake lock functions stand in for the fleet. Loaded by file path (like
`.github/scripts/elect_gpu.py` does) so this suite also collects outside the e2e
container, where `comfyui_superl8/__init__` pulls in torch/ComfyUI deps this module
itself does not need.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fni8_gpu_election",
    os.path.join(os.path.dirname(__file__), "..", "comfyui_superl8", "gpu_election.py"),
)
gpu_election = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gpu_election  # dataclass() needs the module registered
_SPEC.loader.exec_module(gpu_election)

GpuInfo = gpu_election.GpuInfo
NoEligibleGpuError = gpu_election.NoEligibleGpuError
elect_and_lock = gpu_election.elect_and_lock
filter_eligible = gpu_election.filter_eligible
parse_compute_apps_csv = gpu_election.parse_compute_apps_csv
parse_gpu_csv = gpu_election.parse_gpu_csv

PINNED = {"GPU-pinned-0000"}

# A representative fleet: one pinned live-service V100, two CMP 100-210 cards
# (one busy, one healthy), a GV100 with low free memory, and a non-target-arch
# card that happens to also report enough free memory.
FLEET = [
    GpuInfo(0, "GPU-pinned-0000", "Tesla V100-PCIE-16GB", "7.0", 16384, 15000),
    GpuInfo(1, "GPU-cmp-busy-01", "NVIDIA CMP 170HX", "7.0", 16384, 15000),
    GpuInfo(2, "GPU-cmp-ok-02", "NVIDIA CMP 170HX", "7.0", 16384, 15500),
    GpuInfo(3, "GPU-gv100-low-03", "Tesla V100-PCIE-16GB", "7.0", 16384, 12000),
    GpuInfo(4, "GPU-other-arch-04", "NVIDIA A100-PCIE-40GB", "8.0", 40960, 30000),
]


def _always_lock(lock_dir, uuid):
    return object()


def _never_lock(lock_dir, uuid):
    return None


def test_eligible_cmp_and_gv100_cards_are_candidates():
    """A CMP card and a GV100 card that both clear every gate are eligible —
    the old blanket "exclude CMP by index" rule is gone; capability/HBM/process
    checks decide, not the marketing name."""
    eligible = filter_eligible(FLEET, resident_uuids={"GPU-cmp-busy-01"}, pinned_uuids=PINNED)
    uuids = {g.uuid for g in eligible}
    assert "GPU-cmp-ok-02" in uuids
    # gv100-low-03 fails the free-HBM gate, other-arch-04 fails the compute-cap
    # gate, cmp-busy-01 fails the process gate, pinned-0000 is pinned — the only
    # survivor is the healthy CMP card.
    assert uuids == {"GPU-cmp-ok-02"}


def test_resident_process_excludes_card():
    resident = {"GPU-cmp-ok-02"}
    eligible = filter_eligible(FLEET, resident_uuids=resident, pinned_uuids=PINNED)
    assert all(g.uuid != "GPU-cmp-ok-02" for g in eligible)


def test_pinned_uuid_is_never_elected_even_if_otherwise_healthy():
    fleet = [GpuInfo(0, "GPU-pinned-0000", "Tesla V100-PCIE-16GB", "7.0", 16384, 16000)]
    eligible = filter_eligible(fleet, resident_uuids=set(), pinned_uuids={"GPU-pinned-0000"})
    assert eligible == []
    with pytest.raises(NoEligibleGpuError):
        elect_and_lock(
            fleet, resident_uuids=set(), pinned_uuids={"GPU-pinned-0000"}, lock_dir="/unused"
        )


def test_insufficient_free_hbm_excludes_card():
    eligible = filter_eligible(FLEET, resident_uuids=set(), pinned_uuids=PINNED)
    assert all(g.uuid != "GPU-gv100-low-03" for g in eligible)
    # Confirm it's the HBM gate specifically: raising the floor to match its
    # actual free memory brings it back.
    eligible_lenient = filter_eligible(
        FLEET, resident_uuids=set(), pinned_uuids=PINNED, min_free_mib=12000
    )
    assert "GPU-gv100-low-03" in {g.uuid for g in eligible_lenient}


def test_lock_held_skips_to_next_candidate():
    """cmp-ok-02 is the sole eligible candidate for the default fleet; simulate
    its lock already being held and confirm election falls through to failure
    rather than silently reusing a locked card."""
    with pytest.raises(NoEligibleGpuError):
        elect_and_lock(
            FLEET,
            resident_uuids=set(),
            pinned_uuids=PINNED,
            lock_dir="/unused",
            lock_fn=_never_lock,
        )


def test_lock_held_on_top_candidate_falls_through_to_next():
    fleet = [
        FLEET[0],
        GpuInfo(2, "GPU-cmp-ok-02", "NVIDIA CMP 170HX", "7.0", 16384, 15500),
        GpuInfo(5, "GPU-cmp-ok-05", "NVIDIA CMP 170HX", "7.0", 16384, 15000),
    ]
    locked = {"GPU-cmp-ok-02"}

    def lock_fn(lock_dir, uuid):
        return None if uuid in locked else object()

    gpu, handle = elect_and_lock(
        fleet, resident_uuids=set(), pinned_uuids=PINNED, lock_dir="/unused", lock_fn=lock_fn
    )
    assert gpu.uuid == "GPU-cmp-ok-05"


def test_no_eligible_candidate_fails_closed():
    fleet = [GpuInfo(0, "GPU-pinned-0000", "Tesla V100-PCIE-16GB", "7.0", 16384, 16000)]
    with pytest.raises(NoEligibleGpuError):
        elect_and_lock(
            fleet,
            resident_uuids=set(),
            pinned_uuids=PINNED,
            lock_dir="/unused",
            lock_fn=_always_lock,
        )


def test_empty_pinned_list_refuses_rather_than_assumes_nothing_pinned():
    with pytest.raises(NoEligibleGpuError):
        filter_eligible(FLEET, resident_uuids=set(), pinned_uuids=set())


def test_stale_pinned_uuid_refuses_rather_than_assumes_card_was_removed():
    with pytest.raises(NoEligibleGpuError, match="absent from the probed fleet"):
        filter_eligible(FLEET, resident_uuids=set(), pinned_uuids={"GPU-stale"})


def test_priority_prefers_target_hardware_over_non_target_when_both_eligible():
    fleet = [
        FLEET[0],
        GpuInfo(1, "GPU-cmp", "NVIDIA CMP 170HX", "7.0", 16384, 15000),
        GpuInfo(2, "GPU-other", "NVIDIA Mystery", "7.0", 16384, 15500),
    ]
    eligible = filter_eligible(
        fleet, resident_uuids=set(), pinned_uuids=PINNED, required_compute_cap="7.0"
    )
    assert [g.uuid for g in eligible] == ["GPU-cmp", "GPU-other"]


def test_acquire_lock_is_exclusive(tmp_path):
    fh1 = gpu_election.acquire_lock(str(tmp_path), "GPU-x")
    assert fh1 is not None
    fh2 = gpu_election.acquire_lock(str(tmp_path), "GPU-x")
    assert fh2 is None
    fh1.close()
    fh3 = gpu_election.acquire_lock(str(tmp_path), "GPU-x")
    assert fh3 is not None
    fh3.close()


def test_parse_gpu_csv():
    text = (
        "0, GPU-aaaa, Tesla V100-PCIE-16GB, 7.0, 16384, 15000\n"
        "1, GPU-bbbb, NVIDIA CMP 170HX, 7.0, 8192, 7000\n"
    )
    gpus = parse_gpu_csv(text)
    assert len(gpus) == 2
    assert gpus[0] == GpuInfo(0, "GPU-aaaa", "Tesla V100-PCIE-16GB", "7.0", 16384, 15000)
    assert gpus[1].name == "NVIDIA CMP 170HX"


def test_parse_gpu_csv_ignores_blank_lines():
    assert parse_gpu_csv("\n\n") == []


def test_parse_compute_apps_csv():
    text = "GPU-aaaa, 12345\nGPU-aaaa, 12346\nGPU-bbbb, 999\n"
    resident = parse_compute_apps_csv(text)
    assert resident == {"GPU-aaaa", "GPU-bbbb"}


def test_parse_compute_apps_csv_empty_when_no_processes():
    assert parse_compute_apps_csv("") == set()
