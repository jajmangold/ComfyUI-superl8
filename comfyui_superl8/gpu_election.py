# SPDX-License-Identifier: Apache-2.0
"""Fail-closed GPU election for the opt-in real-weight `comfy_e2e` CI lane (#173).

The old election in `.github/workflows/ci.yml` hard-excluded a fixed set of GPU
*indices* as "CMP/marketing cards (no real VRAM)". Indices are not a stable
identity — they can be reassigned across a driver restart or reboot — and the
premise was wrong besides: CMP 100-210 cards are cut-down Volta parts with no
display output, but they carry the same real HBM as a GV100 and are exactly the
kind of card AGENTS.md targets (`CMP 100-210/GV100 sm_70`). A blanket index
exclusion could just as easily exclude a healthy CMP card as include a renumbered
pinned one.

This module replaces that with an election over physical properties that stay
true regardless of index churn:

  UUID         stable hardware identity — pinned live-service cards are excluded
               by UUID, never by index.
  capability   compute capability must match the fleet's target (sm_70 / "7.0"
               per AGENTS.md); this is what actually distinguishes a usable card,
               not a marketing name.
  free HBM     must clear a headroom floor so a DiT load has room to land.
  process      a GPU with any resident compute process is treated as busy, even
               if free-memory bookkeeping hasn't caught up yet.
  admission    a per-UUID advisory lock (see `acquire_lock`) so concurrent CI
               jobs never pile onto the same card.

No candidate surviving every gate -> `NoEligibleGpuError`, never a fallback to a
pinned card, an unfiltered index, or "PR#-mod-anything".
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass

TARGET_COMPUTE_CAP = "7.0"  # CMP 100-210 / GV100 sm_70 (AGENTS.md hard rule)
MIN_FREE_MIB = 14500  # measured LTX stage ceiling; require an effectively drained card


@dataclass(frozen=True)
class GpuInfo:
    """One physical card, as reported by `nvidia-smi --query-gpu`."""

    index: int
    uuid: str
    name: str
    compute_cap: str
    mem_total_mib: int
    mem_free_mib: int


class NoEligibleGpuError(RuntimeError):
    """Raised when no card survives the election. Never caught to fall back to an
    unfiltered or pinned card — the caller must fail the CI job."""


def parse_gpu_csv(text: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.total,memory.free
    --format=csv,noheader,nounits` output."""
    gpus = []
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].strip():
            continue
        idx, uuid, name, cap, total, free = (c.strip() for c in row[:6])
        gpus.append(
            GpuInfo(
                index=int(idx),
                uuid=uuid,
                name=name,
                compute_cap=cap,
                mem_total_mib=int(float(total)),
                mem_free_mib=int(float(free)),
            )
        )
    return gpus


def parse_compute_apps_csv(text: str) -> set[str]:
    """Parse `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits`
    output into the set of GPU UUIDs with at least one resident process."""
    resident = set()
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].strip():
            continue
        resident.add(row[0].strip())
    return resident


def _is_target_card(gpu: GpuInfo) -> bool:
    """True for a full V100 (GV100) or CMP 100-210 card — the fleet's target
    hardware, ranked ahead of any other card that happens to report sm_70."""
    name = gpu.name.upper()
    return "V100" in name or "CMP" in name


def filter_eligible(
    gpus: list[GpuInfo],
    resident_uuids: set[str],
    pinned_uuids: set[str],
    min_free_mib: int = MIN_FREE_MIB,
    required_compute_cap: str = TARGET_COMPUTE_CAP,
) -> list[GpuInfo]:
    """Return eligible cards, ordered by election priority (target hardware first,
    then most free HBM). Does not check or acquire the admission lock.

    Fail-closed: an empty ``pinned_uuids`` is refused outright rather than treated
    as "nothing is pinned" — the live-service card must be named explicitly, never
    assumed absent.
    """
    if not pinned_uuids:
        raise NoEligibleGpuError(
            "refusing to elect a GPU with no pinned live-service UUIDs configured "
            "(set FNI8_PINNED_GPU_UUIDS) — never assume no card is pinned"
        )
    unknown_pins = pinned_uuids - {gpu.uuid for gpu in gpus}
    if unknown_pins:
        raise NoEligibleGpuError(
            "refusing to elect with pinned UUIDs absent from the probed fleet: "
            + ", ".join(sorted(unknown_pins))
        )

    eligible = [
        g
        for g in gpus
        if g.uuid not in pinned_uuids
        and g.compute_cap == required_compute_cap
        and g.mem_free_mib >= min_free_mib
        and g.uuid not in resident_uuids
    ]
    eligible.sort(key=lambda g: (0 if _is_target_card(g) else 1, -g.mem_free_mib))
    return eligible


def acquire_lock(lock_dir: str, uuid: str) -> io.TextIOWrapper | None:
    """Try to take a non-blocking exclusive advisory lock for `uuid`. Returns the
    open file object (keep it referenced — closing it releases the lock) on
    success, or ``None`` if another process already holds it.

    Uses `fcntl.flock`, the same discipline the previous bash election used (one
    lock file per card, held for the run's duration), just keyed by UUID instead
    of index so a reassigned index can't collide two different physical cards
    onto one lock file.
    """
    import fcntl

    os.makedirs(lock_dir, exist_ok=True)
    path = os.path.join(lock_dir, f"card-{uuid}.lock")
    fh = open(path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def elect_and_lock(
    gpus: list[GpuInfo],
    resident_uuids: set[str],
    pinned_uuids: set[str],
    lock_dir: str,
    min_free_mib: int = MIN_FREE_MIB,
    required_compute_cap: str = TARGET_COMPUTE_CAP,
    lock_fn=acquire_lock,
) -> tuple[GpuInfo, io.TextIOWrapper]:
    """Elect a card and hold its admission lock. Returns `(gpu, lock_handle)`;
    the caller must keep `lock_handle` alive for the duration of the run and
    close it to release the card.

    Raises `NoEligibleGpuError` if no candidate clears every gate, or if every
    candidate that does is already locked by another job.
    """
    eligible = filter_eligible(
        gpus, resident_uuids, pinned_uuids, min_free_mib, required_compute_cap
    )
    for gpu in eligible:
        handle = lock_fn(lock_dir, gpu.uuid)
        if handle is not None:
            return gpu, handle
    raise NoEligibleGpuError(
        "No suitable GPU found — all valid cards either have <"
        f"{min_free_mib} MiB free, are pinned, host a resident process, or are "
        "locked by other CI jobs"
    )
