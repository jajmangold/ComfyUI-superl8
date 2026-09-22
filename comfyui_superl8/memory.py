# SPDX-License-Identifier: Apache-2.0
"""Peak HBM tracking for reporting tiling memory savings.

Provides a context manager and standalone helpers that wrap
``torch.cuda.memory_stats`` / ``torch.cuda.max_memory_allocated`` to capture peak
allocated bytes before and after a region of interest. Reports are returned as
structured dicts and a human-readable string.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch


@contextlib.contextmanager
def peak_hbm_monitor(device: int = 0) -> dict[str, float]:
    """Context manager that records peak HBM (GiB) before and after the block.

    Usage::

        info = {}
        with peak_hbm_monitor() as info:
            model(x)
        print(f"peak HBM: {info['peak_gib']:.2f} GiB")

    The dict is populated on exit with keys ``start_gib``, ``end_gib``, ``peak_gib``,
    and ``delta_gib``.
    """
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.max_memory_allocated(device)
    info = {"start_gib": _to_gib(start), "device": device}
    try:
        yield info
    finally:
        peak = torch.cuda.max_memory_allocated(device)
        end = torch.cuda.memory_allocated(device)
        info["end_gib"] = _to_gib(end)
        info["peak_gib"] = _to_gib(peak)
        info["delta_gib"] = _to_gib(peak - start)


def _to_gib(b: int) -> float:
    return b / (1024 ** 3)


def format_hbm_report(info: dict[str, Any], label: str = "") -> str:
    lines = [f"HBM report{' [' + label + ']' if label else ''}:"]
    lines.append(f"  start    : {info.get('start_gib', 0):.2f} GiB")
    lines.append(f"  peak     : {info.get('peak_gib', 0):.2f} GiB")
    lines.append(f"  end      : {info.get('end_gib', 0):.2f} GiB")
    lines.append(f"  delta    : {info.get('delta_gib', 0):.2f} GiB")
    return "\n".join(lines)
