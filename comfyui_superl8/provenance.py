# SPDX-License-Identifier: Apache-2.0
"""Provenance tracking for the Z-Image resident HTTP service (issue #181).

Deterministic, machine-readable provenance for every generation request:
exact model/component revisions, checksums, pipeline semantics, and
request parameters.  No timestamps or RNG leakage — same inputs produce
identical provenance dicts.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Any

# Service version — bumped on API-breaking changes or pipeline behaviour changes.
SERVICE_VERSION = "0.2.0"


def _git_commit() -> str:
    """Best-effort HEAD commit hash. Returns 'unknown' outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _file_sha256(path: str) -> str | None:
    """SHA-256 of a file, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return None


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "not installed"


def _cuda_version() -> str:
    try:
        import torch

        return torch.version.cuda or "none"
    except (ImportError, AttributeError):
        return "unknown"


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "none"


def _python_version() -> str:
    return platform.python_version()


@dataclass(frozen=True)
class Provenance:
    """Deterministic provenance for one generation request."""

    service_version: str = SERVICE_VERSION
    git_commit: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    python_version: str = ""
    gpu_name: str = ""
    pipeline: str = "zimage_turbo_fni8_int8_dp4a"
    dit_format: str = "fni8_b8_w8a8"
    # Pipeline semantics (per AGENTS.md: preserve these exactly)
    lossless_qkv_refusion: bool = True
    fp_normalization: bool = True
    fp_softmax: bool = True
    fp_vae: bool = True
    sqnr_fp_fallback: bool = True
    model_sampling_shift: float | None = None
    sampler_name: str = "res_multistep"
    nfe: int = 8
    explicit_seed: bool = True
    empty_negative: bool = True
    no_cache: bool = True
    cfg: float = 1.0
    # Request parameters
    seed: int = 0
    steps: int = 8
    width: int = 512
    height: int = 512
    in_channels: int = 0
    vae_stride: int = 8
    scheduler: str = "simple"
    # Model provenance
    dit_checksum: str | None = None
    vae_checksum: str | None = None
    te_checksum: str | None = None
    # TE source provenance (#200)
    te_source: str = "local"  # "local" | "sidecar"
    embed_cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_provenance(
    *,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    scheduler: str,
    sampler_name: str = "res_multistep",
    model_sampling_shift: float | None = None,
    dit_path: str | None = None,
    vae_path: str | None = None,
    te_path: str | None = None,
    dit_checksum: str | None = None,
    vae_checksum: str | None = None,
    te_checksum: str | None = None,
    in_channels: int = 4,
    te_source: str = "local",
    embed_cache_hit: bool = False,
) -> Provenance:
    """Build a Provenance record for a generation request.

    All fields are deterministic given the same inputs — no timestamps,
    no RNG, no mutable global state.
    """
    return Provenance(
        git_commit=_git_commit(),
        torch_version=_torch_version(),
        cuda_version=_cuda_version(),
        python_version=_python_version(),
        gpu_name=_gpu_name(),
        seed=seed,
        steps=steps,
        width=width,
        height=height,
        in_channels=in_channels,
        cfg=cfg,
        scheduler=scheduler,
        sampler_name=sampler_name,
        model_sampling_shift=model_sampling_shift,
        dit_checksum=dit_checksum or (_file_sha256(dit_path) if dit_path else None),
        vae_checksum=vae_checksum or (_file_sha256(vae_path) if vae_path else None),
        te_checksum=te_checksum or (_file_sha256(te_path) if te_path else None),
        te_source=te_source,
        embed_cache_hit=embed_cache_hit,
    )
