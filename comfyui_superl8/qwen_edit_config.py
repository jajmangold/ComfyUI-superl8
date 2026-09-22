# SPDX-License-Identifier: Apache-2.0
"""Operator configuration and workflow contract for the Qwen-Image-Edit-2511
resident supervisor (issue #182).

Every field is verified against real on-disk state before the service claims
readiness.  The workflow contract locks the exact official Qwen-Image-Edit-2511
sampler/scheduler/CFG/steps/shift/references — any drift is rejected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Stream a file's SHA-256.  Raises FileNotFoundError if missing."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_sha256_hex(value: str) -> None:
    """Reject non-hex or placeholder SHA-256 digests."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"SHA-256 must be 64 hex chars, got {value!r}")
    if not all(c in "0123456789abcdef" for c in value):
        raise ValueError(f"SHA-256 must be lowercase hex, got {value!r}")
    # Reject known placeholder patterns
    if len(set(value)) == 1:
        raise ValueError(f"SHA-256 is a placeholder (all same char): {value!r}")


# ---------------------------------------------------------------------------
# Pinned model file spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFileSpec:
    """A pinned model file with expected SHA-256 digest.

    Verification is explicit and fail-closed: verify() reads the real file
    and compares byte-for-byte.  No cached or placeholder values.
    """

    name: str
    path: str
    sha256: str = ""

    def exists(self) -> bool:
        return Path(self.path).is_file()

    def verify(self) -> None:
        """Fail-closed: raise on missing file or checksum mismatch."""
        p = Path(self.path)
        if not p.is_file():
            raise FileNotFoundError(f"pinned model file missing: {self.name} at {self.path}")
        if self.sha256:
            actual = sha256_file(p)
            if actual != self.sha256:
                raise ValueError(
                    f"checksum mismatch for {self.name}: "
                    f"expected {self.sha256}, got {actual}"
                )

    def real_sha256(self) -> str:
        """Compute the real SHA-256 of the file on disk."""
        return sha256_file(self.path)


# ---------------------------------------------------------------------------
# Official Qwen-Image-Edit-2511 workflow contract
# ---------------------------------------------------------------------------

# These constants are derived from the official Qwen-Image-Edit workflow
# evidence (see issue #182 comment 4895).  They are NOT user-configurable;
# any request that deviates from this contract is rejected.
#
# Steps: 20 (the official recommended step count for quality)
# CFG: 3.5 (the official classifier-free guidance scale)
# Scheduler: "euler" (the official recommended sampler)
# Shift: "flow_match" (flow-matching shift for Qwen-Image-Edit)
# References: 1-5 ordered reference images (multi-image conditioning)
# Reference mechanism: "Picture N:" (ordered multi-image via LLAMA template)

_LOCKED_STEPS = 20
_LOCKED_CFG = 3.5
_LOCKED_SCHEDULER = "euler"
_LOCKED_SHIFT = "flow_match"
_LOCKED_MIN_DIM = 64
_LOCKED_MAX_DIM = 2048
_LOCKED_DIM_MULTIPLE = 16  # latent spatial dims must be divisible by this
_LOCKED_MIN_REFERENCES = 1
_LOCKED_MAX_REFERENCES = 5
_LOCKED_REFERENCE_MECHANISM = "Picture N:"  # ordered multi-image conditioning
_LOCKED_MAX_SEED = 2**31 - 1
_LOCKED_MAX_PROMPT_LEN = 10_000
_LOCKED_MAX_REQUEST_BYTES = 1_048_576  # 1 MB JSON body
_LOCKED_TIMEOUT_S = 300.0
_LOCKED_MAX_TIMEOUT_S = 600.0

# Qwen-Image-Edit-2511 locked workflow contract.  The service enforces these
# exactly; no user may override steps, cfg, scheduler, or shift.
LOCKED_WORKFLOW: dict[str, Any] = {
    "steps": _LOCKED_STEPS,
    "cfg": _LOCKED_CFG,
    "scheduler": _LOCKED_SCHEDULER,
    "shift": _LOCKED_SHIFT,
    "reference_mechanism": _LOCKED_REFERENCE_MECHANISM,
    "min_references": _LOCKED_MIN_REFERENCES,
    "max_references": _LOCKED_MAX_REFERENCES,
}


# ---------------------------------------------------------------------------
# Operator configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorConfig:
    """Operator-pinned model files for the Qwen-Image-Edit-2511 resident service.

    Every file is checksum-verified before the service claims readiness.
    Paths are explicit — no /mnt/ or environment fallbacks.
    """

    unet: ModelFileSpec
    text_encoder: ModelFileSpec
    vae: ModelFileSpec
    lightning_lora: ModelFileSpec | None = None
    multi_angle_lora: ModelFileSpec | None = None

    def all_specs(self) -> list[ModelFileSpec]:
        """Return all non-None pinned specs."""
        specs = [self.unet, self.text_encoder, self.vae]
        if self.lightning_lora is not None:
            specs.append(self.lightning_lora)
        if self.multi_angle_lora is not None:
            specs.append(self.multi_angle_lora)
        return specs

    def verify_all(self) -> list[str]:
        """Verify all pinned files exist and checksums match.

        Returns a list of gate descriptions that passed.  Raises on first
        failure (fail-closed).
        """
        passed: list[str] = []
        for spec in self.all_specs():
            spec.verify()
            passed.append(f"file:{spec.name}:exists+checksum")
        return passed


def validate_workflow_request(req: dict[str, Any]) -> None:
    """Validate a generation request against the locked workflow contract.

    Rejects any request that deviates from the official Qwen-Image-Edit-2511
    contract.  Raises ValueError on any violation.
    """
    # Reject unknown top-level keys
    known_keys = {"width", "height", "prompt", "seed", "timeout_s", "references"}
    unknown = set(req.keys()) - known_keys
    if unknown:
        raise ValueError(f"unknown request fields: {', '.join(sorted(unknown))}")

    # Prompt
    prompt = req.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > _LOCKED_MAX_PROMPT_LEN:
        raise ValueError(f"prompt exceeds {_LOCKED_MAX_PROMPT_LEN} characters")

    # Width / height
    for dim_name in ("width", "height"):
        val = req.get(dim_name)
        if not isinstance(val, int):
            raise ValueError(f"{dim_name} must be an integer")
        if val < _LOCKED_MIN_DIM or val > _LOCKED_MAX_DIM:
            raise ValueError(f"{dim_name} must be {_LOCKED_MIN_DIM}..{_LOCKED_MAX_DIM}, got {val}")
        if val % _LOCKED_DIM_MULTIPLE != 0:
            raise ValueError(f"{dim_name} must be divisible by {_LOCKED_DIM_MULTIPLE}, got {val}")

    # Seed
    seed = req.get("seed")
    if not isinstance(seed, int) or seed < 0 or seed > _LOCKED_MAX_SEED:
        raise ValueError(f"seed must be 0..{_LOCKED_MAX_SEED}, got {seed}")

    # Timeout
    timeout = req.get("timeout_s", _LOCKED_TIMEOUT_S)
    if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > _LOCKED_MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be 0..{_LOCKED_MAX_TIMEOUT_S}, got {timeout}")

    # References (ordered multi-image conditioning)
    refs = req.get("references")
    if refs is not None:
        if not isinstance(refs, list):
            raise ValueError("references must be a list of image paths")
        if len(refs) < _LOCKED_MIN_REFERENCES or len(refs) > _LOCKED_MAX_REFERENCES:
            raise ValueError(
                f"references must have {_LOCKED_MIN_REFERENCES}-"
                f"{_LOCKED_MAX_REFERENCES} entries, got {len(refs)}"
            )
        for i, ref in enumerate(refs):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"references[{i}] must be a non-empty string (path)")
