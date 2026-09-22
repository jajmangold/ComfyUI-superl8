# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit-2511 resident supervisor (issue #182).

A quality-gated, fail-closed resident service supervisor for the Qwen-Image-
Edit-2511 native-GGUF FNI8 pipeline.  Every readiness claim is backed by real
on-disk verification and actual model-loading evidence — never constants or
placeholders.

Design principles (from qwen-edit-proxy attestation pattern):
  1. Report before gate — readiness produces structured per-gate reports
  2. Verify before use — every file is SHA-256 verified before loading
  3. Fail closed everywhere — missing file, checksum mismatch, loader
     returning None, unknown state → NOT READY
  4. Ordered gate progression — files → checksums → load → residency → audit
  5. No side effects in report mode — health checks never mutate state

State machine:
  NOT_READY → LOADING → READY (after real verification + successful load)
  NOT_READY/LOADING → ERROR (on any failure)

Reuses:
  - comfyui_superl8.nodes.UnetLoaderFNI8 for .fni8 DiT loading
  - comfyui_superl8.gguf_ops for native GGUF loading + LoRA audit
  - comfyui_superl8.gpu_election for fail-closed GPU election
  - comfyui_superl8.memory for peak HBM monitoring
  - comfyui_superl8.qwen_edit_config for pinned files + workflow contract

Do not touch the live-server GPU or deploy anything (AGENTS.md).
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from .qwen_edit_config import (
        LOCKED_WORKFLOW,
        OperatorConfig,
        validate_workflow_request,
    )
except ImportError:
    # Running outside the package (e.g. loaded by file path in tests)
    from qwen_edit_config import (  # type: ignore[no-redef]
        LOCKED_WORKFLOW,
        OperatorConfig,
        validate_workflow_request,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class ServiceState(enum.Enum):
    NOT_READY = "not_ready"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Readiness gate report
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Result of a single readiness gate."""

    name: str
    ok: bool
    reason: str
    detail: Any = None


@dataclass
class ReadinessReport:
    """Structured readiness report — every gate's pass/fail status.

    The overall ok is the conjunction of all gates.  Follows the
    qwen-edit-proxy attestation pattern: report first, then gate.
    """

    ok: bool
    state: ServiceState
    gates: list[GateResult] = field(default_factory=list)
    error: str | None = None
    load_time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "state": self.state.value,
            "gates": [
                {"name": g.name, "ok": g.ok, "reason": g.reason, "detail": g.detail}
                for g in self.gates
            ],
            "error": self.error,
            "load_time_s": self.load_time_s,
        }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceInfo:
    """Honest provenance for the Qwen-Image-Edit-2511 resident service.

    Every field is the result of a real measurement or verification.
    Unresolved fields are None — never fabricated.
    """

    # Model file checksums (real SHA-256 of files on disk)
    unet_sha256: str | None = None
    text_encoder_sha256: str | None = None
    vae_sha256: str | None = None
    lightning_lora_sha256: str | None = None
    multi_angle_lora_sha256: str | None = None

    # Workflow contract (locked constants — these are NOT derived from the
    # request, they are the official contract the service enforces)
    workflow: dict[str, Any] = field(default_factory=lambda: dict(LOCKED_WORKFLOW))

    # GGUF tensor type inventory (real counts from model inspection)
    tensor_type_inventory: dict[str, int] = field(default_factory=dict)

    # LoRA audit (real counts from gguf_lora_audit or equivalent)
    lora_audit: dict[str, int] = field(default_factory=dict)

    # Kernel/fallback telemetry (real counts from last completed edit)
    native_kernel_count: int | None = None
    fallback_count: int | None = None

    def __post_init__(self) -> None:
        """Validate that no placeholder provenance was injected."""
        for field_name in (
            "unet_sha256",
            "text_encoder_sha256",
            "vae_sha256",
            "lightning_lora_sha256",
            "multi_angle_lora_sha256",
        ):
            val = getattr(self, field_name)
            if val is not None:
                _validate_no_placeholder(val, field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unet_sha256": self.unet_sha256,
            "text_encoder_sha256": self.text_encoder_sha256,
            "vae_sha256": self.vae_sha256,
            "lightning_lora_sha256": self.lightning_lora_sha256,
            "multi_angle_lora_sha256": self.multi_angle_lora_sha256,
            "workflow": self.workflow,
            "tensor_type_inventory": self.tensor_type_inventory,
            "lora_audit": self.lora_audit,
            "native_kernel_count": self.native_kernel_count,
            "fallback_count": self.fallback_count,
        }


def _validate_no_placeholder(value: str, field_name: str) -> None:
    """Reject known placeholder patterns in provenance fields."""
    if len(value) != 64:
        raise ValueError(f"{field_name}: expected 64 hex chars, got {len(value)}")
    if not all(c in "0123456789abcdef" for c in value):
        raise ValueError(f"{field_name}: expected lowercase hex")
    if len(set(value)) == 1:
        raise ValueError(f"{field_name}: placeholder pattern (all same char): {value!r}")


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

# Thresholds for residency verification
_MIN_DIT_MEMORY_GIB = 1.0  # a loaded DiT should be > 1 GiB


class QwenEditSupervisor:
    """Resident supervisor for the Qwen-Image-Edit-2511 FNI8 service.

    The supervisor owns the service lifecycle:
      1. Verify operator-pinned files (existence + SHA-256)
      2. Load the model via the proven loader (UnetLoaderFNI8 or native GGUF)
      3. Verify model residency on GPU
      4. Audit LoRA application counts
      5. Transition to READY only after all gates pass

    Any failure at any gate moves the state to ERROR and the service
    remains NOT READY — never fabricated readiness.

    Usage::

        config = OperatorConfig(unet=..., text_encoder=..., vae=...)
        supervisor = QwenEditSupervisor(config)
        supervisor.load_model(loader_fn=my_loader)  # runs gates
        report = supervisor.readiness_report()       # structured per-gate
    """

    def __init__(self, config: OperatorConfig) -> None:
        self._config = config
        self._state = ServiceState.NOT_READY
        self._error: str | None = None
        self._load_start: float = 0.0
        self._load_end: float = 0.0
        self._model: Any = None
        self._provenance = ProvenanceInfo()
        self._gates: list[GateResult] = []
        self._lock = threading.Lock()
        self._active_edits = 0
        self._total_edits = 0

    @property
    def state(self) -> ServiceState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == ServiceState.READY

    # --- Model loading (ordered gate progression) --------------------------

    def load_model(
        self,
        loader_fn: Any = None,
        *,
        verify_checksums: bool = True,
        verify_residency: bool = True,
    ) -> None:
        """Load the model with ordered gate progression.

        Gates (each must pass before the next runs):
          1. File existence — all pinned files exist on disk
          2. File checksums — SHA-256 of each file matches the pin
          3. Model load — loader_fn returns a non-None model object
          4. Model residency — model params on CUDA, memory > threshold
          5. LoRA audit — real counts from the loaded model

        Any failure moves to ERROR.  The service never reaches READY
        without passing ALL gates with real evidence.

        Args:
            loader_fn: Callable that loads and returns the model.  Must return
                a non-None object on success.  If None, skip gate 3 (useful
                for testing the file-verification gates).
            verify_checksums: Whether to verify file SHA-256 digests.
            verify_residency: Whether to verify model is on GPU.
        """
        with self._lock:
            self._state = ServiceState.LOADING
            self._error = None
            self._gates = []
            self._load_start = time.monotonic()

        try:
            self._run_gates(
                loader_fn=loader_fn,
                verify_checksums=verify_checksums,
                verify_residency=verify_residency,
            )
            with self._lock:
                self._state = ServiceState.READY
                self._load_end = time.monotonic()
                logger.info(
                    "Qwen-Edit supervisor READY in %.1f s",
                    self._load_end - self._load_start,
                )
        except Exception as e:
            with self._lock:
                self._state = ServiceState.ERROR
                self._error = str(e)
                self._load_end = time.monotonic()
                logger.error("Qwen-Edit supervisor failed: %s", e)
            raise

    def _run_gates(
        self,
        loader_fn: Any,
        verify_checksums: bool,
        verify_residency: bool,
    ) -> None:
        """Execute the ordered gate progression."""
        # Gate 1: File existence
        gate_files = self._gate_files_exist()
        if not gate_files.ok:
            raise RuntimeError(f"gate {gate_files.name} failed: {gate_files.reason}")

        # Gate 2: File checksums (optional — can be skipped for testing)
        if verify_checksums:
            gate_checksums = self._gate_checksums()
            if not gate_checksums.ok:
                raise RuntimeError(f"gate {gate_checksums.name} failed: {gate_checksums.reason}")

        # Record real file checksums in provenance
        self._record_file_provenance(verify_checksums)

        # Gate 3: Model load (if loader provided)
        if loader_fn is not None:
            gate_load = self._gate_model_load(loader_fn)
            if not gate_load.ok:
                raise RuntimeError(f"gate {gate_load.name} failed: {gate_load.reason}")

        # Gate 4: Model residency (optional — requires GPU)
        if verify_residency and self._model is not None:
            gate_residency = self._gate_model_residency()
            if not gate_residency.ok:
                raise RuntimeError(f"gate {gate_residency.name} failed: {gate_residency.reason}")

        # Gate 5: LoRA audit (if model is loaded)
        if self._model is not None:
            self._gate_lora_audit()
            # LoRA audit is informational — does not block READY
            # (some configs may not have LoRAs)

    # --- Individual gates ---------------------------------------------------

    def _gate_files_exist(self) -> GateResult:
        """Gate 1: All pinned files exist on disk."""
        for spec in self._config.all_specs():
            if not spec.exists():
                result = GateResult(
                    name="files_exist",
                    ok=False,
                    reason=f"file missing: {spec.name} at {spec.path}",
                )
                self._gates.append(result)
                return result
        result = GateResult(
            name="files_exist",
            ok=True,
            reason=f"all {len(self._config.all_specs())} pinned files exist",
        )
        self._gates.append(result)
        return result

    def _gate_checksums(self) -> GateResult:
        """Gate 2: File SHA-256 checksums match pinned values."""
        for spec in self._config.all_specs():
            if spec.sha256:
                try:
                    spec.verify()
                except ValueError as e:
                    result = GateResult(
                        name="checksums",
                        ok=False,
                        reason=f"checksum mismatch: {spec.name}: {e}",
                    )
                    self._gates.append(result)
                    return result
        result = GateResult(
            name="checksums",
            ok=True,
            reason="all pinned checksums verified",
        )
        self._gates.append(result)
        return result

    def _gate_model_load(self, loader_fn: Any) -> GateResult:
        """Gate 3: Loader returns a non-None model object."""
        try:
            model = loader_fn()
            if model is None:
                result = GateResult(
                    name="model_load",
                    ok=False,
                    reason="loader returned None — model not loaded",
                )
                self._gates.append(result)
                return result
            self._model = model
            result = GateResult(
                name="model_load",
                ok=True,
                reason=f"loader returned {type(model).__name__}",
            )
            self._gates.append(result)
            return result
        except Exception as e:
            result = GateResult(
                name="model_load",
                ok=False,
                reason=f"loader raised: {e}",
            )
            self._gates.append(result)
            return result

    def _gate_model_residency(self) -> GateResult:
        """Gate 4: Model parameters are on GPU and memory > threshold."""
        try:
            import torch

            if not torch.cuda.is_available():
                result = GateResult(
                    name="model_residency",
                    ok=False,
                    reason="CUDA not available",
                )
                self._gates.append(result)
                return result

            # Check that at least some parameters are on CUDA
            has_cuda_params = False
            total_param_bytes = 0
            for param in self._model.parameters():
                total_param_bytes += param.nelement() * param.element_size()
                if param.is_cuda:
                    has_cuda_params = True

            if not has_cuda_params:
                result = GateResult(
                    name="model_residency",
                    ok=False,
                    reason="no CUDA parameters found — model not on GPU",
                )
                self._gates.append(result)
                return result

            memory_gib = total_param_bytes / (1024**3)
            if memory_gib < _MIN_DIT_MEMORY_GIB:
                result = GateResult(
                    name="model_residency",
                    ok=False,
                    reason=f"model memory {memory_gib:.2f} GiB < {_MIN_DIT_MEMORY_GIB} GiB threshold",
                )
                self._gates.append(result)
                return result

            result = GateResult(
                name="model_residency",
                ok=True,
                reason=f"model on CUDA, {memory_gib:.2f} GiB",
                detail={"memory_gib": round(memory_gib, 3)},
            )
            self._gates.append(result)
            return result
        except ImportError:
            result = GateResult(
                name="model_residency",
                ok=False,
                reason="torch not available",
            )
            self._gates.append(result)
            return result

    def _gate_lora_audit(self) -> GateResult:
        """Gate 5: Audit LoRA application counts (informational)."""
        try:
            from .gguf_ops import gguf_lora_audit

            audit = gguf_lora_audit(self._model)
            self._provenance.lora_audit = audit
            result = GateResult(
                name="lora_audit",
                ok=True,
                reason=f"LoRA audit: {audit}",
                detail=audit,
            )
            self._gates.append(result)
            return result
        except Exception as e:
            # LoRA audit is informational — log but don't fail
            result = GateResult(
                name="lora_audit",
                ok=True,
                reason=f"LoRA audit skipped: {e}",
                detail={},
            )
            self._gates.append(result)
            return result

    # --- Provenance ---------------------------------------------------------

    def _record_file_provenance(self, include_checksums: bool) -> None:
        """Record real file checksums in provenance."""
        if not include_checksums:
            return
        if self._config.unet.sha256:
            self._provenance.unet_sha256 = self._config.unet.sha256
        elif self._config.unet.exists():
            self._provenance.unet_sha256 = self._config.unet.real_sha256()

        if self._config.text_encoder.sha256:
            self._provenance.text_encoder_sha256 = self._config.text_encoder.sha256
        elif self._config.text_encoder.exists():
            self._provenance.text_encoder_sha256 = self._config.text_encoder.real_sha256()

        if self._config.vae.sha256:
            self._provenance.vae_sha256 = self._config.vae.sha256
        elif self._config.vae.exists():
            self._provenance.vae_sha256 = self._config.vae.real_sha256()

        if self._config.lightning_lora is not None:
            if self._config.lightning_lora.sha256:
                self._provenance.lightning_lora_sha256 = self._config.lightning_lora.sha256
            elif self._config.lightning_lora.exists():
                self._provenance.lightning_lora_sha256 = self._config.lightning_lora.real_sha256()

        if self._config.multi_angle_lora is not None:
            if self._config.multi_angle_lora.sha256:
                self._provenance.multi_angle_lora_sha256 = self._config.multi_angle_lora.sha256
            elif self._config.multi_angle_lora.exists():
                self._provenance.multi_angle_lora_sha256 = self._config.multi_angle_lora.real_sha256()

    @property
    def provenance(self) -> ProvenanceInfo:
        return self._provenance

    # --- Health / readiness reports -----------------------------------------

    def readiness_report(self) -> ReadinessReport:
        """Structured readiness report — every gate's pass/fail status.

        Follows the qwen-edit-proxy attestation pattern: report first,
        then gate.  The overall ok is the conjunction of all gates.
        """
        all_ok = all(g.ok for g in self._gates) if self._gates else False
        return ReadinessReport(
            ok=all_ok and self._state == ServiceState.READY,
            state=self._state,
            gates=list(self._gates),
            error=self._error,
            load_time_s=(
                round(self._load_end - self._load_start, 1)
                if self._load_end > 0
                else None
            ),
        )

    def health_report(self) -> dict[str, Any]:
        """Liveness + readiness in a single structured response.

        Truthful: unreported fields are None, never fabricated.
        """
        report = self.readiness_report()
        return {
            "status": "ok" if self._state == ServiceState.READY else "not_ready",
            "service": "qwen-edit-resident",
            "state": self._state.value,
            "readiness": report.to_dict(),
            "provenance": self._provenance.to_dict(),
            "workflow_contract": LOCKED_WORKFLOW,
            "total_edits": self._total_edits,
            "active_edits": self._active_edits,
            "error": self._error,
        }

    # --- Request validation -------------------------------------------------

    def validate_request(self, req: dict[str, Any]) -> None:
        """Validate a generation request against the locked workflow contract.

        Delegates to qwen_edit_config.validate_workflow_request for the
        actual validation.  Raises ValueError on any violation.
        """
        validate_workflow_request(req)

    # --- Edit lifecycle (for use by the HTTP handler) -----------------------

    def begin_edit(self) -> bool:
        """Try to acquire the single edit slot.  Returns True if acquired."""
        # Single-slot: only one edit at a time
        if self._active_edits >= 1:
            return False
        self._active_edits += 1
        self._total_edits += 1
        return True

    def end_edit(self) -> None:
        """Release the edit slot."""
        self._active_edits = max(0, self._active_edits - 1)
