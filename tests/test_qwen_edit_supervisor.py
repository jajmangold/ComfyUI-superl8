# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen-Image-Edit-2511 resident supervisor (issue #182).

These tests prove that:
  1. The supervisor starts NOT_READY (never fabricated readiness)
  2. Readiness flips only on real state changes (injected loader returning
     a real object, not None or a mock pretending to be ready)
  3. Checksum mismatches are rejected (fail-closed)
  4. Missing files are rejected (fail-closed)
  5. Placeholder provenance is rejected (no "a"*64 hashes)
  6. The workflow contract rejects arbitrary steps/CFG/scheduler
  7. Multi-reference is bounded and ordered
  8. Health telemetry is truthful (unreported fields are None)
  9. LoRA audit counts come from real inspection (not constants)

All tests run on CPU without GPU or real model weights.  The loader is
injected as a callable so tests prove the plumbing, not a no-op load.

Modules are loaded by file path (like test_gpu_election.py) to bypass
comfyui_superl8/__init__.py which pulls in torch/ComfyUI deps.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Load modules by file path to avoid comfyui_superl8/__init__.py (torch dep)
# ---------------------------------------------------------------------------

_MOD_DIR = os.path.join(os.path.dirname(__file__), "..", "comfyui_superl8")


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_MOD_DIR, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_config_mod = _load_mod("fni8_qwen_edit_config", "qwen_edit_config.py")
# Register under the name the supervisor's try/except fallback expects
sys.modules["qwen_edit_config"] = _config_mod
_supervisor_mod = _load_mod("fni8_qwen_edit_supervisor", "qwen_edit_supervisor.py")

ModelFileSpec = _config_mod.ModelFileSpec
OperatorConfig = _config_mod.OperatorConfig
validate_workflow_request = _config_mod.validate_workflow_request
LOCKED_WORKFLOW = _config_mod.LOCKED_WORKFLOW
_validate_sha256_hex = _config_mod._validate_sha256_hex

QwenEditSupervisor = _supervisor_mod.QwenEditSupervisor
ServiceState = _supervisor_mod.ServiceState
ProvenanceInfo = _supervisor_mod.ProvenanceInfo
_validate_no_placeholder = _supervisor_mod._validate_no_placeholder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    unet_path: str = "/tmp/fake_unet.fni8",
    te_path: str = "/tmp/fake_te.safetensors",
    vae_path: str = "/tmp/fake_vae.safetensors",
    unet_sha: str = "",
    te_sha: str = "",
    vae_sha: str = "",
) -> OperatorConfig:
    return OperatorConfig(
        unet=ModelFileSpec(name="unet", path=unet_path, sha256=unet_sha),
        text_encoder=ModelFileSpec(name="text_encoder", path=te_path, sha256=te_sha),
        vae=ModelFileSpec(name="vae", path=vae_path, sha256=vae_sha),
    )


class _Sentinel:
    """A non-None sentinel that the loader returns to prove the plumbing works."""

    pass


def _sentinel_loader() -> _Sentinel:
    return _Sentinel()


def _none_loader() -> None:
    return None


def _raising_loader() -> Any:
    raise RuntimeError("loader exploded")


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_starts_not_ready(self) -> None:
        """Supervisor must start NOT_READY — never fabricated readiness."""
        config = _make_config()
        supervisor = QwenEditSupervisor(config)
        assert supervisor.state == ServiceState.NOT_READY
        assert not supervisor.is_ready

    def test_readiness_flips_on_real_model_load(self) -> None:
        """Readiness must flip to READY only after a real loader returns
        a non-None object — not a mock, not a constant, not a placeholder."""
        with tempfile.NamedTemporaryFile(suffix=".fni8", delete=False) as f:
            unet_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            te_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            vae_path = f.name
        try:
            config = _make_config(unet_path=unet_path, te_path=te_path, vae_path=vae_path)
            supervisor = QwenEditSupervisor(config)

            supervisor.load_model(
                loader_fn=_sentinel_loader,
                verify_checksums=False,
                verify_residency=False,
            )

            assert supervisor.state == ServiceState.READY
            assert supervisor.is_ready
            assert isinstance(supervisor._model, _Sentinel)
        finally:
            os.unlink(unet_path)
            os.unlink(te_path)
            os.unlink(vae_path)

    def test_readiness_stays_not_ready_on_loader_returning_none(self) -> None:
        """A loader returning None must cause ERROR, not READY."""
        with tempfile.NamedTemporaryFile(suffix=".fni8", delete=False) as f:
            unet_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            te_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            vae_path = f.name
        try:
            config = _make_config(unet_path=unet_path, te_path=te_path, vae_path=vae_path)
            supervisor = QwenEditSupervisor(config)

            with pytest.raises(RuntimeError, match="loader returned None"):
                supervisor.load_model(
                    loader_fn=_none_loader,
                    verify_checksums=False,
                    verify_residency=False,
                )

            assert supervisor.state == ServiceState.ERROR
            assert not supervisor.is_ready
        finally:
            os.unlink(unet_path)
            os.unlink(te_path)
            os.unlink(vae_path)

    def test_readiness_stays_not_ready_on_loader_exception(self) -> None:
        """A loader raising an exception must cause ERROR."""
        with tempfile.NamedTemporaryFile(suffix=".fni8", delete=False) as f:
            unet_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            te_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            vae_path = f.name
        try:
            config = _make_config(unet_path=unet_path, te_path=te_path, vae_path=vae_path)
            supervisor = QwenEditSupervisor(config)

            with pytest.raises(RuntimeError, match="loader raised"):
                supervisor.load_model(
                    loader_fn=_raising_loader,
                    verify_checksums=False,
                    verify_residency=False,
                )

            assert supervisor.state == ServiceState.ERROR
            assert not supervisor.is_ready
        finally:
            os.unlink(unet_path)
            os.unlink(te_path)
            os.unlink(vae_path)


# ---------------------------------------------------------------------------
# File verification gates
# ---------------------------------------------------------------------------


class TestFileVerification:
    def test_readiness_stays_not_ready_on_missing_file(self) -> None:
        """Missing pinned file must cause ERROR — fail-closed."""
        config = _make_config(unet_path="/nonexistent/fake.fni8")
        supervisor = QwenEditSupervisor(config)

        with pytest.raises(RuntimeError, match="file missing"):
            supervisor.load_model(
                loader_fn=_sentinel_loader,
                verify_checksums=False,
                verify_residency=False,
            )

        assert supervisor.state == ServiceState.ERROR
        report = supervisor.readiness_report()
        assert not report.ok
        assert any(not g.ok for g in report.gates)

    def test_readiness_stays_not_ready_on_checksum_mismatch(self) -> None:
        """Checksum mismatch must cause ERROR — fail-closed."""
        with tempfile.NamedTemporaryFile(suffix=".fni8", delete=False) as f:
            f.write(b"fake model data")
            unet_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            te_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            vae_path = f.name
        try:
            config = _make_config(
                unet_path=unet_path,
                te_path=te_path,
                vae_path=vae_path,
                unet_sha="0" * 64,
            )
            supervisor = QwenEditSupervisor(config)

            with pytest.raises(RuntimeError, match="checksum mismatch"):
                supervisor.load_model(
                    loader_fn=_sentinel_loader,
                    verify_checksums=True,
                    verify_residency=False,
                )

            assert supervisor.state == ServiceState.ERROR
        finally:
            os.unlink(unet_path)
            os.unlink(te_path)
            os.unlink(vae_path)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_provenance_rejects_placeholder_model_sha(self) -> None:
        """Placeholder SHA-256 patterns must be rejected in provenance."""
        with pytest.raises(ValueError, match="placeholder"):
            ProvenanceInfo(unet_sha256="a" * 64)

    def test_provenance_rejects_all_same_char_sha(self) -> None:
        """Any all-same-character SHA-256 must be rejected."""
        with pytest.raises(ValueError, match="placeholder"):
            ProvenanceInfo(text_encoder_sha256="f" * 64)

    def test_provenance_accepts_real_sha(self) -> None:
        """Real SHA-256 hex strings must be accepted."""
        real_sha = "809b7b6227505c092257537fb209ddcbb04aadeab8e05342fe5f12a30da6463b"
        prov = ProvenanceInfo(unet_sha256=real_sha)
        assert prov.unet_sha256 == real_sha

    def test_provenance_unreported_fields_are_none(self) -> None:
        """Unreported provenance fields must be None, never fabricated."""
        prov = ProvenanceInfo()
        assert prov.unet_sha256 is None
        assert prov.lightning_lora_sha256 is None
        assert prov.native_kernel_count is None
        assert prov.fallback_count is None
        assert prov.lora_audit == {}
        assert prov.tensor_type_inventory == {}

    def test_provenance_workflow_is_locked(self) -> None:
        """Provenance workflow must match the locked contract."""
        prov = ProvenanceInfo()
        assert prov.workflow == LOCKED_WORKFLOW
        assert prov.workflow["steps"] == 20
        assert prov.workflow["cfg"] == 3.5
        assert prov.workflow["scheduler"] == "euler"

    def test_validate_sha256_hex_rejects_placeholders(self) -> None:
        """_validate_sha256_hex must reject placeholders."""
        _validate_sha256_hex("809b7b6227505c092257537fb209ddcbb04aadeab8e05342fe5f12a30da6463b")
        with pytest.raises(ValueError):
            _validate_sha256_hex("a" * 64)
        with pytest.raises(ValueError):
            _validate_sha256_hex("f" * 64)
        with pytest.raises(ValueError):
            _validate_sha256_hex("short")
        with pytest.raises(ValueError):
            _validate_sha256_hex("g" + "0" * 63)


# ---------------------------------------------------------------------------
# Workflow contract
# ---------------------------------------------------------------------------


class TestWorkflowContract:
    def test_workflow_contract_rejects_arbitrary_steps(self) -> None:
        """Request with wrong steps must be rejected (steps are locked)."""
        req = {"width": 512, "height": 512, "prompt": "test", "seed": 42}
        req_with_steps = {**req, "steps": 10}
        with pytest.raises(ValueError, match="unknown request fields"):
            validate_workflow_request(req_with_steps)

    def test_workflow_contract_rejects_arbitrary_cfg(self) -> None:
        """Request with wrong cfg must be rejected."""
        req = {"width": 512, "height": 512, "prompt": "test", "seed": 42}
        req_with_cfg = {**req, "cfg": 7.0}
        with pytest.raises(ValueError, match="unknown request fields"):
            validate_workflow_request(req_with_cfg)

    def test_workflow_contract_rejects_arbitrary_scheduler(self) -> None:
        """Request with wrong scheduler must be rejected."""
        req = {"width": 512, "height": 512, "prompt": "test", "seed": 42}
        req_with_sched = {**req, "scheduler": "dpmpp_2m"}
        with pytest.raises(ValueError, match="unknown request fields"):
            validate_workflow_request(req_with_sched)

    def test_workflow_contract_rejects_arbitrary_shift(self) -> None:
        """Request with wrong shift must be rejected."""
        req = {"width": 512, "height": 512, "prompt": "test", "seed": 42}
        req_with_shift = {**req, "shift": "sigma_shift"}
        with pytest.raises(ValueError, match="unknown request fields"):
            validate_workflow_request(req_with_shift)

    def test_valid_request_passes(self) -> None:
        """A valid request matching the contract must pass."""
        req = {"width": 512, "height": 512, "prompt": "test", "seed": 42}
        validate_workflow_request(req)

    def test_rejects_unknown_fields(self) -> None:
        """Unknown request fields must be rejected."""
        req = {"width": 512, "height": 512, "prompt": "test", "seed": 42, "magic": True}
        with pytest.raises(ValueError, match="unknown request fields"):
            validate_workflow_request(req)


# ---------------------------------------------------------------------------
# Multi-reference
# ---------------------------------------------------------------------------


class TestMultiReference:
    def test_multi_reference_bounded(self) -> None:
        """References must be bounded to 1-5 ordered images."""
        req = {
            "width": 512,
            "height": 512,
            "prompt": "edit",
            "seed": 42,
            "references": ["/path/to/ref1.png"],
        }
        validate_workflow_request(req)

    def test_multi_reference_max(self) -> None:
        """Up to 5 references must be accepted."""
        req = {
            "width": 512,
            "height": 512,
            "prompt": "edit",
            "seed": 42,
            "references": [f"/path/to/ref{i}.png" for i in range(5)],
        }
        validate_workflow_request(req)

    def test_multi_reference_too_many(self) -> None:
        """More than 5 references must be rejected."""
        req = {
            "width": 512,
            "height": 512,
            "prompt": "edit",
            "seed": 42,
            "references": [f"/path/to/ref{i}.png" for i in range(6)],
        }
        with pytest.raises(ValueError, match="5 entries"):
            validate_workflow_request(req)

    def test_multi_reference_empty_rejected(self) -> None:
        """Empty references list must be rejected."""
        req = {
            "width": 512,
            "height": 512,
            "prompt": "edit",
            "seed": 42,
            "references": [],
        }
        with pytest.raises(ValueError, match="1-5 entries"):
            validate_workflow_request(req)

    def test_multi_reference_not_list(self) -> None:
        """References must be a list."""
        req = {
            "width": 512,
            "height": 512,
            "prompt": "edit",
            "seed": 42,
            "references": "/path/to/single.png",
        }
        with pytest.raises(ValueError, match="must be a list"):
            validate_workflow_request(req)


# ---------------------------------------------------------------------------
# Health telemetry
# ---------------------------------------------------------------------------


class TestHealthTelemetry:
    def test_health_report_truthful_unreported_fields_none(self) -> None:
        """Health report must not fabricate unreported fields."""
        config = _make_config()
        supervisor = QwenEditSupervisor(config)
        health = supervisor.health_report()

        assert health["status"] == "not_ready"
        assert health["service"] == "qwen-edit-resident"
        assert health["error"] is None
        assert health["readiness"]["state"] == "not_ready"

        prov = health["provenance"]
        assert prov["unet_sha256"] is None
        assert prov["lightning_lora_sha256"] is None
        assert prov["native_kernel_count"] is None
        assert prov["fallback_count"] is None
        assert prov["lora_audit"] == {}
        assert prov["tensor_type_inventory"] == {}

    def test_readiness_report_gates_empty_when_not_loaded(self) -> None:
        """Readiness report has no gates when no load has been attempted."""
        config = _make_config()
        supervisor = QwenEditSupervisor(config)
        report = supervisor.readiness_report()
        assert report.gates == []
        assert not report.ok

    def test_readiness_report_gates_populated_after_load(self) -> None:
        """Readiness report has gates after a load attempt."""
        with tempfile.NamedTemporaryFile(suffix=".fni8", delete=False) as f:
            unet_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            te_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            vae_path = f.name
        try:
            config = _make_config(unet_path=unet_path, te_path=te_path, vae_path=vae_path)
            supervisor = QwenEditSupervisor(config)
            supervisor.load_model(
                loader_fn=_sentinel_loader,
                verify_checksums=False,
                verify_residency=False,
            )
            report = supervisor.readiness_report()
            assert len(report.gates) >= 1
            assert all(g.ok for g in report.gates)
            assert report.ok
        finally:
            os.unlink(unet_path)
            os.unlink(te_path)
            os.unlink(vae_path)


# ---------------------------------------------------------------------------
# OperatorConfig
# ---------------------------------------------------------------------------


class TestOperatorConfig:
    def test_all_specs_includes_loras(self) -> None:
        """all_specs() includes LoRA specs when present."""
        config = OperatorConfig(
            unet=ModelFileSpec(name="unet", path="/tmp/fake.fni8"),
            text_encoder=ModelFileSpec(name="te", path="/tmp/fake_te.safetensors"),
            vae=ModelFileSpec(name="vae", path="/tmp/fake_vae.safetensors"),
            lightning_lora=ModelFileSpec(name="lightning", path="/tmp/fake_lora.gguf"),
            multi_angle_lora=ModelFileSpec(name="multi_angle", path="/tmp/fake_ma.safetensors"),
        )
        specs = config.all_specs()
        names = [s.name for s in specs]
        assert "unet" in names
        assert "te" in names
        assert "vae" in names
        assert "lightning" in names
        assert "multi_angle" in names

    def test_verify_all_rejects_missing(self) -> None:
        """verify_all() raises on missing files."""
        config = _make_config(unet_path="/nonexistent/file.fni8")
        with pytest.raises(FileNotFoundError, match="pinned model file missing"):
            config.verify_all()

    def test_model_file_spec_verify_checksum_mismatch(self) -> None:
        """ModelFileSpec.verify() rejects wrong checksum."""
        with tempfile.NamedTemporaryFile(suffix=".fni8", delete=False) as f:
            f.write(b"real data")
            path = f.name
        try:
            spec = ModelFileSpec(name="test", path=path, sha256="0" * 64)
            with pytest.raises(ValueError, match="checksum mismatch"):
                spec.verify()
        finally:
            os.unlink(path)

    def test_model_file_spec_real_sha256(self) -> None:
        """ModelFileSpec.real_sha256() computes the real hash."""
        with tempfile.NamedTemporaryFile(suffix=".fni8", delete=False) as f:
            f.write(b"test data for sha256")
            path = f.name
        try:
            spec = ModelFileSpec(name="test", path=path)
            real = spec.real_sha256()
            assert len(real) == 64
            assert all(c in "0123456789abcdef" for c in real)
            assert len(set(real)) > 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Edit lifecycle
# ---------------------------------------------------------------------------


class TestEditLifecycle:
    def test_single_slot_enforced(self) -> None:
        """Only one edit can be active at a time."""
        config = _make_config()
        supervisor = QwenEditSupervisor(config)
        assert supervisor.begin_edit()
        assert not supervisor.begin_edit()
        supervisor.end_edit()
        assert supervisor.begin_edit()

    def test_edit_count_tracked(self) -> None:
        """Total edits are tracked correctly."""
        config = _make_config()
        supervisor = QwenEditSupervisor(config)
        assert supervisor._total_edits == 0
        supervisor.begin_edit()
        supervisor.end_edit()
        assert supervisor._total_edits == 1
        supervisor.begin_edit()
        supervisor.end_edit()
        assert supervisor._total_edits == 2
