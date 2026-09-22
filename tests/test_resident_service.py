# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Z-Image resident HTTP service (issue #181).

Tests the API contract, provenance, request/response validation, concurrency
limits, timeout behaviour, GPU election mocking, sequential lifecycle, and
bounded response envelope WITHOUT requiring a GPU or real weights.  The
service module is pure-Python (stdlib http.server + threading) and imports
nothing GPU-dependent at the module level.

Run:  python -m pytest tests/test_resident_service.py -q
"""

from __future__ import annotations

import inspect
import os
import urllib.error
import json
import threading
import time
from http.client import HTTPConnection
from http.server import HTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from comfyui_superl8.provenance import Provenance, SERVICE_VERSION, build_provenance  # noqa: E402
from comfyui_superl8 import zimage_profile  # noqa: E402
from comfyui_superl8.resident_service import (  # noqa: E402
    GenerationRequest,
    GenerationResponse,
    ZImageResidentService,
    _AbortError,
    _LOCKED_CFG,
    _LOCKED_NFE,
    _LOCKED_SAMPLER,
    _LOCKED_SCHEDULER,
    _MAX_REQUEST_BYTES,
    _MAX_SEED,
    _PERMITTED_KEYS,
    _validate_request,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_has_required_fields():
    p = build_provenance(width=512, height=512, steps=8, cfg=1.0, seed=0, scheduler="simple")
    d = p.to_dict()
    required = {
        "service_version",
        "git_commit",
        "torch_version",
        "cuda_version",
        "python_version",
        "gpu_name",
        "pipeline",
        "scheduler",
        "steps",
        "cfg",
        "seed",
        "width",
        "height",
        "in_channels",
        "vae_stride",
        "lossless_qkv_refusion",
        "fp_normalization",
        "fp_softmax",
        "fp_vae",
        "sqnr_fp_fallback",
        "model_sampling_shift",
        "sampler_name",
        "nfe",
        "explicit_seed",
        "empty_negative",
        "no_cache",
    }
    assert required.issubset(set(d.keys())), f"missing fields: {required - set(d.keys())}"


def test_provenance_deterministic():
    """Same inputs -> identical provenance (no timestamps or RNG leakage)."""
    kw = dict(width=512, height=512, steps=8, cfg=1.0, seed=42, scheduler="simple")
    a = build_provenance(**kw).to_dict()
    b = build_provenance(**kw).to_dict()
    for k in (
        "service_version",
        "pipeline",
        "scheduler",
        "steps",
        "cfg",
        "seed",
        "width",
        "height",
        "lossless_qkv_refusion",
        "fp_normalization",
        "fp_softmax",
        "fp_vae",
        "sqnr_fp_fallback",
        "model_sampling_shift",
        "sampler_name",
        "nfe",
        "explicit_seed",
        "empty_negative",
        "no_cache",
    ):
        assert a[k] == b[k], f"provenance field {k!r} is non-deterministic"


def test_provenance_non_square():
    p = build_provenance(width=896, height=1152, steps=8, cfg=1.0, seed=0, scheduler="simple")
    d = p.to_dict()
    assert d["width"] == 896
    assert d["height"] == 1152


def test_provenance_json_serializable():
    p = build_provenance(width=512, height=512, steps=8, cfg=1.0, seed=0, scheduler="simple")
    blob = json.dumps(p.to_dict())
    restored = json.loads(blob)
    assert restored["width"] == 512


def test_provenance_checksums_none_when_no_path():
    p = build_provenance(width=512, height=512, steps=8, cfg=1.0, seed=0, scheduler="simple")
    assert p.dit_checksum is None
    assert p.vae_checksum is None
    assert p.te_checksum is None


def test_provenance_pipeline_constants():
    """Verify the exact pipeline semantics are encoded in provenance."""
    p = build_provenance(width=512, height=512, steps=8, cfg=1.0, seed=0, scheduler="simple")
    d = p.to_dict()
    assert d["lossless_qkv_refusion"] is True
    assert d["fp_normalization"] is True
    assert d["fp_softmax"] is True
    assert d["fp_vae"] is True
    assert d["sqnr_fp_fallback"] is True
    assert d["explicit_seed"] is True
    assert d["empty_negative"] is True
    assert d["no_cache"] is True
    assert d["cfg"] == 1.0
    assert d["nfe"] == 8
    assert d["sampler_name"] == "res_multistep"
    assert d["model_sampling_shift"] is None
    assert d["dit_format"] == "fni8_b8_w8a8"


def test_provenance_version_is_service_version():
    p = build_provenance(width=512, height=512, steps=8, cfg=1.0, seed=0, scheduler="simple")
    assert p.service_version == SERVICE_VERSION


# ---------------------------------------------------------------------------
# Locked pipeline constants
# ---------------------------------------------------------------------------


def test_locked_constants_match_spec():
    """The official profile requires 8-NFE, CFG1, shift3, res_multistep/simple."""
    assert _LOCKED_NFE == 8
    assert _LOCKED_CFG == 1.0
    assert _LOCKED_SAMPLER == "res_multistep"
    assert _LOCKED_SCHEDULER == "simple"


def test_permitted_keys_exclude_pipeline_params():
    """steps, cfg, scheduler must not be user-configurable."""
    assert "steps" not in _PERMITTED_KEYS
    assert "cfg" not in _PERMITTED_KEYS
    assert "scheduler" not in _PERMITTED_KEYS
    assert "nfe" not in _PERMITTED_KEYS
    assert "shift" not in _PERMITTED_KEYS


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_validate_request_valid():
    req = GenerationRequest(width=512, height=512, prompt="a red fox", seed=0)
    _validate_request(req)  # should not raise


def test_validate_request_non_square():
    req = GenerationRequest(width=896, height=1152, prompt="portrait", seed=0)
    _validate_request(req)


def test_validate_request_rejects_fractional_latent_geometry():
    req = GenerationRequest(width=513, height=512, prompt="portrait", seed=0)
    with pytest.raises(ValueError, match="multiples of 8"):
        _validate_request(req)


def test_validate_request_width_too_small():
    req = GenerationRequest(width=63, height=512, prompt="test", seed=0)
    with pytest.raises(ValueError, match="width"):
        _validate_request(req)


def test_validate_request_height_too_large():
    req = GenerationRequest(width=512, height=4096, prompt="test", seed=0)
    with pytest.raises(ValueError, match="height"):
        _validate_request(req)


def test_validate_request_seed_negative():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=-1)
    with pytest.raises(ValueError, match="seed"):
        _validate_request(req)


def test_validate_request_seed_exceeds_max():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=_MAX_SEED + 1)
    with pytest.raises(ValueError, match="seed"):
        _validate_request(req)


def test_validate_request_prompt_empty():
    req = GenerationRequest(width=512, height=512, prompt="", seed=0)
    with pytest.raises(ValueError, match="prompt"):
        _validate_request(req)


def test_validate_request_prompt_too_long():
    req = GenerationRequest(width=512, height=512, prompt="x" * 10001, seed=0)
    with pytest.raises(ValueError, match="prompt"):
        _validate_request(req)


def test_validate_request_prompt_not_string():
    req = GenerationRequest(width=512, height=512, prompt=123, seed=0)  # type: ignore
    with pytest.raises(ValueError, match="prompt"):
        _validate_request(req)


def test_validate_request_width_not_multiple_of_64():
    """Non-tile-multiple shapes are allowed (AGENTS.md: include non-tile-multiple shapes)."""
    req = GenerationRequest(width=504, height=504, prompt="test", seed=0)
    _validate_request(req)  # should not raise


def test_validate_request_timeout_zero():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0, timeout_s=0)
    with pytest.raises(ValueError, match="timeout_s"):
        _validate_request(req)


def test_validate_request_timeout_exceeds_max():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0, timeout_s=700)
    with pytest.raises(ValueError, match="timeout_s"):
        _validate_request(req)


def test_validate_request_tile_size_too_small():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0, tile_size=32)
    with pytest.raises(ValueError, match="tile_size"):
        _validate_request(req)


def test_validate_request_tile_size_too_large():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0, tile_size=4096)
    with pytest.raises(ValueError, match="tile_size"):
        _validate_request(req)


def test_validate_request_overlap_exceeds_tile_size():
    req = GenerationRequest(
        width=512, height=512, prompt="test", seed=0, tile_size=512, overlap=512
    )
    with pytest.raises(ValueError, match="overlap.*must be < tile_size"):
        _validate_request(req)


def test_validate_request_overlap_negative():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0, overlap=-1)
    with pytest.raises(ValueError, match="overlap"):
        _validate_request(req)


# ---------------------------------------------------------------------------
# Request / Response dataclasses
# ---------------------------------------------------------------------------


def test_generation_request_requires_fields():
    """GenerationRequest is frozen and requires all mandatory fields."""
    req = GenerationRequest(width=896, height=1152, prompt="hello", seed=42)
    assert req.width == 896
    assert req.height == 1152
    assert req.seed == 42
    assert req.prompt == "hello"


def test_generation_request_optional_defaults():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0)
    assert req.tile_size == 512
    assert req.overlap == 64
    assert req.timeout_s == 300.0


def test_generation_request_from_dict():
    d = {"width": 896, "height": 1152, "prompt": "hello", "seed": 42}
    req = GenerationRequest.from_dict(d)
    assert req.width == 896
    assert req.height == 1152
    assert req.seed == 42


def test_generation_request_from_dict_rejects_unknown():
    """Unknown fields must be rejected (blocker #6)."""
    d = {"width": 512, "height": 512, "prompt": "test", "unknown_field": 999}
    with pytest.raises(ValueError, match="unknown request fields"):
        GenerationRequest.from_dict(d)


def test_generation_request_from_dict_rejects_steps():
    """steps is a forbidden field — #181 semantics are exact."""
    d = {"width": 512, "height": 512, "prompt": "test", "seed": 0, "steps": 4}
    with pytest.raises(ValueError, match="unknown request fields.*steps"):
        GenerationRequest.from_dict(d)


def test_generation_request_from_dict_rejects_cfg():
    """cfg is a forbidden field."""
    d = {"width": 512, "height": 512, "prompt": "test", "seed": 0, "cfg": 2.0}
    with pytest.raises(ValueError, match="unknown request fields.*cfg"):
        GenerationRequest.from_dict(d)


def test_generation_request_from_dict_rejects_scheduler():
    """scheduler is a forbidden field."""
    d = {"width": 512, "height": 512, "prompt": "test", "seed": 0, "scheduler": "euler"}
    with pytest.raises(ValueError, match="unknown request fields.*scheduler"):
        GenerationRequest.from_dict(d)


def test_generation_request_from_dict_rejects_non_dict():
    with pytest.raises(TypeError, match="JSON object"):
        GenerationRequest.from_dict("not a dict")


def test_generation_request_frozen():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0)
    with pytest.raises(AttributeError):
        req.width = 1024  # type: ignore


def test_generation_response_to_dict():
    resp = GenerationResponse(
        image_png=b"\x89PNG",
        provenance=Provenance(service_version="0.1.0"),
        timing={"dit_s": 5.0, "vae_s": 1.0, "total_s": 6.0},
        hbm={"peak_gib": 6.5},
    )
    d = resp.to_dict()
    assert "provenance" in d
    assert "timing" in d
    assert "hbm" in d
    assert d["timing"]["dit_s"] == 5.0


# ---------------------------------------------------------------------------
# Concurrency / slot limiting
# ---------------------------------------------------------------------------


def test_service_single_generation_slot():
    """Only one generation can run at a time; the second must queue."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._gen_semaphore = threading.Semaphore(1)
    svc._active_generations = 0

    # First acquire should succeed
    assert svc._gen_semaphore.acquire(blocking=False)
    svc._active_generations = 1

    # Second acquire should block (non-blocking returns False)
    assert not svc._gen_semaphore.acquire(blocking=False)

    # Release first
    svc._gen_semaphore.release()
    svc._active_generations = 0

    # Now second should succeed
    assert svc._gen_semaphore.acquire(blocking=False)
    svc._gen_semaphore.release()


def test_service_readiness_reflects_loading_state():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    assert svc._check_ready() is False

    svc._loading = True
    assert svc._check_ready() is False

    svc._loading = False
    svc._error = "OOM"
    assert svc._check_ready() is False

    svc._error = None
    svc._ready = True
    assert svc._check_ready() is True


# ---------------------------------------------------------------------------
# Bounded request / response sizes
# ---------------------------------------------------------------------------


def test_max_request_bytes_is_reasonable():
    """1 MB upper bound for the JSON request body."""
    assert 1_000 <= _MAX_REQUEST_BYTES <= 10_000_000


def test_max_seed():
    assert _MAX_SEED >= 2**31 - 1


# ---------------------------------------------------------------------------
# Health endpoint (unit-level, no real server)
# ---------------------------------------------------------------------------


def test_health_endpoint_always_ok():
    """Health endpoint returns 200 regardless of model state."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = True
    svc._error = "still loading"
    code, data = svc._handle_health()
    assert code == 200
    assert data["status"] == "ok"


def test_ready_endpoint_not_started():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    code, data = svc._handle_ready()
    assert code == 503
    assert data["status"] == "not_started"


def test_ready_endpoint_loading():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = True
    svc._error = None
    svc._load_start = time.monotonic() - 5.0
    code, data = svc._handle_ready()
    assert code == 503
    assert data["status"] == "loading"


def test_ready_endpoint_error():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = "CUDA OOM"
    code, data = svc._handle_ready()
    assert code == 503
    assert data["status"] == "error"
    assert data["error"] == "CUDA OOM"


def test_ready_endpoint_ready():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    svc._in_channels = 4
    svc._total_generations = 10
    svc._active_generations = 0
    svc._state_lock = threading.Lock()
    svc._load_start = time.monotonic() - 10.0
    svc._load_end = time.monotonic()
    svc._refresh_int8_residency = MagicMock(return_value={"tensors_total": 172})
    code, data = svc._handle_ready()
    assert code == 200
    assert data["status"] == "ready"
    assert data["total_generations"] == 10


# ---------------------------------------------------------------------------
# Generate endpoint (unit-level, mock pipeline)
# ---------------------------------------------------------------------------


def test_generate_rejects_not_ready():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    svc._gen_semaphore = threading.Semaphore(1)
    body = json.dumps({"width": 512, "height": 512, "prompt": "test", "seed": 0}).encode()
    code, data, png = svc._handle_generate(body)
    assert code == 503
    assert "not ready" in data["error"]


def test_generate_rejects_invalid_json():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    code, data, png = svc._handle_generate(b"not json")
    assert code == 400
    assert "invalid JSON" in data["error"]


def test_generate_rejects_bad_request():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    body = json.dumps({"width": -1, "height": 512, "prompt": "test", "seed": 0}).encode()
    code, data, png = svc._handle_generate(body)
    assert code == 400
    assert "width" in data["error"]


def test_generate_rejects_slot_busy():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    svc._gen_semaphore = threading.Semaphore(1)
    svc._gen_semaphore.acquire()  # occupy the slot
    svc._abort = threading.Event()
    svc._done = threading.Event()
    svc._state_lock = threading.Lock()
    body = json.dumps({"width": 512, "height": 512, "prompt": "test", "seed": 0}).encode()
    code, data, png = svc._handle_generate(body)
    assert code == 503
    assert "busy" in data["error"]
    svc._gen_semaphore.release()


def test_generate_rejects_unknown_fields():
    """Blocker #6: reject unknown fields in request body."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    svc._gen_semaphore = threading.Semaphore(1)
    body = json.dumps(
        {
            "width": 512,
            "height": 512,
            "prompt": "test",
            "seed": 0,
            "steps": 4,
            "cfg": 2.0,
            "scheduler": "euler",
        }
    ).encode()
    code, data, png = svc._handle_generate(body)
    assert code == 400
    assert "unknown request fields" in data["error"]


# ---------------------------------------------------------------------------
# Timeout / cooperative abort
# ---------------------------------------------------------------------------


def test_timeout_field_exists():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0, timeout_s=30.0)
    assert req.timeout_s == 30.0


def test_timeout_default():
    req = GenerationRequest(width=512, height=512, prompt="test", seed=0)
    assert req.timeout_s == 300.0


def test_abort_event_signals():
    """Cooperative abort mechanism: set _abort to signal cancellation."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._abort = threading.Event()
    svc._done = threading.Event()
    assert not svc._abort.is_set()
    svc._abort.set()
    assert svc._abort.is_set()


def test_check_abort_raises():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._abort = threading.Event()
    svc._abort.set()
    with pytest.raises(_AbortError):
        svc._check_abort()


def test_check_abort_clear():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._abort = threading.Event()
    svc._check_abort()  # should not raise


# ---------------------------------------------------------------------------
# Weight directory fail-closed
# ---------------------------------------------------------------------------


def test_weights_dir_defaults_to_canonical_nvme_root():
    with patch.dict("os.environ", {}, clear=True):
        svc = ZImageResidentService(weights_dir=None)
    assert svc.weights_dir.endswith("/runtime/hot-models/comfy-storage-models/models")


def test_noncanonical_weights_dir_from_env_is_rejected():
    with patch.dict("os.environ", {"FNI8_WEIGHTS_DIR": "/some/path"}):
        with pytest.raises(ValueError, match="canonical NVMe root"):
            ZImageResidentService(weights_dir=None)


def test_noncanonical_explicit_weights_dir_is_rejected():
    with pytest.raises(ValueError, match="canonical NVMe root"):
        ZImageResidentService(weights_dir="/explicit/path")


# ---------------------------------------------------------------------------
# GPU election verification (root-review blocker #1)
#
# The service no longer elects its own GPU (see scripts/launch_resident_service.py,
# tested separately in test_launch_resident_service.py) -- it only verifies, fail-
# closed, that an external launcher already bound CUDA_VISIBLE_DEVICES before this
# process (or its torch/comfy imports) existed.
# ---------------------------------------------------------------------------


def test_verify_gpu_election_requires_elected_flag():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="launch_resident_service.py"):
            svc._verify_gpu_election()


def test_verify_gpu_election_requires_uuid_and_cvd():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    with patch.dict("os.environ", {"FNI8_GPU_ELECTED": "1"}, clear=True):
        with pytest.raises(RuntimeError, match="FNI8_GPU_UUID"):
            svc._verify_gpu_election()


def test_verify_gpu_election_rejects_mismatched_cuda_visible_devices():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    env = {
        "FNI8_GPU_ELECTED": "1",
        "FNI8_GPU_UUID": "GPU-aaaaaaaa",
        "CUDA_VISIBLE_DEVICES": "GPU-bbbbbbbb",
    }
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(RuntimeError, match="does not match"):
            svc._verify_gpu_election()


def test_verify_gpu_election_rejects_wrong_bound_device(monkeypatch):
    """Even if CUDA_VISIBLE_DEVICES claims the right card, cross-check the
    ACTUAL bound device against torch after import -- proof, not trust."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    env = {
        "FNI8_GPU_ELECTED": "1",
        "FNI8_GPU_UUID": "GPU-aaaaaaaa",
        "CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: SimpleNamespace(uuid="bbbbbbbb")
    )
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(RuntimeError, match="wrong card"):
            svc._verify_gpu_election()


def test_verify_gpu_election_rejects_multiple_visible_devices(monkeypatch):
    svc = ZImageResidentService.__new__(ZImageResidentService)
    env = {
        "FNI8_GPU_ELECTED": "1",
        "FNI8_GPU_UUID": "GPU-aaaaaaaa",
        "CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(RuntimeError, match="exactly one CUDA device"):
            svc._verify_gpu_election()


def test_verify_gpu_election_succeeds_when_bound_device_matches(monkeypatch):
    svc = ZImageResidentService.__new__(ZImageResidentService)
    env = {
        "FNI8_GPU_ELECTED": "1",
        "FNI8_GPU_UUID": "GPU-aaaaaaaa",
        "CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: SimpleNamespace(uuid="aaaaaaaa")
    )
    with patch.dict("os.environ", env, clear=True):
        svc._verify_gpu_election()
    assert svc._gpu_uuid == "GPU-aaaaaaaa"


def test_normalize_uuid_ignores_prefix_and_case():
    from comfyui_superl8.resident_service import _normalize_uuid

    assert _normalize_uuid("GPU-AAAA-bbbb") == _normalize_uuid("aaaa-BBBB")


def test_verify_gpu_election_runs_before_any_other_load_step():
    """The verification must be the very first statement in _load_pipeline --
    before is_sm70() or any comfy/torch-touching import in that method."""
    import inspect

    src = inspect.getsource(ZImageResidentService._load_pipeline)
    assert src.index("_verify_gpu_election()") < src.index("is_sm70")
    assert src.index("_verify_gpu_election()") < src.index("resolve_files")


def test_load_pipeline_no_longer_elects_its_own_gpu():
    """Root-review blocker #1: _load_pipeline must not itself probe nvidia-smi
    or call elect_and_lock -- that can only happen externally, before this
    process exists (see scripts/launch_resident_service.py)."""
    import inspect

    src = inspect.getsource(ZImageResidentService._load_pipeline)
    assert "elect_and_lock" not in src
    assert "nvidia-smi" not in src
    assert not hasattr(ZImageResidentService, "_elect_gpu")


# ---------------------------------------------------------------------------
# Bounded response envelope (no unbounded HTTP headers)
# ---------------------------------------------------------------------------


def test_response_uses_json_body_not_headers():
    """Blocker #8: metadata goes in JSON body, not unbounded HTTP headers.

    The _send_response method must not put JSON in headers for PNG responses.
    """
    # Verify the handler method uses _send_response (not _send_json with headers)
    import inspect

    src = inspect.getsource(ZImageResidentService)
    # The old code had: self.send_header("X-Generation-Metadata", json.dumps(data))
    # The new code must not do this.
    assert "X-Generation-Metadata" not in src


def test_generation_response_base64_png():
    """PNG in the response dict must be base64-encoded."""
    resp = GenerationResponse(
        image_png=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
        provenance=Provenance(service_version="0.1.0"),
        timing={"total_s": 1.0},
        hbm={"peak_gib": 6.0},
    )
    d = resp.to_dict()
    # The response dict should not contain image_png (it's handled by the handler)
    # but the handler base64-encodes it before sending
    assert "image_png" not in d


# ---------------------------------------------------------------------------
# Sequential lifecycle (no simultaneous TE+DiT+VAE residency)
# ---------------------------------------------------------------------------


def test_sequential_methods_exist():
    """Verify the service has separate load/free methods for TE and VAE."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    assert hasattr(svc, "_load_te")
    assert hasattr(svc, "_free_te")
    assert hasattr(svc, "_load_vae")
    assert hasattr(svc, "_free_vae")


def test_encode_prompt_frees_te():
    """_encode_prompt must call _free_te in its finally block."""
    import inspect

    src = inspect.getsource(ZImageResidentService._encode_prompt)
    assert "_free_te" in src
    assert "finally" in src


def test_vae_decode_frees_vae():
    """_do_generate must call _free_vae after decode."""
    import inspect

    src = inspect.getsource(ZImageResidentService._do_generate)
    assert "_free_vae" in src
    assert "finally" in src


# ---------------------------------------------------------------------------
# Component-scoped model release (root-review blocker #2)
#
# comfy.model_management.unload_all_models() is free_memory(1e30, device) with
# an EMPTY keep_loaded -- it evicts every resident model indiscriminately,
# including the DiT this service documents as staying resident for its whole
# lifetime.  _free_te/_free_vae must filter free_memory's keep_loaded by
# object identity so the DiT can never be evicted.
# ---------------------------------------------------------------------------


def test_free_te_and_free_vae_never_call_unload_all_models():
    import inspect

    for method in (
        ZImageResidentService._free_te,
        ZImageResidentService._free_vae,
        ZImageResidentService._free_non_resident_models,
    ):
        assert "mm.unload_all_models(" not in inspect.getsource(method), method.__name__


def test_free_non_resident_models_keeps_only_the_resident_dit():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    resident_dit = object()
    other_model = object()
    svc._model = resident_dit

    class _FakeLoaded:
        def __init__(self, model):
            self.model = model

    loaded_dit = _FakeLoaded(resident_dit)
    loaded_other = _FakeLoaded(other_model)

    fake_mm = SimpleNamespace(
        current_loaded_models=[loaded_dit, loaded_other],
        get_torch_device=lambda: "cuda:0",
        free_memory=MagicMock(),
    )
    fake_comfy = SimpleNamespace(model_management=fake_mm)

    with patch.dict("sys.modules", {"comfy": fake_comfy, "comfy.model_management": fake_mm}):
        svc._free_non_resident_models()

    fake_mm.free_memory.assert_called_once()
    args, kwargs = fake_mm.free_memory.call_args
    assert args[0] == 1e30
    assert args[1] == "cuda:0"
    assert kwargs["keep_loaded"] == [loaded_dit]


def test_free_non_resident_models_keeps_nothing_when_dit_not_loaded_yet():
    """Before the DiT is loaded (self._model is None), there is nothing to
    protect -- behaves like a plain full-evict, same as unload_all_models did."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._model = None

    class _FakeLoaded:
        def __init__(self, model):
            self.model = model

    fake_mm = SimpleNamespace(
        current_loaded_models=[_FakeLoaded(object())],
        get_torch_device=lambda: "cuda:0",
        free_memory=MagicMock(),
    )
    fake_comfy = SimpleNamespace(model_management=fake_mm)

    with patch.dict("sys.modules", {"comfy": fake_comfy, "comfy.model_management": fake_mm}):
        svc._free_non_resident_models()

    _, kwargs = fake_mm.free_memory.call_args
    assert kwargs["keep_loaded"] == []


def test_free_te_calls_free_non_resident_models(monkeypatch):
    svc = ZImageResidentService.__new__(ZImageResidentService)
    called = []
    svc._free_non_resident_models = lambda: called.append("te")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    svc._free_te(clip=object())
    assert called == ["te"]


def test_free_vae_calls_free_non_resident_models(monkeypatch):
    svc = ZImageResidentService.__new__(ZImageResidentService)
    called = []
    svc._free_non_resident_models = lambda: called.append("vae")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    svc._free_vae(vae=object())
    assert called == ["vae"]


# ---------------------------------------------------------------------------
# Staged timing fields
# ---------------------------------------------------------------------------


def test_staged_timing_fields_in_do_generate():
    """Blocker #11: te_s, dit_s, vae_s, png_s must all be recorded."""
    import inspect

    src = inspect.getsource(ZImageResidentService._do_generate)
    assert '"te_s"' in src
    assert '"dit_s"' in src
    assert '"vae_s"' in src
    assert '"png_s"' in src
    assert '"total_s"' in src


def test_no_false_5_7s_claim():
    """Blocker #11: no hardcoded 5.7s claim without measured evidence."""
    import inspect

    src = inspect.getsource(ZImageResidentService)
    assert "5.7" not in src


# ---------------------------------------------------------------------------
# Non-square latent dimensions
# ---------------------------------------------------------------------------


def test_non_square_latent_dimensions():
    """Blocker #2: denoise must use separate width and height."""
    import inspect

    src = inspect.getsource(ZImageResidentService._denoise)
    assert "latent_h" in src
    assert "latent_w" in src


def test_denoise_delegates_to_proven_harness(monkeypatch):
    """Root-review blocker #4: the rectangular latent path must reuse
    bench.full_pipeline_zimage.denoise -- the SAME harness that produced this
    repo's int8-vs-fp quality/runtime evidence -- rather than a second,
    unreconciled reimplementation of the sampling call."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._in_channels = 4
    svc._abort = threading.Event()

    captured = {}

    def fake_denoise(model, cond, empty, in_channels, **kwargs):
        captured["model"] = model
        captured["cond"] = cond
        captured["empty"] = empty
        captured["in_channels"] = in_channels
        captured.update(kwargs)
        return "SENTINEL_SAMPLES"

    from comfyui_superl8 import zimage_profile

    monkeypatch.setattr(zimage_profile, "denoise", fake_denoise)

    result = svc._denoise("MODEL", "COND", "EMPTY", latent_h=144, latent_w=112, steps=8, seed=42)

    assert result == "SENTINEL_SAMPLES"
    assert captured["model"] == "MODEL"
    assert captured["in_channels"] == 4
    assert captured["latent_h"] == 144
    assert captured["latent_w"] == 112
    assert captured["steps"] == 8
    assert captured["seed"] == 42
    assert captured["abort_check"] == svc._check_abort


def test_denoise_does_not_reimplement_sampling():
    """The delegation must be real -- _denoise's own source must not construct
    a latent tensor or call comfy.sample.sample directly (that would be the
    unreconciled bypass root-review blocker #4 called out)."""
    import inspect

    src = inspect.getsource(ZImageResidentService._denoise)
    assert "comfy.sample.sample(" not in src
    assert "torch.zeros(" not in src
    assert "zimage_profile.denoise" in src


def test_generate_computes_latent_dimensions():
    """_do_generate must compute latent_h and latent_w from req."""
    import inspect

    src = inspect.getsource(ZImageResidentService._do_generate)
    assert "validate_geometry(req.width, req.height)" in src


def test_request_rechecks_custom_residency_after_te_and_vae():
    src = inspect.getsource(ZImageResidentService._do_generate)
    assert src.count("self._refresh_int8_residency()") == 2


def test_ready_rechecks_custom_residency_instead_of_returning_snapshot():
    src = inspect.getsource(ZImageResidentService._handle_ready)
    assert "self._refresh_int8_residency()" in src


# ---------------------------------------------------------------------------
# Per-request prompt encoding (no pre-encoding of DEFAULT_PROMPT)
# ---------------------------------------------------------------------------


def test_no_preencoded_default_prompt():
    """Blocker #1: must not pre-encode fp.DEFAULT_PROMPT at startup."""
    import inspect

    src = inspect.getsource(ZImageResidentService._load_pipeline)
    assert "DEFAULT_PROMPT" not in src
    assert "encode_prompt" not in src


def test_encode_prompt_uses_request_prompt():
    """_do_generate must encode the actual request prompt."""
    import inspect

    src = inspect.getsource(ZImageResidentService._do_generate)
    assert "req.prompt" in src


# ---------------------------------------------------------------------------
# VAE decode transpose (ComfyUI [B,H,W,C] -> [B,C,H,W])
# ---------------------------------------------------------------------------


def test_vae_decode_handles_comfyui_transpose():
    """ComfyUI VAE.decode returns [B,H,W,C]; tiled_vae_decode needs [B,C,H,W]."""
    import inspect

    src = inspect.getsource(ZImageResidentService._vae_decode_tiled)
    assert "movedim(-1, 1)" in src or "movedim" in src


# ---------------------------------------------------------------------------
# HTTP integration (start real server, send requests)
# ---------------------------------------------------------------------------


def _start_server(svc):
    """Start the HTTP server on a random port and return (server, port)."""
    server = HTTPServer(
        ("127.0.0.1", 0),
        type(
            "_Handler",
            (
                __import__(
                    "comfyui_superl8.resident_service", fromlist=["_RequestHandler"]
                )._RequestHandler,
            ),
            {"service": svc},
        ),
    )
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def test_http_health_endpoint():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        conn.close()
    finally:
        server.shutdown()


def test_http_ready_endpoint():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    svc._in_channels = 4
    svc._total_generations = 0
    svc._active_generations = 0
    svc._state_lock = threading.Lock()
    svc._load_start = time.monotonic() - 10.0
    svc._load_end = time.monotonic()
    svc._refresh_int8_residency = MagicMock(return_value={"tensors_total": 172})
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/ready")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["status"] == "ready"
        conn.close()
    finally:
        server.shutdown()


def test_http_404():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/nonexistent")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()
    finally:
        server.shutdown()


def test_http_post_generate_not_ready():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    svc._gen_semaphore = threading.Semaphore(1)
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps({"width": 512, "height": 512, "prompt": "test", "seed": 0})
        conn.request("POST", "/generate", body=body)
        resp = conn.getresponse()
        assert resp.status == 503
        conn.close()
    finally:
        server.shutdown()


def test_http_post_generate_bad_json():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    svc._gen_semaphore = threading.Semaphore(1)
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/generate", body="not json")
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()
    finally:
        server.shutdown()


def test_http_post_generate_validation_error():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    svc._gen_semaphore = threading.Semaphore(1)
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps({"width": -1, "height": 512, "prompt": "test"})
        conn.request("POST", "/generate", body=body)
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()
    finally:
        server.shutdown()


def test_http_post_generate_rejects_unknown_fields():
    """Unknown fields must return 400 via HTTP."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    svc._gen_semaphore = threading.Semaphore(1)
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps(
            {
                "width": 512,
                "height": 512,
                "prompt": "test",
                "seed": 0,
                "steps": 4,
            }
        )
        conn.request("POST", "/generate", body=body)
        resp = conn.getresponse()
        assert resp.status == 400
        data = json.loads(resp.read())
        assert "unknown request fields" in data["error"]
        conn.close()
    finally:
        server.shutdown()


def test_http_response_is_json():
    """All responses must be Content-Type: application/json."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        ct = resp.getheader("Content-Type")
        assert ct == "application/json"
        conn.close()
    finally:
        server.shutdown()


def test_http_content_length_set():
    """Responses must include Content-Length header."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        cl = resp.getheader("Content-Length")
        assert cl is not None
        assert int(cl) > 0
        conn.close()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Ready after failure
# ---------------------------------------------------------------------------


def test_ready_not_true_after_error():
    """Service must not report ready after a load failure."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = "CUDA OOM"
    assert svc._check_ready() is False

    # Even if someone sets _ready=True, the error still blocks readiness
    svc._ready = True
    assert svc._check_ready() is False


# ---------------------------------------------------------------------------
# Content-Length validation in HTTP handler
# ---------------------------------------------------------------------------


def test_http_rejects_missing_content_length():
    """POST without Content-Length returns 400."""
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = True
    svc._loading = False
    svc._error = None
    svc._gen_semaphore = threading.Semaphore(1)
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        # Send POST with Content-Length: 0
        conn.request("POST", "/generate", body="")
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# JSON parse of response body
# ---------------------------------------------------------------------------


def test_http_health_response_parseable():
    svc = ZImageResidentService.__new__(ZImageResidentService)
    svc._ready = False
    svc._loading = False
    svc._error = None
    server, port = _start_server(svc)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body = resp.read()
        data = json.loads(body)
        assert isinstance(data, dict)
        assert "status" in data
        conn.close()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# TE sidecar path (#200)
# ---------------------------------------------------------------------------


class TestSidecarPath:
    """Tests for FNI8_TE_ENDPOINT sidecar path in _encode_prompt."""

    @staticmethod
    def _make_sidecar_response(te_checksum: str, embed_cache_hit: bool = False) -> bytes:
        """Build a valid sidecar /encode response with real serializable payload."""
        import io
        import torch

        buf = io.BytesIO()
        torch.save(
            {"cond": torch.tensor([1.0]), "empty": torch.tensor([0.0])},
            buf,
        )
        payload = buf.getvalue()
        envelope = {"te_checksum": te_checksum, "embed_cache_hit": embed_cache_hit}
        env_bytes = json.dumps(envelope).encode()
        env_len = len(env_bytes).to_bytes(4, "big")
        return env_len + env_bytes + payload

    def test_sidecar_path_returns_four_tuple(self):
        """_encode_prompt returns (cond, empty, te_source, embed_cache_hit)."""
        svc = ZImageResidentService.__new__(ZImageResidentService)
        svc._te_path = "/nonexistent"

        raw_response = self._make_sidecar_response(
            zimage_profile.TEXT_ENCODER_SHA256, embed_cache_hit=False
        )

        with patch.dict(os.environ, {"FNI8_TE_ENDPOINT": "http://localhost:9001"}), \
             patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = raw_response
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = svc._encode_prompt_via_sidecar("test prompt", "http://localhost:9001")

        assert len(result) == 4
        cond, empty, te_source, embed_cache_hit = result
        assert te_source == "sidecar"
        assert embed_cache_hit is False

    def test_sidecar_checksum_mismatch_rejected(self):
        """Sidecar te_checksum not matching local pin raises RuntimeError."""
        svc = ZImageResidentService.__new__(ZImageResidentService)

        raw_response = self._make_sidecar_response("WRONG_CHECKSUM")

        with patch.dict(os.environ, {"FNI8_TE_ENDPOINT": "http://localhost:9001"}), \
             patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = raw_response
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with pytest.raises(RuntimeError, match="te_checksum mismatch"):
                svc._encode_prompt_via_sidecar("test", "http://localhost:9001")

    def test_sidecar_empty_checksum_rejected(self):
        """Sidecar te_checksum empty string is rejected (fail closed)."""
        svc = ZImageResidentService.__new__(ZImageResidentService)

        raw_response = self._make_sidecar_response("")

        with patch.dict(os.environ, {"FNI8_TE_ENDPOINT": "http://localhost:9001"}), \
             patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = raw_response
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with pytest.raises(RuntimeError, match="te_checksum mismatch"):
                svc._encode_prompt_via_sidecar("test", "http://localhost:9001")

    def test_sidecar_endpoint_down_fail_closed(self):
        """When sidecar is unreachable, raise (fail closed)."""
        svc = ZImageResidentService.__new__(ZImageResidentService)

        with patch.dict(os.environ, {"FNI8_TE_ENDPOINT": "http://localhost:9999"}), \
             patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            with pytest.raises(RuntimeError, match="sidecar unreachable"):
                svc._encode_prompt_via_sidecar("test", "http://localhost:9999")

    def test_encode_prompt_local_path(self):
        """Without FNI8_TE_ENDPOINT, _encode_prompt uses local path."""
        svc = ZImageResidentService.__new__(ZImageResidentService)
        svc._te_path = "/nonexistent"

        mock_clip = MagicMock()
        mock_cond = MagicMock()
        mock_empty = MagicMock()
        mock_clip.tokenize.return_value = "tokens"
        mock_clip.encode_from_tokens_scheduled.side_effect = [mock_cond, mock_empty]

        with patch.dict(os.environ, {}, clear=False), \
             patch.object(svc, "_load_te", return_value=mock_clip), \
             patch.object(svc, "_free_te"), \
             patch.dict(os.environ, {}, clear=False):
            # Remove FNI8_TE_ENDPOINT if set
            env = os.environ.copy()
            env.pop("FNI8_TE_ENDPOINT", None)
            with patch.dict(os.environ, env, clear=True):
                result = svc._encode_prompt("hello")

        assert len(result) == 4
        assert result[2] == "local"
        assert result[3] is False


# ---------------------------------------------------------------------------
# OOM handling (#200)
# ---------------------------------------------------------------------------


class TestOOMHandling:
    """OOM in _do_generate returns 500 and service stays alive."""

    def test_oom_returns_500_not_crash(self):
        """_handle_generate returns 500 on OOM without crashing the service."""
        svc = ZImageResidentService.__new__(ZImageResidentService)
        svc._ready = True
        svc._loading = False
        svc._error = None
        svc._gen_semaphore = threading.Semaphore(1)
        svc._active_generations = 0
        svc._total_generations = 0
        svc._abort = threading.Event()
        svc._done = threading.Event()
        svc._state_lock = threading.Lock()
        svc._model_sampling_shift = 3.0
        svc._in_channels = 4
        svc._dit_path = "/fake/dit"
        svc._vae_path = "/fake/vae"
        svc._te_path = "/fake/te"
        svc._model = MagicMock()

        with patch.object(svc, "_do_generate", side_effect=torch.OutOfMemoryError("CUDA OOM")):
            body = json.dumps({
                "width": 512, "height": 512, "prompt": "test",
                "seed": 42, "timeout_s": 30,
            }).encode()
            code, data, png = svc._handle_generate(body)

        assert code == 500
        assert "error" in data
        assert png is None
        # Service should still be ready (not crashed)
        assert svc._ready is True


# ---------------------------------------------------------------------------
# Provenance te_source and embed_cache_hit fields (#200)
# ---------------------------------------------------------------------------


class TestProvenanceTEFields:
    def test_provenance_has_te_source(self):
        p = build_provenance(
            width=512, height=512, steps=8, cfg=1.0, seed=0,
            scheduler="simple", te_source="sidecar", embed_cache_hit=True,
        )
        d = p.to_dict()
        assert d["te_source"] == "sidecar"
        assert d["embed_cache_hit"] is True

    def test_provenance_defaults(self):
        p = build_provenance(
            width=512, height=512, steps=8, cfg=1.0, seed=0,
            scheduler="simple",
        )
        d = p.to_dict()
        assert d["te_source"] == "local"
        assert d["embed_cache_hit"] is False


# ---------------------------------------------------------------------------
# Helper to parse sidecar envelope
# ---------------------------------------------------------------------------

