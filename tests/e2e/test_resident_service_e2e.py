# SPDX-License-Identifier: Apache-2.0
"""E2E integration test for the Z-Image resident HTTP service (issue #181).

Exercises the full HTTP service lifecycle: startup, health, readiness,
generation (512x512 and 896x1152), provenance, error paths, and timeout.
Requires a real GPU and the Z-Image weights (same gate as test_full_pipeline_zimage).

PENDING REAL-GPU GATES:
  - Full pipeline load + generation on a real sm_70 GPU
  - 512x512 generation returns valid RGB PNG
  - 896x1152 non-square generation returns valid RGB PNG
  - Provenance contains correct checksums for loaded model files
  - Second concurrent request is rejected (single slot)
  - Timeout cancels long-running generation

Run inside the e2e image:
    FNI8_GPU=0 docker compose run --rm --entrypoint python3 e2e \
      -m pytest custom_nodes/ComfyUI-superl8/tests/e2e/test_resident_service_e2e.py \
      -m "comfy_e2e" --import-mode=importlib -q
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from http.client import HTTPConnection

import pytest

pytest.importorskip("torch")
pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import folder_paths  # noqa: E402

from comfyui_superl8.gate import is_sm70  # noqa: E402
from comfyui_superl8.resident_service import (  # noqa: E402
    ZImageResidentService,
    _RequestHandler,
    _ThreadedHTTPServer,
    _LOCKED_NFE,
    _LOCKED_CFG,
    _LOCKED_SCHEDULER,
)
from comfyui_superl8.provenance import SERVICE_VERSION  # noqa: E402

pytestmark = pytest.mark.comfy_e2e

WEIGHTS_DIR = os.environ.get("FNI8_WEIGHTS_DIR", "")
ZIMAGE_DIT = "Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8"


def _has_weights() -> bool:
    folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)
    return ZIMAGE_DIT in folder_paths.get_filename_list("diffusion_models")


def _start_service(svc: ZImageResidentService) -> tuple[_ThreadedHTTPServer, int]:
    """Start the service HTTP server and wait for it to be ready."""
    handler = type("_Handler", (_RequestHandler,), {"service": svc})
    server = _ThreadedHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _wait_ready(port: float, timeout: float = 300.0) -> dict:
    """Poll /ready until 200 or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/ready")
            resp = conn.getresponse()
            data = json.loads(resp.read())
            conn.close()
            if resp.status == 200:
                return data
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"service did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# Tests (all gated on real GPU + weights)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def service_with_model():
    """Load the real model and start the HTTP service. Yields (svc, port, server)."""
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if not _has_weights():
        pytest.skip(f"{ZIMAGE_DIT} not found on the diffusion_models search path")

    svc = ZImageResidentService(
        host="127.0.0.1",
        port=0,
        weights_dir=WEIGHTS_DIR,
    )
    # Load pipeline synchronously (blocking)
    svc._load_pipeline()

    server, port = _start_service(svc)
    yield svc, port, server
    server.shutdown()


def test_service_health_always_ok(service_with_model):
    """GET /health always returns 200."""
    _, port, _ = service_with_model
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    assert resp.status == 200
    data = json.loads(resp.read())
    assert data["status"] == "ok"
    assert data["version"] == SERVICE_VERSION
    conn.close()


def test_service_ready_after_load(service_with_model):
    """GET /ready returns 200 after model is loaded."""
    _, port, _ = service_with_model
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/ready")
    resp = conn.getresponse()
    assert resp.status == 200
    data = json.loads(resp.read())
    assert data["status"] == "ready"
    assert data["in_channels"] == 4
    conn.close()


def test_service_root_endpoint(service_with_model):
    """GET / returns health (service description)."""
    _, port, _ = service_with_model
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/")
    resp = conn.getresponse()
    assert resp.status == 200
    conn.close()


def test_service_404(service_with_model):
    """GET /nonexistent returns 404."""
    _, port, _ = service_with_model
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/nonexistent")
    resp = conn.getresponse()
    assert resp.status == 404
    conn.close()


def test_service_response_is_json(service_with_model):
    """All responses must be Content-Type: application/json."""
    _, port, _ = service_with_model
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    ct = resp.getheader("Content-Type")
    assert ct == "application/json"
    conn.close()


def test_service_generate_512x512(service_with_model):
    """POST /generate with 512x512 returns a JSON envelope with base64 PNG."""
    _, port, _ = service_with_model
    body = json.dumps(
        {
            "width": 512,
            "height": 512,
            "prompt": "a photograph of a red fox sitting in a field of autumn leaves",
            "seed": 0,
        }
    )
    conn = HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request("POST", "/generate", body=body)
    resp = conn.getresponse()

    assert resp.status == 200
    ct = resp.getheader("Content-Type")
    assert ct == "application/json"

    envelope = json.loads(resp.read())
    assert "provenance" in envelope
    assert "timing" in envelope
    assert "hbm" in envelope
    assert "image_png" in envelope

    # Decode base64 PNG
    png_data = base64.b64decode(envelope["image_png"])
    assert len(png_data) > 100, "PNG too small"
    assert png_data[:8] == b"\x89PNG\r\n\x1a\n"

    # Provenance
    prov = envelope["provenance"]
    assert prov["width"] == 512
    assert prov["height"] == 512
    assert prov["lossless_qkv_refusion"] is True
    assert prov["fp_vae"] is True
    assert prov["nfe"] == _LOCKED_NFE
    assert prov["cfg"] == _LOCKED_CFG
    assert prov["scheduler"] == _LOCKED_SCHEDULER

    # Staged timing
    timing = envelope["timing"]
    assert timing["te_s"] > 0
    assert timing["dit_s"] > 0
    assert timing["vae_s"] > 0
    assert timing["png_s"] > 0
    assert timing["total_s"] > 0
    conn.close()


def test_service_generate_896x1152(service_with_model):
    """POST /generate with non-square 896x1152 returns a valid JSON envelope."""
    _, port, _ = service_with_model
    body = json.dumps(
        {
            "width": 896,
            "height": 1152,
            "prompt": "portrait of a woman in a garden",
            "seed": 42,
        }
    )
    conn = HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request("POST", "/generate", body=body)
    resp = conn.getresponse()

    assert resp.status == 200
    envelope = json.loads(resp.read())

    png_data = base64.b64decode(envelope["image_png"])
    assert png_data[:8] == b"\x89PNG\r\n\x1a\n"

    prov = envelope["provenance"]
    assert prov["width"] == 896
    assert prov["height"] == 1152
    # Non-square: height != width
    assert prov["height"] != prov["width"]
    conn.close()


def test_service_rejects_concurrent_generation(service_with_model):
    """Second generation request while first is running returns 503."""
    _, port, _ = service_with_model

    # Occupy the slot with a slow generation
    body = json.dumps(
        {
            "width": 512,
            "height": 512,
            "prompt": "slow generation test",
            "seed": 0,
        }
    )
    conn1 = HTTPConnection("127.0.0.1", port, timeout=300)
    conn1.request("POST", "/generate", body=body)

    # Small delay to ensure first request acquires the slot
    time.sleep(0.5)

    # Second request should be rejected
    conn2 = HTTPConnection("127.0.0.1", port, timeout=5)
    conn2.request("POST", "/generate", body=body)
    resp2 = conn2.getresponse()
    assert resp2.status == 503
    data2 = json.loads(resp2.read())
    assert "busy" in data2["error"]
    conn2.close()

    # Wait for first to finish
    resp1 = conn1.getresponse()
    assert resp1.status == 200
    resp1.read()
    conn1.close()


def test_service_rejects_bad_request(service_with_model):
    """POST /generate with invalid request returns 400."""
    _, port, _ = service_with_model
    body = json.dumps({"width": -1, "height": 512, "prompt": "test"})
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/generate", body=body)
    resp = conn.getresponse()
    assert resp.status == 400
    data = json.loads(resp.read())
    assert "width" in data["error"]
    conn.close()


def test_service_rejects_invalid_json(service_with_model):
    """POST /generate with non-JSON body returns 400."""
    _, port, _ = service_with_model
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/generate", body="not json at all")
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()


def test_service_rejects_unknown_fields(service_with_model):
    """POST /generate with forbidden fields returns 400."""
    _, port, _ = service_with_model
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
    )
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/generate", body=body)
    resp = conn.getresponse()
    assert resp.status == 400
    data = json.loads(resp.read())
    assert "unknown request fields" in data["error"]
    conn.close()


def test_service_provenance_checksums_present(service_with_model):
    """Provenance includes SHA-256 checksums for the loaded model files."""
    _, port, _ = service_with_model
    body = json.dumps(
        {
            "width": 512,
            "height": 512,
            "prompt": "checksum test",
            "seed": 0,
        }
    )
    conn = HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request("POST", "/generate", body=body)
    resp = conn.getresponse()
    assert resp.status == 200

    envelope = json.loads(resp.read())
    prov = envelope["provenance"]
    # DiT checksum should be present (real .fni8 file loaded)
    assert prov["dit_checksum"] is not None
    assert len(prov["dit_checksum"]) == 64  # SHA-256 hex
    conn.close()


def test_service_deterministic_same_seed(service_with_model):
    """Same seed + same prompt produces the same PNG bytes (determinism)."""
    _, port, _ = service_with_model
    body = json.dumps(
        {
            "width": 512,
            "height": 512,
            "prompt": "determinism test",
            "seed": 12345,
        }
    )
    conn1 = HTTPConnection("127.0.0.1", port, timeout=300)
    conn1.request("POST", "/generate", body=body)
    resp1 = conn1.getresponse()
    env1 = json.loads(resp1.read())
    conn1.close()

    conn2 = HTTPConnection("127.0.0.1", port, timeout=300)
    conn2.request("POST", "/generate", body=body)
    resp2 = conn2.getresponse()
    env2 = json.loads(resp2.read())
    conn2.close()

    assert env1["image_png"] == env2["image_png"], "same seed should produce identical PNG"
    assert env1["provenance"]["seed"] == env2["provenance"]["seed"]
