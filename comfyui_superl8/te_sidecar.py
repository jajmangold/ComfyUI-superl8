# SPDX-License-Identifier: Apache-2.0
"""Z-Image TE encoder HTTP sidecar (#200 Phase 2).

Resident Qwen3-4B text encoder on its own GPU.  The Z-Image fni8 resident
service fetches conditioning over HTTP instead of loading the TE itself,
avoiding the ~85 s/request disk-load+build penalty on a 16 GiB card that
cannot hold DiT+TE simultaneously (issue #200 -- single-card co-residency
is a measured dead end).

Endpoints:
  GET  /health -- liveness (always 200 if process is up)
  GET  /ready  -- readiness (200 when TE is loaded, 503 otherwise)
  POST /encode -- encode a prompt; returns serialized conditioning envelope

Architecture:
  - stdlib http.server + threading (no external deps beyond torch/comfy)
  - Single-slot semaphore (one encode at a time)
  - TE loaded once at startup, stays resident for the service lifetime
  - Empty-string conditioning cached once at startup
  - All tensors moved to CPU before serialization (network transfer)
  - SHA-256 embedding LRU cache keyed with pipelines#117 compute_cache_key
    derivation: sha256(prompt \\0 '' \\0 te_model_revision)
  - CUDA_VISIBLE_DEVICES binding via --gpus flag (no GPU election needed)
  - /encode response includes te_checksum in JSON envelope for provenance

Cache key contract (replicates pipelines text_embed_bundle.compute_cache_key):
    material = f"{prompt}\\0{neg}\\0{te_model_revision}".encode("utf-8")
    cache_key = hashlib.sha256(material).hexdigest()

The negative prompt is always '' for the Z-Image CFG=1 path.  te_model_revision
is the TEXT_ENCODER_SHA256 from zimage_profile.  This derivation MUST match
pipelines#117 exactly -- see contract test in tests/test_te_sidecar.py.

Usage:
    CUDA_VISIBLE_DEVICES=<uuid> python3 -m comfyui_superl8.te_sidecar \\
        --host 0.0.0.0 --port 9001

The sidecar does NOT perform GPU election -- it is pinned via
CUDA_VISIBLE_DEVICES externally (e.g. docker --gpus).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

_MAX_REQUEST_BYTES = 1_048_576  # 1 MB JSON body
_MAX_PROMPT_LEN = 10_000
_DEFAULT_CACHE_SIZE = 2048


# ---------------------------------------------------------------------------
# Cache key derivation (matches pipelines text_embed_bundle.compute_cache_key)
# pipelines#117: sha256(prompt \0 neg \0 te_model_revision)
# ---------------------------------------------------------------------------


def compute_cache_key(prompt: str, te_model_revision: str) -> str:
    """Return the hex SHA-256 cache key for a text-embed bundle.

    Replicates the exact derivation from content-factory-pipelines
    text_embed_bundle.compute_cache_key (pipelines#117).  The negative
    prompt is always '' for the Z-Image CFG=1 path.

    Contract: changing prompt, te_model_revision, or (hypothetically) neg
    MUST produce a different cache address.  Verified by
    tests/test_te_sidecar.py::test_cache_key_contract_matches_pipelines.
    """
    material = f"{prompt}\0\0{te_model_revision}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


# ---------------------------------------------------------------------------
# LRU cache for embeddings
# ---------------------------------------------------------------------------


class EmbeddingLRUCache:
    """Thread-safe LRU cache for serialized conditioning tensors.

    Keys are SHA-256 hex digests (compute_cache_key).  Values are the raw
    bytes that would be returned by /encode (the torch.save envelope).

    Worst-case memory: max_size entries * average blob size.  A typical
    Z-Image conditioning blob at 512x512 is ~50-100 KB; at the default
    max_size=2048 that is ~100-200 MB host RAM worst-case.  The cache is
    bounded and evicts LRU on overflow -- memory usage is capped regardless
    of blob size.
    """

    def __init__(self, max_size: int = _DEFAULT_CACHE_SIZE):
        self._max_size = max_size
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> bytes | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


# ---------------------------------------------------------------------------
# TE loader -- reuses the proven ComfyUI helpers exactly as bench/full_pipeline_zimage.py
# ---------------------------------------------------------------------------


def _load_te_clip():
    """Load the Qwen3-4B text encoder via ComfyUI's Z-Image CLIP path.

    Returns (clip, checksum) -- the clip model object ready for tokenize +
    encode_from_tokens_scheduled, and the TEXT_ENCODER_SHA256 from profile.
    This is the SAME code path as bench/full_pipeline_zimage.py:encode_prompt
    and resident_service.py:_load_te -- clip_type=None triggers ComfyUI's
    TEModel.QWEN3_4B detection, building the Z-Image TE with the <|im_start|>
    template and hidden layer -2.
    """
    import comfy.sd
    import comfy.utils
    from comfyui_superl8 import zimage_profile

    te_path = os.path.join(
        zimage_profile.CANONICAL_MODEL_ROOT,
        zimage_profile.TEXT_ENCODER_RELATIVE_PATH,
    )
    if not os.path.isfile(te_path):
        raise FileNotFoundError(f"Qwen3-4B TE not found: {te_path}")

    logger.info("loading Qwen3-4B TE from %s ...", te_path)
    t0 = time.monotonic()
    merged = comfy.utils.load_torch_file(te_path, safe_load=True)
    clip = comfy.sd.load_text_encoder_state_dicts([merged], clip_type=None)
    logger.info("TE loaded in %.1f s", time.monotonic() - t0)
    return clip, zimage_profile.TEXT_ENCODER_SHA256


def _move_tensors_to_cpu(obj):
    """Recursively move every torch.Tensor in a nested structure to CPU.

    ComfyUI conditioning is list[tuple[tensor, dict]] -- the dict may contain
    cross-attention control tensors.  We walk the entire structure and move
    every tensor to CPU so torch.save produces a CPU-only checkpoint.
    """
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_tensors_to_cpu(item) for item in obj]
        return type(obj)(moved)
    return obj


def _parse_sidecar_response(raw: bytes) -> dict:
    """Parse the sidecar's /encode response envelope.

    Format: 4-byte big-endian length prefix + JSON envelope + torch.save payload.

    Returns dict with keys: 'te_checksum', 'embed_cache_hit', 'cache_key', 'payload'.
    Raises RuntimeError on malformed response.
    """
    if len(raw) < 4:
        raise RuntimeError(f"sidecar response too short ({len(raw)} bytes)")
    env_len = int.from_bytes(raw[:4], "big")
    if len(raw) < 4 + env_len:
        raise RuntimeError(
            f"sidecar envelope length ({env_len}) exceeds response size ({len(raw)})"
        )
    env_bytes = raw[4 : 4 + env_len]
    payload = raw[4 + env_len :]
    envelope = json.loads(env_bytes)
    envelope["payload"] = payload
    return envelope


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TEEncoderSidecar:
    """Resident HTTP daemon for Z-Image TE encoding.

    Loads the Qwen3-4B text encoder once at startup and serves encode
    requests over HTTP.  The Z-Image fni8 resident service calls this
    instead of loading the TE itself.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9001,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ):
        self.host = host
        self.port = port

        # State
        self._ready = False
        self._loading = False
        self._error: str | None = None
        self._load_start: float = 0.0
        self._load_end: float = 0.0
        self._te_checksum: str = ""

        # TE objects (populated by _load_te)
        self._clip = None
        self._empty_cond = None  # cached "" encoding

        # Embedding LRU cache
        self._cache = EmbeddingLRUCache(max_size=cache_size)

        # Single encode slot
        self._encode_semaphore = threading.Semaphore(1)

    def _check_ready(self) -> bool:
        return self._ready and not self._loading and self._error is None

    def _load_te(self) -> None:
        """Load TE in background thread at startup."""
        self._loading = True
        self._load_start = time.monotonic()
        try:
            clip, checksum = _load_te_clip()
            self._clip = clip
            self._te_checksum = checksum

            # Cache empty-string conditioning once
            logger.info("caching empty-string conditioning ...")
            empty_tokens = clip.tokenize("")
            self._empty_cond = _move_tensors_to_cpu(
                clip.encode_from_tokens_scheduled(empty_tokens)
            )

            self._ready = True
            self._loading = False
            self._load_end = time.monotonic()
            logger.info(
                "TE sidecar ready in %.1f s (checksum=%s)",
                self._load_end - self._load_start,
                checksum[:16],
            )
        except Exception as e:
            self._loading = False
            self._error = str(e)
            self._load_end = time.monotonic()
            logger.error("TE load failed: %s", e)
            raise

    def _handle_health(self) -> tuple[int, dict]:
        """GET /health -- liveness (always 200 if process is up)."""
        return 200, {"status": "ok", "te_checksum": self._te_checksum}

    def _handle_ready(self) -> tuple[int, dict]:
        """GET /ready -- readiness (200 when TE is loaded)."""
        if self._error:
            return 503, {"status": "error", "error": self._error}
        if self._loading:
            elapsed = time.monotonic() - self._load_start
            return 503, {"status": "loading", "elapsed_s": round(elapsed, 1)}
        if self._ready:
            return 200, {
                "status": "ready",
                "te_checksum": self._te_checksum,
                "load_time_s": round(self._load_end - self._load_start, 1),
                "cache_stats": self._cache.stats,
            }
        return 503, {"status": "not_started"}

    def _handle_encode(self, body: bytes) -> tuple[int, dict | None, bytes | None]:
        """POST /encode -- encode a prompt.  Returns (code, json_err, octet_stream)."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return 400, {"error": f"invalid JSON: {e}"}, None

        if not isinstance(data, dict):
            return 400, {"error": "request body must be a JSON object"}, None

        # Reject unknown fields
        known = {"prompt"}
        unknown = set(data.keys()) - known
        if unknown:
            return 400, {"error": f"unknown fields: {', '.join(sorted(unknown))}"}, None

        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return 400, {"error": "prompt must be a non-empty string"}, None
        if len(prompt) > _MAX_PROMPT_LEN:
            return 400, {"error": f"prompt exceeds {_MAX_PROMPT_LEN} characters"}, None

        if not self._check_ready():
            return 503, {"error": "TE not ready", "detail": self._error}, None

        # Check embedding cache
        cache_key = compute_cache_key(prompt, self._te_checksum)
        cached = self._cache.get(cache_key)
        cache_hit = cached is not None

        if cache_hit:
            logger.info("cache hit for prompt (key=%s...)", cache_key[:16])
            # Wrap cached payload with envelope including te_checksum
            envelope = {
                "te_checksum": self._te_checksum,
                "embed_cache_hit": True,
                "cache_key": cache_key,
            }
            # Prepend envelope length + JSON envelope before cached payload
            env_bytes = json.dumps(envelope).encode()
            env_len = len(env_bytes).to_bytes(4, "big")
            return 200, None, env_len + env_bytes + cached

        # Acquire single encode slot
        acquired = self._encode_semaphore.acquire(blocking=False)
        if not acquired:
            return 503, {"error": "encode slot busy, try again later"}, None

        try:
            import torch

            t0 = time.monotonic()
            tokens = self._clip.tokenize(prompt)
            cond = _move_tensors_to_cpu(
                self._clip.encode_from_tokens_scheduled(tokens)
            )
            elapsed = time.monotonic() - t0
            logger.info("encoded prompt in %.2f s (len=%d)", elapsed, len(prompt))

            # Serialize: torch.save of {"cond", "empty"} -- all tensors on CPU
            buf = io.BytesIO()
            torch.save({"cond": cond, "empty": self._empty_cond}, buf)
            payload = buf.getvalue()

            # Cache the result
            self._cache.put(cache_key, payload)

            # Wrap with envelope
            envelope = {
                "te_checksum": self._te_checksum,
                "embed_cache_hit": False,
                "cache_key": cache_key,
            }
            env_bytes = json.dumps(envelope).encode()
            env_len = len(env_bytes).to_bytes(4, "big")
            return 200, None, env_len + env_bytes + payload
        except Exception as e:
            logger.error("encode failed: %s\n%s", e, traceback.format_exc())
            return 500, {"error": str(e)}, None
        finally:
            self._encode_semaphore.release()


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _RequestHandler(BaseHTTPRequestHandler):
    """Dispatch requests to the sidecar service."""

    service: TEEncoderSidecar

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_octet(self, code: int, data: dict | None, payload: bytes):
        if data is not None:
            # Error case: send JSON
            self._send_json(code, data)
        else:
            # Success case: send octet-stream
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            code, data = self.service._handle_health()
            self._send_json(code, data)
        elif self.path == "/ready":
            code, data = self.service._handle_ready()
            self._send_json(code, data)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/encode":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self._send_json(400, {"error": "missing Content-Length"})
                return
            if length > _MAX_REQUEST_BYTES:
                self._send_json(
                    413,
                    {"error": f"request body too large ({length} > {_MAX_REQUEST_BYTES})"},
                )
                return
            body = self.rfile.read(length)
            code, err_data, payload = self.service._handle_encode(body)
            self._send_octet(code, err_data, payload)
        else:
            self._send_json(404, {"error": "not found"})


class _ThreadedHTTPServer(HTTPServer):
    """HTTPServer with daemon threads so shutdown is clean."""

    allow_reuse_address = True

    def process_request(self, request, client_address):
        t = threading.Thread(
            target=self.process_request_thread,
            args=(request, client_address),
            daemon=True,
        )
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import torch  # noqa: F401 -- verify CUDA is available before loading TE

    ap = argparse.ArgumentParser(description="Z-Image TE encoder HTTP sidecar")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--cache-size", type=int, default=_DEFAULT_CACHE_SIZE)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not torch.cuda.is_available():
        logger.error("CUDA is not available -- refusing to start")
        sys.exit(1)
    logger.info(
        "CUDA device: %s (%s)",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).uuid,
    )

    svc = TEEncoderSidecar(
        host=args.host, port=args.port, cache_size=args.cache_size
    )

    # Load TE in background thread
    loader = threading.Thread(target=svc._load_te, daemon=True)
    loader.start()

    # Start HTTP server
    handler = type(
        "_Handler",
        (_RequestHandler,),
        {"service": svc},
    )
    server = _ThreadedHTTPServer((args.host, args.port), handler)
    logger.info("TE sidecar listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
