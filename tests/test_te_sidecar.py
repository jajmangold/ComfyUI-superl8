# SPDX-License-Identifier: Apache-2.0
"""Unit tests for comfyui_superl8.te_sidecar (#200).

Tests the cache key contract, LRU cache, request validation, sidecar
service internals, and envelope format WITHOUT requiring a GPU or real
weights.

Run:  python -m pytest tests/test_te_sidecar.py -q
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from comfyui_superl8.te_sidecar import (  # noqa: E402
    EmbeddingLRUCache,
    TEEncoderSidecar,
    _MAX_PROMPT_LEN,
    _parse_sidecar_response,
    compute_cache_key,
)


# ---------------------------------------------------------------------------
# Cache key contract (must match pipelines#117 exactly)
# ---------------------------------------------------------------------------


class TestCacheKeyContract:
    """Verify that compute_cache_key matches the pipelines#117 derivation."""

    def test_deterministic(self):
        k1 = compute_cache_key("hello world", "abc123")
        k2 = compute_cache_key("hello world", "abc123")
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex

    def test_different_prompt_different_key(self):
        k1 = compute_cache_key("prompt A", "rev1")
        k2 = compute_cache_key("prompt B", "rev1")
        assert k1 != k2

    def test_different_revision_different_key(self):
        k1 = compute_cache_key("prompt", "rev1")
        k2 = compute_cache_key("prompt", "rev2")
        assert k1 != k2

    def test_matches_pipelines_derivation(self):
        """Pin a known-good literal hash so drift from pipelines#117 fails the test.

        The expected hash was computed from:
            prompt = "a cute cat"
            te_model_revision = "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a"
            material = f"{prompt}\\0\\0{te_model_revision}".encode("utf-8")
            hashlib.sha256(material).hexdigest()
        """
        prompt = "a cute cat"
        te_rev = "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a"
        # Pre-computed literal -- if this diverges from pipelines#117, the test fails
        known_good = "3dbdf1b040765e881228c19ab71e4c0a99af3549737af1a4981ea357936d414c"
        result = compute_cache_key(prompt, te_rev)
        assert len(result) == 64, "must be SHA-256 hex"
        # The actual computed value must be stable; replace the literal if the
        # derivation is intentionally changed (coordinate with pipelines#117).
        assert result == known_good, (
            f"compute_cache_key({prompt!r}, {te_rev[:16]}...) = {result!r} "
            f"!= pinned literal {known_good!r} -- derivation drifted from pipelines#117?"
        )

    def test_empty_prompt(self):
        k = compute_cache_key("", "rev")
        assert len(k) == 64

    def test_unicode_prompt(self):
        k = compute_cache_key("\u4e16\u754c\u4f60\u597d", "rev")
        assert len(k) == 64


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------


class TestEmbeddingLRUCache:
    def test_put_get(self):
        c = EmbeddingLRUCache(max_size=10)
        c.put("key1", b"payload1")
        assert c.get("key1") == b"payload1"

    def test_miss(self):
        c = EmbeddingLRUCache(max_size=10)
        assert c.get("nonexistent") is None

    def test_eviction(self):
        c = EmbeddingLRUCache(max_size=2)
        c.put("a", b"1")
        c.put("b", b"2")
        c.put("c", b"3")  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == b"2"
        assert c.get("c") == b"3"

    def test_lru_ordering(self):
        c = EmbeddingLRUCache(max_size=2)
        c.put("a", b"1")
        c.put("b", b"2")
        c.get("a")  # "a" becomes most recent
        c.put("c", b"3")  # evicts "b" (least recent)
        assert c.get("a") == b"1"
        assert c.get("b") is None
        assert c.get("c") == b"3"

    def test_update_existing(self):
        c = EmbeddingLRUCache(max_size=2)
        c.put("a", b"1")
        c.put("a", b"2")
        assert c.get("a") == b"2"
        assert c.stats["size"] == 1

    def test_stats(self):
        c = EmbeddingLRUCache(max_size=10)
        c.put("a", b"1")
        c.get("a")
        c.get("b")
        stats = c.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_thread_safety(self):
        c = EmbeddingLRUCache(max_size=100)
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    c.put(f"key-{start + i}", b"x")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    c.get(f"key-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(100,)),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestSidecarRequestValidation:
    def test_empty_body(self):
        svc = TEEncoderSidecar()
        svc._ready = True
        code, err, _ = svc._handle_encode(b"")
        assert code == 400
        assert "invalid JSON" in err["error"]

    def test_not_json(self):
        svc = TEEncoderSidecar()
        svc._ready = True
        code, err, _ = svc._handle_encode(b"not json")
        assert code == 400

    def test_not_object(self):
        svc = TEEncoderSidecar()
        svc._ready = True
        code, err, _ = svc._handle_encode(b"[]")
        assert code == 400
        assert "JSON object" in err["error"]

    def test_unknown_fields(self):
        svc = TEEncoderSidecar()
        svc._ready = True
        code, err, _ = svc._handle_encode(b'{"prompt": "x", "unknown": 1}')
        assert code == 400
        assert "unknown fields" in err["error"]

    def test_empty_prompt(self):
        svc = TEEncoderSidecar()
        svc._ready = True
        code, err, _ = svc._handle_encode(b'{"prompt": ""}')
        assert code == 400
        assert "non-empty" in err["error"]

    def test_whitespace_only_prompt(self):
        svc = TEEncoderSidecar()
        svc._ready = True
        code, err, _ = svc._handle_encode(b'{"prompt": "   "}')
        assert code == 400

    def test_prompt_too_long(self):
        svc = TEEncoderSidecar()
        svc._ready = True
        body = json.dumps({"prompt": "x" * (_MAX_PROMPT_LEN + 1)}).encode()
        code, err, _ = svc._handle_encode(body)
        assert code == 400
        assert "exceeds" in err["error"]

    def test_not_ready(self):
        svc = TEEncoderSidecar()
        # _ready defaults to False
        code, err, _ = svc._handle_encode(b'{"prompt": "test"}')
        assert code == 503
        assert "not ready" in err["error"]


# ---------------------------------------------------------------------------
# Health/ready endpoints
# ---------------------------------------------------------------------------


class TestSidecarEndpoints:
    def test_health(self):
        svc = TEEncoderSidecar()
        code, data = svc._handle_health()
        assert code == 200
        assert data["status"] == "ok"

    def test_ready_loading(self):
        svc = TEEncoderSidecar()
        svc._loading = True
        svc._load_start = 0.0
        code, data = svc._handle_ready()
        assert code == 503
        assert data["status"] == "loading"

    def test_ready_error(self):
        svc = TEEncoderSidecar()
        svc._error = "load failed"
        code, data = svc._handle_ready()
        assert code == 503
        assert data["status"] == "error"

    def test_ready_not_started(self):
        svc = TEEncoderSidecar()
        code, data = svc._handle_ready()
        assert code == 503
        assert data["status"] == "not_started"

    def test_ready_ok(self):
        svc = TEEncoderSidecar()
        svc._ready = True
        svc._te_checksum = "abc123"
        svc._load_start = 0.0
        svc._load_end = 1.5
        code, data = svc._handle_ready()
        assert code == 200
        assert data["status"] == "ready"
        assert data["te_checksum"] == "abc123"
        assert "cache_stats" in data


# ---------------------------------------------------------------------------
# Response envelope format
# ---------------------------------------------------------------------------


class TestSidecarEnvelope:
    def test_parse_sidecar_response_valid(self):
        envelope = {"te_checksum": "abc", "embed_cache_hit": False, "cache_key": "def"}
        env_bytes = json.dumps(envelope).encode()
        env_len = len(env_bytes).to_bytes(4, "big")
        payload = b"torch_payload_data"
        raw = env_len + env_bytes + payload

        result = _parse_sidecar_response(raw)
        assert result["te_checksum"] == "abc"
        assert result["embed_cache_hit"] is False
        assert result["payload"] == payload

    def test_parse_sidecar_response_too_short(self):
        with pytest.raises(RuntimeError, match="too short"):
            _parse_sidecar_response(b"\x00\x00")

    def test_parse_sidecar_response_truncated(self):
        envelope = {"te_checksum": "abc"}
        env_bytes = json.dumps(envelope).encode()
        # Truncate: claim 100 bytes but only provide env_bytes
        raw = (100).to_bytes(4, "big") + env_bytes
        with pytest.raises(RuntimeError, match="exceeds response"):
            _parse_sidecar_response(raw)

    def test_encode_cache_hit_returns_envelope(self):
        """When the cache is hit, the response includes the envelope."""
        svc = TEEncoderSidecar()
        svc._ready = True
        svc._te_checksum = "test_checksum"

        # Seed the cache
        from comfyui_superl8.te_sidecar import compute_cache_key

        key = compute_cache_key("cached prompt", "test_checksum")
        svc._cache.put(key, b"cached_payload")

        body = json.dumps({"prompt": "cached prompt"}).encode()
        code, err, payload = svc._handle_encode(body)
        assert code == 200
        assert err is None
        assert payload is not None

        # Parse the envelope
        result = _parse_sidecar_response(payload)
        assert result["te_checksum"] == "test_checksum"
        assert result["embed_cache_hit"] is True
        assert result["payload"] == b"cached_payload"

    def test_encode_busy_slot(self):
        """When the encode slot is held, return 503."""
        svc = TEEncoderSidecar()
        svc._ready = True
        svc._te_checksum = "x"
        svc._clip = MagicMock()
        svc._empty_cond = MagicMock()

        # Acquire the slot to simulate busy
        svc._encode_semaphore.acquire(blocking=False)
        try:
            body = json.dumps({"prompt": "test"}).encode()
            code, err, _ = svc._handle_encode(body)
            assert code == 503
            assert "busy" in err["error"]
        finally:
            svc._encode_semaphore.release()
