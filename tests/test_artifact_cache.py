# SPDX-License-Identifier: Apache-2.0
"""Tests for comfyui_superl8.artifact_cache — the provenance cache-key and CAS glue.

TDD: the stale-hit regression test is the acceptance gate. A cache hit that
ignores model revision silently reuses a latent from a DIFFERENT model.
"""

from __future__ import annotations



from comfyui_superl8.artifact_cache import (
    CASStore,
    compute_conditioning_key,
    compute_latent_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_latent(seed: int = 42, width: int = 512, height: int = 512):
    """Minimal latent dict matching ComfyUI's LATENT type."""
    import torch

    return {"samples": torch.randn(1, 4, height // 8, width // 8, generator=torch.Generator().manual_seed(seed))}


def _fake_conditioning(prompt: str = "hello", seed: int = 42):
    """Minimal conditioning list matching ComfyUI's CONDITIONING type."""
    import torch

    return [[torch.randn(1, 77, 768, generator=torch.Generator().manual_seed(seed)), {"pooled_output": torch.randn(1, 768)}]]


# ---------------------------------------------------------------------------
# Cache key determinism
# ---------------------------------------------------------------------------

class TestCacheKeyDeterminism:
    def test_same_inputs_same_key(self):
        key1 = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=_fake_latent(seed=42),
        )
        key2 = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=_fake_latent(seed=42),
        )
        assert key1 == key2

    def test_different_model_revision_different_key(self):
        key1 = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=_fake_latent(seed=42),
        )
        key2 = compute_latent_key(
            model_revision="rev-2",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=_fake_latent(seed=42),
        )
        assert key1 != key2

    def test_different_seed_different_key(self):
        key1 = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=_fake_latent(seed=42),
        )
        key2 = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=99,
            width=512,
            height=512,
            lora_hashes=[],
            latent=_fake_latent(seed=99),
        )
        assert key1 != key2

    def test_different_lora_different_key(self):
        key1 = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=["abc123"],
            latent=_fake_latent(seed=42),
        )
        key2 = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=["def456"],
            latent=_fake_latent(seed=42),
        )
        assert key1 != key2

    def test_conditioning_key_deterministic(self):
        cond = _fake_conditioning("a cat")
        key1 = compute_conditioning_key(
            model_revision="rev-1",
            prompt="a cat",
            neg_prompt="blurry",
            lora_hashes=[],
            conditioning=cond,
        )
        key2 = compute_conditioning_key(
            model_revision="rev-1",
            prompt="a cat",
            neg_prompt="blurry",
            lora_hashes=[],
            conditioning=cond,
        )
        assert key1 == key2

    def test_conditioning_key_model_rev_matters(self):
        cond = _fake_conditioning("a cat")
        key1 = compute_conditioning_key(
            model_revision="rev-1",
            prompt="a cat",
            neg_prompt="blurry",
            lora_hashes=[],
            conditioning=cond,
        )
        key2 = compute_conditioning_key(
            model_revision="rev-2",
            prompt="a cat",
            neg_prompt="blurry",
            lora_hashes=[],
            conditioning=cond,
        )
        assert key1 != key2


# ---------------------------------------------------------------------------
# CAS store round-trip
# ---------------------------------------------------------------------------

class TestCASStore:
    def test_store_and_load_latent(self, tmp_path):
        store = CASStore(tmp_path)
        latent = _fake_latent(seed=42)
        key = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=latent,
        )
        store.store_latent(key, latent, model_revision="rev-1", provenance={"sampler": "euler"})
        loaded = store.load_latent(key)
        assert loaded is not None
        assert loaded["samples"].shape == latent["samples"].shape
        assert (loaded["samples"] == latent["samples"]).all()

    def test_store_and_load_conditioning(self, tmp_path):
        store = CASStore(tmp_path)
        cond = _fake_conditioning("hello")
        key = compute_conditioning_key(
            model_revision="rev-1",
            prompt="hello",
            neg_prompt="",
            lora_hashes=[],
            conditioning=cond,
        )
        store.store_conditioning(key, cond, model_revision="rev-1", provenance={"prompt": "hello"})
        loaded = store.load_conditioning(key)
        assert loaded is not None
        assert len(loaded) == len(cond)
        assert (loaded[0][0] == cond[0][0]).all()

    def test_load_miss_returns_none(self, tmp_path):
        store = CASStore(tmp_path)
        assert store.load_latent("nonexistent") is None
        assert store.load_conditioning("nonexistent") is None

    def test_content_addressed_dedup(self, tmp_path):
        store = CASStore(tmp_path)
        latent = _fake_latent(seed=42)
        key = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=latent,
        )
        store.store_latent(key, latent, model_revision="rev-1", provenance={})
        store.store_latent(key, latent, model_revision="rev-1", provenance={})
        artifacts = list((tmp_path / "cas" / "artifacts").rglob("*.zst"))
        assert len(artifacts) == 1


# ---------------------------------------------------------------------------
# THE ACCEPTANCE GATE: stale-hit regression
# ---------------------------------------------------------------------------

class TestStaleHitRegression:
    """Changing the model revision MUST miss the cache.

    A cache hit that ignores model revision silently reuses a latent from a
    DIFFERENT model -> stale, wrong output. The attestation rev MUST be in
    every key. This is the one way the architecture is unsound.
    """

    def test_latent_stale_hit_misses(self, tmp_path):
        """Store with rev-1, load with rev-2 -> MUST be a miss."""
        store = CASStore(tmp_path)
        latent = _fake_latent(seed=42)
        key_rev1 = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=latent,
        )
        # Store under rev-1
        store.store_latent(key_rev1, latent, model_revision="rev-1", provenance={})

        # Compute key for rev-2 (same params otherwise)
        key_rev2 = compute_latent_key(
            model_revision="rev-2",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=latent,
        )

        # Key itself is different (model_rev is in the hash)
        assert key_rev1 != key_rev2, (
            "model_revision MUST change the cache key; "
            "same key for different revisions = silent staleness"
        )

        # Loading the rev-2 key returns None (never stored)
        assert store.load_latent(key_rev2) is None

        # Loading the old rev-1 key with rev-2 verification also fails
        loaded = store.load_latent(key_rev1, expect_model_revision="rev-2")
        assert loaded is None, (
            "metadata verification MUST reject stale hits; "
            "loaded a rev-1 latent when rev-2 was expected"
        )

    def test_conditioning_stale_hit_misses(self, tmp_path):
        """Store conditioning with rev-1, load with rev-2 -> MUST be a miss."""
        store = CASStore(tmp_path)
        cond = _fake_conditioning("a cat")
        key_rev1 = compute_conditioning_key(
            model_revision="rev-1",
            prompt="a cat",
            neg_prompt="",
            lora_hashes=[],
            conditioning=cond,
        )
        store.store_conditioning(key_rev1, cond, model_revision="rev-1", provenance={})

        key_rev2 = compute_conditioning_key(
            model_revision="rev-2",
            prompt="a cat",
            neg_prompt="",
            lora_hashes=[],
            conditioning=cond,
        )

        # Different key
        assert key_rev1 != key_rev2

        # New key is a miss
        assert store.load_conditioning(key_rev2) is None

        # Old key with rev-2 verification is also a miss
        loaded = store.load_conditioning(key_rev1, expect_model_revision="rev-2")
        assert loaded is None

    def test_hit_when_rev_matches(self, tmp_path):
        """Store with rev-1, load with rev-1 -> MUST be a hit."""
        store = CASStore(tmp_path)
        latent = _fake_latent(seed=42)
        key = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=[],
            latent=latent,
        )
        store.store_latent(key, latent, model_revision="rev-1", provenance={})
        loaded = store.load_latent(key, expect_model_revision="rev-1")
        assert loaded is not None
        assert (loaded["samples"] == latent["samples"]).all()

    def test_lora_change_also_misses(self, tmp_path):
        """Different LoRAs produce different keys even with same model_rev."""
        store = CASStore(tmp_path)
        latent = _fake_latent(seed=42)
        key_lora_a = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=["lora_a"],
            latent=latent,
        )
        key_lora_b = compute_latent_key(
            model_revision="rev-1",
            sampler="euler",
            scheduler="simple",
            steps=8,
            cfg=1.0,
            seed=42,
            width=512,
            height=512,
            lora_hashes=["lora_b"],
            latent=latent,
        )
        assert key_lora_a != key_lora_b
        store.store_latent(key_lora_a, latent, model_revision="rev-1", provenance={})
        assert store.load_latent(key_lora_b) is None
