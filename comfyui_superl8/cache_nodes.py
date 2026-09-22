# SPDX-License-Identifier: Apache-2.0
"""ComfyUI CAS-backed cache nodes for pipeline artifact caching.

CASSaveConditioning / CASLoadConditioning -- text-embed bundle caching.
CASSaveLatent / CASLoadLatent -- latent / keyframe / shot caching.

Cache key = SHA-256 of (ALL inputs + model revision + sampler params + seed + LoRAs).
A cache hit that ignores model revision silently reuses a latent from a DIFFERENT
model -> stale, wrong output.  The attestation rev MUST be in every key.

These nodes are thin ComfyUI-facing wrappers over comfyui_superl8.artifact_cache.
No CUDA, no model execution, no fork of core ComfyUI types.
"""

from __future__ import annotations

from pathlib import Path

from .artifact_cache import CASStore, compute_conditioning_key, compute_latent_key


def _default_store() -> CASStore:
    """Default CAS store rooted under ComfyUI's output directory."""
    try:
        import folder_paths

        base = Path(folder_paths.get_output_directory()) / "cas_cache"
    except Exception:
        base = Path("cas_cache")
    return CASStore(base)


# ---------------------------------------------------------------------------
# Latent nodes
# ---------------------------------------------------------------------------


class CASSaveLatent:
    """Save a LATENT to the content-addressed cache.

    The cache key is computed from all inputs (including model_revision).
    Returns the key string so downstream nodes can pass it to CASLoadLatent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model_revision": ("STRING", {"default": "", "multiline": False}),
                "sampler_name": ("STRING", {"default": "euler", "multiline": False}),
                "scheduler": ("STRING", {"default": "simple", "multiline": False}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
            },
            "optional": {
                "loras_json": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("cache_key", "latent")
    FUNCTION = "save"
    CATEGORY = "fni8/cache"
    TITLE = "CAS Save Latent"

    def save(
        self,
        latent,
        model_revision,
        sampler_name,
        scheduler,
        steps,
        cfg,
        seed,
        width,
        height,
        loras_json="[]",
    ):
        import json

        import torch

        lora_hashes = json.loads(loras_json) if isinstance(loras_json, str) else []
        samples = latent.get("samples")
        if samples is None or not isinstance(samples, torch.Tensor):
            raise ValueError("latent must contain a 'samples' tensor")

        key = compute_latent_key(
            model_revision=model_revision,
            sampler=sampler_name,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            seed=seed,
            width=width,
            height=height,
            lora_hashes=lora_hashes,
            latent={"samples": samples},
        )
        store = _default_store()
        store.store_latent(
            key,
            {"samples": samples},
            model_revision=model_revision,
            provenance={
                "sampler": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "width": width,
                "height": height,
                "lora_hashes": lora_hashes,
            },
        )
        return (key, latent)


class CASLoadLatent:
    """Load a LATENT from the content-addressed cache.

    Takes a cache_key (from CASSaveLatent) and model_revision.
    Returns the cached latent on hit, or an empty latent on miss.
    Also returns a status string ("hit" or "miss") for conditional routing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_key": ("STRING", {"default": "", "multiline": False}),
                "model_revision": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "load"
    CATEGORY = "fni8/cache"
    TITLE = "CAS Load Latent"

    def load(self, cache_key, model_revision):
        import torch

        if not cache_key:
            return ({"samples": torch.zeros(1, 4, 1, 1)}, "miss")
        store = _default_store()
        loaded = store.load_latent(cache_key, expect_model_revision=model_revision)
        if loaded is None:
            return ({"samples": torch.zeros(1, 4, 1, 1)}, "miss")
        return ({"samples": loaded["samples"]}, "hit")


# ---------------------------------------------------------------------------
# Conditioning nodes
# ---------------------------------------------------------------------------


class CASSaveConditioning:
    """Save a CONDITIONING bundle to the content-addressed cache.

    The cache key is computed from all inputs (including model_revision).
    Returns the key string so downstream nodes can pass it to CASLoadConditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "model_revision": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "neg_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "loras_json": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "CONDITIONING")
    RETURN_NAMES = ("cache_key", "conditioning")
    FUNCTION = "save"
    CATEGORY = "fni8/cache"
    TITLE = "CAS Save Conditioning"

    def save(self, conditioning, model_revision, prompt, neg_prompt, loras_json="[]"):
        import json

        lora_hashes = json.loads(loras_json) if isinstance(loras_json, str) else []
        key = compute_conditioning_key(
            model_revision=model_revision,
            prompt=prompt,
            neg_prompt=neg_prompt,
            lora_hashes=lora_hashes,
            conditioning=conditioning,
        )
        store = _default_store()
        store.store_conditioning(
            key,
            conditioning,
            model_revision=model_revision,
            provenance={
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "lora_hashes": lora_hashes,
            },
        )
        return (key, conditioning)


class CASLoadConditioning:
    """Load a CONDITIONING bundle from the content-addressed cache.

    Takes a cache_key (from CASSaveConditioning) and model_revision.
    Returns the cached conditioning on hit, or empty conditioning on miss.
    Also returns a status string ("hit" or "miss") for conditional routing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_key": ("STRING", {"default": "", "multiline": False}),
                "model_revision": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "status")
    FUNCTION = "load"
    CATEGORY = "fni8/cache"
    TITLE = "CAS Load Conditioning"

    def load(self, cache_key, model_revision):
        import torch

        if not cache_key:
            empty = [[torch.zeros(1, 77, 768), {"pooled_output": torch.zeros(1, 768)}]]
            return (empty, "miss")
        store = _default_store()
        loaded = store.load_conditioning(cache_key, expect_model_revision=model_revision)
        if loaded is None:
            empty = [[torch.zeros(1, 77, 768), {"pooled_output": torch.zeros(1, 768)}]]
            return (empty, "miss")
        return (loaded, "hit")


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "CASSaveLatent": CASSaveLatent,
    "CASLoadLatent": CASLoadLatent,
    "CASSaveConditioning": CASSaveConditioning,
    "CASLoadConditioning": CASLoadConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CASSaveLatent": "CAS Save Latent",
    "CASLoadLatent": "CAS Load Latent",
    "CASSaveConditioning": "CAS Save Conditioning",
    "CASLoadConditioning": "CAS Load Conditioning",
}
