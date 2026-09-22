# SPDX-License-Identifier: Apache-2.0
"""Content-addressed artifact cache for ComfyUI pipeline stages.

Cache key = SHA-256 of (ALL inputs + model revision + sampler params + seed + LoRAs).
A cache hit that ignores model revision silently reuses a latent from a DIFFERENT
model -> stale, wrong output.  The attestation rev MUST be in every key.

Storage: filesystem CAS with zstd-compressed safetensors blobs.
No ComfyUI dependency -- pure Python + torch + safetensors + zstd.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as _sf_load, save as _sf_save


# ---------------------------------------------------------------------------
# Zstd compression helpers
# ---------------------------------------------------------------------------

def _zstd_compress(data: bytes, level: int = 3) -> bytes:
    """Compress bytes with zstd.  Falls back to subprocess, then raw."""
    try:
        import zstandard as zstd

        return zstd.ZstdCompressor(level=level).compress(data)
    except ImportError:
        pass
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_in:
        tmp_in.write(data)
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path + ".zst"
    try:
        subprocess.run(
            ["zstd", "-f", "-o", tmp_out_path, tmp_in_path],
            check=True,
            capture_output=True,
        )
        return Path(tmp_out_path).read_bytes()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return data
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


def _zstd_decompress(data: bytes) -> bytes:
    """Decompress zstd bytes.  Falls back to subprocess, then raw (uncompressed)."""
    try:
        import zstandard as zstd

        return zstd.ZstdDecompressor().decompress(data)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp_in:
        tmp_in.write(data)
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path + ".out"
    try:
        subprocess.run(
            ["zstd", "-d", "-f", "-o", tmp_out_path, tmp_in_path],
            check=True,
            capture_output=True,
        )
        return Path(tmp_out_path).read_bytes()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return data
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tensor content hashing
# ---------------------------------------------------------------------------

def _tensor_content_hash(tensors: dict[str, torch.Tensor]) -> str:
    """SHA-256 of the raw bytes of all tensors (sorted by key for determinism)."""
    h = hashlib.sha256()
    for key in sorted(tensors):
        h.update(key.encode())
        h.update(tensors[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------

def compute_latent_key(
    *,
    model_revision: str,
    sampler: str,
    scheduler: str,
    steps: int,
    cfg: float,
    seed: int,
    width: int,
    height: int,
    lora_hashes: list[str],
    latent: dict[str, torch.Tensor],
) -> str:
    """Compute a deterministic cache key for a latent artifact.

    The key is a SHA-256 hex digest of a canonical JSON object containing ALL
    inputs that affect the output, including model revision.  Changing any
    input produces a different key.
    """
    payload = {
        "artifact_type": "shot-latent/v1",
        "model_revision": model_revision,
        "sampler": sampler,
        "scheduler": scheduler,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "width": width,
        "height": height,
        "lora_hashes": sorted(lora_hashes),
        "content_hash": _tensor_content_hash(latent),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_conditioning_key(
    *,
    model_revision: str,
    prompt: str,
    neg_prompt: str,
    lora_hashes: list[str],
    conditioning: list,
) -> str:
    """Compute a deterministic cache key for a conditioning artifact.

    Same contract as compute_latent_key: model revision is load-bearing.
    """
    cond_tensors: dict[str, torch.Tensor] = {}
    for i, item in enumerate(conditioning):
        if isinstance(item, (list, tuple)):
            for j, t in enumerate(item):
                if isinstance(t, torch.Tensor):
                    cond_tensors[f"{i}_{j}"] = t
                elif isinstance(t, dict):
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            cond_tensors[f"{i}_{j}_{k}"] = v
        elif isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    cond_tensors[f"{i}_{k}"] = v

    payload = {
        "artifact_type": "text-embed/v1",
        "model_revision": model_revision,
        "prompt": prompt,
        "neg_prompt": neg_prompt,
        "lora_hashes": sorted(lora_hashes),
        "content_hash": _tensor_content_hash(cond_tensors) if cond_tensors else "",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# CAS (content-addressed store) -- filesystem backend
# ---------------------------------------------------------------------------

@dataclass
class CASMetadata:
    """Provenance record stored alongside each artifact."""

    key: str
    model_revision: str
    artifact_type: str
    provenance: dict[str, Any] = field(default_factory=dict)


class CASStore:
    """Filesystem-backed content-addressed artifact store.

    Layout::

        base_dir/
          cas/
            artifacts/
              ab/
                abcdef...safetensors.zst   (compressed blob)
            metadata/
              abcdef...json               (provenance record)

    Blobs are deduplicated by content hash (the key IS the hash).
    """

    def __init__(self, base_dir: Path | str):
        self._base = Path(base_dir)
        self._artifacts_dir = self._base / "cas" / "artifacts"
        self._metadata_dir = self._base / "cas" / "metadata"

    def _artifact_path(self, key: str) -> Path:
        return self._artifacts_dir / key[:2] / f"{key}.safetensors.zst"

    def _metadata_path(self, key: str) -> Path:
        return self._metadata_dir / f"{key}.json"

    def _write_metadata(self, key: str, meta: CASMetadata) -> None:
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path(key).write_text(
            json.dumps(
                {
                    "key": meta.key,
                    "model_revision": meta.model_revision,
                    "artifact_type": meta.artifact_type,
                    "provenance": meta.provenance,
                },
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
        )

    def _read_metadata(self, key: str) -> CASMetadata | None:
        path = self._metadata_path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return CASMetadata(
            key=data["key"],
            model_revision=data["model_revision"],
            artifact_type=data["artifact_type"],
            provenance=data.get("provenance", {}),
        )

    def store_latent(
        self,
        key: str,
        latent: dict[str, torch.Tensor],
        *,
        model_revision: str,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Store a latent dict to CAS, compressed as zstd(safetensors)."""
        path = self._artifact_path(key)
        if path.exists():
            return  # already content-addressed; dedup by construction
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = _sf_save(
            {k: v.detach().cpu().contiguous() for k, v in latent.items()},
        )
        compressed = _zstd_compress(buf)
        path.write_bytes(compressed)
        self._write_metadata(
            key,
            CASMetadata(
                key=key,
                model_revision=model_revision,
                artifact_type="shot-latent/v1",
                provenance=provenance or {},
            ),
        )

    def load_latent(
        self,
        key: str,
        *,
        expect_model_revision: str | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Load a latent from CAS.  Returns None on miss.

        If *expect_model_revision* is given, the stored metadata must match;
        a mismatch is treated as a miss (staleness guard).
        """
        path = self._artifact_path(key)
        if not path.exists():
            return None
        if expect_model_revision is not None:
            meta = self._read_metadata(key)
            if meta is None or meta.model_revision != expect_model_revision:
                return None
        compressed = path.read_bytes()
        raw = _zstd_decompress(compressed)
        return _sf_load(raw)

    def store_conditioning(
        self,
        key: str,
        conditioning: list,
        *,
        model_revision: str,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Store a conditioning list to CAS, compressed as zstd(safetensors)."""
        path = self._artifact_path(key)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        for i, item in enumerate(conditioning):
            if isinstance(item, (list, tuple)):
                for j, t in enumerate(item):
                    if isinstance(t, torch.Tensor):
                        tensors[f"{i}_{j}"] = t.detach().cpu().contiguous()
                    elif isinstance(t, dict):
                        for k, v in t.items():
                            if isinstance(v, torch.Tensor):
                                tensors[f"{i}_{j}_{k}"] = v.detach().cpu().contiguous()
            elif isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        tensors[f"{i}_{k}"] = v.detach().cpu().contiguous()
        buf = _sf_save(tensors)
        compressed = _zstd_compress(buf)
        path.write_bytes(compressed)
        self._write_metadata(
            key,
            CASMetadata(
                key=key,
                model_revision=model_revision,
                artifact_type="text-embed/v1",
                provenance=provenance or {},
            ),
        )

    def load_conditioning(
        self,
        key: str,
        *,
        expect_model_revision: str | None = None,
    ) -> list | None:
        """Load conditioning from CAS.  Returns None on miss.

        If *expect_model_revision* is given, the stored metadata must match.
        """
        path = self._artifact_path(key)
        if not path.exists():
            return None
        if expect_model_revision is not None:
            meta = self._read_metadata(key)
            if meta is None or meta.model_revision != expect_model_revision:
                return None
        compressed = path.read_bytes()
        raw = _zstd_decompress(compressed)
        tensors = _sf_load(raw)
        result: list = []
        idx = 0
        while True:
            prefix = f"{idx}_"
            if not any(k.startswith(prefix) for k in tensors):
                break
            inner: list = []
            j = 0
            while True:
                tensor_key = f"{idx}_{j}"
                dict_prefix = f"{idx}_{j}_"
                if tensor_key in tensors:
                    inner.append(tensors[tensor_key])
                    j += 1
                elif any(k.startswith(dict_prefix) for k in tensors):
                    d: dict[str, torch.Tensor] = {}
                    for k in sorted(tensors):
                        if k.startswith(dict_prefix):
                            d[k[len(dict_prefix) :]] = tensors[k]
                    inner.append(d)
                    j += 1
                else:
                    break
            result.append(inner)
            idx += 1
        return result
