# SPDX-License-Identifier: Apache-2.0
"""Z-Image resident HTTP service (issue #181).

A bounded, single-slot resident HTTP daemon around the proven full Z-Image
.fni8 int8 dp4a pipeline.  Per-request prompt encoding (no pre-encoded
DEFAULT_PROMPT); non-square latent support (width and height independent),
shared with the proven square benchmark through ``zimage_profile.denoise``;
bounded cooperative timeout with a real,
provable overrun ceiling (see below); externally-verified fail-closed GPU
election (this process only checks, it never elects -- see
``scripts/launch_resident_service.py``); component-scoped model release that
never evicts the resident DiT; strict request-field rejection; sequential
model lifecycle (TE freed after encoding, VAE freed after decode); bounded
typed JSON response envelope (no unbounded HTTP headers); staged timing
(TE/DiT/VAE/PNG); and comprehensive input validation.

Endpoints:
  GET  /       — service description
  GET  /health — liveness (always 200 if process is up)
  GET  /ready  — readiness (200 when model is resident, 503 otherwise)
  POST /generate — run one generation (bounded to one at a time)

Architecture:
  - stdlib http.server + threading (no external deps beyond torch/PIL)
  - Single generation slot via threading.Semaphore(1)
  - Request/response bounded (_MAX_REQUEST_BYTES, _MAX_DIM, _MAX_STEPS)
  - Deterministic machine-readable provenance on every response
  - Sequential model lifecycle: TE->free, DiT resident, VAE->free after decode

Timeout semantics (honest bound, not an instant kill):
  Abort is checked (a) between the TE/DiT/VAE/PNG pipeline stages, (b) after
  every DiT sampling step (comfy.sample.sample's per-step callback), and
  (c) before every VAE tile decode.  A timeout can therefore only fire once
  the currently in-flight unit of work -- one full TE forward pass, one
  sampling step, or one VAE tile -- has actually finished; the maximum
  overrun past `timeout_s` is bounded by the slowest of those three, NOT by
  the whole remaining stage.  `_do_generate` only reports `aborted` (and only
  after `_check_abort` has actually raised out of the sampler/tiler, i.e.
  after compute for that unit of work has genuinely stopped) -- it never
  reports cancellation before compute stops.  There is no hard kill (no
  subprocess/thread termination): a pathological single step or tile that
  never returns is not interrupted.  If that bound is ever unacceptable for a
  given deployment, the caller must add a real process-level boundary (e.g.
  running this service under a supervisor that SIGKILLs on `timeout_s +
  measured_max_step_time`), not rely on this cooperative check alone.

Reuses proven repository loaders and harnesses:
  - comfyui_superl8.zimage_profile: canonical files, loader, denoise, and gates
    vae_decode_tiled, save_png, _int8_attn_engaged
  - comfyui_superl8.vae_tiled: tiled_vae_decode (extended with abort_check)
  - comfyui_superl8.memory: peak_hbm_monitor, format_hbm_report
  - comfyui_superl8.provenance: build_provenance, Provenance

GPU binding is performed OUTSIDE this process: launch via
``scripts/launch_resident_service.py``, never ``python -m
comfyui_superl8.resident_service`` directly (see ``_verify_gpu_election``).

Do not touch the live-server GPU or deploy anything (AGENTS.md).
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import logging
import os
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .provenance import Provenance, build_provenance, SERVICE_VERSION
from . import zimage_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits (bounded request/response sizes)
# ---------------------------------------------------------------------------

_MAX_REQUEST_BYTES = 1_048_576  # 1 MB JSON body
_MIN_DIM = 64
_MAX_DIM = 2048
_MIN_STEPS = 1
_MAX_STEPS = 50
_MAX_SEED = 2**31 - 1
_MAX_PROMPT_LEN = 10_000
_DEFAULT_TIMEOUT_S = 300.0  # 5 minutes
_MAX_TIMEOUT_S = 600.0  # 10 minutes hard ceiling
_MAX_TILE_SIZE = 2048
_MIN_TILE_SIZE = 64
_MAX_OVERLAP = 512
_MIN_OVERLAP = 0
_MAX_PNG_BYTES = 50_000_000  # 50 MiB response ceiling
_MAX_GENERATION_COUNTER = 2**63 - 1

# Pipeline constants (per #181: exact 8-NFE, CFG1, res_multistep/simple,
# shift3, explicit seed, empty negative, no cache).
# These are NOT user-configurable; any drift is rejected.
_LOCKED_NFE = zimage_profile.STEPS
_LOCKED_CFG = zimage_profile.CFG
_LOCKED_SAMPLER = zimage_profile.SAMPLER
_LOCKED_SCHEDULER = zimage_profile.SCHEDULER
_EXPECTED_MODEL_SAMPLING_SHIFT = zimage_profile.FLOW_SHIFT


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    """Incoming generation request (JSON body).  Frozen to prevent mutation."""

    width: int
    height: int
    prompt: str
    seed: int
    timeout_s: float = _DEFAULT_TIMEOUT_S
    tile_size: int = 512
    overlap: int = 64

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationRequest:
        """Parse from a JSON dict.  Rejects unknown keys and forbidden
        pipeline-parameter drift (steps, cfg, scheduler, etc.)."""
        if not isinstance(d, dict):
            raise TypeError("request body must be a JSON object")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d.keys()) - known
        if unknown:
            raise ValueError(f"unknown request fields: {', '.join(sorted(unknown))}")
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class GenerationResponse:
    """Outgoing generation response."""

    image_png: bytes = b""
    provenance: Provenance = field(default_factory=Provenance)
    timing: dict[str, float] = field(default_factory=dict)
    hbm: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "timing": self.timing,
            "hbm": self.hbm,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Only these request keys are permitted.  steps, cfg, scheduler, and other
# pipeline parameters are NOT permitted — the #181 semantics are exact and
# non-negotiable.
_PERMITTED_KEYS = frozenset(
    {
        "width",
        "height",
        "prompt",
        "seed",
        "timeout_s",
        "tile_size",
        "overlap",
    }
)


def _validate_request(req: GenerationRequest) -> None:
    """Validate a generation request; raises ValueError on any violation."""
    if not isinstance(req.prompt, str) or not req.prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(req.prompt) > _MAX_PROMPT_LEN:
        raise ValueError(f"prompt exceeds {_MAX_PROMPT_LEN} characters")
    if not isinstance(req.width, int) or not isinstance(req.height, int):
        raise ValueError("width and height must be integers")
    if req.width < _MIN_DIM or req.width > _MAX_DIM:
        raise ValueError(f"width must be {_MIN_DIM}..{_MAX_DIM}, got {req.width}")
    if req.height < _MIN_DIM or req.height > _MAX_DIM:
        raise ValueError(f"height must be {_MIN_DIM}..{_MAX_DIM}, got {req.height}")
    zimage_profile.validate_geometry(req.width, req.height)
    if not isinstance(req.seed, int) or req.seed < 0 or req.seed > _MAX_SEED:
        raise ValueError(f"seed must be 0..{_MAX_SEED}, got {req.seed}")
    if (
        not isinstance(req.timeout_s, (int, float))
        or req.timeout_s <= 0
        or req.timeout_s > _MAX_TIMEOUT_S
    ):
        raise ValueError(f"timeout_s must be 0..{_MAX_TIMEOUT_S}, got {req.timeout_s}")
    if (
        not isinstance(req.tile_size, int)
        or req.tile_size < _MIN_TILE_SIZE
        or req.tile_size > _MAX_TILE_SIZE
    ):
        raise ValueError(
            f"tile_size must be {_MIN_TILE_SIZE}..{_MAX_TILE_SIZE}, got {req.tile_size}"
        )
    if not isinstance(req.overlap, int) or req.overlap < _MIN_OVERLAP or req.overlap > _MAX_OVERLAP:
        raise ValueError(f"overlap must be {_MIN_OVERLAP}..{_MAX_OVERLAP}, got {req.overlap}")
    if req.overlap >= req.tile_size:
        raise ValueError(f"overlap ({req.overlap}) must be < tile_size ({req.tile_size})")


def _normalize_uuid(u: str) -> str:
    """Normalize a GPU UUID for comparison: nvidia-smi reports `GPU-xxxx...`,
    `CUDA_VISIBLE_DEVICES` accepts that same form, and `torch.cuda
    .get_device_properties().uuid` stringifies without the `GPU-` prefix.  Strip
    the prefix and lowercase so all three compare equal for the same card."""
    u = u.strip()
    if u.upper().startswith("GPU-"):
        u = u[4:]
    return u.lower()


def _move_tensors_to_cuda(obj, device):
    """Recursively move every torch.Tensor in a nested structure to device."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cuda(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_tensors_to_cuda(item, device) for item in obj]
        return type(obj)(moved)
    return obj


from .te_sidecar import _parse_sidecar_response  # noqa: E402


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ZImageResidentService:
    """Resident HTTP daemon for Z-Image generation.

    Sequential model lifecycle per request:
      1. Load TE -> encode prompt -> free TE  (TE is ~8 GiB, freed to make room)
      2. Denoise with DiT (resident in VRAM)
      3. Load VAE -> tiled decode -> free VAE  (VAE is ~1-2 GiB)

    On a 16 GiB card: TE (~8 GiB) + DiT (~4-5 GiB) = ~12-13 GiB during
    encoding; TE freed; DiT + VAE (~1-2 GiB) = ~5-7 GiB during decode.
    No simultaneous TE+DiT+VAE residency — the documented 16 GiB envelope.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8410,
        weights_dir: str | None = None,
    ):
        self.host = host
        self.port = port

        # Production is fixed to the checksummed NVMe model root.
        if weights_dir is None:
            weights_dir = os.environ.get(
                "FNI8_WEIGHTS_DIR", zimage_profile.CANONICAL_MODEL_ROOT
            )
        canonical = os.path.realpath(zimage_profile.CANONICAL_MODEL_ROOT)
        if os.path.realpath(weights_dir) != canonical:
            raise ValueError(f"weights_dir must be canonical NVMe root {canonical}")
        self.weights_dir = canonical

        # State
        self._ready = False
        self._loading = False
        self._error: str | None = None
        self._load_start: float = 0.0
        self._load_end: float = 0.0

        # Pipeline objects (populated by _load_pipeline)
        self._model = None
        self._vae = None
        self._in_channels: int = 4
        self._te_path: str | None = None
        self._vae_path: str | None = None
        self._dit_path: str | None = None
        self._int8_inventory: dict[str, object] = {}
        self._runtime_int8: dict[str, object] = {}
        self._model_sampling_shift: float | None = None
        self._clip_model: Any = None  # TE for per-request encoding

        # GPU election is verified (not performed) by this process; the UUID is
        # populated by _verify_gpu_election() and the lock itself is held
        # externally, by scripts/launch_resident_service.py, for this process's
        # entire lifetime.
        self._gpu_uuid: str | None = None

        # Single generation slot
        self._gen_semaphore = threading.Semaphore(1)
        self._active_generations = 0
        self._total_generations = 0

        # Cooperative abort (per-request)
        self._abort = threading.Event()
        self._done = threading.Event()
        self._state_lock = threading.Lock()

        # Model file paths for provenance
        self._dit_path: str | None = None
        self._vae_path: str | None = None
        self._te_path: str | None = None

    def _check_ready(self) -> bool:
        return self._ready and not self._loading and self._error is None

    # --- GPU election verification (fail-closed) ----------------------------
    #
    # Root-review blocker #1: this process cannot safely elect its own GPU.  By
    # the time any method on this class runs, `comfyui_superl8/__init__.py` has
    # already been imported (Python imports the parent package before this
    # submodule) and has already pulled in `.nodes` / `.gguf_ops` / etc., which
    # import `comfy`/`torch` and can lazily initialize a CUDA context on
    # whatever the driver enumerates as device 0 — the *unfiltered* fleet.
    # Setting an env var afterward, from inside this same process, binds
    # nothing.
    #
    # The actual election + `CUDA_VISIBLE_DEVICES` binding now happens
    # *externally*, before this process (or its torch/comfy imports) exists at
    # all: `scripts/launch_resident_service.py` is stdlib-only (no torch, no
    # comfy, no `comfyui_superl8` package import), probes the fleet, elects and
    # locks a card by UUID, sets `CUDA_VISIBLE_DEVICES` to that UUID, and only
    # then spawns `python -m comfyui_superl8.resident_service` as a *separate
    # child process* that inherits the restricted environment. This service
    # only ever verifies — and refuses to run without proof — that binding
    # happened; it never attempts to perform it itself.

    def _verify_gpu_election(self) -> None:
        """Fail-closed: refuse to load the pipeline unless a real, external,
        pre-Torch GPU election already bound this process to exactly one card.

        Must run before any torch/comfy import in `_load_pipeline` touches CUDA.
        """
        if os.environ.get("FNI8_GPU_ELECTED") != "1":
            raise RuntimeError(
                "GPU was not elected before this process started: launch this "
                "service via `python3 scripts/launch_resident_service.py -- "
                "--weights-dir ...`, which elects and locks a card by UUID and "
                "sets CUDA_VISIBLE_DEVICES before comfyui_superl8.resident_service "
                "(and its torch/comfy imports) is even loaded. Refusing to start "
                "without a proven external device binding — setting FNI8_GPU "
                "after process startup does not bind an already-importing Torch "
                "process to a card."
            )
        expected_uuid = os.environ.get("FNI8_GPU_UUID", "").strip()
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not expected_uuid or not cvd:
            raise RuntimeError(
                "FNI8_GPU_ELECTED=1 but FNI8_GPU_UUID/CUDA_VISIBLE_DEVICES are "
                "unset; refusing to start (see scripts/launch_resident_service.py)"
            )
        if _normalize_uuid(cvd) != _normalize_uuid(expected_uuid):
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES ({cvd!r}) does not match the elected UUID "
                f"({expected_uuid!r}); refusing to start with a mismatched binding"
            )

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this process")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "expected exactly one CUDA device visible after election, got "
                f"{torch.cuda.device_count()} — CUDA_VISIBLE_DEVICES was not honored"
            )
        bound_uuid = _normalize_uuid(str(torch.cuda.get_device_properties(0).uuid))
        if bound_uuid != _normalize_uuid(expected_uuid):
            raise RuntimeError(
                f"bound CUDA device UUID ({bound_uuid}) does not match the "
                f"elected UUID ({_normalize_uuid(expected_uuid)}) — this process "
                "saw the wrong card"
            )
        self._gpu_uuid = expected_uuid
        logger.info("verified external GPU election: bound to %s", expected_uuid)

    # --- Pipeline lifecycle (sequential eviction) --------------------------

    def _load_pipeline(self) -> None:
        """Load the Z-Image pipeline with sequential model lifecycle.

        Runs in a background thread at startup.  Loads DiT (resident) and
        prepares TE/VAE paths for per-request loading/eviction.
        """
        self._loading = True
        self._load_start = time.monotonic()
        try:
            # Verify FIRST, before any import in this method (or its transitive
            # imports) can touch torch/CUDA. `comfyui_superl8/__init__.py` has already
            # been imported by this point (parent-package-before-submodule), so the
            # earliest possible point left to check is right here — but the actual
            # binding must already have happened externally (see
            # scripts/launch_resident_service.py); this can only verify, not elect.
            self._verify_gpu_election()

            from .gate import is_sm70

            if not is_sm70():
                raise RuntimeError("requires a Volta/CMP (sm_70) GPU")

            files = zimage_profile.resolve_files(self.weights_dir, verify=True)
            self._dit_path = files.dit
            self._vae_path = files.vae
            self._te_path = files.text_encoder

            # Step 2: Load int8 DiT (stays resident for the service lifetime)
            logger.info("Loading int8 Z-Image DiT...")
            t_dit = time.monotonic()
            self._model = zimage_profile.load_int8_model(self.weights_dir)
            self._model_sampling_shift = zimage_profile.assert_model_sampling_shift(
                self._model
            )
            import comfy.model_management as mm

            mm.load_models_gpu([self._model], force_full_load=True)
            self._int8_inventory = zimage_profile.move_superl8_tensors_(
                self._model.model.diffusion_model, mm.get_torch_device()
            )
            self._in_channels = self._model.model.diffusion_model.in_channels
            logger.info(
                "DiT loaded in %.1f s (in_channels=%d)", time.monotonic() - t_dit, self._in_channels
            )

            self._ready = True
            self._loading = False
            self._load_end = time.monotonic()
            logger.info(
                "Pipeline loaded in %.1f s (in_channels=%d)",
                self._load_end - self._load_start,
                self._in_channels,
            )
        except Exception as e:
            self._loading = False
            self._error = str(e)
            self._load_end = time.monotonic()
            logger.error("Pipeline load failed: %s", e)
            raise

    # --- Per-request TE (loaded, used, freed) ------------------------------

    def _load_te(self) -> Any:
        """Load the Qwen3-4B text encoder.  Returns clip model.

        TE is loaded per-request and freed after encoding to stay within
        the 16 GiB sequential eviction envelope (~8 GiB TE + ~5 GiB DiT).
        """
        import comfy.sd
        import comfy.utils

        if self._te_path is None:
            raise RuntimeError("canonical text encoder was not resolved")
        state = comfy.utils.load_torch_file(self._te_path, safe_load=True)
        clip = comfy.sd.load_text_encoder_state_dicts([state], clip_type=None)
        return clip

    def _free_non_resident_models(self) -> None:
        """Evict every ComfyUI-managed model EXCEPT the resident DiT.

        Root-review blocker #2: `comfy.model_management.unload_all_models()` is
        `free_memory(1e30, device)` with an EMPTY `keep_loaded` — it evicts every
        entry in `current_loaded_models` indiscriminately, including the DiT this
        service documents as staying resident for its whole lifetime, the moment
        ComfyUI's own sampling path has registered it there. `LoadedModel`
        equality (and `free_memory`'s `keep_loaded` membership test) is Python
        object identity on the underlying model, so filtering by `is self._model`
        is exact -- not a heuristic -- and reuses ComfyUI's own eviction/refcount
        logic for everything else (TE, VAE, or any stray registration).
        """
        import comfy.model_management as mm

        device = mm.get_torch_device()
        keep = [
            loaded
            for loaded in mm.current_loaded_models
            if self._model is not None and loaded.model is self._model
        ]
        mm.free_memory(1e30, device, keep_loaded=keep)

    def _refresh_int8_residency(self) -> dict[str, object]:
        """Re-read custom code and scale placement; never trust a startup snapshot."""
        if self._model is None:
            raise RuntimeError("resident Z-Image DiT is not loaded")
        self._int8_inventory = zimage_profile.assert_int8_inventory(
            self._model.model.diffusion_model, require_cuda=True
        )
        return self._int8_inventory

    def _free_te(self, clip: Any) -> None:
        """Free the text encoder from VRAM.  Never evicts the resident DiT."""
        try:
            self._free_non_resident_models()
        except Exception:
            logger.warning("TE eviction failed; VRAM may not be reclaimed", exc_info=True)
        del clip
        import torch

        torch.cuda.empty_cache()

    def _load_vae(self) -> Any:
        """Load the VAE.  Returns comfy.sd.VAE.

        VAE is loaded per-request and freed after decode.
        """
        import comfy.sd
        import comfy.utils

        if self._vae_path is None:
            raise RuntimeError("canonical VAE was not resolved")
        return comfy.sd.VAE(sd=comfy.utils.load_torch_file(self._vae_path, safe_load=True))

    def _free_vae(self, vae: Any) -> None:
        """Free the VAE from VRAM.  Never evicts the resident DiT."""
        try:
            self._free_non_resident_models()
        except Exception:
            logger.warning("VAE eviction failed; VRAM may not be reclaimed", exc_info=True)
        del vae
        import torch

        torch.cuda.empty_cache()

    # --- Prompt encoding (per-request, actual prompt, no pre-encoding) -----

    def _encode_prompt(self, prompt: str) -> tuple[Any, Any]:
        """Encode the user's exact prompt.

        When FNI8_TE_ENDPOINT is set, fetches conditioning from the TE
        sidecar over HTTP (#200 Phase 2).  Otherwise loads the TE locally
        (existing path, ~85 s per request on 16 GiB cards).

        Returns (cond, empty, te_source, embed_cache_hit) where:
          - cond/empty: conditioning tensors on CUDA
          - te_source: 'sidecar' or 'local'
          - embed_cache_hit: bool (True if sidecar served from embedding cache)
        """
        te_endpoint = os.environ.get("FNI8_TE_ENDPOINT")
        if te_endpoint:
            return self._encode_prompt_via_sidecar(prompt, te_endpoint)

        # Local path: load TE, encode, free TE (existing #181 behavior)
        clip = self._load_te()
        try:
            tokens = clip.tokenize(prompt)
            cond = clip.encode_from_tokens_scheduled(tokens)
            empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
            return cond, empty, "local", False
        finally:
            self._free_te(clip)

    def _encode_prompt_via_sidecar(
        self, prompt: str, endpoint: str
    ) -> tuple[Any, Any, str, bool]:
        """Fetch conditioning from the TE sidecar over HTTP.

        POSTs the prompt to ``endpoint/encode``, deserializes the torch.save'd
        response, validates the te_checksum against our local pin, and moves
        tensors to this service's CUDA device.  Fail closed on any sidecar
        error -- raise, do not silently fall back to the local path.
        """
        import torch

        url = f"{endpoint.rstrip('/')}/encode"
        payload = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        logger.info("fetching TE from sidecar at %s ...", url)
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 200:
                    body = resp.read()
                    try:
                        err = json.loads(body)
                    except Exception:
                        err = {"detail": body.decode("utf-8", errors="replace")}
                    raise RuntimeError(
                        f"sidecar encode failed ({resp.status}): {err}"
                    )
                raw = resp.read()
        except urllib.error.URLError as e:
            raise RuntimeError(f"sidecar unreachable at {url}: {e}") from e
        elapsed = time.monotonic() - t0

        # Parse envelope: 4-byte length + JSON + torch.save payload
        envelope = _parse_sidecar_response(raw)
        te_checksum = envelope.get("te_checksum", "")
        embed_cache_hit = envelope.get("embed_cache_hit", False)
        torch_payload = envelope["payload"]

        # Verify te_checksum against our local pin (fail closed).
        # An empty/missing te_checksum from the sidecar is rejected -- the
        # sidecar MUST echo the checksum it was loaded with.
        expected_checksum = zimage_profile.TEXT_ENCODER_SHA256
        if te_checksum != expected_checksum:
            raise RuntimeError(
                f"sidecar te_checksum mismatch: got {te_checksum[:16] if te_checksum else '(empty)'}..., "
                f"expected {expected_checksum[:16]}... (model drift?)"
            )

        logger.info(
            "sidecar TE fetched in %.2f s (%d bytes, cache_hit=%s)",
            elapsed,
            len(torch_payload),
            embed_cache_hit,
        )

        # Deserialize -- weights_only=False is acceptable here because we
        # trust the sidecar (same host, same deployment).  The payload was
        # produced by torch.save of {"cond", "empty"} with all tensors on CPU.
        buf = io.BytesIO(torch_payload)
        data = torch.load(buf, weights_only=False, map_location="cpu")
        cond = _move_tensors_to_cuda(data["cond"], torch.device("cuda"))
        empty = _move_tensors_to_cuda(data["empty"], torch.device("cuda"))
        return cond, empty, "sidecar", bool(embed_cache_hit)

    # --- Denoise (non-square, exact #181 semantics) ------------------------

    def _denoise(
        self,
        model: Any,
        cond: Any,
        empty: Any,
        latent_h: int,
        latent_w: int,
        steps: int,
        seed: int,
    ) -> Any:
        """Run denoise with explicit width x height latent (non-square capable).

        Delegates to ``zimage_profile.denoise`` -- the SAME production function
        used by the int8-vs-fp benchmark -- rather than reimplementing the
        sampling call separately (root-review
        blocker #4: a second, unreconciled rectangular path has no relationship
        to the evidence gathered for the square case). ``abort_check`` is called
        after every sampling step; raising from it (via ``_check_abort``)
        propagates out of ``comfy.sample.sample`` immediately, bounding a
        mid-denoise timeout to at most one step's duration.
        """
        return zimage_profile.denoise(
            model,
            cond,
            empty,
            self._in_channels,
            latent_h=latent_h,
            latent_w=latent_w,
            steps=steps,
            seed=seed,
            abort_check=self._check_abort,
        )

    # --- VAE decode (tiled, with ComfyUI output transpose) -----------------

    def _vae_decode_tiled(
        self,
        vae: Any,
        latent: Any,
        tile_size: int,
        overlap: int,
    ) -> Any:
        """Decode via tiled VAE with correct ComfyUI output transpose.

        ComfyUI VAE.decode returns [B,H,W,C]; tiled_vae_decode expects
        [B,C,H,W] from the decode function. ``_check_abort`` is invoked before
        every tile, bounding a mid-decode timeout to at most one tile's
        duration.
        """
        from comfyui_superl8.vae_tiled import tiled_vae_decode

        def decode_fn(lat):
            with torch.no_grad():
                img = vae.decode(lat.to("cuda"))
            if img.dim() == 4 and img.shape[-1] in (1, 3):
                img = img.movedim(-1, 1)  # [B,H,W,C] -> [B,C,H,W]
            return img.detach().to(lat.device)

        import torch

        with torch.no_grad():
            img_bchw = tiled_vae_decode(
                decode_fn,
                latent.to("cuda"),
                abort_check=self._check_abort,
                tile_size=tile_size,
                overlap=overlap,
            )
        return img_bchw

    # --- Generation (cooperative abort, staged timing) ---------------------

    def _do_generate(self, req: GenerationRequest) -> GenerationResponse:
        """Run one generation. Called with _gen_semaphore held.

        Staged timing: te_s, dit_s, vae_s, png_s, total_s.
        Cooperative abort via threading.Event.

        OOM handling: on torch.OutOfMemoryError the service returns 500,
        clears CUDA cache, and stays alive (does not crash).  Observed
        during single-card TE experiments (#200 comment 3).
        """
        from .memory import peak_hbm_monitor
        from PIL import Image

        import torch

        # _encode_prompt returns (cond, empty, te_source, embed_cache_hit)
        te_source = "local"
        embed_cache_hit = False

        provenance = build_provenance(
            width=req.width,
            height=req.height,
            steps=_LOCKED_NFE,
            cfg=_LOCKED_CFG,
            seed=req.seed,
            scheduler=_LOCKED_SCHEDULER,
            sampler_name=_LOCKED_SAMPLER,
            model_sampling_shift=self._model_sampling_shift,
            dit_path=self._dit_path,
            vae_path=self._vae_path,
            te_path=self._te_path,
            dit_checksum=zimage_profile.DIT_SHA256,
            vae_checksum=zimage_profile.VAE_SHA256,
            te_checksum=zimage_profile.TEXT_ENCODER_SHA256,
            in_channels=self._in_channels,
            te_source=te_source,
            embed_cache_hit=embed_cache_hit,
        )

        timing: dict[str, float] = {}
        hbm: dict[str, float] = {}
        t0 = time.monotonic()
        self._abort.clear()
        self._done.clear()

        try:
            torch.cuda.reset_peak_memory_stats()

            latent_h, latent_w = zimage_profile.validate_geometry(req.width, req.height)

            # --- Stage 1: TE encode (load TE, encode, free TE) ---
            self._check_abort()
            t_te_start = time.monotonic()
            with peak_hbm_monitor() as hbm_te:
                cond, empty, te_source, embed_cache_hit = self._encode_prompt(req.prompt)
            self._refresh_int8_residency()
            t_te_end = time.monotonic()
            timing["te_s"] = t_te_end - t_te_start
            hbm["te_peak_gib"] = hbm_te.get("peak_gib", 0.0)

            # Update provenance with actual te_source and cache hit
            provenance = dataclasses.replace(
                provenance, te_source=te_source, embed_cache_hit=embed_cache_hit
            )

            # --- Stage 2: DiT denoise ---
            self._check_abort()
            t_dit_start = time.monotonic()
            with peak_hbm_monitor() as hbm_dit:
                samples = self._denoise(
                    self._model,
                    cond,
                    empty,
                    latent_h,
                    latent_w,
                    _LOCKED_NFE,
                    req.seed,
                )
            t_dit_end = time.monotonic()
            timing["dit_s"] = t_dit_end - t_dit_start
            hbm["dit_peak_gib"] = hbm_dit.get("peak_gib", 0.0)

            if not torch.isfinite(samples).all():
                raise RuntimeError("denoise produced non-finite values")
            self._runtime_int8 = zimage_profile.assert_runtime_int8_engagement(
                self._model.model.diffusion_model
            )

            # --- Stage 3: VAE decode (load VAE, decode, free VAE) ---
            self._check_abort()
            t_vae_start = time.monotonic()
            vae = self._load_vae()
            try:
                with peak_hbm_monitor() as hbm_v:
                    img_bchw = self._vae_decode_tiled(
                        vae,
                        samples,
                        req.tile_size,
                        req.overlap,
                    )
            finally:
                self._free_vae(vae)
            self._refresh_int8_residency()
            t_vae_end = time.monotonic()
            timing["vae_s"] = t_vae_end - t_vae_start
            hbm["vae_peak_gib"] = hbm_v.get("peak_gib", 0.0)

            # --- Stage 4: PNG encode ---
            self._check_abort()
            t_png_start = time.monotonic()
            x = img_bchw[0].detach().clamp(0, 1).float().cpu().movedim(0, -1).numpy()
            pil_img = Image.fromarray((x * 255).round().astype("uint8"))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=False)
            png_bytes = buf.getvalue()
            t_png_end = time.monotonic()
            timing["png_s"] = t_png_end - t_png_start

            # Validate PNG structure
            if len(png_bytes) < 8 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError("generated PNG has invalid magic bytes")
            if len(png_bytes) > _MAX_PNG_BYTES:
                raise RuntimeError(f"PNG too large: {len(png_bytes)} > {_MAX_PNG_BYTES}")

            # --- Peak HBM ---
            peak_all = torch.cuda.max_memory_allocated() / (1024**3)
            hbm["peak_gib"] = peak_all

            timing["total_s"] = time.monotonic() - t0

            with self._state_lock:
                self._total_generations += 1

            self._done.set()

            return GenerationResponse(
                image_png=png_bytes,
                provenance=provenance,
                timing=timing,
                hbm=hbm,
            )
        except _AbortError:
            timing["total_s"] = time.monotonic() - t0
            timing["aborted"] = 1.0
            self._done.set()
            raise RuntimeError(f"generation aborted after {timing['total_s']:.1f}s (timeout)")
        except torch.OutOfMemoryError as e:
            timing["total_s"] = time.monotonic() - t0
            timing["oom"] = 1.0
            logger.error("OOM during generation: %s", e)
            torch.cuda.empty_cache()
            self._done.set()
            raise RuntimeError(
                f"OOM after {timing['total_s']:.1f}s: {e}"
            ) from e
        except Exception as e:
            timing["total_s"] = time.monotonic() - t0
            self._done.set()
            raise RuntimeError(f"generation failed after {timing['total_s']:.1f}s: {e}") from e

    def _check_abort(self) -> None:
        """Raise if cooperative abort has been signalled."""
        if self._abort.is_set():
            raise _AbortError("generation aborted")

    # --- HTTP handlers ---------------------------------------------------

    def _handle_health(self) -> tuple[int, dict[str, Any]]:
        """GET /health — liveness probe (always 200 if process is up)."""
        return 200, {
            "status": "ok",
            "service": "zimage-resident",
            "version": SERVICE_VERSION,
        }

    def _handle_ready(self) -> tuple[int, dict[str, Any]]:
        """GET /ready — readiness probe (200 when model is resident)."""
        if self._error:
            return 503, {"status": "error", "error": self._error}
        if self._loading:
            elapsed = time.monotonic() - self._load_start
            return 503, {"status": "loading", "elapsed_s": round(elapsed, 1)}
        if self._ready:
            try:
                int8_inventory = self._refresh_int8_residency()
            except RuntimeError as error:
                return 503, {"status": "error", "error": str(error)}
            with self._state_lock:
                total = self._total_generations
                active = self._active_generations
            return 200, {
                "status": "ready",
                "in_channels": self._in_channels,
                "total_generations": total,
                "active_generations": active,
                "load_time_s": round(self._load_end - self._load_start, 1),
                "model_format": "fni8",
                "model_sha256": zimage_profile.DIT_SHA256,
                "int8_inventory": int8_inventory,
                "runtime_int8": getattr(self, "_runtime_int8", {}),
            }
        return 503, {"status": "not_started"}

    def _handle_generate(self, body: bytes) -> tuple[int, dict[str, Any], bytes | None]:
        """POST /generate — run one generation.  Returns (code, json, png_bytes)."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return 400, {"error": f"invalid JSON: {e}"}, None

        try:
            req = GenerationRequest.from_dict(data)
        except (TypeError, ValueError) as e:
            return 400, {"error": f"invalid request: {e}"}, None

        try:
            _validate_request(req)
        except ValueError as e:
            return 400, {"error": str(e)}, None

        if not self._check_ready():
            return 503, {"error": "service not ready", "detail": self._error}, None

        # Try to acquire the single generation slot
        acquired = self._gen_semaphore.acquire(blocking=False)
        if not acquired:
            return 503, {"error": "generation slot busy, try again later"}, None

        # Start watchdog for cooperative timeout
        self._abort.clear()
        self._done.clear()
        watchdog = threading.Thread(
            target=self._watchdog,
            args=(req.timeout_s,),
            daemon=True,
        )
        watchdog.start()

        with self._state_lock:
            self._active_generations += 1
        try:
            resp = self._do_generate(req)
            return 200, resp.to_dict(), resp.image_png
        except Exception as e:
            logger.error("generation failed: %s\n%s", e, traceback.format_exc())
            return 500, {"error": str(e)}, None
        finally:
            self._abort.set()  # signal watchdog to exit
            self._done.set()
            with self._state_lock:
                self._active_generations -= 1
            self._gen_semaphore.release()

    def _watchdog(self, timeout_s: float) -> None:
        """Cooperative timeout watchdog.  Signals abort after timeout_s.

        Signalling abort is not the same as stopping compute: `_check_abort`
        (wired into the sampler's per-step callback and the VAE tiler's
        per-tile loop) is what actually raises and unwinds the in-flight call.
        See the module docstring's "Timeout semantics" section for the bound.
        """
        deadline = time.monotonic() + timeout_s
        while not self._done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("generation timeout after %.1fs", timeout_s)
                self._abort.set()
                return
            self._done.wait(timeout=min(remaining, 0.5))


class _AbortError(RuntimeError):
    """Raised when cooperative abort is signalled."""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _RequestHandler(BaseHTTPRequestHandler):
    """Dispatch requests to the service methods.

    Responses use a bounded typed JSON envelope.  For PNG responses the body
    is ``{"provenance": {...}, "timing": {...}, "hbm": {...}, "image_png":
    "<base64>"}`` — no unbounded HTTP headers.
    """

    service: ZImageResidentService

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)

    def _send_response(self, code: int, data: dict[str, Any], png: bytes | None = None):
        """Send a bounded JSON envelope.  PNG is base64-encoded inside the body."""
        if png is not None:
            data["image_png"] = base64.b64encode(png).decode("ascii")
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "":
            code, data = self.service._handle_health()
            self._send_response(code, data)
        elif self.path == "/health":
            code, data = self.service._handle_health()
            self._send_response(code, data)
        elif self.path == "/ready":
            code, data = self.service._handle_ready()
            self._send_response(code, data)
        else:
            self._send_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self._send_response(400, {"error": "missing Content-Length"})
                return
            if length > _MAX_REQUEST_BYTES:
                self._send_response(
                    413, {"error": f"request body too large ({length} > {_MAX_REQUEST_BYTES})"}
                )
                return
            body = self.rfile.read(length)
            code, data, png = self.service._handle_generate(body)
            self._send_response(code, data, png=png)
        else:
            self._send_response(404, {"error": "not found"})


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
    ap = argparse.ArgumentParser(description="Z-Image resident HTTP service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8410)
    ap.add_argument(
        "--weights-dir",
        default=zimage_profile.CANONICAL_MODEL_ROOT,
        help="Canonical NVMe model root (alternate roots are rejected)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    svc = ZImageResidentService(
        host=args.host,
        port=args.port,
        weights_dir=args.weights_dir,
    )

    # Load pipeline in background thread
    loader = threading.Thread(target=svc._load_pipeline, daemon=True)
    loader.start()

    # Start HTTP server
    handler = type(
        "_Handler",
        (_RequestHandler,),
        {"service": svc},
    )
    server = _ThreadedHTTPServer((args.host, args.port), handler)
    logger.info("Z-Image resident service listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
