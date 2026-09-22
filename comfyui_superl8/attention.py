# SPDX-License-Identifier: Apache-2.0
"""Bidirectional int8 dp4a attention for DiT self-attention — pure, ComfyUI-free.

DiT self-attention has NO causal mask (every token attends every token), which is
exactly `superl8.attn_int8_fwd(causal=False, rotate=True)`. The Hadamard rotation
mitigates channel outliers in Q/K (the dominant int8 error source — measured 1.4x
lower rel-err on real DiT K activations); it is logit-invariant and adds no
measurable perf cost. This wraps the kernel to match the
shapes ComfyUI hands a custom attention callback, with an fp fallback for masked
attention, cross-attention, or head dims the kernel doesn't support.

On top of that shape/mask fallback there is a numeric **SQNR gate** (issue #82),
the attention analogue of the per-layer linear gate in `ops.py`: the int8 dp4a output
is compared against the tiled fp16/bf16 reference; if its signal-to-quantization-noise
ratio is below `ATTN_SQNR_FLOOR_DB` the call-site is demoted to that reference kernel for
the rest of the run.
Un-gated int8 self-attention quality was previously unvalidated (esp. video), so
without this a badly-quantized attention block would silently ship garbage.

**Multi-timestep calibration (issue: single-activation gate).** A DiT's activation
distribution SHIFTS across the denoising trajectory (early/mid/late timesteps) and
across resolutions, so a decision locked from the FIRST activation a call-site sees is
not trustworthy — this repo's own evidence shows a one-step latent cosine of 1.0 that
MISSED a severe multi-step image collapse (`docs/int8-dit-validation.md`). When
`calib_samples > 1` the gate therefore gathers SQNR over several forwards (successive
denoise steps, each possibly a different resolution -> its own signature) and finalizes
the int8/fp decision on the **WORST** SQNR seen (conservative), instead of the first.
`calib_samples == 1` (the default) reproduces the legacy first-activation behavior, so
the change is opt-in and backward-compatible. `revalidate_every > 0` additionally
re-checks a passing call-site periodically and demotes it if a later activation shifts
unusually (drops below the floor, or far below its calibrated worst).
"""

from __future__ import annotations

import os

import torch

# Set FNI8_ATTN_DEBUG=1 to log, per attention call-site, which kernel path was chosen
# (int8 dp4a vs the O(N) fp16 kernel vs torch SDPA) and the q/k/v shapes.
_ATTN_DEBUG = os.environ.get("FNI8_ATTN_DEBUG", "") not in ("", "0", "false", "False")


def _dbg(route: str, q, k):
    if _ATTN_DEBUG:
        print(f"[fni8-attn] route={route:10s} q={tuple(q.shape)} k={tuple(k.shape)}",
              flush=True)


# fni8's int8 attention head-dim allowlist (D must be one of these).
SUPPORTED_HEAD_DIMS = (32, 64, 72, 80, 128, 256)

# Minimum acceptable signal-to-quantization-noise ratio (dB) for int8 self-attention
# vs the fp SDPA reference. 20 dB == quantization-noise power 1% of signal power
# (rel-RMS error ~0.1, cosine-sim ~0.995) — at/above the strictness of the linear
# gate's cos>=0.99 bar (cos 0.99 == 17.0 dB). Tunable; the validation run in
# docs/int8-dit-validation.md reports the SQNR actually observed on real DiTs.
ATTN_SQNR_FLOOR_DB = 20.0


def attn_sqnr(y_int8: torch.Tensor, y_fp: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio (dB) of an int8-attn output vs its fp
    reference: ``10*log10(mean(y_fp^2) / mean((y_int8 - y_fp)^2))``. Higher is better;
    +inf when the two are bit-identical. Computed in fp32 regardless of input dtype.

    Returns ``-inf`` (a definitive gate FAIL -> fp fallback) when either tensor is
    non-finite: an SQNR that can't be trusted must never accidentally pass the gate
    (``nan >= floor`` is False in Python, so nan already falls back, but -inf makes the
    intent explicit and keeps the 'below floor' accounting honest)."""
    a = y_int8.detach().float()
    b = y_fp.detach().float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("-inf")
    signal = b.pow(2).mean()
    noise = (a - b).pow(2).mean()
    if noise.item() == 0.0:
        return float("inf")
    return (10.0 * torch.log10(signal / noise)).item()


class Int8AttnGate:
    """SQNR fallback gate for int8 self-attention, mirroring the per-layer linear gate
    (`ops.py`): measure int8 vs fp, keep int8 only if it clears the floor, else demote
    the call-site to the memory-efficient fp16/bf16 kernel for the rest of the run.

    Keyed by shape and, when ``sites_per_signature > 1``, a round-robin block-site
    ordinal. This prevents one block's verdict from authorizing every other block with
    the same shape. With
    ``calib_samples == 1`` (default) the decision is finalized from the first call for a
    signature; with ``calib_samples > 1`` it is calibrated across that many forwards
    (successive denoise timesteps — and different resolutions land on different
    signatures) and finalized on the **WORST** SQNR observed, which is conservative and
    catches a per-timestep collapse a single-activation gate would miss. A passing
    signature then runs int8-only; a failing one falls back to fp16/bf16. When
    ``revalidate_every > 0`` a passing signature is periodically re-measured and demoted
    if a later activation shifts unusually. One gate instance is created per patched
    model (see `nodes.FNI8AttentionPatch`), so decisions never leak across models.

    **Self-fallback detection (Qwen-Image et al.):** `superl8.attn_int8_fwd` has its OWN
    inner SageAttention accuracy gate — when a Q row is outlier-dominated
    (``max|q|/median|q|`` large; Qwen-Image's post-RoPE Q measures ~5e4) int8 QK would
    crush the non-outlier channels, so the kernel quantizes NOTHING and returns plain fp
    SDPA. That output is bit-identical to our fp reference, i.e. SQNR ``+inf``. Naively
    that "passes" the floor and the site is cached as int8 — but then every later denoise
    step re-pays the kernel's per-call outlier scan (a median over the whole Q) + dispatch
    only to fall back to the same fp SDPA again. So a ``+inf`` SQNR is treated as a FAIL
    for ROUTING (``passed=False``): the site is pinned to fp SDPA and skips the wasted
    kernel round-trip, with bit-identical output. A genuinely-engaged int8 site has a
    FINITE SQNR (Z-Image self-attn ~47 dB) and is unaffected."""

    def __init__(self, sqnr_floor_db: float = ATTN_SQNR_FLOOR_DB, *,
                 calib_samples: int = 1, revalidate_every: int = 0,
                 revalidate_drop_db: float = 6.0, sites_per_signature: int = 1):
        self.sqnr_floor_db = sqnr_floor_db
        # How many forwards (denoise timesteps) to sample per signature before locking
        # the int8/fp decision on the WORST SQNR seen. 1 == legacy first-activation gate.
        self.calib_samples = max(1, int(calib_samples))
        # Re-measure a passing signature every N calls and demote if it shifted; 0 == off.
        self.revalidate_every = max(0, int(revalidate_every))
        self.revalidate_drop_db = float(revalidate_drop_db)
        self.sites_per_signature = max(1, int(sites_per_signature))
        # Base shape signature -> next stable block-site ordinal. Calls of a given
        # signature occur in stable transformer-block order on every denoise step.
        self._site_cursor: dict = {}
        # signature -> (passed: bool, worst_sqnr_db: float)   [FINALIZED decisions]
        self.decisions: dict = {}
        # signature -> [worst_db, n_samples]                  [calibration IN PROGRESS]
        self._calib: dict = {}
        # signature -> calls since last (re)validation         [revalidation cadence]
        self._since_reval: dict = {}
        # True once _maybe_save has written the cache (one save per gate lifetime)
        self._persisted: bool = False
        # Dirty flag: set when decisions change, cleared after save
        self._dirty: bool = False

    # -- persistence (issue #196) ------------------------------------------------

    @staticmethod
    def _cache_path(gpu_id: str | None = None) -> str:
        """Return the per-GPU cache file path for SQNR attention verdicts."""
        import os
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fni8")
        if gpu_id is None:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_id = torch.cuda.get_device_name(0).replace(" ", "_")
            except Exception:
                gpu_id = "cpu"
        return os.path.join(cache_dir, f"sqnr-attn-{gpu_id}.json")

    @staticmethod
    def _gpu_id() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0).replace(" ", "_")
        except Exception:
            pass
        return "cpu"

    def save(self, path: str | None = None) -> str | None:
        """Persist finalized decisions to disk. Returns the path written, or None
        if nothing was saved (empty decisions or write failure)."""
        if not self.decisions:
            return None
        import json
        import os

        path = path or self._cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}
        for k, (passed, sqnr_db) in self.decisions.items():
            # Tuple keys -> JSON array; strings stay strings
            key_json = list(k) if isinstance(k, tuple) else k
            data[str(key_json)] = [passed, sqnr_db]
        payload = {
            "version": 1,
            "gpu": self._gpu_id(),
            "sqnr_floor_db": self.sqnr_floor_db,
            "decisions": data,
        }
        try:
            with open(path, "w") as f:
                json.dump(payload, f)
            return path
        except OSError:
            return None

    def load(self, path: str | None = None) -> bool:
        """Load persisted decisions from disk. Returns True if decisions were
        loaded successfully; False if the file is missing, stale, or corrupt.
        Only loads decisions whose keys match the current gate's fingerprint
        (same GPU, same sqnr_floor)."""
        import json
        import os

        path = path or self._cache_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("version") != 1:
            return False
        if payload.get("gpu") != self._gpu_id():
            return False
        if payload.get("sqnr_floor_db") != self.sqnr_floor_db:
            return False
        loaded = 0
        for k_str, (passed, sqnr_db) in payload.get("decisions", {}).items():
            try:
                k_list = json.loads(k_str.replace("'", '"'))
                k = tuple(
                    str(x) if isinstance(x, str) else int(x)
                    for x in k_list
                )
            except Exception:
                continue
            self.decisions[k] = (passed, float(sqnr_db))
            loaded += 1
        return loaded > 0

    def site_key(self, signature):
        """Return the stable round-robin site key for this shape signature."""
        if self.sites_per_signature == 1:
            return signature
        site = self._site_cursor.get(signature, 0)
        self._site_cursor[signature] = (site + 1) % self.sites_per_signature
        return (*signature, site)

    def _maybe_save(self):
        """Persist finalized verdicts to disk when all calibrations are done and the
        decisions have changed since the last save."""
        if self._calib or not self.decisions or not self._dirty:
            return
        self.save()
        self._dirty = False

    def cached(self, key):
        """Prior FINALIZED decision for this signature, or None if unseen / still
        calibrating."""
        d = self.decisions.get(key)
        return None if d is None else d[0]

    @staticmethod
    def _is_pass(sqnr_db: float, floor: float) -> bool:
        # A FINITE SQNR at/above the floor means int8 genuinely engaged AND is accurate.
        # +inf means the kernel's own gate rejected int8 and already ran fp (bit-identical
        # output) — route such sites to fp directly so we stop re-dispatching the kernel
        # (and its per-call outlier median) every step. See the class docstring.
        return floor <= sqnr_db < float("inf")

    def record(self, key, sqnr_db: float) -> bool:
        """Finalize the decision for `key` from a single SQNR (immediate). Used by the
        default single-sample path and by direct callers/tests."""
        passed = self._is_pass(sqnr_db, self.sqnr_floor_db)
        self.decisions[key] = (passed, sqnr_db)
        self._dirty = True
        return passed

    def observe(self, key, sqnr_db: float):
        """Fold one measured SQNR into multi-timestep calibration for `key`, tracking the
        WORST (most conservative) value across up to `calib_samples` forwards. Since the
        DiT's activation distribution shifts along the denoise trajectory, a per-timestep
        collapse only shows up in a later sample — so the decision is locked on the worst,
        never the first. Finalizes via `record` on the worst once enough samples are
        gathered and returns the finalized bool; returns None while still calibrating."""
        st = self._calib.get(key)
        if st is None:
            st = [sqnr_db, 1]
            self._calib[key] = st
        else:
            st[0] = min(st[0], sqnr_db)
            st[1] += 1
        if st[1] >= self.calib_samples:
            passed = self.record(key, st[0])
            del self._calib[key]
            return passed
        return None

    def interim_pass(self, key) -> bool:
        """Worst-so-far verdict to USE for the current forward while `key` is still
        calibrating (before the decision finalizes)."""
        st = self._calib.get(key)
        if st is None:
            return True
        return self._is_pass(st[0], self.sqnr_floor_db)

    def revalidation_due(self, key) -> bool:
        """True when a finalized-PASS signature is due for a periodic re-measurement
        (opt-in via `revalidate_every`); advances the per-key counter as a side effect."""
        if self.revalidate_every <= 0 or self.cached(key) is not True:
            return False
        n = self._since_reval.get(key, 0) + 1
        if n >= self.revalidate_every:
            self._since_reval[key] = 0
            return True
        self._since_reval[key] = n
        return False

    def revalidate(self, key, sqnr_db: float) -> bool:
        """Re-check a finalized-PASS signature against a fresh activation. Demote to fp if
        the new SQNR fell below the floor (or is +inf self-fallback), or dropped more than
        `revalidate_drop_db` below its calibrated worst — an activation that shifted
        unusually. Records the worse of the two and returns the (possibly updated) bool."""
        prev = self.decisions.get(key)
        prev_worst = prev[1] if prev else sqnr_db
        ok = self._is_pass(sqnr_db, self.sqnr_floor_db)
        if not ok or sqnr_db < prev_worst - self.revalidate_drop_db:
            self.decisions[key] = (False, min(sqnr_db, prev_worst))
            self._dirty = True
            return False
        self.decisions[key] = (True, min(sqnr_db, prev_worst))
        self._dirty = True
        return True


def _sdpa(q, k, v, mask=None):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _prep_additive_mask(mask, B, heads, M, N, dtype):
    """Shape a ComfyUI attention mask into fni8's additive-mask layout
    ``[B|1, heads|1, M, N]`` (natural-log bias), using VIEWS only so a broadcast
    mask is never materialized. Returns ``None`` (→ caller uses torch SDPA) when
    the mask can't be mapped without a dtype-cast/expand that would defeat the
    memory win (mismatched dtype, or trailing dims that aren't ``[M, N]``).

    A boolean mask (True = attend) is converted to an additive ``0 / -inf`` bias
    in ``dtype`` — this materializes, but bool masks are small/rare here."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        mask = torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(
            ~mask, float("-inf")
        )
    if mask.dtype != dtype:
        return None  # a cast would materialize an expanded mask — fall back to SDPA
    if mask.dim() == 2:
        mask = mask.view(1, 1, *mask.shape)
    elif mask.dim() == 3:
        if mask.shape[0] == B:
            mask = mask.view(B, 1, *mask.shape[1:])
        elif mask.shape[0] == B * heads:
            mask = mask.view(B, heads, *mask.shape[1:])
        else:
            mask = mask.view(1, mask.shape[0], *mask.shape[1:])
    if mask.dim() != 4 or mask.shape[-2] != M or mask.shape[-1] != N:
        return None
    if not (mask.shape[0] in (1, B) and mask.shape[1] in (1, heads)):
        return None
    return mask


def _fp16_attn(q, k, v, mask=None):
    """Memory-efficient fp16/bf16 fallback for the non-int8 path (masked, or
    outlier-gated). Uses fni8's tiled half2 FA-2 prefill kernel — O(N) memory,
    unlike torch SDPA which has no flash backend on Volta and materializes the
    O(N^2) score matrix (the LTX/Wan video-DiT OOM, issue #97). q/k/v are
    ``[B, heads, M/N, D]`` (self- or cross-attention). Falls back to torch SDPA
    only when the shape/dtype is genuinely unsupported by the kernel (mixed
    dtype, unsupported head dim, or a mask that can't be mapped without
    materializing)."""
    B, heads, M, D = q.shape
    N = k.shape[2]
    supported = (
        q.is_cuda
        and D in SUPPORTED_HEAD_DIMS
        and q.dtype == k.dtype == v.dtype
        and q.dtype in (torch.float16, torch.bfloat16)
    )
    add_mask = (
        _prep_additive_mask(mask, B, heads, M, N, q.dtype)
        if (mask is not None and supported)
        else None
    )
    # Check support BEFORE importing fni8 so CPU / fp32 / unsupported paths never
    # touch the CUDA extension (keeps the pure-torch unit tests fni8-free).
    if not supported or (mask is not None and add_mask is None):
        return _sdpa(q, k, v, mask)  # genuinely unsupported -> fp SDPA
    import superl8

    # The tiled half2 FA-2 prefill kernel is a recent fni8 addition; an older baked
    # `fni8` build (its master can lag the node pack's #132 wiring) may not export it.
    # Degrade gracefully to torch SDPA rather than crash — correct, just O(N^2) memory
    # (fine for the short image-DiT sequences this fp16 fallback actually fires on).
    if not hasattr(fni8, "attn_fp16_fwd"):
        return _sdpa(q, k, v, mask)
    return superl8.attn_fp16_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(), causal=False, mask=add_mask
    )


def fni8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    *,
    skip_reshape: bool = False,
    gate: "Int8AttnGate | None" = None,
    sta_layout: tuple[int, int, int] | None = None,
    sta_tile: tuple[int, int, int] | None = None,
    **_,
) -> torch.Tensor:
    """Bidirectional attention. q,k,v either [B, S, heads*D] (ComfyUI default) or
    already [B, heads, S, D] when skip_reshape. Returns [B, S, heads*D].

    When `gate` is provided the int8 path is SQNR-gated per call-site (see
    `Int8AttnGate`); with `gate=None` the int8 kernel is used whenever the shape/mask
    are supported (the raw, un-gated path used by unit tests).

    **Sliding-Tile Attention (STA)** — when both ``sta_layout=(F, H, W)`` and
    ``sta_tile=(tile_f, tile_h, tile_w)`` are set, the attention is restricted
    to a local 3D window around each token.  Quality is measured via masked
    SDPA (the existing int8 kernel does not yet support block-sparse masks).
    The ``gate`` parameter is shared with the STA gate when applicable."""
    if sta_layout is not None and sta_tile is not None:
        return _sta_attn_path(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            gate=gate,
            sta_layout=sta_layout,
            sta_tile=sta_tile,
        )

    if skip_reshape:
        B, H, S, D = q.shape
        inner = H * D
    else:
        B, S, inner = q.shape
        D = inner // heads
        q = q.view(B, S, heads, D).transpose(1, 2)
        # Derive K/V sequence length from THEIR OWN tensor, never from Q's S: LTX
        # cross-attention has video queries (S=17408) attending a 1024-token Gemma
        # context, so K/V are shorter than Q. Reusing Q's S here reshaped a 1024-token
        # K/V as [1, 17408, 32, 128] -> "shape ... is invalid for input of size
        # 4194304" (the render-log crash). K/V heads may also differ from Q (GQA).
        S_kv = k.shape[1]
        k = k.view(B, S_kv, k.shape[-1] // D, D).transpose(1, 2)
        v = v.view(B, S_kv, v.shape[-1] // D, D).transpose(1, 2)

    # q/k/v are now [B, heads, M, D] and [B, kv_heads, N, D].
    M, N = q.shape[2], k.shape[2]
    # DP4A handles unmasked, supported-D, equal-length self-attention at video lengths.
    # Its SQNR reference is the tiled O(N)-memory fp16/bf16 kernel, so calibration does
    # not require an arbitrary sequence cutoff or allocate torch SDPA's O(N^2) scores.
    # Masked/cross attention and unsupported head dims use that same fp kernel directly.
    use_int8 = mask is None and D in SUPPORTED_HEAD_DIMS and M == N
    if use_int8:
        try:
            _dbg("int8", q, k)
            out = _int8_attn_gated(q, k, v, heads, D, gate)
        except Exception:
            # ANY failure from the int8 path (e.g. the kernel RuntimeError on a shape it
            # doesn't support) falls to the O(N) fp16 kernel — not torch SDPA, which
            # would materialize the O(N^2) score matrix and OOM (#97).
            _dbg("fp16(rescue)", q, k)
            out = _fp16_attn(q, k, v, mask)
    else:
        _dbg("fp16" if mask is None else "fp16(masked)", q, k)
        out = _fp16_attn(q, k, v, mask)

    return out.transpose(1, 2).reshape(B, S, inner)


def _sta_attn_path(q, k, v, heads, mask, skip_reshape, gate, sta_layout, sta_tile):
    """Route to Sliding-Tile Attention (quality reference via masked SDPA)."""
    from .sta import StaGate, sta_attention

    F, H, W = sta_layout
    tile_f, tile_h, tile_w = sta_tile

    if skip_reshape:
        B, nh, N, D = q.shape
    else:
        B, S, inner = q.shape
        D = inner // heads
        N = S
        nh = heads
        q = q.view(B, N, nh, D).transpose(1, 2)
        k = k.view(B, N, k.shape[-1] // D, D).transpose(1, 2)
        v = v.view(B, N, v.shape[-1] // D, D).transpose(1, 2)

    sta_gate: StaGate | None = None
    if gate is not None:
        sta_gate = StaGate(sqnr_floor_db=gate.sqnr_floor_db)

    out = sta_attention(q, k, v, F, H, W, tile_f, tile_h, tile_w, gate=sta_gate)

    if skip_reshape:
        return out
    return out.transpose(1, 2).reshape(B, N, nh * D)


def _int8_dp4a(q, k, v):
    import superl8

    # Preserve the caller's native dtype (fp16 or bf16) instead of forcing fp16:
    # bf16-native DiTs (Z-Image, Flux, ...) overflow to inf under a bf16->fp16
    # downcast here, which is what produced black images before fni8's kernels
    # gained bf16 GEMM/FA output support.
    in_dtype = q.dtype
    # This wrapper compares the actual quantized output against the tiled fp reference
    # and caches the output-SQNR verdict. Ask fni8 for the real DP4A result instead of
    # paying its conservative any-row pre-detector, whose false-positive probability
    # grows with video sequence length; the default remains enabled for other callers.
    return superl8.attn_int8_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        causal=False,
        rotate=True,
        internal_accuracy_gate=False,
    ).to(in_dtype)


def _int8_attn_gated(q, k, v, heads, D, gate):
    """Return the int8 attn output for [B, heads, S, D] q/k/v, applying the SQNR gate
    when one is supplied. Falls back to tiled fp16/bf16 for demoted call-sites."""
    if gate is None:
        return _int8_dp4a(q, k, v)

    B, S = q.shape[0], q.shape[2]
    key = gate.site_key((B, heads, S, D, str(q.dtype)))
    cached = gate.cached(key)

    # Finalized FAIL -> always fp16/bf16 (skip the wasted int8 kernel round-trip).
    if cached is False:
        # Demoted call-site: the memory-efficient fp16 kernel, not O(N^2) SDPA (#97).
        return _fp16_attn(q, k, v)
    # Finalized PASS and no periodic revalidation due -> int8 only.
    if cached is True and not gate.revalidation_due(key):
        return _int8_dp4a(q, k, v)

    # A measurement is needed this forward: either still CALIBRATING (cached is None —
    # gathering worst-case SQNR across several denoise timesteps) or a REVALIDATION of a
    # passing site is due. Both need the fp reference alongside the int8 output.
    int8_out = _int8_dp4a(q, k, v)
    try:
        fp_ref = _fp16_attn(q, k, v)
    except Exception:
        # Can't build a reference (e.g. GQA without enable_gqa) — don't gate, don't
        # cache; keep the shape/mask-supported int8 output.
        return int8_out
    sqnr_db = attn_sqnr(int8_out, fp_ref)

    if cached is True:
        # Revalidation of an already-passing site: keep int8 only if it still clears bar.
        return int8_out if gate.revalidate(key, sqnr_db) else fp_ref

    # Calibration: fold this timestep's SQNR into the running worst-case for the site.
    verdict = gate.observe(key, sqnr_db)
    if verdict is None:                       # still calibrating -> use worst-so-far
        verdict = gate.interim_pass(key)
    gate._maybe_save()  # persist when all signatures are finalized
    return int8_out if verdict else fp_ref


def make_attn1_replace_callback(
    gate: "Int8AttnGate | None" = None,
    sta_layout: tuple[int, int, int] | None = None,
    sta_tile: tuple[int, int, int] | None = None,
):
    """Build a `set_model_attn1_replace` callback bound to a per-model SQNR gate.

    When ``sta_layout`` and ``sta_tile`` are set, the callback applies
    Sliding-Tile Attention (see ``sta.py``)."""

    def attn1_replace_callback(q, k, v, extra_options):
        return fni8_attention(
            q, k, v, extra_options["n_heads"], gate=gate, sta_layout=sta_layout, sta_tile=sta_tile
        )

    return attn1_replace_callback


# Backward-compatible ungated callback (raw int8, no SQNR gate) — kept for callers/tests
# that imported it directly. Production wiring uses `make_attn1_replace_callback(gate)`.
def attn1_replace_callback(q, k, v, extra_options):
    """The signature ComfyUI's `set_model_attn1_replace` expects: already-projected
    q,k,v [B, S, inner] + extra_options{n_heads, dim_head}. Returns the attention
    output pre-`to_out`."""
    return fni8_attention(q, k, v, extra_options["n_heads"])


def make_fni8_attn_override(
    gate: "Int8AttnGate | None" = None,
    sta_layout: tuple[int, int, int] | None = None,
    sta_tile: tuple[int, int, int] | None = None,
):
    """Build an `optimized_attention_override` hook bound to a per-model SQNR gate.

    When ``sta_layout`` and ``sta_tile`` are set, the override applies
    Sliding-Tile Attention (see ``sta.py``)."""

    def fni8_attn_override(func, *args, **kwargs):
        """ComfyUI `transformer_options["optimized_attention_override"]` hook. ComfyUI's
        `wrap_attn` decorator invokes it as ``override(orig_fn, q, k, v, heads, ...)`` at
        every attention call (comfy/ldm/modules/attention.py), reaching every DiT regardless
        of whether it imported `optimized_attention` by value — which is why this hook is
        used instead of monkeypatching the module global (43 of ComfyUI's 44 DiTs import it
        by value, so a global swap is invisible to them).

        Routes bidirectional, supported-head-dim self-attention through the int8 dp4a kernel
        (SQNR-gated per call-site). MASKED attention (a real additive mask, e.g. LTX's
        guided-generation `self_attention_mask`) is now routed too — through the O(N)-memory
        fp16 kernel (`_fp16_attn`), NOT torch SDPA which has no flash backend on Volta and
        OOMs on long video-DiT attention (#97). Only `skip_output_reshape` callers (whose
        output layout `fni8_attention` can't produce) and unsupported head dims fall back to
        ComfyUI's own `func`.

        The silent `except: pass` that used to wrap the fni8 call was REMOVED (#97): it hid
        real kernel failures behind an fp fallback. `fni8_attention` already degrades to fp
        SDPA internally for genuinely-unsupported shapes/dtypes/masks, so any exception that
        does escape is a real bug and must surface loudly."""
        try:
            q, k, v = args[0], args[1], args[2]
            heads = args[3] if len(args) > 3 else kwargs["heads"]
        except (IndexError, KeyError):
            return func(*args, **kwargs)
        mask = kwargs.get("mask", args[4] if len(args) > 4 else None)
        skip_reshape = kwargs.get("skip_reshape", False)
        D = q.shape[-1] if skip_reshape else q.shape[-1] // heads
        if not kwargs.get("skip_output_reshape", False) and D in SUPPORTED_HEAD_DIMS:
            # No try/except: fni8_attention falls back to fp SDPA internally for
            # unsupported shapes/masks; a raised error is a real bug, kept loud.
            return fni8_attention(
                q,
                k,
                v,
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                gate=gate,
                sta_layout=sta_layout,
                sta_tile=sta_tile,
            )
        return func(*args, **kwargs)

    return fni8_attn_override


# Backward-compatible ungated override — production wiring uses
# `make_fni8_attn_override(gate)` so the SQNR gate is engaged.
fni8_attn_override = make_fni8_attn_override(None)
