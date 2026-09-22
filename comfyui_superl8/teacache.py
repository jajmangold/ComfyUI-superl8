# SPDX-License-Identifier: Apache-2.0
"""TeaCache-style cross-step feature caching for the fni8 diffusion DiTs — pure logic,
ComfyUI-free (mirrors `attention.py`).

Consecutive denoise steps feed the DiT a slowly-changing latent, so consecutive DiT
predictions are very similar. TeaCache (Timestep-Embedding-Aware Cache, 2411.19108)
watches the timestep-conditioned input delta across steps and **skips recomputing the
DiT when the accumulated relative change is below a threshold**, reusing the previous
step's cached DiT output. FLOPs are the actual DiT bottleneck on this fleet (the int8
dp4a compute win is real but the model is still compute-bound), so skipping whole steps
stacks *multiplicatively* on int8: int8 (~2x) x TeaCache (~1.5-2x) ~= 3-4x.

Method (following the TeaCache paper's shape, honest about the deltas for this pack):

  * **Skip signal** = relative-L1 change of the model input between consecutive steps:
    ``rel_l1 = mean(|s_t - s_{t-1}|) / mean(|s_{t-1}|)``. The paper taps the first block's
    timestep-*modulated* normalized hidden state; we tap the model *input* latent ``x``
    (reachable at the `diffusion_model` wrapper seam for EVERY DiT without per-arch block
    surgery). Its rel-L1 across steps IS the FBCache-style skip signal and is already
    implicitly timestep-dependent (the trajectory changes fastest early, slowest late).
  * **Strategy** — ``"tea"`` accumulates the raw ``rel_l1``; ``"taylor"`` (default) scales
    it by a running ``dy/dx`` sensitivity estimate so the metric predicts *output* change,
    not input change (TaylorSeer, higher quality at the same step reduction). Adopted from
    the sibling draft (#94), which had the right controller idea but wired it to a seam
    (`ModelPatcher.apply_model`) that ComfyUI's sampler never calls — this module uses the
    `WrappersMP.DIFFUSION_MODEL` seam `BaseModel._forward` actually reads, the same family
    as the pack's proven `optimized_attention_override`.
  * **Cumulative threshold** — accumulate the metric every step; while the accumulator
    stays below ``rel_l1_thresh`` the step is SKIPPED (cached output reused); when it
    crosses, the DiT is recomputed and the accumulator resets. Bigger threshold => more
    skips => faster but lower quality. This is the speed/quality dial.
  * **SQNR spot-check gate** — the first time we have a fresh compute to compare against the
    previous cached output, measure their SQNR; if it is below ``sqnr_floor_db`` (20 dB,
    same bar as the attention gate) the model's step-to-step outputs are too different to
    cache safely, so caching is DISABLED for the rest of the run (never ship a divergent
    result silently). Also adopted from #94; consistent with `attention.attn_sqnr`.

Honesty (this is a caching accelerator — it TRADES quality for speed):
  * We cache the **whole DiT output** (prediction reuse), not the paper's block-stack
    *residual*. Prediction reuse is fully arch-agnostic (one code path for single-stream
    Z-Image, double-stream Qwen-Image-Edit, audio+video LTX-2.3) at a small quality cost vs
    residual caching, which the per-model latent-cosine gate measures directly.
  * A skipped step also skips the int8 dp4a path for that step, so the composed
    int8+TeaCache result must clear the SAME final-latent quality gate as int8-only. The
    bench (`bench/teacache_sweep.py`) reports int8-only vs int8+TeaCache speedup AND the
    latent cosine at each threshold so the knee is chosen by measurement.

The controller keeps independent cache state per ``cond_or_uncond`` stream (CFG runs the
cond and uncond forwards separately; their inputs must not be diffed against each other)
and resets at the start of each generation (detected via the sampler's sigma schedule).
"""
from __future__ import annotations

import torch

# Default cumulative rel-L1 threshold. 0.0 disables skipping (every step computed).
DEFAULT_REL_L1_THRESH = 0.15
# Always compute the first N steps of a generation. Few-step distilled models
# (Z-Image-Turbo = 8 steps) are sensitive to an early skip, and step 0 has nothing to reuse.
DEFAULT_WARMUP_STEPS = 1
# Never skip more than this many steps in a row — bounds cache staleness / error drift.
DEFAULT_MAX_CONSECUTIVE_SKIPS = 3
# SQNR floor (dB) for the spot-check gate — same bar as the attention SQNR gate.
CACHE_SQNR_FLOOR_DB = 20.0


def cache_sqnr(cached: torch.Tensor, actual: torch.Tensor) -> float:
    """Signal-to-error ratio (dB) of a cached step vs the full-model output — the same
    formula as `attention.attn_sqnr`: ``10*log10(mean(actual^2)/mean((cached-actual)^2))``.
    +inf when bit-identical, -inf when either tensor is non-finite (a definitive FAIL)."""
    a = cached.detach().float()
    b = actual.detach().float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("-inf")
    signal = b.pow(2).mean()
    noise = (a - b).pow(2).mean()
    if noise.item() == 0.0:
        return float("inf")
    return (10.0 * torch.log10(signal / noise)).item()


def _rel_l1(cur: torch.Tensor, prev: torch.Tensor) -> float:
    """Relative-L1 distance ``mean(|cur - prev|) / mean(|prev|)`` in fp32 — the TeaCache
    skip signal (how much the input moved since the last step)."""
    cur = cur.detach().float()
    prev = prev.detach().float()
    denom = prev.abs().mean()
    if denom.item() == 0.0:
        return float("inf")
    return (cur - prev).abs().mean().div(denom).item()


class _StreamState:
    """Per-stream (cond / uncond) TeaCache state."""

    __slots__ = ("prev_signal", "accumulated", "cached_output", "consecutive_skips",
                 "last_computed_signal", "sensitivity")

    def __init__(self):
        self.prev_signal: torch.Tensor | None = None       # input at the previous step
        self.accumulated: float = 0.0
        self.cached_output: torch.Tensor | None = None      # output at the last COMPUTED step
        self.last_computed_signal: torch.Tensor | None = None  # input at the last COMPUTED step
        self.consecutive_skips: int = 0
        self.sensitivity: float = 1.0  # TaylorSeer dy/dx EMA

    def reset(self):
        self.prev_signal = None
        self.accumulated = 0.0
        self.cached_output = None
        self.last_computed_signal = None
        self.consecutive_skips = 0
        self.sensitivity = 1.0


class TeaCacheController:
    """Cross-step DiT output cache with a cumulative rel-L1 skip signal (TeaCache).

    One instance per patched model (see `nodes.FNI8TeaCache`). Keeps independent state per
    ``cond_or_uncond`` stream and resets each generation. Not thread-safe — a single
    sequential denoise loop is the only caller.
    """

    def __init__(self, rel_l1_thresh: float = DEFAULT_REL_L1_THRESH,
                 warmup_steps: int = DEFAULT_WARMUP_STEPS,
                 max_consecutive_skips: int = DEFAULT_MAX_CONSECUTIVE_SKIPS,
                 strategy: str = "taylor", sqnr_floor_db: float = CACHE_SQNR_FLOOR_DB,
                 mode: str = "output", poly_coeffs=None):
        if strategy not in ("tea", "taylor"):
            raise ValueError(f"Unknown strategy '{strategy}'; choose 'tea' or 'taylor'")
        if mode not in ("output", "residual"):
            raise ValueError(f"Unknown mode '{mode}'; choose 'output' or 'residual'")
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.warmup_steps = int(warmup_steps)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.strategy = strategy
        self.sqnr_floor_db = float(sqnr_floor_db)
        self.mode = mode
        self.poly_coeffs = poly_coeffs
        self._sensitivity_decay = 0.9
        self._streams: dict = {}
        self._gen_sigma0: float | None = None
        self.step_index: int = 0
        self.caching_disabled: bool = False   # set True if the SQNR gate fails
        self._sqnr_checked: bool = False
        # Residual-mode block-cache state (single-stream models; see
        # `install_residual_block_cache`). skip_active is set per-forward by the wrapper and
        # read by the block wrappers; _residual is the cached transformer-block delta.
        self.skip_active: bool = False
        self._skipping: bool = False
        self._residual: torch.Tensor | None = None
        self._resid_h_in: torch.Tensor | None = None
        # Telemetry (read by the bench / node for the honest speedup report).
        self.computed_steps: int = 0
        self.skipped_steps: int = 0
        self.gate_sqnr_db: float | None = None

    # -- generation lifecycle -------------------------------------------------
    def _maybe_reset_generation(self, transformer_options: dict):
        """Detect the start of a new sampling run from the sigma schedule and reset all
        stream caches (ComfyUI puts the full schedule in ``sample_sigmas`` and the current
        sigma in ``sigmas``; a fresh run begins when they coincide at the first entry)."""
        sched = transformer_options.get("sample_sigmas")
        cur = transformer_options.get("sigmas")
        if sched is None or cur is None:
            return
        try:
            sigma0 = float(sched.flatten()[0])
            cur0 = float(cur.flatten()[0])
        except Exception:
            return
        is_first = abs(cur0 - sigma0) < 1e-9
        if is_first and (self._gen_sigma0 is None or self.step_index > 0):
            for st in self._streams.values():
                st.reset()
            self.step_index = 0
            self.computed_steps = 0
            self.skipped_steps = 0
            self.caching_disabled = False
            self._sqnr_checked = False
            self.gate_sqnr_db = None
            self._residual = None
            self._resid_h_in = None
            self._skipping = False
        self._gen_sigma0 = sigma0

    def _stream(self, key) -> _StreamState:
        st = self._streams.get(key)
        if st is None:
            st = _StreamState()
            self._streams[key] = st
        return st

    def reset(self):
        for st in self._streams.values():
            st.reset()
        self.step_index = 0
        self.computed_steps = 0
        self.skipped_steps = 0
        self.caching_disabled = False
        self._sqnr_checked = False
        self.gate_sqnr_db = None
        self._residual = None
        self._resid_h_in = None
        self._skipping = False

    # -- the core decision ----------------------------------------------------
    def _metric(self, rel: float, st: _StreamState) -> float:
        rel = _poly_rescale(rel, self.poly_coeffs)
        if self.strategy == "taylor":
            return st.sensitivity * rel
        return rel

    def should_skip(self, x: torch.Tensor, transformer_options: dict):
        """Decide whether this DiT step can reuse the cached output. Returns
        ``(skip: bool, stream_state)``. On a non-skip the caller must stash the fresh
        output via `record_output` so future steps can reuse it and update the gate."""
        self._maybe_reset_generation(transformer_options)
        key = tuple(transformer_options.get("cond_or_uncond", (0,)))
        st = self._stream(key)
        signal = x

        # "Do we have something cached to reuse?" — the cached whole-output (output mode) or
        # the cached block residual (residual mode).
        if self.mode == "residual":
            have_cache = self._residual is not None
        else:
            have_cache = (st.cached_output is not None
                          and st.cached_output.shape == x.shape)

        skip = False
        if self.caching_disabled:
            skip = False
        elif not have_cache or st.prev_signal is None:
            skip = False  # nothing to reuse yet
        elif self.step_index < self.warmup_steps:
            skip = False
        else:
            rel = _rel_l1(signal, st.prev_signal)
            st.accumulated += self._metric(rel, st)
            if (st.accumulated < self.rel_l1_thresh
                    and st.consecutive_skips < self.max_consecutive_skips):
                skip = True

        # prev_signal advances every step (skipped or not) so the accumulator always
        # measures against the most recent input.
        st.prev_signal = signal.detach()
        return skip, st

    def record_output(self, st: _StreamState, output: torch.Tensor, signal: torch.Tensor):
        """Called after a real DiT compute with the step's input ``signal`` and the fresh
        ``output``: run the SQNR gate, update the TaylorSeer sensitivity, cache the output,
        and reset the accumulator."""
        out = output.detach()
        # SQNR spot-check gate: the first time we can compare a fresh compute against the
        # previous cached output, verify they are close enough to cache. If not, the model's
        # step-to-step outputs diverge too fast — disable caching for the rest of the run.
        if (not self._sqnr_checked and st.cached_output is not None
                and out.shape == st.cached_output.shape):
            sqnr_db = cache_sqnr(st.cached_output, out)
            self.gate_sqnr_db = sqnr_db
            self._sqnr_checked = True
            if sqnr_db < self.sqnr_floor_db:
                self.caching_disabled = True
        # TaylorSeer sensitivity update: dy/dx = (output change) / (input change) measured
        # between the two most recent COMPUTED steps (st.cached_output / st.last_computed_signal
        # still hold the PREVIOUS compute here). sensitivity>1 => the model amplifies input
        # changes (be more conservative); <1 => it damps them (skip more freely).
        if (self.strategy == "taylor" and st.cached_output is not None
                and st.last_computed_signal is not None
                and out.shape == st.cached_output.shape
                and signal.shape == st.last_computed_signal.shape):
            dx = _rel_l1(signal, st.last_computed_signal)
            dy = _rel_l1(out, st.cached_output)
            if dx > 1e-10:
                instant = dy / dx
                st.sensitivity = (self._sensitivity_decay * st.sensitivity
                                  + (1.0 - self._sensitivity_decay) * instant)
        st.cached_output = out
        st.last_computed_signal = signal.detach()
        st.accumulated = 0.0
        st.consecutive_skips = 0

    def note_step(self, skipped: bool, st: _StreamState):
        if skipped:
            self.skipped_steps += 1
            st.consecutive_skips += 1
        else:
            self.computed_steps += 1

    def stats(self) -> dict:
        total = self.computed_steps + self.skipped_steps
        return {
            "computed": self.computed_steps,
            "skipped": self.skipped_steps,
            "total": total,
            "skip_frac": (self.skipped_steps / total) if total else 0.0,
            "rel_l1_thresh": self.rel_l1_thresh,
            "strategy": self.strategy,
            "caching_disabled": self.caching_disabled,
            "gate_sqnr_db": self.gate_sqnr_db,
        }


def _poly_rescale(x: float, coeffs) -> float:
    """Optional per-model polynomial rescale ``c0 + c1*x + ...`` of the raw rel-L1
    (TeaCache's calibrated signal->output-change map). ``coeffs=None`` => identity."""
    if not coeffs:
        return x
    y = 0.0
    for c in reversed(coeffs):  # Horner
        y = y * x + c
    return y


def make_teacache_wrapper(controller: TeaCacheController, dit=None):
    """Build a ComfyUI ``WrappersMP.DIFFUSION_MODEL`` wrapper bound to ``controller``.

    In residual mode, ``dit`` is the single-stream diffusion model whose block ModuleList is
    wrapped for residual caching; the wrappers are installed/restored AROUND each ``_forward``
    call so the mutation never leaks to other clones sharing the module.

    Registered on a ModelPatcher via ``add_wrapper_with_key`` (see `nodes.FNI8TeaCache`);
    ComfyUI merges it into ``transformer_options["wrappers"]`` at sample time and every DiT
    invokes it as ``wrapper(executor, x, timesteps, *rest, **kwargs)`` around its
    ``_forward`` (comfy.patcher_extension.WrapperExecutor). The wrapper reads the latent
    input ``x`` (args[0]) and the sampler context from ``transformer_options`` to decide
    whether to run ``executor`` (real DiT compute) or return the cached output."""

    def teacache_wrapper(executor, *args, **kwargs):
        to = kwargs.get("transformer_options")
        if to is None:
            for a in args:
                if isinstance(a, dict) and ("wrappers" in a or "sample_sigmas" in a):
                    to = a
                    break
        if to is None or not args or not isinstance(args[0], torch.Tensor):
            return executor(*args, **kwargs)

        x = args[0]
        skip, st = controller.should_skip(x, to)
        is_cond_stream = tuple(to.get("cond_or_uncond", (0,)))[0] == 0

        if controller.mode == "residual":
            # The block wrappers implement the skip: on a skip they reuse the cached
            # transformer-block residual but STILL run `_forward`'s cheap timestep-conditioned
            # prologue/epilogue at the current step — higher fidelity than reusing the whole
            # output. Install/restore the block wrappers around this single _forward so the
            # shared module is never left patched for other clones.
            target = dit if dit is not None else getattr(executor, "class_obj", None)
            blocks = find_block_list(target) if target is not None else None
            controller.skip_active = skip
            restore = install_residual_block_cache(blocks, controller) if blocks else None
            try:
                out = executor(*args, **kwargs)
            finally:
                if restore is not None:
                    restore()
            controller.note_step(skip, st)
            if is_cond_stream:
                controller.step_index += 1
            return out

        # output mode: reuse the whole cached prediction (arch-agnostic).
        if skip:
            controller.note_step(True, st)
            if is_cond_stream:
                controller.step_index += 1
            return st.cached_output.to(dtype=x.dtype, device=x.device)
        out = executor(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            controller.record_output(st, out, x)
        controller.note_step(False, st)
        if is_cond_stream:
            controller.step_index += 1
        return out

    return teacache_wrapper


def find_block_list(dit):
    """Return the DiT's transformer-block ``nn.ModuleList`` if it is a single-stream stack
    whose blocks take and return one hidden tensor (Lumina / Z-Image ``.layers``), else None.
    Residual caching only supports this shape today (double-stream Qwen and audio+video LTX
    thread two/three tensors and need per-stream residuals — a follow-up)."""
    layers = getattr(dit, "layers", None)
    try:
        import torch.nn as nn
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    except Exception:
        pass
    return None


def install_residual_block_cache(blocks, controller):
    """Wrap each block's ``forward`` in ``blocks`` (a single-stream ModuleList) so the
    transformer-block *residual* is cached and reused on skipped steps, while the DiT's own
    ``_forward`` still runs the timestep-conditioned input/output projections every step.

    On a COMPUTED step: block 0 records the stack input ``h_in``; the last block records
    ``residual = h_out - h_in``. On a SKIPPED step (``controller.skip_active``): block 0
    returns ``h_in + residual`` and every later block is the identity, so the stack output is
    ``h_in + residual`` without running any block. Returns a callable that restores the
    original forwards."""
    n = len(blocks)
    originals = [blk.forward for blk in blocks]

    def make(i, orig):
        def fwd(*args, **kwargs):
            h = args[0]
            if i == 0:
                controller._resid_h_in = h
                controller._skipping = (controller.skip_active
                                        and controller._residual is not None
                                        and not controller.caching_disabled)
            if controller._skipping:
                if i == 0:
                    return h + controller._residual.to(dtype=h.dtype, device=h.device)
                return h  # identity — do not run the block
            out = orig(*args, **kwargs)
            if i == n - 1 and isinstance(out, torch.Tensor) \
                    and isinstance(controller._resid_h_in, torch.Tensor) \
                    and out.shape == controller._resid_h_in.shape:
                controller._residual = (out - controller._resid_h_in).detach()
            return out
        return fwd

    for i, blk in enumerate(blocks):
        blk.forward = make(i, originals[i])

    def restore():
        for blk, orig in zip(blocks, originals):
            blk.forward = orig

    return restore
