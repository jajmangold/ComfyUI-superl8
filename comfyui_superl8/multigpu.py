# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU strategy for the PCIe-1.0-x1 fleet (~250 MB/s, no NVLink).

The verdict is the SAME as fni8's LLM transport verdict: anything that communicates
per-layer dies; anything per-step / per-handoff survives. Encoded here as a ranked
policy the distribution nodes consult, plus a pure device-placement planner.

  ADOPT   cfg_parallel      cond/uncond on 2 GPUs, 1 latent AllGather/step (~free 2x)
  ADOPT   component_parallel text-encoder / DiT / VAE on separate GPUs; ZERO loop comm
  ADOPT   pipefusion        pipeline stages, stage-boundary patch activations only,
                            displaced-patch/stale-KV so stages never stall (const in L)
  ADOPT   pipeline_parallel DiT transformer blocks split across GPUs; the hidden state
                            is compressed with the fni8 transport codec at each stage
                            boundary (one handoff/step, not per-layer). A memory-FIT
                            strategy — it makes a DiT that exceeds one card runnable, not
                            a speedup (stages serialise, each step pays one codec handoff).
  AVOID   usp               Ulysses/Ring sequence-parallel — per-attn-layer all-to-all
  AVOID   tensor_parallel   per-layer all-reduce (dead, same as LLM TP)
  FIT-ONLY block_offload    DisTorch-style weight streaming: bandwidth-bound; ~1GB
                            shard ≈ 4s/step at 250 MB/s. Never a distribution strategy.
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch

STRATEGIES = {
    "cfg_parallel": "adopt",
    "component_parallel": "adopt",
    "pipefusion": "adopt",
    "pipeline_parallel": "adopt",
    "usp": "avoid",
    "tensor_parallel": "avoid",
    "block_offload": "fit_only",
}


@dataclass
class Placement:
    """Component-parallel plan: which device holds each model component. Zero
    denoise-loop traffic — only 2-3 one-time handoffs per generation."""

    text_encoder: str = "cuda:0"
    dit: str = "cuda:0"
    vae: str = "cuda:0"


@dataclass(frozen=True)
class LTXComponentPlacement:
    """Resident LTX stages separated at small latent boundaries."""

    dit: str = "cuda:0"
    upscaler: str = "cuda:0"
    video_vae: str = "cuda:0"
    audio_vae: str = "cuda:0"


@dataclass
class CFGParallelPlan:
    """CFG-parallel plan: which device handles the cond and uncond branches."""

    cond_device: str = "cuda:0"
    uncond_device: str = "cuda:1"


def available_devices() -> list[str]:
    """Return list of available CUDA devices (e.g. ``['cuda:0', 'cuda:1']``)."""
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def plan_components(devices: list[str]) -> Placement:
    """Place text-encoder / DiT / VAE across the available devices. DiT gets the
    primary compute GPU; encoder and VAE spill to others to cut peak VRAM."""
    if not devices:
        return Placement()
    dit = devices[0]
    enc = devices[1 % len(devices)] if len(devices) > 1 else dit
    vae = devices[2 % len(devices)] if len(devices) > 2 else enc
    return Placement(text_encoder=enc, dit=dit, vae=vae)


def plan_ltx_components(devices: list[str]) -> LTXComponentPlacement:
    """Assign serial LTX stages while keeping every heavyweight component resident.

    Additional cards peel stages off the DiT in priority order. With two cards all
    post-DiT work shares the second card; four cards isolate the upscalers and both
    decoders so video/audio decode can overlap the next window's denoise.
    """
    if not devices:
        devices = ["cuda:0"]
    dit = devices[0]
    post = devices[1] if len(devices) > 1 else dit
    return LTXComponentPlacement(
        dit=dit,
        upscaler=post,
        video_vae=devices[2] if len(devices) > 2 else post,
        audio_vae=devices[3] if len(devices) > 3 else post,
    )


def plan_cfg_parallel(devices: list[str]) -> CFGParallelPlan:
    """Plan CFG-parallel device assignment. Uncond gets a secondary GPU when
    available; both land on the primary when only one device exists."""
    if not devices:
        # No CUDA visible → no parallelism possible; collapse both branches onto
        # the single logical device so callers fall back to single-GPU execution.
        return CFGParallelPlan(cond_device="cuda:0", uncond_device="cuda:0")
    cond = devices[0]
    uncond = devices[1 % len(devices)] if len(devices) > 1 else cond
    return CFGParallelPlan(cond_device=cond, uncond_device=uncond)


def is_link_tolerant(strategy: str) -> bool:
    """True if the strategy survives a 250 MB/s link (per-step/handoff, not per-layer)."""
    return STRATEGIES.get(strategy) in ("adopt",)


def build_cfg_parallel_replica(diffusion_model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Deep-copy a diffusion model and move it to ``device``.

    This is a one-time copy at graph-build time (not per-step). The replica shares
    zero tensor storage with the original, so both can run concurrently on separate
    GPUs without device-conflict issues."""
    replica = copy.deepcopy(diffusion_model).to(device)
    replica.requires_grad_(False)
    replica.eval()
    return replica


def _move_cond(obj: Any, device: torch.device | str) -> Any:
    """Move tensors inside a conditioning dict/list to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_cond(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_cond(v, device) for v in obj]
    return obj


def make_cfg_parallel_wrapper(
    primary_model: Any,
    replica_dit: torch.nn.Module,
    plan: CFGParallelPlan,
) -> Callable:
    """Build a ``model_options['model_function_wrapper']`` hook that dispatches the
    batched cond+uncond forward across two model replicas on separate devices.

    ``primary_model`` is the ComfyUI ``ModelPatcher`` instance for the cond branch.
    ``replica_dit`` is a deep-copied diffusion model on the uncond device.

    The wrapper:
      1. Splits the batch by ``cond_or_uncond`` into cond and uncond halves.
      2. Runs the cond half on the primary device via ``primary_model.apply_model``.
      3. Runs the uncond half on the secondary device via the replica's forward,
         with conditioning tensors moved to that device.
      4. Merges the results back into the original batch order.

    Both branches run concurrently in Python threads. The cond path goes through
    ComfyUI's full ``apply_model`` infrastructure; the uncond path bypasses it and
    calls the diffusion model directly (safe since the replica is a bare module with
    no patcher hooks or device-offload logic).

    When only one unique value of ``cond_or_uncond`` is present (cfg=1, single-branch
    sampling, or only one GPU available) the wrapper passes through to
    ``primary_model.apply_model`` with no threading overhead."""

    replica_dit = replica_dit
    cond_device = plan.cond_device
    uncond_device = plan.uncond_device

    def _uncond_forward(x: torch.Tensor, t: Any, c: list) -> torch.Tensor:
        """Run the replica diffusion model for the uncond branch."""
        x = x.to(uncond_device)
        t = t.to(uncond_device) if torch.is_tensor(t) else t
        c_moved = [_move_cond(entry, uncond_device) for entry in c]
        with torch.no_grad():
            return replica_dit(x, t, c_moved)

    def wrapper(func, kwargs):
        input_x = kwargs["input"]
        c = kwargs["c"]
        cond_or_uncond = kwargs["cond_or_uncond"]
        if not isinstance(cond_or_uncond, list):
            cond_or_uncond = list(cond_or_uncond)

        unique = set(cond_or_uncond)
        if len(unique) <= 1:
            return primary_model.apply_model(
                input_x, kwargs["timestep"], c, cond_or_uncond=cond_or_uncond
            )

        cond_idx = [i for i, v in enumerate(cond_or_uncond) if v == 0]
        uncond_idx = [i for i, v in enumerate(cond_or_uncond) if v == 1]
        if not cond_idx or not uncond_idx:
            return primary_model.apply_model(
                input_x, kwargs["timestep"], c, cond_or_uncond=cond_or_uncond
            )

        timestep = kwargs["timestep"]
        results = [None, None]
        errors = [None, None]

        def _run_cond():
            try:
                x_d = input_x[cond_idx].to(cond_device)
                c_d = [c[i] for i in cond_idx] if isinstance(c, list) else c
                results[0] = primary_model.apply_model(x_d, timestep, c_d, cond_or_uncond=[0])
            except Exception as e:
                errors[0] = e

        def _run_uncond():
            try:
                x_d = input_x[uncond_idx]
                c_d = [c[i] for i in uncond_idx] if isinstance(c, list) else c
                results[1] = _uncond_forward(x_d, timestep, c_d)
            except Exception as e:
                errors[1] = e

        t1 = threading.Thread(target=_run_cond)
        t2 = threading.Thread(target=_run_uncond)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if errors[0] is not None:
            raise errors[0]
        if errors[1] is not None:
            raise errors[1]

        ref = results[0]
        output = torch.zeros(
            len(cond_or_uncond),
            *ref.shape[1:],
            dtype=ref.dtype,
            device=ref.device,
        )
        output[cond_idx] = results[0]
        output[uncond_idx] = results[1].to(ref.device)
        return output

    return wrapper


# ---------------------------------------------------------------------------
# Pipeline-parallel: transport codec + block split across GPUs
#
# The DiT is too big for one 16 GB card (LTX-2.3: 23.5 GB int8). Split its
# transformer blocks into contiguous stages, one stage per GPU, and hand the
# boundary hidden state across the PCIe-1.0-x1 link COMPRESSED via the fni8
# transport codec (`superl8.transport`, the same int8/int4 wire codec the serve side
# uses). One codec handoff per stage boundary per denoise step — never per-layer.
# ---------------------------------------------------------------------------


@dataclass
class TransportStats:
    """One cross-GPU hidden-state handoff, measured. `raw_bytes` is what an
    uncompressed fp move would cost; `wire_bytes` is what the codec actually sent."""

    scheme: str
    src: str
    dst: str
    raw_bytes: int
    wire_bytes: int
    compress_ms: float
    move_ms: float
    decompress_ms: float
    cos: float = float("nan")
    sqnr_db: float = float("nan")

    @property
    def ratio(self) -> float:
        """Compression ratio (raw / wire); 1.0 when the codec is unavailable."""
        return self.raw_bytes / self.wire_bytes if self.wire_bytes else 1.0

    @property
    def link_ms(self) -> float:
        """Modelled time to push `wire_bytes` over the 250 MB/s PCIe-1.0-x1 link."""
        return effective_transfer_ms(self.wire_bytes)

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "src": self.src,
            "dst": self.dst,
            "raw_bytes": self.raw_bytes,
            "wire_bytes": self.wire_bytes,
            "ratio": round(self.ratio, 3),
            "cos": round(self.cos, 6),
            "sqnr_db": round(self.sqnr_db, 2),
            "compress_ms": round(self.compress_ms, 3),
            "move_ms": round(self.move_ms, 3),
            "decompress_ms": round(self.decompress_ms, 3),
            "link_ms_modelled": round(self.link_ms, 3),
        }


def effective_transfer_ms(nbytes: int, link_bytes_per_s: float = 250_000_000.0) -> float:
    """Modelled wall time to move `nbytes` over the fleet's 250 MB/s PCIe-1.0-x1 link.
    Mirrors `superl8.transport.effective_transfer_ms` (used as the fallback if the fni8
    package is unavailable, so the planner stays importable without it)."""
    try:
        from superl8.transport import effective_transfer_ms as _f

        return _f(nbytes, link_bytes_per_s=link_bytes_per_s)
    except Exception:
        return 1000.0 * nbytes / link_bytes_per_s


class TransportCodec:
    """Compress/move/decompress a hidden-state tensor across the GPU boundary using
    the real fni8 transport codec (`superl8.transport.compress_activation`).

    `transport(x, dst)` compresses `x` on its own device, moves ONLY the packed
    codes + per-row scales across the link (that is the whole point on a 250 MB/s
    link), and decompresses on `dst`. Returns `(reconstructed_on_dst, TransportStats)`.

    Schemes: ``int8`` (2x vs fp16, the default — matches the DiT's own weight codec),
    ``int4`` / ``nf4`` (~4x, more loss), ``fp16`` (lossless baseline). When
    `superl8.transport` is unavailable the codec degrades to a plain `.to(dst)` device
    move (ratio 1.0) so the split still runs — the transport is then uncompressed,
    which the stats make explicit."""

    def __init__(
        self, scheme: str = "int8", group_size: int | None = None, report_quality: bool = True
    ):
        self.scheme = scheme
        self.group_size = group_size
        self.report_quality = report_quality

    @staticmethod
    def available() -> bool:
        try:
            import superl8.transport  # noqa: F401

            return True
        except Exception:
            return False

    def transport(self, x: torch.Tensor, dst: str | torch.device):
        src = str(x.device)
        dst = str(dst)
        if not self.available():
            return self._plain_move(x, src, dst)

        from superl8.transport import (
            Compressed,
            compress_activation,
            decompress_activation,
            reconstruction_report,
        )

        t0 = _now()
        c = compress_activation(x, scheme=self.scheme, group_size=self.group_size)
        _sync(x.device)
        t1 = _now()
        # Move ONLY the wire payload (codes + scales) across the link — not the
        # full-precision tensor. This is the compressed cross-GPU handoff.
        c_dst = Compressed(
            c.scheme,
            c.shape,
            c.dtype,
            c.group_size,
            c.payload.to(dst, non_blocking=True),
            c.scales.to(dst, non_blocking=True) if c.scales.numel() else c.scales,
            c.d,
        )
        _sync(dst)
        t2 = _now()
        y = decompress_activation(c_dst)
        _sync(dst)
        t3 = _now()

        cos = float("nan")
        sqnr = float("nan")
        if self.report_quality:
            rep = reconstruction_report(x, c)
            cos = float(rep.get("cos", float("nan")))
            xr = decompress_activation(c).float()
            xf = x.float()
            noise = (xr - xf).pow(2).sum().item()
            sqnr = 10.0 * math.log10(xf.pow(2).sum().item() / noise) if noise > 0 else float("inf")
        stats = TransportStats(
            scheme=self.scheme,
            src=src,
            dst=dst,
            raw_bytes=x.numel() * x.element_size(),
            wire_bytes=int(c.on_wire_bytes),
            compress_ms=(t1 - t0) * 1e3,
            move_ms=(t2 - t1) * 1e3,
            decompress_ms=(t3 - t2) * 1e3,
            cos=cos,
            sqnr_db=sqnr,
        )
        return y, stats

    def _plain_move(self, x, src, dst):
        t0 = _now()
        y = x.to(dst, non_blocking=True)
        _sync(dst)
        t1 = _now()
        nb = x.numel() * x.element_size()
        return y, TransportStats(
            scheme="none",
            src=src,
            dst=dst,
            raw_bytes=nb,
            wire_bytes=nb,
            compress_ms=0.0,
            move_ms=(t1 - t0) * 1e3,
            decompress_ms=0.0,
            cos=1.0,
            sqnr_db=float("inf"),
        )


def _now() -> float:
    import time

    return time.perf_counter()


def _sync(device) -> None:
    if torch.cuda.is_available():
        d = torch.device(device)
        if d.type == "cuda":
            torch.cuda.synchronize(d)


@dataclass
class PipelinePlan:
    """How a DiT's transformer blocks are split across devices. Each stage owns a
    contiguous block range `(first_block, num_blocks)` on `devices[stage]`. The
    hidden state flows stage 0 → 1 → … → N-1 and (for a wrap-back split) home again;
    one codec handoff per boundary per step. `transport_scheme` names the codec."""

    devices: list[str]
    stage_ranges: list[tuple[int, int]]
    transport_scheme: str = "int8"

    @property
    def num_stages(self) -> int:
        return len(self.stage_ranges)

    def stage_of_block(self, block_idx: int) -> int:
        for s, (start, count) in enumerate(self.stage_ranges):
            if start <= block_idx < start + count:
                return s
        return self.num_stages - 1


def plan_pipeline(
    num_blocks: int,
    devices: list[str] | None = None,
    transport_scheme: str = "int8",
    head_weight_blocks: float = 0.0,
) -> PipelinePlan:
    """Split `num_blocks` transformer blocks across `devices` as evenly as possible.

    `head_weight_blocks` biases the FIRST stage to own FEWER blocks, accounting for
    non-block weight it also carries (embeddings / connectors / prologue / output all
    live on stage 0 in the LTX split). Expressed in block-equivalents: e.g. 12 means
    "stage 0 already carries ~12 blocks' worth of other weight, so give it 12 fewer".

    Falls back to a 2-device `cuda:0`/`cuda:1` plan when fewer than 2 devices resolve."""
    if devices is None:
        n = max(2, torch.cuda.device_count() if torch.cuda.is_available() else 2)
        devices = [f"cuda:{i}" for i in range(n)]
    else:
        devices = list(devices)
    if len(devices) < 2:
        devices = ["cuda:0", "cuda:1"]

    n_stages = len(devices)
    # Even base split, then move `head_weight_blocks/2` blocks off stage 0 onto the
    # others so the physical VRAM per card balances once the non-block weight is added.
    base = [
        num_blocks // n_stages + (1 if i < num_blocks % n_stages else 0) for i in range(n_stages)
    ]
    shift = int(round(head_weight_blocks / 2.0))
    if n_stages == 2 and 0 < shift < base[0]:
        base[0] -= shift
        base[1] += shift
    stage_ranges: list[tuple[int, int]] = []
    start = 0
    for count in base:
        stage_ranges.append((start, count))
        start += count
    return PipelinePlan(devices, stage_ranges, transport_scheme=transport_scheme)


def count_dit_blocks(model) -> int:
    """Count a diffusion model's transformer blocks by inspecting its submodules /
    state-dict keys. Handles LTX/Wan (`transformer_blocks.N` / `blocks.N`), MMDiT
    (`double_blocks.N` + `single_blocks.N`, e.g. Flux) and UNet (`input/output_blocks`)."""
    import re

    tb = getattr(model, "transformer_blocks", None)
    if tb is not None:
        try:
            return len(tb)
        except TypeError:
            pass
    sd = model.state_dict() if hasattr(model, "state_dict") else {}
    keys = list(sd.keys())
    for pat in (r"transformer_blocks\.(\d+)", r"^blocks\.(\d+)"):
        idx = {int(m.group(1)) for k in keys for m in [re.search(pat, k)] if m}
        if idx:
            return max(idx) + 1
    double = {int(m.group(1)) for k in keys for m in [re.search(r"double_blocks\.(\d+)", k)] if m}
    single = {int(m.group(1)) for k in keys for m in [re.search(r"single_blocks\.(\d+)", k)] if m}
    if double or single:
        return len(double) + len(single)
    inp = {int(m.group(1)) for k in keys for m in [re.search(r"input_blocks\.(\d+)", k)] if m}
    out = {int(m.group(1)) for k in keys for m in [re.search(r"output_blocks\.(\d+)", k)] if m}
    total = len(inp) + len(out) + (1 if any("middle_block." in k for k in keys) else 0)
    return max(total, 1)


def move_module_to_device(module: torch.nn.Module, device: str | torch.device) -> None:
    """Move every parameter and buffer of `module` to `device` IN PLACE, preserving
    tensor subclasses (int8 `FNI8Tensor` weights must stay int8 — a plain
    `module.to(device)` routes through `nn.Module._apply`, which can strip the subclass
    and its per-row scale). Reassigns `_parameters` / `_buffers` directly, the same
    escape hatch `loader.assign_int8_weights` uses."""
    device = torch.device(device)
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is not None and p.device != device:
                sub._parameters[name] = p.to(device=device)
        for name, b in list(sub._buffers.items()):
            if b is not None and b.device != device:
                sub._buffers[name] = b.to(device=device)


def module_vram_bytes(module: torch.nn.Module) -> int:
    """Resident byte footprint of a module's parameters + buffers (int8 codes count as
    1 byte/elt; the fp32 per-row scale is added for `FNI8Tensor` weights)."""
    total = 0
    seen: set[int] = set()
    for t in list(module.parameters()) + list(module.buffers()):
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
        # NB: `torch.Tensor.q_scale` is a builtin METHOD on every tensor, so a plain
        # `getattr(t, "q_scale", None)` returns that method, never None — only an
        # FNI8Tensor shadows it with a real scale tensor (set in __init__).
        sc = getattr(t, "q_scale", None)
        if isinstance(sc, torch.Tensor):
            total += sc.numel() * sc.element_size()
    return total


def component_parallel_apply(model, placement: Placement) -> None:
    """Move a ComfyUI model's sub-modules to their assigned devices per the
    ``Placement`` plan. Operates in-place on the loaded model components.

    The DiT's ``diffusion_model`` is moved to its assigned device; the ModelPatcher's
    ``load_device`` attribute is updated so ComfyUI doesn't try to move it back.

    This is a one-time handoff — there is zero denoise-loop traffic between devices."""
    dit = getattr(model, "diffusion_model", None) or getattr(model, "model", None)
    if dit is not None and hasattr(dit, "to"):
        dit.to(placement.dit)
    if hasattr(model, "load_device"):
        model.load_device = torch.device(placement.dit)
