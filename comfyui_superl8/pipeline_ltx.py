# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel execution for the LTX-2 audio+video DiT (`comfy.ldm.lightricks.
av_model.LTXAVModel`), the one published `.fni8` whose int8 DiT (23.5 GB) exceeds a
single 16 GB card.

The split is physical: the 48 `transformer_blocks` are distributed contiguously across
GPUs (stage 0 → 1 → …); the prologue (`_process_input` / `_prepare_*`), the
embeddings connectors and the output head all stay on stage 0. At each denoise step the
per-block hidden state `(vx, ax)` — the ONLY tensor that evolves block-to-block — is
handed across the GPU boundary COMPRESSED with the fni8 transport codec
(`multigpu.TransportCodec` over `superl8.transport`); the constant sidecar tensors
(context, positional embeddings, timesteps, masks) are moved once per boundary.

This makes LTX-2.3 RUNNABLE across 2 cards (a memory-fit strategy, not a speedup:
stages serialise and each step pays one codec handoff over the 250 MB/s link).
"""
from __future__ import annotations

import types
from typing import Any

import torch

from .multigpu import (
    PipelinePlan,
    TransportCodec,
    move_module_to_device,
)


def _to_dev(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors inside tuples/lists/dicts to `device`; pass the rest
    through untouched. Used for the constant sidecar tensors at a stage boundary."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, tuple):
        return tuple(_to_dev(v, device) for v in obj)
    if isinstance(obj, list):
        return [_to_dev(v, device) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dev(v, device) for k, v in obj.items()}
    return obj


def distribute_ltxav(dit: torch.nn.Module, plan: PipelinePlan) -> dict:
    """Physically place an LTXAVModel across the plan's devices IN PLACE.

    Stage-0 device holds everything EXCEPT the blocks assigned to later stages: the
    prologue, all embedding / connector / adaLN modules, the output head, and stage-0's
    own block range. Every later stage holds only its contiguous block range.

    Returns a per-device resident-byte report (measured after placement)."""
    from .multigpu import module_vram_bytes

    blocks = dit.transformer_blocks
    dev0 = torch.device(plan.devices[0])

    # Move the whole model to stage-0 device first (prologue/embeds/connectors/output),
    # then relocate each block to its stage device. Blocks dominate VRAM, so this never
    # materialises the full 23.5 GB on one card as long as the caller placed the model
    # on CPU before calling (see the loader's keep_on_cpu path).
    for i, block in enumerate(blocks):
        stage = plan.stage_of_block(i)
        move_module_to_device(block, plan.devices[stage])
    # Everything that is not a transformer block goes to stage 0.
    for name, child in dit.named_children():
        if name == "transformer_blocks":
            continue
        move_module_to_device(child, dev0)
    # top-level parameters/buffers (scale_shift_table, etc.) -> stage 0
    for pname, p in list(dit._parameters.items()):
        if p is not None:
            dit._parameters[pname] = p.to(device=dev0)
    for bname, b in list(dit._buffers.items()):
        if b is not None:
            dit._buffers[bname] = b.to(device=dev0)

    report: dict[str, int] = {}
    for d in plan.devices:
        report[d] = 0
    report[str(dev0)] += sum(
        module_vram_bytes(c) for n, c in dit.named_children()
        if n != "transformer_blocks"
    )
    for i, block in enumerate(blocks):
        d = plan.devices[plan.stage_of_block(i)]
        report[d] += module_vram_bytes(block)
    return report


def _make_split_process_blocks(plan: PipelinePlan, codec: TransportCodec):
    """Build the instance-bound replacement for `LTXAVModel._process_transformer_blocks`
    that runs each contiguous block range on its stage device, handing `(vx, ax)` across
    each boundary through the transport codec and moving the constant sidecars once."""

    def _process_transformer_blocks(
        self, x, context, attention_mask, timestep, pe,
        transformer_options={}, self_attention_mask=None, **kwargs,
    ):
        vx = x[0]
        ax = x[1]
        v_context = context[0]
        a_context = context[1]
        v_timestep = timestep[0]
        a_timestep = timestep[1]
        v_pe, av_cross_video_freq_cis = pe[0]
        a_pe, av_cross_audio_freq_cis = pe[1]
        (
            av_ca_audio_scale_shift_timestep,
            av_ca_video_scale_shift_timestep,
            av_ca_a2v_gate_noise_timestep,
            av_ca_v2a_gate_noise_timestep,
        ) = timestep[2]
        v_prompt_timestep = timestep[3]
        a_prompt_timestep = timestep[4]

        # The constant per-step sidecars — moved as a set when we cross onto a new device.
        sidecars = {
            "v_context": v_context, "a_context": a_context,
            "attention_mask": attention_mask,
            "v_timestep": v_timestep, "a_timestep": a_timestep,
            "v_pe": v_pe, "a_pe": a_pe,
            "v_cross_pe": av_cross_video_freq_cis, "a_cross_pe": av_cross_audio_freq_cis,
            "v_cross_scale_shift_timestep": av_ca_video_scale_shift_timestep,
            "a_cross_scale_shift_timestep": av_ca_audio_scale_shift_timestep,
            "v_cross_gate_timestep": av_ca_a2v_gate_noise_timestep,
            "a_cross_gate_timestep": av_ca_v2a_gate_noise_timestep,
            "self_attention_mask": self_attention_mask,
            "v_prompt_timestep": v_prompt_timestep,
            "a_prompt_timestep": a_prompt_timestep,
        }

        home_device = torch.device(plan.devices[0])
        stats: list = []
        moved_to: dict[str, dict] = {str(home_device): sidecars}
        cur_stage = 0

        def _sidecars_on(device: torch.device) -> dict:
            key = str(device)
            if key not in moved_to:
                moved_to[key] = _to_dev(sidecars, device)
            return moved_to[key]

        for i, block in enumerate(self.transformer_blocks):
            stage = plan.stage_of_block(i)
            if stage != cur_stage:
                # ---- stage boundary: hand (vx, ax) across the link via the codec ----
                dst = torch.device(plan.devices[stage])
                vx, s_v = codec.transport(vx.contiguous(), dst)
                ax, s_a = codec.transport(ax.contiguous(), dst)
                stats.append(s_v)
                stats.append(s_a)
                cur_stage = stage
            sc = _sidecars_on(torch.device(plan.devices[stage]))
            vx, ax = block(
                (vx, ax),
                v_context=sc["v_context"], a_context=sc["a_context"],
                attention_mask=sc["attention_mask"],
                v_timestep=sc["v_timestep"], a_timestep=sc["a_timestep"],
                v_pe=sc["v_pe"], a_pe=sc["a_pe"],
                v_cross_pe=sc["v_cross_pe"], a_cross_pe=sc["a_cross_pe"],
                v_cross_scale_shift_timestep=sc["v_cross_scale_shift_timestep"],
                a_cross_scale_shift_timestep=sc["a_cross_scale_shift_timestep"],
                v_cross_gate_timestep=sc["v_cross_gate_timestep"],
                a_cross_gate_timestep=sc["a_cross_gate_timestep"],
                transformer_options=transformer_options,
                self_attention_mask=sc["self_attention_mask"],
                v_prompt_timestep=sc["v_prompt_timestep"],
                a_prompt_timestep=sc["a_prompt_timestep"],
            )

        # Bring the final hidden state home so the output head (on stage 0) can run.
        if cur_stage != 0:
            vx, s_v = codec.transport(vx.contiguous(), home_device)
            ax, s_a = codec.transport(ax.contiguous(), home_device)
            stats.append(s_v)
            stats.append(s_a)

        self._fni8_pipeline_last_stats = [s.as_dict() for s in stats]
        return [vx, ax]

    return _process_transformer_blocks


def install_ltxav_pipeline(dit: torch.nn.Module, plan: PipelinePlan,
                           codec: TransportCodec | None = None) -> dict:
    """Distribute an LTXAVModel across `plan.devices` and install the split-execution
    `_process_transformer_blocks`. Returns the per-device resident-byte report.

    After this call `dit._forward(...)` runs the blocks split across GPUs with a codec
    handoff at each stage boundary; `dit._fni8_pipeline_last_stats` holds the transport
    metrics from the most recent forward."""
    codec = codec or TransportCodec(scheme=plan.transport_scheme)
    report = distribute_ltxav(dit, plan)
    dit._process_transformer_blocks = types.MethodType(
        _make_split_process_blocks(plan, codec), dit
    )
    dit._fni8_pipeline_plan = plan
    dit._fni8_pipeline_last_stats = []
    return report
