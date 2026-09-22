# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from bench.full_pipeline_ltx import flow_match_denoise, ltx_sigma_schedule


def test_ltx_sigma_schedule_matches_comfy_resolution_shift():
    got = ltx_sigma_schedule(steps=4, tokens=4096, device="cpu")
    base = torch.linspace(1.0, 0.0, 5)
    shift = 2.05
    es = torch.exp(torch.tensor(shift))
    expected = torch.where(base != 0, es / (es + (1.0 / base - 1.0)), base)

    assert torch.allclose(got, expected)
    assert got[0] == 1 and got[-1] == 0
    assert bool(torch.all(got[:-1] > got[1:]))


def test_flow_match_denoise_applies_cfg_and_explicit_sigmas():
    timesteps = []

    class _FakeDit:
        def forward(self, *, x, timestep, context, **kwargs):
            timesteps.append(timestep.item())
            value = context[:, :1, :1].reshape(1, 1, 1, 1, 1)
            video = torch.ones_like(x[0]) * value
            audio = torch.ones_like(x[1]) * value.flatten()[0]
            return [video, audio]

    vx = torch.zeros(1, 1, 1, 1, 1)
    ax = torch.zeros(1, 1, 1, 1)
    cond = torch.tensor([[[2.0]]])
    uncond = torch.tensor([[[1.0]]])
    sigmas = [1.0, 0.5, 0.0]

    video = flow_match_denoise(
        _FakeDit(),
        vx,
        ax,
        cond,
        steps=2,
        frame_rate=25,
        audio_length=1,
        device="cpu",
        uncond_context=uncond,
        cfg=3.0,
        sigmas=sigmas,
    )

    # v = 1 + 3*(2-1) = 4; total dt = -1.
    assert torch.equal(video, torch.full_like(video, -4.0))
    assert timesteps == [1.0, 1.0, 0.5, 0.5]


def test_flow_match_denoise_can_return_synchronized_audio_trajectory():
    class _FakeDit:
        def forward(self, *, x, **kwargs):
            return [torch.ones_like(x[0]), torch.full_like(x[1], 2.0)]

    video, audio = flow_match_denoise(
        _FakeDit(),
        torch.zeros(1, 1, 1, 1, 1),
        torch.zeros(1, 1, 2, 1),
        torch.zeros(1, 1, 1),
        steps=1,
        frame_rate=25,
        audio_length=2,
        device="cpu",
        sigmas=[1.0, 0.0],
        return_audio=True,
    )

    assert torch.equal(video, torch.full_like(video, -1.0))
    assert torch.equal(audio, torch.full_like(audio, -2.0))
