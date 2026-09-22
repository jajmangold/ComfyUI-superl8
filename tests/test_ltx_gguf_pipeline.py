# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import wave

import pytest
import torch

from bench.full_pipeline_ltx_gguf import (
    OFFICIAL_DISTILLED_SIGMAS,
    STAGE2_DISTILLED_SIGMAS,
    audio_latents_for_video,
    configure_vae_residency,
    normalize_connector_state_dict,
    prepare_refinement_latent,
    project_gemma_hidden_states,
    refinement_frame_rate,
    refinement_sigmas,
    stage_devices,
    resolve_vae_dtype,
    save_wav_pcm16,
    temporal_upscaled_latent_frames,
    two_stage_latent_shapes,
    vae_chunk_bytes,
)
from bench.full_pipeline_ltx import build_latents


def test_normalize_connector_state_dict_strips_only_dit_prefix():
    state = {
        "model.diffusion_model.video_embeddings_connector.weight": torch.ones(1),
        "model.diffusion_model.audio_embeddings_connector.bias": torch.ones(1),
    }
    assert set(normalize_connector_state_dict(state)) == {
        "video_embeddings_connector.weight",
        "audio_embeddings_connector.bias",
    }


def test_project_gemma_hidden_states_matches_dual_linear_contract():
    hidden = torch.tensor([[[[3.0, 4.0]], [[0.0, 2.0]]]])  # [B,L=2,T=1,H=2]
    state = {
        "text_embedding_projection.video_aggregate_embed.weight": torch.ones(3, 4),
        "text_embedding_projection.video_aggregate_embed.bias": torch.zeros(3),
        "text_embedding_projection.audio_aggregate_embed.weight": torch.full((1, 4), 2.0),
        "text_embedding_projection.audio_aggregate_embed.bias": torch.ones(1),
    }

    got = project_gemma_hidden_states(hidden, state)

    x = hidden.movedim(1, -1)
    x = x * torch.rsqrt(x.pow(2).mean(dim=2, keepdim=True) + 1e-6)
    x = x.flatten(start_dim=2)
    video = torch.nn.functional.linear(
        x * (3 / 2) ** 0.5,
        state["text_embedding_projection.video_aggregate_embed.weight"],
        state["text_embedding_projection.video_aggregate_embed.bias"],
    )
    audio = torch.nn.functional.linear(
        x * (1 / 2) ** 0.5,
        state["text_embedding_projection.audio_aggregate_embed.weight"],
        state["text_embedding_projection.audio_aggregate_embed.bias"],
    )
    assert torch.allclose(got, torch.cat((video, audio), dim=-1))


def test_official_distilled_schedule_has_eight_steps():
    assert len(OFFICIAL_DISTILLED_SIGMAS) == 9
    assert OFFICIAL_DISTILLED_SIGMAS[0] == 1.0
    assert OFFICIAL_DISTILLED_SIGMAS[-1] == 0.0


def test_audio_latents_match_decoded_video_duration():
    assert audio_latents_for_video(latent_video_frames=9, frame_rate=25) == 65


def test_build_latents_accepts_rectangular_spatial_shape():
    class Dit:
        in_channels = 4
        num_audio_channels = 2
        audio_frequency_bins = 8

    video, audio = build_latents(
        Dit(), "cpu", torch.float32, (20, 12), frames=3, audio_frames=5, seed=7
    )

    assert video.shape == (1, 4, 3, 20, 12)
    assert audio.shape == (1, 2, 5, 8)


def test_two_stage_portrait_shapes_are_half_then_full_resolution():
    assert two_stage_latent_shapes(1024, 576) == ((16, 9), (32, 18))


def test_refinement_latent_matches_official_noise_blend():
    source = torch.full((1, 2, 1, 2, 2), 4.0)
    noise = torch.full_like(source, -2.0)

    refined = prepare_refinement_latent(source, noise, STAGE2_DISTILLED_SIGMAS[0])

    assert torch.equal(refined, torch.full_like(source, -1.1))


def test_refinement_presets_keep_the_official_start_and_endpoint():
    assert refinement_sigmas("coherent-fast") == (0.85, 0.725, 0.0)
    assert refinement_sigmas("official") == STAGE2_DISTILLED_SIGMAS


def test_temporal_x2_doubles_intervals_without_duplicating_first_frame():
    assert temporal_upscaled_latent_frames(5) == 9


def test_temporal_upscale_before_refine_uses_final_frame_rate():
    assert refinement_frame_rate(12, 24, temporal_before_refine=True) == 24
    assert refinement_frame_rate(12, 24, temporal_before_refine=False) == 12


def test_vae_dtype_names_resolve_to_supported_torch_dtypes():
    assert resolve_vae_dtype("auto") is None
    assert resolve_vae_dtype("bf16") is torch.bfloat16
    assert resolve_vae_dtype("fp32") is torch.float32


def test_vae_chunk_budget_uses_binary_mebibytes_and_rejects_nonpositive_values():
    assert vae_chunk_bytes(192) == 192 * 1024**2
    with pytest.raises(ValueError):
        vae_chunk_bytes(0)


def test_vae_residency_control_forces_or_restores_dynamic_weight_streaming():
    class VAE:
        disable_offload = False
        first_stage_model = object()
        device = torch.device("cuda")
        patcher = "dynamic"

    vae = VAE()
    configure_vae_residency(
        vae,
        resident=True,
        patcher_factory=lambda model, **kwargs: (model, kwargs),
        offload_device=torch.device("cpu"),
    )
    assert vae.disable_offload is True
    assert vae.patcher == (
        vae.first_stage_model,
        {"load_device": vae.device, "offload_device": torch.device("cpu")},
    )
    configure_vae_residency(vae, resident=False)
    assert vae.disable_offload is False


def test_vae_residency_can_target_a_dedicated_stage_device():
    class VAE:
        disable_offload = False
        first_stage_model = object()
        device = torch.device("cuda:0")

    vae = VAE()
    configure_vae_residency(
        vae,
        resident=True,
        load_device="cuda:2",
        patcher_factory=lambda model, **kwargs: kwargs,
        offload_device=torch.device("cpu"),
    )
    assert vae.device == torch.device("cuda:2")
    assert vae.patcher["load_device"] == torch.device("cuda:2")


def test_stage_devices_parse_explicit_logical_cuda_roles():
    assert stage_devices("cuda:0,cuda:1, cuda:2") == ["cuda:0", "cuda:1", "cuda:2"]
    with pytest.raises(ValueError):
        stage_devices("cuda:0,cpu")


def test_save_wav_pcm16_writes_stereo_header(tmp_path):
    path = tmp_path / "audio.wav"
    save_wav_pcm16(path, torch.zeros(2, 480), sample_rate=48_000)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 48_000
        assert handle.getnframes() == 480
