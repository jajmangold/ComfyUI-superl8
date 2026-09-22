# SPDX-License-Identifier: Apache-2.0
"""One-card LTX-2.3 distilled pipeline through stock ComfyUI-GGUF + fni8 DP4A.

The distilled GGUF contains the transformer but intentionally omits the one-shot Gemma
text projection and video/audio embedding connectors. Load those matching official
assets, build prompt context once, evict them, then place the native k-quant transformer
on the card. This avoids the old experiment's mismatched connector bundle and ad-hoc
per-forward dequant overrides.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import sys
import time
import wave

import torch

OFFICIAL_DISTILLED_SIGMAS = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)
STAGE2_DISTILLED_SIGMAS = (0.85, 0.725, 0.4219, 0.0)


def refinement_sigmas(preset: str) -> tuple[float, ...]:
    """Return the measured coherent-fast schedule or all three official evaluations."""
    return {
        "coherent-fast": (0.85, 0.725, 0.0),
        "official": STAGE2_DISTILLED_SIGMAS,
    }[preset]


def resolve_vae_dtype(name: str) -> torch.dtype | None:
    """Map the benchmark's explicit VAE precision dial to Comfy's constructor."""
    return {
        "auto": None,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def vae_chunk_bytes(mebibytes: int) -> int:
    """Convert the user-facing LTX decoder budget to bytes with basic validation."""
    if mebibytes <= 0:
        raise ValueError("VAE chunk budget must be positive")
    return mebibytes * 1024**2


def stage_devices(spec: str) -> list[str]:
    """Parse logical CUDA devices exposed to the staged pipeline process."""
    devices = [item.strip() for item in spec.split(",") if item.strip()]
    if not devices or any(not item.startswith("cuda:") for item in devices):
        raise ValueError("stage devices must be a comma-separated list of cuda:N devices")
    return devices


def configure_vae_residency(
    vae,
    *,
    resident: bool,
    patcher_factory=None,
    offload_device=None,
    load_device=None,
) -> None:
    """Keep all VAE weights on device instead of streaming them over PCIe per layer."""
    if load_device is not None:
        vae.device = torch.device(load_device)
    vae.disable_offload = resident
    if not resident:
        return
    if patcher_factory is None:
        import comfy.model_management
        import comfy.model_patcher

        patcher_factory = comfy.model_patcher.ModelPatcher
        offload_device = comfy.model_management.vae_offload_device()
    vae.patcher = patcher_factory(
        vae.first_stage_model,
        load_device=vae.device,
        offload_device=offload_device,
    )


def two_stage_latent_shapes(height: int, width: int):
    """Return Wan2GP-compatible half/full latent shapes for an x2 portrait run."""
    if height % 64 or width % 64:
        raise ValueError("two-stage height and width must be multiples of 64")
    return (height // 64, width // 64), (height // 32, width // 32)


def temporal_upscaled_latent_frames(frames: int) -> int:
    """LTX temporal x2 doubles frame intervals and keeps one shared endpoint."""
    if frames < 1:
        raise ValueError("latent frame count must be positive")
    return frames * 2 - 1


def refinement_frame_rate(
    conditioned_frame_rate: int,
    output_frame_rate: int,
    *,
    temporal_before_refine: bool,
) -> int:
    return output_frame_rate if temporal_before_refine else conditioned_frame_rate


def prepare_refinement_latent(
    source: torch.Tensor, noise: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Apply LTX's stage-two Gaussian noiser to an upscaled latent."""
    if source.shape != noise.shape:
        raise ValueError("source and refinement noise shapes must match")
    return noise * sigma + source * (1.0 - sigma)


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def audio_latents_for_video(
    latent_video_frames: int,
    frame_rate: int,
    *,
    sample_rate: int = 16_000,
    hop_length: int = 160,
    audio_downsample: int = 4,
) -> int:
    """Match Comfy's LTX audio duration to the video VAE's decoded frame count."""
    decoded_frames = (latent_video_frames - 1) * 8 + 1
    latents_per_second = sample_rate / hop_length / audio_downsample
    return math.ceil((decoded_frames / frame_rate) * latents_per_second)


def normalize_connector_state_dict(state: dict[str, torch.Tensor]):
    prefix = "model.diffusion_model."
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state.items()
    }


def project_gemma_hidden_states(
    hidden_states: torch.Tensor, projection_state: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Comfy LTX dual-linear contract: [B,L,T,H] -> raw [B,T,6144]."""
    prefix = "text_embedding_projection."
    vw = projection_state[prefix + "video_aggregate_embed.weight"].float()
    vb = projection_state[prefix + "video_aggregate_embed.bias"].float()
    aw = projection_state[prefix + "audio_aggregate_embed.weight"].float()
    ab = projection_state[prefix + "audio_aggregate_embed.bias"].float()
    x = hidden_states.float().movedim(1, -1)
    source_dim = x.shape[-2]
    x = x * torch.rsqrt(x.pow(2).mean(dim=2, keepdim=True) + 1e-6)
    x = x.flatten(start_dim=2)
    video = torch.nn.functional.linear(x * math.sqrt(vw.shape[0] / source_dim), vw, vb)
    audio = torch.nn.functional.linear(x * math.sqrt(aw.shape[0] / source_dim), aw, ab)
    return torch.cat((video, audio), dim=-1)


def _load_torch_file(path: str):
    import comfy.utils

    return comfy.utils.load_torch_file(path, safe_load=True)


def load_stock_gguf_model(path: str):
    """Load through ComfyUI-GGUF's public node after installing the fni8 patch."""
    import folder_paths

    custom_nodes = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if custom_nodes not in sys.path:
        sys.path.insert(0, custom_nodes)
    directory, name = os.path.dirname(path), os.path.basename(path)
    folder_paths.add_model_folder_path("unet", directory)
    folder_paths.add_model_folder_path("diffusion_models", directory)
    gguf_nodes = importlib.import_module("ComfyUI-GGUF.nodes")
    from comfyui_superl8.gguf_ops import enable_fni8_gguf

    assert enable_fni8_gguf(), "fni8 native GGUF patch did not engage"
    return gguf_nodes.UnetLoaderGGUF().load_unet(name)[0]


def build_context(
    dit,
    hidden_path: str,
    projection_path: str,
    connector_path: str,
    device: str,
    dtype: torch.dtype,
):
    """Project cached Gemma hidden states, run matching connectors once, then return context."""
    connector_state = normalize_connector_state_dict(_load_torch_file(connector_path))
    incompatible = dit.load_state_dict(connector_state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if "embedding" in key]
    if unexpected:
        raise RuntimeError(f"connector keys did not map into LTX DiT: {unexpected[:8]}")

    cached = torch.load(hidden_path, map_location="cpu", weights_only=True)
    hidden = cached["hidden_states"] if isinstance(cached, dict) else cached
    raw = project_gemma_hidden_states(hidden, _load_torch_file(projection_path))
    raw = raw.to(device=device, dtype=dtype)
    dit.video_embeddings_connector.to(device)
    dit.audio_embeddings_connector.to(device)

    def run_connector(module, value):
        result = module(value)
        return result[0] if isinstance(result, (tuple, list)) else result

    with torch.no_grad():
        video = run_connector(dit.video_embeddings_connector, raw[:, :, : dit.cross_attention_dim])
        audio = run_connector(dit.audio_embeddings_connector, raw[:, :, dit.cross_attention_dim :])
    context = torch.cat((video, audio), dim=-1).to(dtype)
    if not bool(torch.isfinite(context).all()):
        raise RuntimeError("LTX text context is non-finite")
    return context


def evict_one_shot_conditioners(dit):
    import torch.nn as nn

    dit.video_embeddings_connector = nn.Identity()
    dit.audio_embeddings_connector = nn.Identity()
    for name in ("caption_projection", "audio_caption_projection"):
        if hasattr(dit, name):
            setattr(dit, name, nn.Identity())
    gc.collect()
    torch.cuda.empty_cache()


def load_official_audio_vae(path: str):
    """Load only the official audio VAE and vocoder subtrees, not the 46 GB DiT."""
    import comfy.sd
    import comfy.utils
    from safetensors.torch import safe_open

    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            if key.startswith(("audio_vae.", "vocoder.")):
                state[key] = handle.get_tensor(key)
    if not state:
        raise RuntimeError("no `audio_vae.` or `vocoder.` subtree in the audio bundle")
    state = comfy.utils.state_dict_prefix_replace(
        state,
        {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."},
        filter_keys=True,
    )
    vae = comfy.sd.VAE(sd=state, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def load_video_latent_statistics(path: str):
    """Read only the two LTX video-latent normalization vectors."""
    from safetensors import safe_open

    prefix = "vae.per_channel_statistics."
    with safe_open(path, framework="pt", device="cpu") as handle:
        mean = handle.get_tensor(prefix + "mean-of-means")
        std = handle.get_tensor(prefix + "std-of-means")
    return mean, std


def load_latent_upsampler(path: str):
    """Load Comfy's LTX latent upsampler from its checkpoint metadata."""
    import comfy.utils
    from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

    state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    config = json.loads(metadata["config"])
    model = LatentUpsampler.from_config(config).to(dtype=torch.bfloat16)
    model.load_state_dict(state)
    return model.eval()


def spatially_upsample_latent(
    latent: torch.Tensor,
    upsampler,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Match LTX/Wan2GP latent unnormalize -> learned x2 -> normalize."""
    mean = mean.view(1, -1, 1, 1, 1).to(latent)
    std = std.view(1, -1, 1, 1, 1).to(latent)
    with torch.no_grad():
        upscaled = upsampler(latent * std + mean)
    return (upscaled - mean) / std


def decode_audio(vae, latent: torch.Tensor, device: str = "cuda"):
    """Decode LTX audio latent [B,C,T,F] to [B,C,samples] and its output rate."""
    with torch.no_grad():
        waveform = vae.decode(latent.to(device)).movedim(-1, 1).cpu()
    sample_rate = int(vae.first_stage_model.output_sample_rate)
    return waveform, sample_rate


def save_wav_pcm16(path, waveform: torch.Tensor, sample_rate: int):
    """Write [channels, samples] float audio without an optional codec dependency."""
    pcm = (
        waveform.detach()
        .float()
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .transpose(0, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(waveform.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def validate_cli_args(parser, args):
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be provided together")
    if args.height is not None and (args.height % 32 or args.width % 32):
        parser.error("--height and --width must be multiples of 32")
    if args.upscaler and (args.height is None or args.vae_stats is None):
        parser.error("two-stage mode requires --height/--width and --vae-stats")
    if args.upscaler and (args.height % 64 or args.width % 64):
        parser.error("two-stage --height and --width must be multiples of 64")
    if args.temporal_upscaler and args.vae_stats is None:
        parser.error("temporal upscaling requires --vae-stats")
    if args.temporal_before_refine and not args.temporal_upscaler:
        parser.error("--temporal-before-refine requires --temporal-upscaler")
    if not args.no_decode and not args.audio_bundle:
        parser.error("--audio-bundle is required unless --no-decode is set")


def run_checkpoint_upsampler(
    latent,
    checkpoint_path,
    stats_path,
    *,
    expected_frames=None,
    device="cuda:0",
    cache=None,
):
    mean, std = load_video_latent_statistics(stats_path)
    cache_key = (checkpoint_path, device)
    upsampler = cache.get(cache_key) if cache is not None else None
    if upsampler is None:
        upsampler = load_latent_upsampler(checkpoint_path).to(device)
        if cache is not None:
            cache[cache_key] = upsampler
    latent = latent.to(device)
    latent = spatially_upsample_latent(latent, upsampler, mean, std)
    torch.cuda.synchronize()
    if expected_frames is not None and latent.shape[2] != expected_frames:
        raise RuntimeError(
            f"latent upscaler returned {latent.shape[2]} frames; expected {expected_frames}"
        )
    if cache is None:
        del upsampler
        gc.collect()
        torch.cuda.empty_cache()
    del mean, std
    return latent


def run_two_stage_refinement(
    dit,
    latent,
    audio_latent,
    context,
    options,
    args,
    dtype,
    audio_frames,
    output_latent_frames,
    output_frame_rate,
    refinement_hw,
    wall,
    *,
    dit_device="cuda:0",
    upscaler_device="cuda:0",
    upscaler_cache=None,
):
    from bench.full_pipeline_ltx import build_latents, flow_match_denoise

    t0 = time.time()
    latent = run_checkpoint_upsampler(
        latent,
        args.upscaler,
        args.vae_stats,
        device=upscaler_device,
        cache=upscaler_cache,
    )
    wall["latent-upscale"] = time.time() - t0

    if args.temporal_upscaler and args.temporal_before_refine:
        t0 = time.time()
        latent = run_checkpoint_upsampler(
            latent,
            args.temporal_upscaler,
            args.vae_stats,
            expected_frames=output_latent_frames,
            device=upscaler_device,
            cache=upscaler_cache,
        )
        wall["temporal-upscale-pre-refine"] = time.time() - t0

    refinement_video_noise, refinement_audio_noise = build_latents(
        dit,
        dit_device,
        dtype,
        refinement_hw,
        latent.shape[2],
        audio_frames,
        args.seed + 1,
    )
    stage2_sigmas = refinement_sigmas(args.refinement_preset)
    sigma0 = stage2_sigmas[0]
    latent = prepare_refinement_latent(latent.to(dit_device), refinement_video_noise, sigma0)
    audio_latent = audio_latent.to(dit_device)
    audio_latent = prepare_refinement_latent(audio_latent, refinement_audio_noise, sigma0)
    t0 = time.time()
    latent, audio_latent = flow_match_denoise(
        dit,
        latent,
        audio_latent,
        context,
        len(stage2_sigmas) - 1,
        refinement_frame_rate(
            args.frame_rate,
            output_frame_rate,
            temporal_before_refine=args.temporal_before_refine,
        ),
        audio_frames,
        dit_device,
        transformer_options=options,
        sigmas=stage2_sigmas,
        return_audio=True,
    )
    torch.cuda.synchronize()
    wall["stage2-refine"] = time.time() - t0
    if not bool(torch.isfinite(latent).all()):
        raise RuntimeError("stage-two video latent is non-finite")
    if not bool(torch.isfinite(audio_latent).all()):
        raise RuntimeError("stage-two audio latent is non-finite")
    return latent, audio_latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--connector", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--hidden", default="/dev/shm/ltx_cond/hs_fox.pt")
    parser.add_argument("--vae-bundle", required=True)
    parser.add_argument(
        "--vae-dtype",
        choices=("auto", "bf16", "fp32"),
        default="fp32",
        help="video VAE compute dtype; fp32 may beat crippled BF16 tensor cores on CMP GV100",
    )
    parser.add_argument(
        "--vae-chunk-mib",
        type=int,
        default=192,
        help="LTX decoder intermediate chunk budget (fleet knee: 192 MiB)",
    )
    parser.add_argument(
        "--vae-resident",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fully load the 2.7 GiB video VAE to avoid PCIe 1.0 x1 weight streaming",
    )
    parser.add_argument("--vae-stats", help="official video VAE safetensors")
    parser.add_argument("--audio-bundle")
    parser.add_argument("--upscaler", help="LTX learned x2 spatial upscaler")
    parser.add_argument("--temporal-upscaler", help="LTX learned x2 temporal upscaler")
    parser.add_argument("--temporal-before-refine", action="store_true")
    parser.add_argument(
        "--refinement-preset",
        choices=("coherent-fast", "official"),
        default="coherent-fast",
        help="two-evaluation measured default or three-evaluation official schedule",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--hw", type=int, default=32)
    parser.add_argument("--height", type=int, help="decoded pixel height (multiple of 32)")
    parser.add_argument("--width", type=int, help="decoded pixel width (multiple of 32)")
    parser.add_argument("--audio-frames", type=int)
    parser.add_argument("--frame-rate", type=int, default=25)
    parser.add_argument("--output-frame-rate", type=int)
    parser.add_argument(
        "--stage-devices",
        default="cuda:0",
        help="logical CUDA roles in DiT, upscaler, video-VAE, audio-VAE order",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--outdir", default=os.environ.get("FNI8_OUTPUT_DIR", "output/ltx_fni8_gguf"))
    args = parser.parse_args()
    validate_cli_args(parser, args)

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bench.full_pipeline_ltx import (
        build_latents,
        decode_video,
        flow_match_denoise,
        load_bundle_video_vae,
        save_frames,
    )
    from comfyui_superl8.attention import Int8AttnGate, make_fni8_attn_override
    from comfyui_superl8.int8_linear import sqnr_calib_samples
    from comfyui_superl8.multigpu import plan_ltx_components
    from comfyui_superl8.nodes import _attention_sites_per_signature

    try:
        placement = plan_ltx_components(stage_devices(args.stage_devices))
    except ValueError as exc:
        parser.error(str(exc))
    role_devices = tuple(dict.fromkeys(vars(placement).values()))
    os.makedirs(args.outdir, exist_ok=True)
    for device in role_devices:
        torch.cuda.reset_peak_memory_stats(device)
    wall = {}
    t0 = time.time()
    model = load_stock_gguf_model(args.gguf)
    dit = model.model.diffusion_model
    dtype = torch.bfloat16
    context = build_context(dit, args.hidden, args.projection, args.connector, placement.dit, dtype)
    evict_one_shot_conditioners(dit)
    dit.to(placement.dit)
    wall["load+context"] = time.time() - t0

    gate = Int8AttnGate(
        calib_samples=sqnr_calib_samples(),
        sites_per_signature=_attention_sites_per_signature(model),
    )
    options = {"optimized_attention_override": make_fni8_attn_override(gate)}
    audio_frames = args.audio_frames
    output_latent_frames = (
        temporal_upscaled_latent_frames(args.frames) if args.temporal_upscaler else args.frames
    )
    output_frame_rate = args.output_frame_rate
    if output_frame_rate is None:
        output_frame_rate = args.frame_rate * (2 if args.temporal_upscaler else 1)
    if audio_frames is None:
        audio_frames = audio_latents_for_video(output_latent_frames, output_frame_rate)
    if args.upscaler:
        latent_hw, refinement_hw = two_stage_latent_shapes(args.height, args.width)
    else:
        latent_hw = args.hw if args.height is None else (args.height // 32, args.width // 32)
    vx, ax = build_latents(
        dit, placement.dit, dtype, latent_hw, args.frames, audio_frames, args.seed
    )
    sigmas = OFFICIAL_DISTILLED_SIGMAS
    if args.steps != len(sigmas) - 1:
        sigmas = None
    t0 = time.time()
    latent, audio_latent = flow_match_denoise(
        dit,
        vx,
        ax,
        context,
        args.steps,
        args.frame_rate,
        audio_frames,
        placement.dit,
        transformer_options=options,
        sigmas=sigmas,
        return_audio=True,
    )
    torch.cuda.synchronize()
    wall["stage1-denoise" if args.upscaler else "denoise"] = time.time() - t0
    assert bool(torch.isfinite(latent).all())
    assert bool(torch.isfinite(audio_latent).all())

    upscaler_cache = {} if placement.upscaler != placement.dit else None
    if args.upscaler:
        latent, audio_latent = run_two_stage_refinement(
            dit,
            latent,
            audio_latent,
            context,
            options,
            args,
            dtype,
            audio_frames,
            output_latent_frames,
            output_frame_rate,
            refinement_hw,
            wall,
            dit_device=placement.dit,
            upscaler_device=placement.upscaler,
            upscaler_cache=upscaler_cache,
        )

    if args.temporal_upscaler and not args.temporal_before_refine:
        t0 = time.time()
        latent = run_checkpoint_upsampler(
            latent,
            args.temporal_upscaler,
            args.vae_stats,
            expected_frames=output_latent_frames,
            device=placement.upscaler,
            cache=upscaler_cache,
        )
        wall["temporal-upscale"] = time.time() - t0

    frames = None
    audio = None
    audio_rate = None
    staged = len(role_devices) > 1
    if not staged:
        del dit, model
        gc.collect()
        torch.cuda.empty_cache()
    if not args.no_decode:
        t0 = time.time()
        vae = load_bundle_video_vae(args.vae_bundle, dtype=resolve_vae_dtype(args.vae_dtype))
        configure_vae_residency(
            vae,
            resident=args.vae_resident,
            load_device=placement.video_vae,
        )
        print(f"  video VAE dtype={vae.vae_dtype}", flush=True)
        frames = decode_video(
            vae,
            latent.float(),
            latent.shape[2],
            max_chunk_bytes=vae_chunk_bytes(args.vae_chunk_mib),
            device=placement.video_vae,
        )
        torch.cuda.synchronize()
        wall["vae"] = time.time() - t0
        save_frames(frames, args.outdir)
        if placement.audio_vae == placement.video_vae:
            del vae
            gc.collect()
            torch.cuda.empty_cache()
        t0 = time.time()
        audio_vae = load_official_audio_vae(args.audio_bundle)
        configure_vae_residency(
            audio_vae,
            resident=True,
            load_device=placement.audio_vae,
        )
        audio, audio_rate = decode_audio(
            audio_vae, audio_latent.float(), device=placement.audio_vae
        )
        audio_path = os.path.join(args.outdir, "ltx23_audio.wav")
        save_wav_pcm16(audio_path, audio[0], audio_rate)
        torch.cuda.synchronize()
        wall["audio"] = time.time() - t0
        print(f"  saved synchronized audio -> {audio_path}", flush=True)

    peaks = {device: torch.cuda.max_memory_allocated(device) / (1024**3) for device in role_devices}
    passed = sum(decision[0] for decision in gate.decisions.values())
    print("=" * 72)
    mode = "two-stage" if args.upscaler else "single-stage"
    if args.temporal_upscaler:
        mode += "+temporal-x2"
    execution = "component-staged" if staged else "one-card"
    print(f"LTX-2.3 distilled native-GGUF {execution} {mode} RESULT")
    print(f"  placement={placement}")
    print(f"  latent={tuple(latent.shape)} finite=True std={latent.float().std().item():.4f}")
    print(f"  latent sha256={tensor_sha256(latent)}")
    print(f"  audio latent sha256={tensor_sha256(audio_latent)}")
    if frames is not None:
        print(f"  frames={tuple(frames.shape)} std={frames.float().std().item():.4f}")
    if audio is not None:
        print(
            f"  audio={tuple(audio.shape)} rate={audio_rate} std={audio.float().std().item():.4f}"
        )
    print(f"  audio latent frames={audio_frames}")
    print(f"  conditioned fps={args.frame_rate}, output fps={output_frame_rate}")
    print(f"  attention DP4A sites={passed}/{len(gate.decisions)}")
    print("  wall(s): " + ", ".join(f"{key}={value:.1f}" for key, value in wall.items()))
    print("  peak HBM=" + ", ".join(f"{d}:{peak:.2f} GiB" for d, peak in peaks.items()))
    print("=" * 72)


if __name__ == "__main__":
    main()
