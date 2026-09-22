# SPDX-License-Identifier: Apache-2.0
"""Single-card end-to-end LTX-2.3 (audio+video DiT) generation on the int8 dp4a path.

This is the LTX-2.3 analogue of ``bench/full_pipeline_zimage.py`` — but LTX-2.3 is a
*video* DiT whose stock ComfyUI sampler path (`comfy.sample.sample`) is broken on the
pinned build (a KeyError deep in the CFG plumbing for the dual video/audio conditioning),
so this harness drives the denoise the way the prior LTX / Z-Image bring-up agents did:
**straight through the diffusion model's own forward** (`dit.forward(x=[vx, ax], ...)`),
a flow-matching Euler loop with NO `comfy.sample` / `KSampler` in the path. That is the
same faithful forward `bench/validate_ltx23_split.py` exercises, wrapped in a real
multi-step denoise + VAE-decode-to-frames + a quality gate.

    (Gemma-3 text context)  ->  int8 dp4a LTX-2.3 DiT (N-step flow-match denoise)
                            ->  LTX-2 video VAE decode (from the bundle)  ->  RGB frames

Fit strategy (measured, not assumed):
  * The published int8 DiT is 23.5 GB (b8) — it does NOT fit one 16 GB card. When the DiT
    does not fit, this harness transparently falls back to the 2-GPU pipeline-parallel
    split (`install_ltxav_pipeline`) so the FULL pipeline (denoise + decode + gate) can be
    validated on TODAY's weights. That is a validation topology, not the deliverable.
  * The deliverable single-card path engages automatically the moment a DiT that FITS one
    card is passed via ``--unet`` (the sibling mixed-precision shrink's ``<=14 GB``
    per_row_i8 DiT) — then everything runs on ``cuda:0`` alone.

HONEST GAPS (both external to this harness — see docs/ltx23-single-card-findings.md):
  1. No Gemma-3 text encoder is on the fleet, so real *prompt* conditioning is not yet
     possible. `--placeholder-te` (default when no TE is wired) uses a deterministic
     random text context of the correct shape — exactly the stable operating point
     `validate_ltx23_split.py` uses. The int8-vs-fp quality gate is valid under it (both
     paths see identical conditioning); only *prompt faithfulness* needs the real TE.
  2. Audio decode (vocoder) is not driven here; the video stream is the target. The audio
     latent is carried through the DiT (the AV cross-attention is load-bearing for the
     video too) but only the video VAE is decoded to frames.

Run inside the e2e image (needs the mounted weights + Volta/CMP sm_70 GPU(s))::

    FNI8_GPU=13 docker compose run --rm --entrypoint bash e2e -c \
      'cd /opt/ComfyUI && PYTHONPATH=/opt/ComfyUI:custom_nodes/ComfyUI-superl8 python3 \
       custom_nodes/ComfyUI-superl8/bench/full_pipeline_ltx.py --steps 8 --frames 25 --hw 64'

  For the 2-card validation topology on today's 23.5 GB b8, expose two free CMP cards
  (e.g. FNI8_GPU=12,13) — the harness auto-splits when the DiT does not fit one card.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

import torch

WEIGHTS_DIR = os.environ.get("FNI8_WEIGHTS_DIR", "")
LTX23_DIT = "Lightricks__LTX-2.3.dit.b8.fni8"  # base 22B (per_row_i8, 23.5 GB, 2-card)
# The single-card DiT: b4-conn-i4 (per_group_i4 W4A8, ~11 GB) — pass via --unet. The comfy
# DiT W4A8 path (this PR) runs its linears on fni8's gemm_w4a8 dp4a kernel.
CARD_BUDGET_GB = 15.0  # 16 GB card, headroom for activations

# Block-extrapolated time ladder (PR #114) to validate the real wall-time against.
# 512^2 x 97f int8 single-card ~= 3 min extrapolated @ 30 steps (base). Per-step scales
# with token count (H*W*T) and step count. Printed alongside the measured wall-time.
LADDER_NOTE = "PR#114 ladder: 512^2 x 97f int8 ~= 3 min extrapolated (30 steps, single card)"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
def build_text_context(dit, device, dtype, seed: int, length: int = 128):
    """Placeholder Gemma-3 text context of the shape LTX-2.3 expects
    (`cross_attention_dim + audio_cross_attention_dim`). Deterministic per seed so the
    int8 and fp runs see byte-identical conditioning (the quality gate depends on it).

    Swap this for the real Gemma-3-12B encoder output once a TE is provisioned — the DiT
    forward signature is unchanged (`context=<[B, L, ctx_dim]>`)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    # LTX-2.3 `_prepare_context` splits `context` into a video + audio stream. When
    # `caption_proj_before_connector` is False (the LTX-2.3 case), BOTH streams are the
    # raw text-encoder dim `caption_channels` (Gemma-3-12B hidden = 3840) and the
    # per-stream Embeddings1DConnector projects them up to the 4096/2048 cross-attn dims.
    # So the placeholder context must be sized 2*caption_channels, NOT the post-connector
    # `cross_attention_dim + audio_cross_attention_dim` (which double-counts the projection
    # and mis-splits — see comfy.ldm.lightricks.av_model._prepare_context).
    cc = getattr(dit, "caption_channels", None)
    if cc is not None and getattr(dit, "caption_proj_before_connector", False) is False:
        ctx_dim = cc * 2
    else:
        ctx_dim = dit.cross_attention_dim + dit.audio_cross_attention_dim
    return torch.randn(1, length, ctx_dim, generator=g, dtype=torch.float32).to(device, dtype)


def build_latents(
    dit,
    device,
    dtype,
    hw: int | tuple[int, int],
    frames: int,
    audio_frames: int,
    seed: int,
):
    """Initial noise latents for the video and audio streams, at the model's own dims.

    Video latent: [B, in_channels, T, H, W]; audio latent: [B, aC, aT, aF]. Flow-matching
    starts from pure noise at sigma=1 and integrates to sigma=0."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    latent_height, latent_width = (hw, hw) if isinstance(hw, int) else hw
    C = dit.in_channels
    vx = torch.randn(
        1,
        C,
        frames,
        latent_height,
        latent_width,
        generator=g,
        dtype=torch.float32,
    ).to(device, dtype)
    aC = dit.num_audio_channels
    aF = dit.audio_frequency_bins
    ax = torch.randn(1, aC, audio_frames, aF, generator=g, dtype=torch.float32).to(device, dtype)
    return vx, ax


# ---------------------------------------------------------------------------
# Denoise — flow-matching Euler, driven straight through dit.forward
# ---------------------------------------------------------------------------
def ltx_sigma_schedule(
    steps: int,
    tokens: int,
    device,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
):
    """Resolution-shifted schedule used by ComfyUI's ``LTXVScheduler``."""
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    slope = (max_shift - base_shift) / (4096 - 1024)
    sigma_shift = tokens * slope + (base_shift - slope * 1024)
    exp_shift = torch.exp(torch.tensor(sigma_shift, device=device))
    return torch.where(
        sigmas != 0,
        exp_shift / (exp_shift + (1.0 / sigmas - 1.0)),
        sigmas,
    )


def flow_match_denoise(
    dit,
    vx,
    ax,
    context,
    steps: int,
    frame_rate: int,
    audio_length: int,
    device,
    transformer_options=None,
    uncond_context=None,
    cfg: float = 1.0,
    sigmas=None,
    return_audio: bool = False,
):
    """Flow-matching Euler with LTX's shifted schedule and optional CFG.

    This is the `apply_model`-equivalent at the diffusion-model level — no comfy.sample /
    KSampler (broken on this build for LTX's dual conditioning).

    `transformer_options` is threaded to the DiT so the production
    `optimized_attention_override` (the fni8 int8/fp16 attention wrapper) is engaged — the
    same hook `nodes.FNI8AttentionPatch` installs on the KSampler path. Without it the
    blocks run ComfyUI's native attention, which is NOT what we mean to validate."""
    topts = transformer_options if transformer_options is not None else {}
    if sigmas is None:
        tokens = int(vx.shape[2] * vx.shape[3] * vx.shape[4])
        sigma_t = ltx_sigma_schedule(steps, tokens, device)
    else:
        sigma_t = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
        if sigma_t.numel() != steps + 1:
            raise ValueError("sigmas must contain steps + 1 values")
    use_cfg = cfg > 1.0 and uncond_context is not None

    def velocity(ctx, timestep):
        out = dit.forward(
            x=[vx, ax],
            timestep=timestep,
            context=ctx,
            attention_mask=None,
            frame_rate=frame_rate,
            audio_length=audio_length,
            transformer_options=topts,
        )
        video = out[0] if isinstance(out, (list, tuple)) else out
        audio = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else None
        return video, audio

    for i in range(steps):
        dt = sigma_t[i + 1] - sigma_t[i]
        # LTX-2.3 uses ModelSamplingFlux, whose timestep() is the identity. Passing the
        # legacy diffusion convention (sigma * 1000) drives the model far outside its
        # trained time domain and produces finite but visually meaningless noise.
        t = sigma_t[i].reshape(1).to(device)
        with torch.no_grad():
            v_v, v_a = velocity(context, t)
            if use_cfg:
                u_v, u_a = velocity(uncond_context, t)
                v_v = u_v + cfg * (v_v - u_v)
                if v_a is not None and u_a is not None:
                    v_a = u_a + cfg * (v_a - u_a)
        vx = vx + dt * v_v.to(vx.dtype)
        # Integrate the audio stream too when the forward returns its velocity — the AV
        # cross-attention is load-bearing for the video, so keep audio on its own trajectory
        # rather than frozen. Video remains the decode target.
        if v_a is not None:
            ax = ax + dt * v_a.to(ax.dtype)
    return (vx, ax) if return_audio else vx


# ---------------------------------------------------------------------------
# VAE — extracted from the LTX-2.3 bundle
# ---------------------------------------------------------------------------
def load_bundle_video_vae(path: str, dtype: torch.dtype | None = None):
    """Extract the LTX-2 *video* VAE from the bundled `.fni8` (`vae.` sub-tree, stored
    raw/fp) and build a `comfy.sd.VAE` from it — no separate VAE file needed.

    Builds the VAE from the converter-PERSISTED `metadata['config']['vae']` (a
    `CausalVideoAutoencoder` config) rather than comfy's shape/`version` auto-detect: the
    LTX-2.3 VAE matches none of comfy's built-in presets (0/1/2), so auto-detect mis-sizes
    the decoder and dies at `decoder.up_blocks.4` (512-vs-256 channel mismatch). When the
    `.fni8` predates the converter persisting the config, `ltx2_vae_metadata` returns None
    and comfy falls back to auto-detect (the honest degraded path)."""
    import comfy.sd

    from comfyui_superl8.loader import load_fni8_dit, ltx2_vae_metadata

    sd = load_fni8_dit(
        path, device="cpu", strip_prefix="vae.", keep_only_prefixed=True, dequant_fp=()
    )
    if not sd:
        raise RuntimeError("no `vae.` sub-tree in the bundle")
    return comfy.sd.VAE(sd=sd, metadata=ltx2_vae_metadata(path), dtype=dtype)


@contextlib.contextmanager
def _ltx_vae_chunk_budget(max_chunk_bytes: int | None):
    """Temporarily tune Comfy's native LTX temporal chunker for a sequential decode."""
    if max_chunk_bytes is None:
        yield
        return

    import comfy.ldm.lightricks.vae.causal_video_autoencoder as ltx_vae

    original = ltx_vae.get_max_chunk_size
    ltx_vae.get_max_chunk_size = lambda _device: max_chunk_bytes
    try:
        yield
    finally:
        ltx_vae.get_max_chunk_size = original


def decode_video(
    vae,
    latent,
    chunk: int,
    max_chunk_bytes: int | None = None,
    device: str = "cuda",
):
    """Decode a video latent [B,C,T,H,W] -> frames via the pack's 3D-causal chunked decoder
    (keeps peak HBM to a temporal window)."""
    from comfyui_superl8.vae_tiled import chunked_vae_decode_3d

    with torch.no_grad(), _ltx_vae_chunk_budget(max_chunk_bytes):
        return chunked_vae_decode_3d(
            lambda lat: vae.decode(lat.to(device)), latent.to(device), chunk_size=chunk
        )


# ---------------------------------------------------------------------------
# DiT loading + fit
# ---------------------------------------------------------------------------
def load_dit(unet_name: str):
    from comfyui_superl8.nodes import UnetLoaderFNI8, _is_ltxav

    (model,) = UnetLoaderFNI8().load(unet_name, arch="ltx_video")
    dit = model.model.diffusion_model
    assert _is_ltxav(dit), f"expected LTXAV DiT, got {type(dit).__name__}"
    return model, dit


def dit_resident_gb(dit) -> float:
    from comfyui_superl8.multigpu import module_vram_bytes

    return module_vram_bytes(dit) / 1e9


def place_dit(dit, budget_gb: float):
    """Single card if the DiT fits the budget, else 2-GPU pipeline-parallel split.
    Returns the string 'single' or 'split-N'."""
    from comfyui_superl8.multigpu import (
        TransportCodec,
        available_devices,
        count_dit_blocks,
        plan_pipeline,
    )
    from comfyui_superl8.pipeline_ltx import install_ltxav_pipeline

    weight_gb = dit_resident_gb(dit)
    devs = available_devices()
    if weight_gb <= budget_gb:
        dit.to("cuda:0")
        return "single", {"cuda:0": weight_gb}
    if len(devs) < 2:
        raise RuntimeError(
            f"DiT is {weight_gb:.1f} GB (> {budget_gb} GB budget) and only one GPU is "
            f"visible — pass a DiT that fits one card (the shrunk <=14 GB per_row_i8 "
            f"weights) or expose a second GPU for the validation split."
        )
    n_blocks = count_dit_blocks(dit)
    plan = plan_pipeline(n_blocks, devices=devs[:2], transport_scheme="int8", head_weight_blocks=10)
    report = install_ltxav_pipeline(dit, plan, TransportCodec(scheme="int8"))
    return f"split-{len(devs[:2])}", {d: b / 1e9 for d, b in report.items()}


def dequantize_dit_inplace(dit, dtype=torch.bfloat16) -> int:
    """Dequantize every quantized FNI8Tensor weight -> fp in place, for the fp reference
    run (identical architecture + conditioning; only the matmul/attention precision
    differs). Scheme-aware: per_row_i8 (W8A8) and per_group_i4 (W4A8) both dequant here."""
    from comfyui_superl8.superl8_tensor import FNI8Tensor
    from comfyui_superl8.int8_linear import dequantize_qtensor_data

    n = 0
    for mod in dit.modules():
        for pname, p in list(mod._parameters.items()):
            if isinstance(p, FNI8Tensor) and p.q_scale is not None:
                w = dequantize_qtensor_data(
                    p.int8_data(),
                    p.q_scale,
                    getattr(p, "q_scheme", "per_row_i8"),
                    getattr(p, "q_group_size", 0),
                    dtype,
                )
                mod._parameters[pname] = torch.nn.Parameter(w, requires_grad=False)
                n += 1
    return n


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--unet",
        default=LTX23_DIT,
        help="DiT .fni8 name on the diffusion_models path (default: base b8; "
        "pass the shrunk <=14 GB per_row_i8 DiT for the single-card path)",
    )
    ap.add_argument("--steps", type=int, default=8, help="denoise steps (base~40, distilled~8)")
    ap.add_argument(
        "--sigmas",
        default="",
        help="explicit comma-separated sigma schedule (must contain steps+1 values)",
    )
    ap.add_argument("--frames", type=int, default=25, help="video latent T (frames)")
    ap.add_argument("--hw", type=int, default=64, help="video latent H=W (px = 32x-ish)")
    ap.add_argument("--audio-frames", type=int, default=32)
    ap.add_argument("--frame-rate", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=4, help="VAE temporal chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-gb", type=float, default=CARD_BUDGET_GB)
    ap.add_argument("--no-fp-ref", action="store_true", help="skip the fp reference/gate")
    ap.add_argument("--no-decode", action="store_true", help="skip VAE decode (denoise only)")
    ap.add_argument("--outdir", default=os.environ.get("FNI8_PIPE_OUT", ".pipe_out"))
    args = ap.parse_args()

    sys.path.insert(0, _repo_root())
    os.makedirs(args.outdir, exist_ok=True)

    import folder_paths

    folder_paths.add_model_folder_path("diffusion_models", WEIGHTS_DIR)
    # Allow an absolute --unet path (e.g. a RAM-staged /dev/shm copy, to dodge a contended
    # HDD): register its directory as a search path and reduce --unet to the basename.
    if os.path.isabs(args.unet) and os.path.exists(args.unet):
        folder_paths.add_model_folder_path("diffusion_models", os.path.dirname(args.unet))
        args.unet = os.path.basename(args.unet)

    from comfyui_superl8.gate import is_sm70
    from bench.quality import cosine_similarity

    if not is_sm70():
        print("SKIP: needs a Volta/CMP (sm_70) GPU")
        return
    if args.unet not in folder_paths.get_filename_list("diffusion_models"):
        print(f"SKIP: {args.unet} not on the diffusion_models path")
        return
    path = folder_paths.get_full_path("diffusion_models", args.unet)

    print("=" * 72)
    print(f"LTX-2.3 single-card e2e   unet={args.unet}")
    print(f"  {LADDER_NOTE}")
    print("=" * 72)

    torch.cuda.reset_peak_memory_stats()
    wall = {}

    t0 = time.time()
    model, dit = load_dit(args.unet)
    wall["load"] = time.time() - t0
    dtype = next((p.dtype for p in dit.parameters() if p.dtype.is_floating_point), torch.bfloat16)
    print(
        f"[load] {len(list(dit.parameters()))} params, dtype={dtype}, "
        f"resident={dit_resident_gb(dit):.2f} GB, in {wall['load']:.1f}s"
    )

    topo, report = place_dit(dit, args.budget_gb)
    print(f"[place] topology={topo}  per-card GB={ {k: round(v, 2) for k, v in report.items()} }")
    dev0 = "cuda:0"

    # ---- int8 denoise -------------------------------------------------------
    # Install the production fni8 attention override (the int8 dp4a / O(N) fp16 wrapper),
    # the same hook nodes.FNI8AttentionPatch puts on the KSampler path. This is what makes
    # LTX's 17408-token UNMASKED self-attention route to superl8.attn_fp16_fwd (O(N) memory)
    # instead of ComfyUI's native O(N^2) attention. FNI8_ATTN_DEBUG=1 logs each dispatch.
    from comfyui_superl8.attention import Int8AttnGate, make_fni8_attn_override
    from comfyui_superl8.int8_linear import sqnr_calib_samples
    from comfyui_superl8.nodes import _attention_sites_per_signature

    gate = Int8AttnGate(
        calib_samples=sqnr_calib_samples(),
        sites_per_signature=_attention_sites_per_signature(model),
    )
    topts = {"optimized_attention_override": make_fni8_attn_override(gate)}

    ctx = build_text_context(dit, dev0, dtype, args.seed)
    vx, ax = build_latents(dit, dev0, dtype, args.hw, args.frames, args.audio_frames, args.seed)
    print(
        f"[denoise] int8: {args.steps} steps, video latent {tuple(vx.shape)}, "
        f"audio {tuple(ax.shape)}"
    )
    sigmas = [float(value) for value in args.sigmas.split(",")] if args.sigmas else None
    t0 = time.time()
    lat_int8 = flow_match_denoise(
        dit,
        vx,
        ax,
        ctx,
        args.steps,
        args.frame_rate,
        args.audio_frames,
        dev0,
        transformer_options=topts,
        sigmas=sigmas,
    )
    torch.cuda.synchronize()
    wall["denoise_int8"] = time.time() - t0
    finite = bool(torch.isfinite(lat_int8).all())
    print(
        f"  int8 latent {tuple(lat_int8.shape)} finite={finite} "
        f"std={float(lat_int8.float().std()):.4f} in {wall['denoise_int8']:.1f}s "
        f"({wall['denoise_int8'] / args.steps:.2f}s/step)"
    )
    passed = sum(decision[0] for decision in gate.decisions.values())
    sqnrs = [decision[1] for decision in gate.decisions.values()]
    worst = min(sqnrs, default=float("nan"))
    print(f"  attention SQNR sites: {passed}/{len(gate.decisions)} DP4A; worst={worst:.1f} dB")
    assert finite, "int8 latent non-finite"

    # ---- VAE decode ---------------------------------------------------------
    frames_img = None
    if not args.no_decode:
        t0 = time.time()
        vae = load_bundle_video_vae(path)
        frames_img = decode_video(vae, lat_int8.float(), args.chunk)
        torch.cuda.synchronize()
        wall["vae"] = time.time() - t0
        print(f"[vae] decoded frames {tuple(frames_img.shape)} in {wall['vae']:.1f}s")
        save_frames(frames_img, args.outdir)

    # ---- fp reference + quality gate ---------------------------------------
    cos_lat = None
    if not args.no_fp_ref and topo == "single":
        # Dequantize in place for the fp reference (only feasible single-card; the split
        # topology mutates forward, so the gate runs on the single-card path).
        ndq = dequantize_dit_inplace(dit, dtype)
        print(f"[fp-ref] dequantized {ndq} int8 linears -> {dtype}")
        vx2, ax2 = build_latents(
            dit, dev0, dtype, args.hw, args.frames, args.audio_frames, args.seed
        )
        lat_fp = flow_match_denoise(
            dit,
            vx2,
            ax2,
            ctx,
            args.steps,
            args.frame_rate,
            args.audio_frames,
            dev0,
            transformer_options=topts,
            sigmas=sigmas,
        )
        cos_lat = cosine_similarity(lat_int8, lat_fp)
    elif topo != "single":
        print(
            "[fp-ref] skipped: quality gate runs on the single-card topology "
            "(the split mutates the forward). Use validate_ltx23_split.py for the "
            "split's int8-vs-fp cosine."
        )

    # ---- verdict ------------------------------------------------------------
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print("=" * 72)
    print("LTX-2.3 single-card e2e RESULT")
    print(f"  topology              : {topo}")
    print(
        "  wall-time (s)         : "
        + ", ".join(f"{k}={v:.1f}" for k, v in wall.items())
        + f"  TOTAL={sum(wall.values()):.1f}"
    )
    if frames_img is not None:
        struct = frames_img.float().std().item() > 1e-3 and bool(torch.isfinite(frames_img).all())
        print(f"  frames finite+structured: {struct}  shape={tuple(frames_img.shape)}")
    if cos_lat is not None:
        gate = cos_lat >= 0.985
        print(
            f"  latent cosine int8-vs-fp: {cos_lat:.6f}   GATE(>=0.985): "
            f"{'PASS' if gate else 'FAIL'}"
        )
    print(f"  peak HBM              : {peak:.2f} GiB (card = 16 GiB)")
    print("=" * 72)


def save_frames(frames_img, outdir):
    """Save decoded frames as individual PNGs (comfy IMAGE is [B,H,W,C] or [B,C,H,W])."""
    from PIL import Image

    x = frames_img.detach().float().cpu()
    if x.dim() == 5:  # drop the batch dim -> a 4-D [T,*,*,*] stack
        x = x[0]  # comfy VideoVAE returns [B,T,H,W,C] (channel-LAST)
    if x.dim() == 4 and x.shape[-1] in (1, 3):  # [T,H,W,C] (channel-last)
        frames = [x[t] for t in range(x.shape[0])]
    elif x.dim() == 4 and x.shape[0] in (1, 3):  # [C,T,H,W] (channel-first)
        frames = [x[:, t].movedim(0, -1) for t in range(x.shape[1])]
    elif x.dim() == 4:  # [T,C,H,W]
        frames = [x[t].movedim(0, -1) for t in range(x.shape[0])]
    else:
        frames = [x.reshape(x.shape[-2], x.shape[-1], -1)]
    for i, f in enumerate(frames[:64]):
        arr = (f.clamp(0, 1).numpy() * 255).round().astype("uint8")
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        Image.fromarray(arr).save(os.path.join(outdir, f"ltx23_frame_{i:03d}.png"))
    print(f"  saved {min(len(frames), 64)} frame PNG(s) -> {outdir}")


if __name__ == "__main__":
    main()
