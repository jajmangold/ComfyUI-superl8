# SPDX-License-Identifier: Apache-2.0
"""ComfyUI node classes. Thin wiring over the pure logic (int8_linear / attention /
loader / arch / gate). ComfyUI is imported lazily, so this module loads without it
and NODE_CLASS_MAPPINGS is always exported.
"""

from __future__ import annotations

import torch

from . import arch
from ._comfy_compat import COMFY, require_comfy
from .attention import Int8AttnGate, make_attn1_replace_callback
from .gate import require_sm70
from .guidance_harvest import GUIDANCE_DISTILLED_ARCHS, _patch_model, mark_guidance_distilled

if COMFY:
    import folder_paths

    # ComfyUI's diffusion_models folder listing is filtered by extension (like
    # ComfyUI-GGUF does for .gguf); without this a .fni8 checkpoint never shows up
    # in UnetLoaderFNI8's dropdown or folder_paths.get_filename_list at all.
    folder_paths.supported_pt_extensions.add(".fni8")


class UnetLoaderFNI8:
    """Load a `.fni8`-quantized DiT as a ComfyUI MODEL running int8 dp4a ops."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            names = folder_paths.get_filename_list("diffusion_models")
        except Exception:
            names = []
        return {
            "required": {
                "unet_name": (names,),
                "arch": (["auto", *arch.families()],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "fni8"
    TITLE = "FNI8 DiT Loader (int8 dp4a)"

    def load(self, unet_name, arch="auto"):
        require_comfy()
        require_sm70()
        import comfy.sd
        import folder_paths

        import torch

        from .superl8_tensor import FNI8Tensor
        from .loader import (
            _HEAD_FP_DEQUANT,
            assign_int8_weights,
            dit_bundle_prefix,
            fni8_dit_native_dtype,
            fuse_attention_qkv_int8,
            load_fni8_dit,
            ltx2_detection_metadata,
            ltx2_persisted_transformer_config,
            remap_diffusers_to_native,
        )
        from .ops import FNI8Ops

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        a = arch if arch != "auto" else None
        # `dequant_fp` names int8 weights to dequantize back to fp at load: always the
        # output head (`_HEAD_FP_DEQUANT`), plus any arch-specific genuinely-fp linears
        # (`DiTArch.fp_dequant`). Archs whose ComfyUI model fuses `to_q`/`to_k`/`to_v`
        # into one `qkv` are handled separately below (`fuse_attn_qkv`): the loader
        # re-fuses their separate per-row-int8 projections losslessly so attention stays
        # int8, rather than dropping it to fp.
        arch_obj = arch_get(arch)
        dequant_fp = tuple(_HEAD_FP_DEQUANT) + (arch_obj.fp_dequant if arch_obj else ())
        # Run the DiT in its native dtype (bf16 for modern DiTs). Computed up front because
        # the qkv re-fusion dequantizes the diffusers projections to this dtype for comfy's
        # (fp) convert. Volta has no bf16 hardware so ComfyUI would otherwise default the
        # unet to fp16 -- which overflows bf16-native DiTs (Z-Image, Flux, ...) to NaN ->
        # BLACK images. Our dp4a path never touches the (gimped) 16-bit tensor cores.
        native = fni8_dit_native_dtype(path)
        dtype = torch.bfloat16 if native == "bfloat16" else torch.float16
        # Bundled full checkpoints (LTX-2/2.3) pack the DiT under `model.diffusion_model.`
        # next to a VAE/vocoder/audio-VAE the DiT loader must not carry. Extract ONLY the
        # DiT sub-tree (already ComfyUI-native layout) and skip the rest on disk. Plain
        # DiTs keep the arch's own `diffusion_model.` strip.
        bundle = dit_bundle_prefix(path)
        if bundle:
            sd = load_fni8_dit(
                path,
                device="cpu",
                strip_prefix=bundle,
                keep_only_prefixed=True,
                dequant_fp=dequant_fp,
            )
        else:
            prefix = "diffusion_model." if a and globals_arch_prefix(a) else ""
            sd = load_fni8_dit(path, device="cpu", strip_prefix=prefix, dequant_fp=dequant_fp)
        # Video DiTs (Wan/LTX-Video v1) ship in diffusers layout, which ComfyUI's
        # `detect_unet_config` doesn't recognize -> rewrite keys to native (issue #30).
        # No-op for image DiTs, for already-native checkpoints, and for LTX-2 bundles
        # (their extracted DiT is already ComfyUI-native — see the remap signature gate).
        sd = remap_diffusers_to_native(sd, a)
        # Archs whose ComfyUI model FUSES separate diffusers `to_q`/`to_k`/`to_v` into one
        # `attention.qkv` (Z-Image S3-DiT): losslessly re-fuse the separate per-row-int8
        # projections into a single int8 `qkv` (concat codes + per-row scales -> zero added
        # error) so attention runs dp4a too, not just the FFN (#103 had to keep it fp). The
        # helper also dequantizes the diffusers projections left in `sd` to fp so comfy's
        # convert still BUILDS the model correctly; the fused int8 weights (native-keyed)
        # then overwrite those fp params via `assign_int8_weights`, like the int8 FFN.
        fused_attn = fuse_attention_qkv_int8(sd, dtype=dtype) if (
            arch_obj and arch_obj.fuse_attn_qkv) else {}
        # Keep the int8 weights aside: comfy.sd builds the model with assign=False, so
        # its load_state_dict would copy the int8 codes into fp params (dropping the
        # per-row scale). We re-attach them as FNI8Tensor after the build so the dp4a
        # path actually engages (see loader.assign_int8_weights).
        int8_weights = {k: v for k, v in sd.items() if isinstance(v, FNI8Tensor)}
        int8_weights.update(fused_attn)  # native-keyed fused qkv + out (overwrite fp)
        # LTX-2 bundles need a transformer-config flag (cross_attention_adaln -> 9-row
        # adaLN tables) that the `.fni8` dropped; recover it from the tensor shapes and
        # hand it to detect_unet_config as the `metadata['config']` it would otherwise
        # read from the diffusers checkpoint. None (and thus a no-op) for every other DiT.
        persisted = ltx2_persisted_transformer_config(path) if bundle else None
        metadata = ltx2_detection_metadata(sd, persisted=persisted) if bundle else None
        # per_group_i4 (W4A8) weights are stored PACKED (uint8 [N, K//2]); comfy's
        # `detect_unet_config` reads Linear `.shape`s and its shape-checked load_state_dict
        # compares each weight against the model's logical [N, K] — the packed [N, K//2]
        # both mis-detects the config AND raises a size mismatch (fatal even under
        # strict=False). Swap each packed i4 weight for a LOGICAL-shaped placeholder so
        # detection + build + copy all see [N, K]; `assign_int8_weights` then installs the
        # real packed FNI8Tensor after the build. The placeholder is a 0-stride `expand`
        # of a scalar (zero real storage), so this adds no memory for the ~22B DiT — the
        # copied-in zeros are overwritten by the FNI8Tensor before the model is used.
        # (per_row_i8 codes are already [N, K], so those pass through comfy's copy as before.)
        build_sd = {}
        for k, v in sd.items():
            if isinstance(v, FNI8Tensor) and getattr(v, "q_scheme", "") == "per_group_i4":
                n_out, cols = int(v.shape[0]), int(v.shape[1])
                build_sd[k] = torch.zeros((), dtype=dtype).expand(n_out, cols * 2)
            else:
                build_sd[k] = v
        model = comfy.sd.load_diffusion_model_state_dict(
            dict(build_sd),
            model_options={"custom_operations": FNI8Ops, "dtype": dtype},
            metadata=metadata,
        )
        if model is None:
            raise RuntimeError(
                f"ComfyUI could not detect a DiT config for '{unet_name}' (arch={arch}). "
                "If this is a video DiT, its diffusers layout may need a key remap in "
                "comfyui_superl8.loader.remap_diffusers_to_native (issue #30)."
            )
        if int8_weights:
            assign_int8_weights(model.model.diffusion_model, int8_weights)
        # Wire int8 FlashAttention into the loaded model (per-arch attn_seam).
        # This makes a cloned MODEL whose self-attention runs through
        # superl8.attn_int8_fwd instead of fp SDPA.  The clone preserves model_options
        # (deep-copied), so any flags set before the clone survive.
        (model,) = FNI8AttentionPatch().patch(model, arch=arch)
        # Auto-mark guidance-distilled arches so the uncond branch is skipped
        # (guidance-distillation harvest, issue #27). This happens after the patch
        # so the cloned model carries the flag directly.
        arch_hint = (arch_obj.name if arch_obj else arch).lower()
        if any(g in arch_hint for g in GUIDANCE_DISTILLED_ARCHS):
            mark_guidance_distilled(model)
        _patch_model(model)
        return (model,)


class UnetLoaderFNI8GGUF:
    """Load a GGUF DiT via ComfyUI-GGUF's loader with FNI8GGUFOps (int8 dp4a k-quant).

    Uses FNI8GGUFOps (subclass of GGMLOps) as custom_operations, so every GGUF-linear
    runs the fused dp4a native path instead of stock dequant-to-fp.  The rest of the
    ComfyUI-GGUF pipeline (state-dict load, model detection, weight placement) is
    untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            folder_paths.supported_pt_extensions.add(".gguf")
            names = folder_paths.get_filename_list("diffusion_models")
        except Exception:
            names = []
        return {
            "required": {
                "unet_name": (names,),
                "arch": (["auto", *arch.families()],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "fni8"
    TITLE = "FNI8 GGUF DiT Loader (int8 dp4a k-quant)"

    def load(self, unet_name, arch="auto"):
        require_comfy()
        require_sm70()
        import sys
        import comfy.sd
        import folder_paths

        from .gguf_ops import enable_fni8_gguf

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        # Ensure the fni8 monkeypatch is on GGMLOps.Linear (idempotent).
        enable_fni8_gguf()

        # Find ComfyUI-GGUF's loader and ops modules from sys.modules (they were
        # loaded by ComfyUI's custom node system; importlib can't handle the hyphen).
        gguf_mod = None
        gguf_ops_mod = None
        for name, mod in list(sys.modules.items()):
            if "gguf" in name.lower() and hasattr(mod, "gguf_sd_loader"):
                gguf_mod = mod
            if name.endswith(".ops") and hasattr(mod, "GGMLOps"):
                if isinstance(mod.GGMLOps, type) and hasattr(mod.GGMLOps, "Linear"):
                    gguf_ops_mod = mod
            if gguf_mod and gguf_ops_mod:
                break

        if gguf_mod is None:
            raise RuntimeError("ComfyUI-GGUF extension not installed (gguf_sd_loader unavailable)")
        if gguf_ops_mod is None:
            raise RuntimeError("ComfyUI-GGUF extension not installed (GGMLOps unavailable)")

        sd, extra = gguf_mod.gguf_sd_loader(path)
        ops = gguf_ops_mod.GGMLOps()

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops},
        )
        if model is None:
            raise RuntimeError(
                f"ComfyUI could not detect a DiT config for '{unet_name}' (arch={arch}). "
                "Ensure ComfyUI-GGUF is installed and the file is a valid GGUF."
            )

        a = arch_get(arch)
        (model,) = FNI8AttentionPatch().patch(model, arch=arch)
        arch_hint = (a.name if a else arch).lower()
        if any(g in arch_hint for g in GUIDANCE_DISTILLED_ARCHS):
            mark_guidance_distilled(model)
        _patch_model(model)
        return (model,)


class FNI8ComponentLoader:
    """Load the text-encoder + fp VAE a DiT arch expects, so a full graph (DiT + TE +
    VAE) runs without guessing which `CLIPLoader`/`DualCLIPLoader` `type` or VAE file
    an arch needs (see `arch.py`'s per-arch `text_encoder`/`clip_type`/`vae` fields and
    the README precision table). The VAE is convolutional, not a transformer — it
    always stays fp. The TE defaults to fp too, but `clip_precision="int8_dp4a"` routes
    its `Linear` layers through the same `FNI8Ops` int8 dp4a GEMM the DiT uses (via
    `custom_operations`, same plumbing ComfyUI-GGUF's `CLIPLoaderGGUF` uses for
    quantized CLIP): quantized lazily from the loaded fp weight on first forward, so no
    pre-quantized `.fni8` TE file is needed. `FNI8Ops` only patches `nn.Linear` —
    LayerNorm/Embedding modules are untouched, so norms/embeddings stay fp for free.
    TE attention (the actual softmax(QK^T)V compute, not the QKV/out-proj linears)
    stays fp regardless — only `FNI8AttentionPatch` (DiT-only today) touches that."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            te_names = folder_paths.get_filename_list("text_encoders")
            vae_names = folder_paths.get_filename_list("vae")
        except Exception:
            te_names = vae_names = []
        return {
            "required": {
                "arch": (arch.families(),),
                "clip_name1": (te_names,),
                "vae_name": (vae_names,),
                "clip_precision": (["fp", "int8_dp4a"],),
            },
            "optional": {
                "clip_name2": (["none", *te_names],),
            },
        }

    RETURN_TYPES = ("CLIP", "VAE")
    RETURN_NAMES = ("clip", "vae")
    FUNCTION = "load"
    CATEGORY = "fni8"
    TITLE = "FNI8 Text-Encoder + VAE Loader (fp or int8 dp4a TE, matched to arch)"

    def load(self, arch, clip_name1, vae_name, clip_precision="fp", clip_name2="none"):
        require_comfy()
        import comfy.sd
        import comfy.utils
        import folder_paths

        a = arch_get(arch)
        if a is None:
            raise ValueError(f"Unknown arch '{arch}'; see comfyui_superl8.arch.families()")

        clip_type = _resolve_clip_type(a)
        clip_names = (
            [clip_name1] if not clip_name2 or clip_name2 == "none" else [clip_name1, clip_name2]
        )
        clip_paths = [folder_paths.get_full_path("text_encoders", n) for n in clip_names]
        vae_path = folder_paths.get_full_path("vae", vae_name)
        for p, kind in [*((p, "text encoder") for p in clip_paths), (vae_path, "VAE")]:
            _reject_fni8(p, kind)

        clip_model_options = {}
        if clip_precision == "int8_dp4a":
            require_sm70()
            from .ops import FNI8Ops

            clip_model_options["custom_operations"] = FNI8Ops

        clip = comfy.sd.load_text_encoder_state_dicts(
            [comfy.utils.load_torch_file(p, safe_load=True) for p in clip_paths],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=clip_model_options,
        )
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path, safe_load=True))
        return (clip, vae)


class FNI8AttentionPatch:
    """Swap a MODEL's self-attention for bidirectional int8 dp4a FlashAttention.

    **Sliding-Tile Attention (STA)** — when ``sta_enabled`` and the model is a
    video DiT with a Conv3d patch_embedding, self-attention is restricted to a
    local 3D spatio-temporal tile, cutting FLOPs with minimal quality loss.

    STA quality is measured via masked SDPA (the int8 kernel does not yet support
    block-sparse masks). A ``StaGate`` fallback demotes any call-site where STA
    degrades quality below the SQNR floor back to dense attention. This is
    POST-1.0 / draft-PR work — the quality-cost frontier is being charted.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",), "arch": (["auto", *arch.families()],)},
            "optional": {
                "sta_enabled": (("disabled", "enabled"),),
                "sta_tile_f": ("INT", {"default": 9, "min": 1, "max": 256}),
                "sta_tile_h": ("INT", {"default": 8, "min": 1, "max": 256}),
                "sta_tile_w": ("INT", {"default": 8, "min": 1, "max": 256}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "fni8"
    TITLE = "FNI8 Attention Patch (bidirectional int8 FA + optional STA)"

    def patch(
        self, model, arch="auto", sta_enabled="disabled", sta_tile_f=9, sta_tile_h=8, sta_tile_w=8
    ):
        require_comfy()
        require_sm70()
        m = model.clone()
        a = arch_get(arch)
        if a is not None:
            seam = a.attn_seam
        else:
            seam = _detect_seam(m)

        sta_layout: tuple[int, int, int] | None = None
        sta_tile: tuple[int, int, int] | None = None
        if sta_enabled == "enabled":
            sta_layout = _infer_sta_layout(m)
            sta_tile = (sta_tile_f, sta_tile_h, sta_tile_w)

        # One SQNR gate per patched model: it compares int8 dp4a vs fp SDPA per attention
        # shape signature and demotes the site to fp if its SQNR is below the floor (issue
        # #82). By default the verdict is taken from the first call; set
        # $FNI8_SQNR_CALIB_SAMPLES>1 to instead calibrate across several denoise timesteps
        # and decide on the WORST SQNR (a per-timestep collapse a single-activation gate
        # misses — see docs/int8-dit-validation.md), and $FNI8_SQNR_REVALIDATE_EVERY>0 to
        # periodically re-check passing sites. Fresh instance so decisions never leak
        # across models.
        from .int8_linear import sqnr_calib_samples

        gate = Int8AttnGate(
            calib_samples=sqnr_calib_samples(),
            revalidate_every=_sqnr_revalidate_every(),
            sites_per_signature=_attention_sites_per_signature(m),
        )
        # Issue #196: try to restore previously-calibrated verdicts from disk so the
        # ~200s cold-start calibration only runs once per GPU + model combination.
        gate.load()
        if seam == "attn1_replace":
            callback = make_attn1_replace_callback(gate, sta_layout=sta_layout, sta_tile=sta_tile)
            for block in _self_attn_blocks(m):
                m.set_model_attn1_replace(callback, *block)
        else:
            # Flux/SD3-style DiTs bypass attn1_replace -> wrap optimized_attention.
            _wrap_optimized_attention(m, gate, sta_layout=sta_layout, sta_tile=sta_tile)
        # Wire post-denoise save hook so finalized verdicts persist across restarts.
        # (save is triggered automatically by _int8_attn_gated._maybe_save after all
        # signatures are finalized)
        return (m,)


def globals_arch_prefix(name):
    a = arch.get(name)
    return bool(a and a.key_prefix)


def arch_get(name):
    return None if name == "auto" else arch.get(name)


def _sqnr_revalidate_every(default: int = 0) -> int:
    """How often (in calls) a passing attention site is re-measured, from
    ``$FNI8_SQNR_REVALIDATE_EVERY`` (default 0 == off). See `attention.Int8AttnGate`."""
    import os

    try:
        return max(0, int(os.environ.get("FNI8_SQNR_REVALIDATE_EVERY", default)))
    except (TypeError, ValueError):
        return default


def _resolve_clip_type(a):
    """Map an arch's best-effort `clip_type` hint to `comfy.sd.CLIPType`. When the
    hint is empty (unconfirmed arch) returns ``None`` so ComfyUI auto-detects the
    text encoder from the state dict."""
    import comfy.sd

    if not a.clip_type:
        return None
    try:
        return comfy.sd.CLIPType[a.clip_type.upper()]
    except KeyError:
        raise ValueError(
            f"comfy.sd.CLIPType has no member '{a.clip_type.upper()}' for arch "
            f"'{a.name}'. This pack's clip_type hints are best-effort against a "
            "specific ComfyUI version — check yours (CLIPLoader/DualCLIPLoader `type` "
            "dropdown) and file an issue if it has changed."
        ) from None


def _reject_fni8(path, kind):
    """`.fni8` is the DiT-only int8 container (see loader.py) — the TE/VAE must stay
    fp, so catch a misplaced `.fni8` here instead of failing deep inside comfy.sd with
    a confusing parse error."""
    if path and path.endswith(".fni8"):
        raise ValueError(
            f"{kind} '{path}' is a `.fni8` (int8 DiT) file — the text encoder and VAE "
            "must stay fp. Point this input at the arch's original fp checkpoint."
        )


def _self_attn_blocks(model):
    """Yield (block_name, index) tuples for BasicTransformerBlock-style models.
    Enumerate the actual input/middle/output blocks from the diffusion_model so we
    never register spurious blocks for blocks that don't exist."""
    dit = model.model.diffusion_model
    blocks = []
    for kind, attr in [("input", "input_blocks"), ("output", "output_blocks")]:
        container = getattr(dit, attr, None)
        if container is not None:
            for i in range(len(container)):
                blocks.append((kind, i))
    if getattr(dit, "middle_block", None) is not None:
        blocks.append(("middle", 0))
    if not blocks:
        for kind in ("input", "output"):
            container = getattr(dit, f"{kind}_block", None)
            if container is not None:
                for i in range(len(container)):
                    blocks.append((kind, i))
        if getattr(dit, "middle_block", None) is not None:
            blocks.append(("middle", 0))
    return blocks


def _attention_sites_per_signature(model) -> int:
    """Best-effort count of same-shape self-attention sites in one DiT forward.

    The optimized-attention override is shared by the whole model. A round-robin site
    ordinal therefore needs the transformer-block count so equal-shaped blocks retain
    independent SQNR decisions across denoise steps.
    """
    dit = model.model.diffusion_model
    for attr in ("transformer_blocks", "blocks", "joint_blocks"):
        blocks = getattr(dit, attr, None)
        if blocks is not None:
            try:
                count = len(blocks)
            except TypeError:
                continue
            if count:
                return count
    split_count = sum(
        len(blocks)
        for attr in ("double_blocks", "single_blocks")
        if (blocks := getattr(dit, attr, None)) is not None
    )
    if split_count:
        return split_count
    return max(1, len(_self_attn_blocks(model)))


def _infer_sta_layout(model) -> tuple[int, int, int] | None:
    """Best-effort 3D token layout inference for video DiTs.

    Tries to extract the spatio-temporal grid dimensions ``(F, H, W)`` from
    the model's ``patch_embedding`` (Conv3d) structure.  Returns ``None`` when
    the model does not appear to be a video DiT with a detectable layout;
    callers should fall back to dense attention (no STA).

    The heuristic assumes roughly 1:1 spatiotemporal ratio after patch embed
    for typical 81-frame 480p latent video (conservative — real operators
    should pass an explicit layout through ``transformer_options``).
    """
    dit = model.model.diffusion_model
    pe = getattr(dit, "patch_embedding", None)
    if pe is not None and hasattr(pe, "kernel_size") and len(pe.kernel_size) == 3:
        ps_t, ps_h, ps_w = pe.kernel_size
        # Infer from a typical video latent. This is a fallback heuristic;
        # production users should provide explicit F, H, W.
        return (81 // ps_t, 60 // ps_h, 40 // ps_w)
    return None


def _detect_seam(model):
    """Heuristic: if the diffusion_model has a standard UNet block layout we assume
    attn1_replace; otherwise object_patch (global improved-modules attention)."""
    dit = model.model.diffusion_model
    if hasattr(dit, "input_blocks") or hasattr(dit, "output_blocks"):
        return "attn1_replace"
    if hasattr(dit, "middle_block"):
        return "attn1_replace"
    return "object_patch"


def _wrap_optimized_attention(model, gate=None, sta_layout=None, sta_tile=None):
    """Route int8 dp4a self-attention into DiTs that call ComfyUI's `optimized_attention`
    directly (object-patch seam: Z-Image / Flux / SD3 / …), via ComfyUI's
    `transformer_options["optimized_attention_override"]` hook.

    NOT a global monkeypatch: 43 of ComfyUI's 44 DiT modules import `optimized_attention`
    BY VALUE (`from ...attention import optimized_attention`), so reassigning the module
    global is invisible to them (the reason the prior `forward`-patch + global-swap never
    engaged — ComfyUI also samples via `apply_model`→`diffusion_model`, not `forward`).
    ComfyUI's own `wrap_attn` decorator instead checks this override in `transformer_options`
    at each attention call and invokes it as `override(orig_fn, q, k, v, heads, ...)` —
    reaching every DiT regardless of import style, and scoped to this model's
    `transformer_options` (no cross-model leakage)."""
    from .attention import make_fni8_attn_override

    # fresh dict (don't mutate a transformer_options possibly shared with the source model)
    to = dict(model.model_options.get("transformer_options") or {})
    to["optimized_attention_override"] = make_fni8_attn_override(
        gate, sta_layout=sta_layout, sta_tile=sta_tile
    )
    model.model_options["transformer_options"] = to
    model.model_options["fni8_attn_wrapped"] = True


# ---------------------------------------------------------------------------
# Tiled / chunked VAE decode + tiled DiT (MultiDiffusion-style)
# ---------------------------------------------------------------------------


class TiledVAEDecode:
    """Decode a latent image through the VAE with spatial tiling, reducing peak HBM
    from ~37 GiB at 2048 px (FLUX) down to a tile-size budget (~2 GiB at 512 px).

    Uses overlapping tiles with a linear weight ramp to blend tile seams. No host
    offload — all operations stay on GPU.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "fni8/tiling"
    TITLE = "FNI8 Tiled VAE Decode"

    def decode(self, vae, samples, tile_size=512, overlap=64):
        from .memory import peak_hbm_monitor, format_hbm_report

        latent = samples["samples"]
        info = {}
        with peak_hbm_monitor() as info:
            result = _tiled_vae_decode(vae, latent, tile_size, overlap)
        info["hbm_report"] = format_hbm_report(info, "tiled vae decode")
        return (result,)


class ApplyTiledDiT:
    """Patch a diffusion MODEL to run each denoising step with MultiDiffusion-style
    spatial tiling. Compatible with ``FNI8AttentionPatch`` — the attention patch works
    through ``transformer_options`` and does not conflict with tiling.

    Tiling caps the activation working set to the tile area so resolutions that
    OOM today (e.g. 2048 px FLUX on 16 GB) run within budget. Overlap + weight-ramp
    blend avoids visible seams.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
                "overlap": ("INT", {"default": 32, "min": 4, "max": 256, "step": 4}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "fni8/tiling"
    TITLE = "FNI8 Apply Tiled DiT"

    def patch(self, model, tile_size=256, overlap=32):
        require_comfy()
        m = model.clone()
        _patch_dit_forward(m, tile_size, overlap)
        return (m,)


class VAETiledChunked:
    """Decode a video latent through a 3D-causal VAE in temporal chunks with
    feature-cache propagation.

    The full video activation set never materializes — each chunk of ``chunk_size``
    frames is decoded independently and the causal conv hidden states are carried
    across chunk boundaries. Falls back to :class:`TiledVAEDecode` per-chunk for
    very wide frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "chunk_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "fni8/tiling"
    TITLE = "FNI8 VAE Chunked Decode (video, 3D-causal)"

    def decode(self, vae, samples, chunk_size=4):
        from .memory import peak_hbm_monitor, format_hbm_report
        from .vae_tiled import chunked_vae_decode_3d

        latent = samples["samples"]
        info = {}
        with peak_hbm_monitor() as info:
            result = chunked_vae_decode_3d(
                lambda lat: vae.decode(lat),
                latent,
                chunk_size=chunk_size,
            )
        info["hbm_report"] = format_hbm_report(info, "chunked vae decode")
        return (result,)


def _patch_dit_forward(model, tile_size: int, overlap: int):
    """Replace the diffusion model's forward method with a tiled version.

    The tiled version crops the input latent into overlapping tiles, runs the
    original forward on each tile, and blends the results with a linear weight
    ramp (MultiDiffusion 2302.08113). Peak HBM scales with tile area, not the
    full latent.
    """
    from .dit_tiled import tiled_dit_forward

    dit = model.model.diffusion_model
    orig_forward = dit.forward

    def tiled_forward(x, timestep, **kwargs):
        return tiled_dit_forward(
            lambda tx, tt, **kw: orig_forward(tx, tt, **kw),
            x,
            timestep,
            tile_size=tile_size,
            overlap=overlap,
            **kwargs,
        )

    dit.forward = tiled_forward


def _tiled_vae_decode(vae, latent, tile_size, overlap):
    """Helper: run tiled VAE decode using comfy.sd.VAE.decode."""
    from .vae_tiled import tiled_vae_decode

    return tiled_vae_decode(
        lambda lat: vae.decode(lat),
        latent,
        tile_size=tile_size,
        overlap=overlap,
    )


class FNI8Multigpu:
    """Multi-GPU execution node: dispatches the denoise step across devices.

    Two strategies:
      - ``component_parallel`` (VALIDATED): text-encoder / DiT / VAE placed on
        separate GPUs to cut peak VRAM on the primary card (zero denoise-loop
        traffic, no batch split, no numeric change — a pure device move).
      - ``cfg_parallel`` (NOT quality-validated on the int8 path — see below):
        cond/uncond intended to run on separate GPUs concurrently. MEASURED on real
        int8 Z-Image: splitting the batched cond+uncond forward into two batch-1
        forwards does NOT reproduce the single-GPU batched output — cos ≈ 0.88 vs
        the cos ≥ 0.99 quality bar (the int8 DiT forward is batch-size-dependent;
        the raw dp4a kernels are batch-independent, but the full model prediction at
        B=1 differs from B=2). It also cannot deep-copy an int8 FNI8Tensor DiT onto
        the second GPU. So this strategy SAFELY FALLS BACK to single-GPU rather than
        silently shipping a divergent image. Kept as a documented, guarded path
        pending a batch-invariant int8 DiT forward.

    Falls back to the unmodified model when only one GPU is available, making it
    safe to include in graphs regardless of the hardware."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strategy": (["none", "cfg_parallel", "component_parallel"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "fni8"
    TITLE = "FNI8 Multi-GPU"

    def apply(self, model, strategy):
        require_comfy()
        if strategy == "none":
            return (model,)
        if strategy == "cfg_parallel":
            return self._enable_cfg_parallel(model)
        if strategy == "component_parallel":
            return self._enable_component_parallel(model)
        return (model,)

    def _enable_cfg_parallel(self, model):
        """CFG-parallel: intended to run cond/uncond on separate GPUs concurrently.

        GUARDED / SAFE-FALLBACK. Two hard blockers, both MEASURED on real int8
        Z-Image (see the class docstring and docs/multigpu-cfg-parallel-finding.md):

          1. Quality: splitting the batched cond+uncond forward into two batch-1
             forwards yields cos ≈ 0.88 vs the single-GPU batched reference — below
             the cos ≥ 0.99 bar. The int8 DiT forward is batch-size-dependent even
             though the dp4a kernels are batch-independent.
          2. Mechanics: the int8 DiT weight is an ``FNI8Tensor`` subclass that has no
             ``__deepcopy__``, so a second-GPU replica cannot be deep-copied.

        Rather than silently ship a divergent image (or crash on the deepcopy), this
        returns the model unchanged. The planning helpers and wrapper logic remain in
        ``multigpu`` (unit-tested) so a future batch-invariant forward can re-enable
        this path by measurement, not by assumption."""
        import logging

        from .multigpu import available_devices, plan_cfg_parallel

        devices = available_devices()
        plan = plan_cfg_parallel(devices)
        if plan.cond_device == plan.uncond_device:
            return (model,)

        logging.getLogger(__name__).warning(
            "FNI8Multigpu: cfg_parallel is NOT quality-validated on the int8 DiT path "
            "(measured cos ~0.88 vs the 0.99 bar; the int8 forward is batch-size "
            "dependent, and int8 FNI8Tensor weights can't be deep-copied to a second "
            "GPU). Falling back to single-GPU execution. Use component_parallel for a "
            "validated multi-GPU VRAM win."
        )
        return (model,)

    def _enable_component_parallel(self, model):
        """Place the diffusion model on a secondary GPU to free VRAM on the
        primary card for the text encoder / VAE / other graph nodes.

        Only moves the model when at least two GPUs are available."""
        from .multigpu import available_devices, plan_components

        devices = available_devices()
        placement = plan_components(devices)
        if placement.dit == "cuda:0" and placement.text_encoder == "cuda:0":
            return (model,)

        dit = model.model.diffusion_model
        dit.to(placement.dit)
        if hasattr(model, "load_device"):
            model.load_device = torch.device(placement.dit)

        mo = dict(model.model_options)
        mo["fni8_component_parallel"] = True
        m = model.clone()
        m.model_options = mo
        return (m,)


class FNI8PipelineParallel:
    """Split a DiT's transformer blocks across GPUs (pipeline-parallel) so a DiT that
    exceeds one card becomes runnable, handing the boundary hidden state across the link
    COMPRESSED with the fni8 transport codec (`multigpu.TransportCodec` over
    `superl8.transport`). One codec handoff per stage boundary per denoise step — never
    per-layer (the 250 MB/s PCIe-1.0-x1 verdict).

    Target: LTX-2.3, whose int8 DiT is 23.5 GB (16.6 video + 6.9 audio) and does NOT fit
    a 16 GB card. This node physically distributes the model — prologue / embeddings /
    connectors / output head + stage-0 blocks on GPU 0, the rest of the blocks on the
    later GPUs — and installs the split `_process_transformer_blocks`. Each card then
    holds ~half the weight; `install_ltxav_pipeline`'s report confirms per-card VRAM.

    STATUS / honest scope:
      * The split + transport mechanism is LIVE for `LTXAVModel`: after this node the
        model's `diffusion_model._forward(...)` runs blocks split across GPUs with a
        codec handoff at each boundary, and `diffusion_model._fni8_pipeline_last_stats`
        holds the transport metrics (bytes / ratio / latency / SQNR). This is validated
        by the direct-drive e2e (`tests/e2e/test_pipeline_ltx.py`).
      * It is a memory-FIT strategy, NOT a speedup: stages serialise and each step pays
        one codec handoff. It only helps when the DiT does not fit one card.
      * ComfyUI's own sampler moves a MODEL to a single `load_device` via
        `model_management`, which would re-collapse the split (and OOM). Driving the
        split through the stock `KSampler` therefore needs a model-management bypass —
        follow-up. Use the distributed `diffusion_model` directly (the e2e does) to run
        the split forward today.
      * Non-LTXAV DiTs: the plan is computed and stashed in `transformer_options`
        (`fni8_pipeline_plan`) but not auto-installed (each arch's block loop differs);
        wiring another arch means adding its split `_process_*` like `pipeline_ltx`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "num_gpus": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1}),
                "transport_scheme": (["int8", "int4", "nf4", "fp16"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "fni8"
    TITLE = "FNI8 Pipeline Parallel (LTX-2 direct-drive only; not run by stock KSampler)"
    DESCRIPTION = (
        "Splits an oversized DiT's transformer blocks across GPUs so it becomes runnable "
        "(a memory-FIT strategy, NOT a speedup). HONEST SCOPE: the split actually EXECUTES "
        "only for the LTX-2 audio+video DiT (LTXAVModel) when its distributed "
        "`diffusion_model` is driven directly (see tests/e2e/test_pipeline_ltx.py). Stock "
        "ComfyUI KSampler moves the MODEL back to a single load_device via model_management, "
        "which re-collapses the split (and can OOM) — a KSampler bypass is follow-up. For any "
        "NON-LTXAV DiT this node only computes a plan and stashes it; nothing executes it. "
        "It does NOT make PP work through a normal KSampler graph today."
    )

    def patch(self, model, num_gpus=2, transport_scheme="int8"):
        require_comfy()
        import logging

        from .multigpu import (
            TransportCodec,
            available_devices,
            count_dit_blocks,
            plan_pipeline,
        )

        m = model.clone()
        dit = m.model.diffusion_model
        devs = available_devices()
        if len(devs) < 2:
            logging.getLogger(__name__).warning(
                "FNI8PipelineParallel: needs >=2 CUDA GPUs to split; returning model "
                "unchanged (single-GPU)."
            )
            return (m,)
        devices = devs[:num_gpus]
        num_blocks = count_dit_blocks(dit)
        # Stage 0 also carries the prologue/embeddings/connectors/output head. For
        # LTX-2.3 that non-block weight is ~5 blocks' worth of VRAM (connectors ~4 GB +
        # prologue/output ~0.85 GB, ~0.39 GB/block) -> bias ~10 blocks off stage 0.
        head_bias = 10.0 if _is_ltxav(dit) else 0.0
        plan = plan_pipeline(
            num_blocks,
            devices=devices,
            transport_scheme=transport_scheme,
            head_weight_blocks=head_bias,
        )

        to = dict(m.model_options.get("transformer_options") or {})
        to["fni8_pipeline_plan"] = plan
        to["fni8_pipeline_num_blocks"] = num_blocks
        m.model_options["transformer_options"] = to

        if _is_ltxav(dit):
            from .pipeline_ltx import install_ltxav_pipeline

            report = install_ltxav_pipeline(dit, plan, TransportCodec(scheme=transport_scheme))
            m.model_options["fni8_pipeline_vram"] = report
            # The split EXECUTES only when the distributed diffusion_model is driven
            # directly. Stock ComfyUI KSampler re-collapses it onto one load_device, so
            # flag that this MODEL requires direct-drive and say so loudly — never let the
            # node silently imply PP runs in a normal KSampler graph.
            m.model_options["fni8_pipeline_requires_direct_drive"] = True
            gb = {d: round(b / 1e9, 2) for d, b in report.items()}
            logging.getLogger(__name__).info(
                f"FNI8PipelineParallel: split LTXAV {num_blocks} blocks across "
                f"{plan.stage_ranges} on {plan.devices}; per-card weight GB={gb}"
            )
            logging.getLogger(__name__).warning(
                "FNI8PipelineParallel: LTXAV blocks are split across %s, but stock ComfyUI "
                "KSampler moves the MODEL to a single load_device (model_management) which "
                "will RE-COLLAPSE the split and can OOM. The split only runs when the "
                "distributed diffusion_model is driven directly (see "
                "tests/e2e/test_pipeline_ltx.py); a KSampler bypass is follow-up.",
                plan.devices,
            )
        else:
            logging.getLogger(__name__).warning(
                "FNI8PipelineParallel: %s is not an LTXAVModel — a plan was computed and "
                "stashed in transformer_options['fni8_pipeline_plan'] but NOTHING executes "
                "it (per-arch split wiring is follow-up). This node is a NO-OP for this "
                "architecture; it does not run pipeline-parallel here.",
                type(dit).__name__,
            )
        return (m,)


class FNI8TeaCache:
    """Cross-step DiT output caching (TeaCache) — skips recomputing the whole DiT on
    denoise steps whose timestep-conditioned input barely changed, reusing the previous
    step's prediction. FLOPs are the DiT bottleneck, so skipped steps stack
    *multiplicatively* on the int8 dp4a win (int8 ~2x x TeaCache ~1.5-2x ~= 3-4x).

    Arch-agnostic: hooks ComfyUI's ``WrappersMP.DIFFUSION_MODEL`` seam (the same
    `transformer_options` plumbing `FNI8AttentionPatch` uses), so one code path covers
    single-stream Z-Image, double-stream Qwen-Image-Edit, and audio+video LTX-2.3.

    ``rel_l1_thresh`` is the speed/quality dial: higher => more skips => faster but lower
    fidelity. Caching TRADES quality for speed and the error accumulates across steps, so
    gate the FINAL latent (cosine >= 0.985 vs the no-cache baseline) at the chosen
    threshold — see bench/teacache_sweep.py for the per-model knee. Compatible with
    ``FNI8AttentionPatch``; place this node after it (order does not matter — different
    seams). ``rel_l1_thresh=0`` disables skipping (passthrough)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0,
                                            "step": 0.01,
                                            "tooltip": "Cumulative rel-L1 skip threshold. "
                                            "Higher = more skips = faster/lower quality. "
                                            "0 disables caching."}),
                "warmup_steps": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1,
                                         "tooltip": "Always compute the first N steps."}),
                "max_consecutive_skips": ("INT", {"default": 3, "min": 1, "max": 20,
                                                  "step": 1,
                                                  "tooltip": "Hard cap on skips in a row "
                                                  "(bounds cache staleness)."}),
                "strategy": (["taylor", "tea"], {"tooltip": "taylor scales the skip signal "
                                                 "by a running output/input sensitivity "
                                                 "(higher quality); tea uses the raw "
                                                 "rel-L1."}),
                "mode": (["auto", "residual", "output"], {"tooltip": "residual caches the "
                         "transformer-block delta and re-runs the timestep-conditioned "
                         "projections every step (higher quality, single-stream DiTs only); "
                         "output reuses the whole prediction (arch-agnostic). auto = "
                         "residual when supported, else output."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "fni8"
    TITLE = "FNI8 TeaCache (cross-step DiT cache)"

    def patch(self, model, rel_l1_thresh=0.15, warmup_steps=1, max_consecutive_skips=3,
              strategy="taylor", mode="auto"):
        require_comfy()
        import comfy.patcher_extension

        from .teacache import TeaCacheController, find_block_list, make_teacache_wrapper

        m = model.clone()
        if rel_l1_thresh <= 0.0:
            return (m,)  # passthrough: no wrapper registered
        dit = m.model.diffusion_model
        # residual caching needs a single-stream block ModuleList; fall back to whole-output
        # caching (arch-agnostic) when the arch isn't supported or the user forces it.
        single_stream = find_block_list(dit) is not None
        if mode == "auto":
            resolved = "residual" if single_stream else "output"
        elif mode == "residual" and not single_stream:
            resolved = "output"
        else:
            resolved = mode
        controller = TeaCacheController(
            rel_l1_thresh=rel_l1_thresh, warmup_steps=warmup_steps,
            max_consecutive_skips=max_consecutive_skips, strategy=strategy, mode=resolved)
        # Stash on the patcher so a bench/test can read the skip telemetry after sampling.
        m.model_options["fni8_teacache_controller"] = controller
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "fni8_teacache",
            make_teacache_wrapper(controller, dit=dit),
        )
        return (m,)


def _is_ltxav(dit) -> bool:
    """True if the diffusion model is an LTX-2 audio+video DiT (has the split-execution
    hook `_process_transformer_blocks` and an `audio_adaln_single` submodule)."""
    return (
        hasattr(dit, "transformer_blocks")
        and hasattr(dit, "_process_transformer_blocks")
        and any("audio_adaln_single" in n for n, _ in dit.named_modules())
    )


class FNI8StepController:
    """Unified step-controller — the SINGLE KNOB for step-budget optimization.

    Coordinates the three step-cutting techniques so they STACK instead of CANCEL:
    TeaCache/TaylorSeer (temporal), step-adaptive W4A8→int8 precision (cost), and the
    AYS scheduler / distilled checkpoints (temporal). Given the model and a target
    step budget it emits one coordinated plan and wires it onto the model, guarding
    against the two anti-synergies (spent temporal pool at low step counts; W4A8 and
    cache errors co-aligning on the same step). See ``step_controller.py``.

    Drop this AFTER ``UnetLoaderFNI8`` and feed its MODEL output to the sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "arch": (
                    ["zimage", "qwen_image", "flux1", "flux2", "ltx", "wan", "sd3"],
                    {"default": "qwen_image"},
                ),
                "target_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "distilled": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "allow_w4a8": ("BOOLEAN", {"default": True}),
                "allow_cache": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "plan")
    FUNCTION = "control"
    CATEGORY = "fni8"
    TITLE = "FNI8 Step Controller (unified step-budget knob)"

    def control(self, model, arch, target_steps, distilled, allow_w4a8=True, allow_cache=True):
        require_comfy()
        from .step_controller import ModelProfile, QualityGate, apply_plan, plan_steps

        prof = ModelProfile(
            arch=arch,
            distilled=distilled,
            is_video=arch in ("ltx", "wan"),
            memory_bound=arch in ("ltx", "wan"),
        )
        gate = QualityGate(allow_w4a8=allow_w4a8, allow_cache=allow_cache)
        plan = plan_steps(prof, target_steps, gate)
        patched, handles = apply_plan(model, plan)

        lines = [repr(plan), ""]
        lines += [f"- {r}" for r in plan.rationale]
        if plan.warnings:
            lines += [""] + [f"! {w}" for w in plan.warnings]
        if handles.get("skipped"):
            lines += ["", "(unwired, module not present: " + "; ".join(handles["skipped"]) + ")"]
        return (patched, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "UnetLoaderFNI8": UnetLoaderFNI8,
    "UnetLoaderFNI8GGUF": UnetLoaderFNI8GGUF,
    "FNI8StepController": FNI8StepController,
    "FNI8ComponentLoader": FNI8ComponentLoader,
    "FNI8AttentionPatch": FNI8AttentionPatch,
    "TiledVAEDecode": TiledVAEDecode,
    "ApplyTiledDiT": ApplyTiledDiT,
    "VAETiledChunked": VAETiledChunked,
    "FNI8Multigpu": FNI8Multigpu,
    "FNI8PipelineParallel": FNI8PipelineParallel,
    "FNI8TeaCache": FNI8TeaCache,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
