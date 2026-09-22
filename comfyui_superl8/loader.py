# SPDX-License-Identifier: Apache-2.0
"""`.fni8` DiT weight loader + a DiT-state-dict quantizer.

Reuses the SAME `.fni8` container as the LLM side (`fni8.format`), so one converter
and one format serve both worlds. `quantize_dit_state_dict` turns a ComfyUI/diffusers
DiT state dict (fp16) into int8 `per_row_i8` QTensors keyed to the model's own keys;
`load_fni8_dit` reads them back. Norms/embeddings/modulation stay fp16 (raw).
"""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from superl8 import QTensor

# A 2-D weight (in % 4 == 0) is a quantizable DiT linear UNLESS its name marks it as
# a norm, an adaLN/modulation projection, a positional/patch/time/text embedder — those
# stay fp (numerically load-bearing or memory-bound). The output-head projections
# (`final_layer`, `proj_out`, `context_embedder`, `guidance_in`) were previously in the
# skip-list (Nunchaku heuristic) but are recalibrated to int8 here: the 40:1 int8-vs-bf16
# speedup on this fleet makes it worth trying, and the runtime SQNR gate in `FNI8Ops`
# catches any per-instance outliers. See `bench/recalibrate_skip_list.py`.
_SKIP_HINTS = ("norm", "modulation", "adaln", "ada_ln", "scale_shift_table",
               "pos_embed", "patch_embed",
               "_embed", "embedder", "time_in", "txt_in", "img_in", "vector_in",
               "guidance_in", "context_embedder", "final_layer", "proj_out")

# Recalibrated projections: these were in `_SKIP_HINTS` (always bf16) but are genuine
# linears that pass per-op SQNR (cos >= 0.99) on representative activations.  The broad
# `_embed`/`embedder` patterns in `_SKIP_HINTS` would still catch `context_embedder` etc.,
# so this override list is checked FIRST to exempt them from the skip-list.  The runtime
# SQNR gate (`FNI8Ops.Linear.forward_comfy_cast_weights`) catches any per-instance outliers.
# Recalibrated: bench/recalibrate_skip_list.py
_LOADER_SKIP_OVERRIDE = ("final_layer", "proj_out", "context_embedder", "guidance_in")


def _is_dit_linear(name: str, w: torch.Tensor) -> bool:
    if w.dim() != 2 or w.shape[-1] % 4 != 0:
        return False
    n = name.lower()
    # Override: recalibrated projections go through to int8 (runtime gate catches issues).
    if any(s in n for s in _LOADER_SKIP_OVERRIDE):
        return True
    return not any(s in n for s in _SKIP_HINTS)


def quantize_dit_state_dict(sd: dict, *, keep_fp16: tuple = ()) -> dict:
    """DiT state dict -> {name: QTensor}. Linears int8; the rest stay raw in the
    model's NATIVE dtype. `keep_fp16` forces extra name substrings to stay raw.

    Raw tensors keep their source dtype (do NOT force fp16): modern DiTs (Z-Image,
    Flux, SD3, Wan, ...) are bf16-native, and fp16 overflows their activations to
    NaN -> black images. The int8 scales are computed in fp32 regardless."""
    from superl8 import QTensor
    from superl8.quant.core import quantize_int8_rowwise

    out: dict[str, "QTensor"] = {}
    for name, w in sd.items():
        w = w.detach().cpu()
        if _is_dit_linear(name, w) and not any(s in name for s in keep_fp16):
            q, s = quantize_int8_rowwise(w)
            out[name] = QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")
        else:
            out[name] = QTensor(w, None, scheme="raw")   # keep native dtype (bf16/fp16)
    return out


def quantize_dit_state_dict_w4(
    sd: dict, *, group_size: int = 128, keep_fp16: tuple = ()
) -> dict:
    """DiT state dict -> {name: QTensor}. Linears int4 (W4A8 group-wise packed);
    norms/embeds/keep_fp16 stay raw. Half the weight memory of the W8A8 path
    (``quantize_dit_state_dict``); each linear is routed to fni8's ``gemm_w4a8`` dp4a
    kernel at runtime, so the ``per_group_i4`` QTensor MUST carry ``codebook='int4'``.

    Each linear weight is quantized per-group of ``group_size`` columns with symmetric
    int4 range [-7, 7], then packed (two 4-bit values per byte) into a
    ``scheme='per_group_i4'`` QTensor."""
    from superl8 import QTensor

    from .int8_linear import quantize_weight_int4

    out: dict[str, "QTensor"] = {}
    for name, w in sd.items():
        w = w.detach().cpu()
        if _is_dit_linear(name, w) and not any(s in name for s in keep_fp16):
            packed, scale = quantize_weight_int4(w, group_size=group_size)
            out[name] = QTensor(
                packed.contiguous(), scale.contiguous(),
                scheme="per_group_i4", group_size=group_size, codebook="int4",
            )
        else:
            out[name] = QTensor(w, None, scheme="raw")   # keep native dtype (bf16/fp16)
    return out


def _dominant_dtype(sd: dict) -> str:
    """The DiT's native compute dtype (the modal dtype of its raw float tensors)."""
    from collections import Counter

    c = Counter(str(w.dtype).replace("torch.", "") for w in sd.values()
                if getattr(w, "is_floating_point", lambda: False)())
    return c.most_common(1)[0][0] if c else "bfloat16"


def save_dit_fni8(path: str, sd: dict, *, arch: str | None = None, keep_fp16: tuple = ()):
    """Quantize a DiT state dict and write a `.fni8` (bytes == resident dp4a layout).
    Records `native_dtype` so the loader can run the model in the dtype it needs
    (bf16 for modern DiTs) instead of Volta's fp16 default that black-screens them."""
    from superl8 import save_fni8

    native_dtype = _dominant_dtype(sd)
    qsd = quantize_dit_state_dict(sd, keep_fp16=keep_fp16)
    save_fni8(path, qsd, meta={"kind": "dit", "arch": arch or "unknown",
                               "native_dtype": native_dtype})
    return qsd


def fni8_dit_native_dtype(path: str, default: str = "bfloat16") -> str:
    """Read the DiT's native compute dtype from a `.fni8` (meta `native_dtype`).
    Defaults to bf16 — the safe choice for modern DiTs, and fp16-native DiTs
    tolerate bf16 too (more exponent range, no overflow)."""
    from superl8 import FQReader

    with FQReader(path) as r:
        meta = r.header.get("__meta__") or {}
    return meta.get("native_dtype", default)


# The token -> pixel-patch output-head projections. When a `.fni8` bakes these as int8
# (a save-time bet, `_LOADER_SKIP_OVERRIDE` above), int8 error there lands DIRECTLY on
# each output pixel-patch, so we DEQUANTIZE them back to fp at load — their share of the
# DiT FLOPs is tiny (the int8 win is in the transformer blocks, not the head). Applies
# only where the head is actually int8 (e.g. Flux); it is a no-op on Z-Image, whose
# published `.fni8` already keeps `final_layer` raw/fp. (Z-Image's real collapse was a
# separate qkv-fusion bug in the ATTENTION path — see `DiTArch.fp_dequant` and
# docs/zimage-full-pipeline-findings.md — fixed by per-arch `fp_dequant`, not this list.)
_HEAD_FP_DEQUANT: tuple[str, ...] = ("final_layer", "proj_out", "scale_shift_table")
# NOTE `scale_shift_table` (adaLN modulation tables) is ALSO in `_HEAD_FP_DEQUANT` because
# already-shipped `.fni8`s (LTX-2.3) baked these tiny [n_ada, dim] modulation tables to
# int8 before `scale_shift_table` entered `_SKIP_HINTS`. They are NOT a FLOP-bulk linear —
# they are per-token adaLN scale/shift/gate rows that the block INDEXES (`table[slice]`),
# and the indexed-then-dequant path on an int8 `FNI8Tensor` is broken (and would put int8
# error straight onto every token's modulation). Dequant them to fp at load so the block
# drives them as plain fp tensors; freshly-converted `.fni8`s keep them raw fp instead.


def _dequant_qtensor(qt, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize a quantized QTensor back to a plain fp tensor. `per_row_i8` is
    int8 [out,in] * fp32 per-row scale [out]; `per_group_i4` unpacks the packed nibbles
    and applies the per-group scale (the sub-8-bit DiT head/adaLN tables the loader routes
    to fp). Any other quantized scheme raises rather than silently corrupting the weight."""
    scheme = getattr(qt, "scheme", "per_row_i8")
    if scheme == "per_group_i4":
        from .int8_linear import dequantize_int4_weight

        N, cols = qt.data.shape
        return dequantize_int4_weight(
            qt.data, qt.scale, qt.group_size, N, cols * 2, dtype
        )
    if scheme != "per_row_i8":
        raise ValueError(f"_dequant_qtensor: unsupported scheme {scheme!r}")
    return (qt.data.float() * qt.scale.unsqueeze(1)).to(dtype)


# LTX-2.3 ships the caption/text projections OUTSIDE the DiT sub-tree, at the archive
# TOP LEVEL under `text_embedding_projection.{video,audio}_aggregate_embed.*` — a name
# ComfyUI's `comfy.ldm.lightricks.av_model` LTXAV model does NOT build. The model instead
# expects the single-linear `NormSingleLinearTextProjection` under
# `caption_projection.linear_1.*` (video text) and `audio_caption_projection.linear_1.*`
# (audio text). Without this rename the model random-inits those projections -> the
# conditioning path is garbage -> the latent goes NaN. Pure rename: the checkpoint stores
# them as raw fp `nn.Linear` [out, in] exactly like comfy, so no transpose/split is
# needed. (`caption_channels`=188160 must ALSO be injected — see `ltx2_detection_metadata`
# — or comfy builds the Linear at its 3840 default and the [out, 188160] weight mismatches.)
_LTX2_CAPTION_REMAP: dict[str, str] = {
    "text_embedding_projection.video_aggregate_embed.weight": "caption_projection.linear_1.weight",
    "text_embedding_projection.video_aggregate_embed.bias": "caption_projection.linear_1.bias",
    "text_embedding_projection.audio_aggregate_embed.weight": "audio_caption_projection.linear_1.weight",
    "text_embedding_projection.audio_aggregate_embed.bias": "audio_caption_projection.linear_1.bias",
}


def load_fni8_dit(path: str, *, device: str = "cuda", strip_prefix: str = "",
                  keep_only_prefixed: bool = False,
                  dequant_fp: tuple[str, ...] = _HEAD_FP_DEQUANT) -> dict:
    """Load a `.fni8` DiT -> {key: QTensor(int8) | Tensor(fp16)}, ready to feed the
    ComfyUI ops patch. Optionally strip a key prefix (e.g. 'diffusion_model.').

    `keep_only_prefixed` loads ONLY the tensors whose name starts with `strip_prefix`
    and skips the rest entirely (never touches them on disk). This is the memory-cheap
    path for a bundled full checkpoint (LTX-2/2.3: the DiT lives under
    `model.diffusion_model.*` alongside a `vae.`/`audio_vae.`/`vocoder.` the DiT loader
    must not carry) — we read only the ~23 GB DiT, not the whole ~28 GB archive.

    `dequant_fp` is a tuple of name substrings whose int8 weights are DEQUANTIZED back to
    fp on load (the output-head projections, `_HEAD_FP_DEQUANT`) so they run on the fp
    path even though the shipped `.fni8` baked them int8. Pass `()` to force the whole DiT
    int8 (used by the bisect harness to reproduce the collapse)."""
    from superl8 import FQReader

    from .superl8_tensor import FNI8Tensor

    out: dict = {}
    with FQReader(path) as r:
        for name in r.names:
            # LTX-2.3's caption/text projections live at the archive top level (outside
            # `model.diffusion_model.`), so `keep_only_prefixed` would drop them; capture
            # + rename them onto the keys the LTXAV model builds (see _LTX2_CAPTION_REMAP).
            remap_key = _LTX2_CAPTION_REMAP.get(name)
            if remap_key is None and keep_only_prefixed and not (
                strip_prefix and name.startswith(strip_prefix)):
                continue
            qt = r.get_qtensor(name, device=device)
            if remap_key is not None:
                key = remap_key
            else:
                key = name[len(strip_prefix):] if strip_prefix and name.startswith(strip_prefix) else name
            if getattr(qt, "scheme", None) == "raw":
                out[key] = qt.data                              # norms/embeds: plain tensor
            elif dequant_fp and any(s in key.lower() for s in dequant_fp):
                out[key] = _dequant_qtensor(qt)                 # head proj: int8 -> fp
            else:
                # Quantized linear -> tensor-like, tagged with its scheme so the forward
                # path dispatches W8A8 (per_row_i8) vs W4A8 (per_group_i4) dp4a.
                out[key] = FNI8Tensor(
                    qt.data, qt.scale,
                    scheme=getattr(qt, "scheme", "per_row_i8"),
                    group_size=getattr(qt, "group_size", 0),
                    codebook=getattr(qt, "codebook", ""),
                )
    return out


# The DiT prefix inside a bundled full-checkpoint `.fni8` (e.g. LTX-2/2.3, which pack
# the video DiT, the VAE, the audio VAE, the vocoder and the text-embedding projection
# into one file, exactly as ComfyUI's own `CheckpointLoaderSimple` expects). ComfyUI's
# `model_detection.unet_prefix_from_state_dict` uses the same string for ldm/sgm models.
_BUNDLE_DIT_PREFIX = "model.diffusion_model."


def dit_bundle_prefix(path: str) -> str:
    """Return the `model.diffusion_model.` sub-prefix if a `.fni8` is a bundled full
    checkpoint (its DiT nested under that prefix, LTX-2/2.3 style), else "". Reads only
    the key inventory from the header — no tensor data is loaded."""
    from superl8 import FQReader

    with FQReader(path) as r:
        for name in r.names:
            if name.startswith(_BUNDLE_DIT_PREFIX):
                return _BUNDLE_DIT_PREFIX
    return ""


# LTX-2 adaLN-single modulation-row count (comfy.ldm.lightricks.av_model): a plain LTX
# block has ADALN_BASE_PARAMS_COUNT=6 scale/shift/gate rows; LTX-2.3 turns on
# prompt-conditioned cross-attention adaLN, so each block's `scale_shift_table` gains 3
# more -> ADALN_CROSS_ATTN_PARAMS_COUNT=9 (and the model grows `prompt_adaln_single`).
_ADALN_CROSS_ATTN_ROWS = 9
# The Embeddings1DConnector packs its channel dim as num_heads * head_dim; LTX-2's
# connectors use head_dim 128 (comfy default `connector_attention_head_dim`).
_CONNECTOR_HEAD_DIM = 128

# LTX-2 (`ltxav`) architecture-family FORWARD scalars that comfy's `LTXAVModel` ctor
# defaults get WRONG (its defaults target no particular checkpoint) but every shipped
# LTX-2 checkpoint sets — and which the `.fni8` quantization dropped. Verified constant
# across the official `ltx2_19b_config.json` / `ltx2_22b_config.json` `transformer`
# blocks, and they do NOT manifest as any tensor shape, so they cannot be inferred from
# the weights:
#   * causal_temporal_positioning=True   — temporal RoPE uses causal frame indices.
#   * use_middle_indices_grid=True       — the positional grid is built from mid-cell
#                                          indices (comfy default False shifts every RoPE
#                                          position -> wrong attention geometry).
#   * av_ca_timestep_scale_multiplier=1000.0 — the audio<->video cross-attention timestep
#                                          is scaled by 1000 (comfy default 1.0 collapses
#                                          `av_ca_factor = av_ca_ts_mult/ts_mult` from 1.0
#                                          to 0.001 -> the AV cross-attn adaLN is ~dead).
#   * rope_type="split"                 — LTX-2.3 was trained with split rotary layout;
#                                          comfy's interleaved default changes attention
#                                          geometry while remaining finite.
#   * frequencies_precision="float64"  — build the long positional grid at the
#                                          checkpoint's precision instead of comfy's
#                                          float32 default.
# A converter that records the real diffusers `transformer` config in the `.fni8` meta
# supersedes these (see `ltx2_persisted_transformer_config` + the `persisted` arg below);
# they are the honest fallback for the already-shipped `.fni8`s that predate that.
_LTXAV_FORWARD_SCALARS: dict = {
    "causal_temporal_positioning": True,
    "use_middle_indices_grid": True,
    "av_ca_timestep_scale_multiplier": 1000.0,
    "rope_type": "split",
    "frequencies_precision": "float64",
}


def ltx2_persisted_transformer_config(path: str) -> dict | None:
    """Read the diffusers `transformer` config a converter persisted into a `.fni8`'s
    meta (`meta['config']['transformer']`), or None if absent. This is the DURABLE
    source of the scalar forward config (`rope_type`, timestep multipliers,
    `causal_temporal_positioning`, …) that `ltx2_detection_metadata` cannot infer from
    tensor shapes alone — once present it is authoritative over the family-scalar
    fallback. Reads only the header, no tensor data."""
    from superl8 import FQReader

    with FQReader(path) as r:
        meta = r.header.get("__meta__") or {}
    cfg = meta.get("config")
    if isinstance(cfg, str):
        import json

        try:
            cfg = json.loads(cfg)
        except (ValueError, TypeError):
            return None
    if isinstance(cfg, dict):
        tf = cfg.get("transformer", cfg)
        if isinstance(tf, dict) and tf:
            return tf
    return None


def ltx2_persisted_vae_config(path: str) -> dict | None:
    """Read the 3D-causal *video-VAE* config a converter persisted into a `.fni8`'s meta
    (`meta['config']['vae']`), or None if absent. LTX-2.3's flat single-file release ships
    NO diffusers `config.json` — the ONLY place its VAE architecture lives is the
    checkpoint's safetensors `__metadata__['config']` blob, which the converter now copies
    verbatim into the `.fni8` meta (a `CausalVideoAutoencoder` config: `latent_channels`,
    the `encoder_blocks`/`decoder_blocks` compression schedule, `norm_layer`, `patch_size`,
    …). Without it `comfy.sd.VAE` falls back to sizing the LTX video VAE from a single conv
    shape + a built-in `version` preset (0/1/2), NONE of which matches LTX-2.3 — it mis-sizes
    the decoder and fails at `decoder.up_blocks.4` (512-vs-256 channel mismatch). Reads only
    the header, no tensor data."""
    from superl8 import FQReader

    with FQReader(path) as r:
        meta = r.header.get("__meta__") or {}
    cfg = meta.get("config")
    if isinstance(cfg, str):
        import json

        try:
            cfg = json.loads(cfg)
        except (ValueError, TypeError):
            return None
    if isinstance(cfg, dict):
        vae = cfg.get("vae")
        if isinstance(vae, dict) and vae:
            return vae
    return None


# The canonical LTX-2.3 `CausalVideoAutoencoder` config, packaged as durable loader data
# (`comfyui_superl8/data/ltx23_vae_config.json`), captured verbatim from the official
# `Lightricks/LTX-2.3` checkpoint's safetensors `__metadata__['config']['vae']`. It is the
# FALLBACK for already-shipped `.fni8`s that predate the converter persisting the VAE config
# (their meta is only kind/repo/bits/native_dtype) — exactly analogous to the family-scalar
# fallback `ltx2_detection_metadata` uses for the transformer config. `latent_channels=128`,
# 9 decoder blocks (comfy's version presets are 7 -> the 512-vs-256 mismatch), 8x temporal /
# 32x spatial compression. Verified: `VideoVAE(config=this)` loads the real bundle's 170
# `vae.*` weights with 0 missing / 0 unexpected (tests/e2e/test_ltx_vae_config.py).
_CANONICAL_LTX23_VAE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "data", "ltx23_vae_config.json")


def ltx2_canonical_vae_config() -> dict | None:
    """The known-good LTX-2.3 video-VAE config packaged with this node pack, or None if the
    data file is missing. Used only as a fallback when a `.fni8` has no persisted VAE config
    AND the bundle is an LTX-2.3 checkpoint (`meta['repo']`)."""
    import json

    try:
        with open(_CANONICAL_LTX23_VAE_CONFIG_PATH) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) and cfg.get("_class_name") == "CausalVideoAutoencoder" else None


def _ltx2_bundle_repo(path: str) -> str:
    """The `.fni8` meta `repo` string (e.g. `Lightricks/LTX-2.3`), or ''. Header-only read."""
    from superl8 import FQReader

    with FQReader(path) as r:
        meta = r.header.get("__meta__") or {}
    return str(meta.get("repo") or "")


def ltx2_vae_metadata(path: str) -> dict | None:
    """The `metadata=` blob `comfy.sd.VAE` expects (a `{'config': <json str>}`) to BUILD the
    LTX video VAE from the config instead of its shape/`version` auto-detect. `comfy.sd.VAE`
    reads `json.loads(metadata['config']).get('vae')` and hands that dict straight to
    `comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE(config=...)`, so a
    config-complete `.fni8` builds the EXACT LTX-2.3 VAE (no channel mismatch).

    Config source, most-authoritative first:
      1. the converter-PERSISTED `meta['config']['vae']` (durable, checkpoint-exact);
      2. FALLBACK — the packaged canonical LTX-2.3 config, when the `.fni8` predates the
         converter persisting a config AND `meta['repo']` names an `LTX-2.3` bundle (every
         already-shipped LTX-2.3 `.fni8`: base/distilled share this VAE). This is what turns
         a pre-persist bundle into a decodable clip instead of the auto-detect crash.
    Returns None only for a non-LTX-2.3 `.fni8` with no persisted config, so the caller
    cleanly falls back to comfy's auto-detect (the honest degraded path)."""
    import json

    vae = ltx2_persisted_vae_config(path)
    if not vae and "LTX-2.3" in _ltx2_bundle_repo(path):
        vae = ltx2_canonical_vae_config()
    return {"config": json.dumps({"vae": vae})} if vae else None


def _count_connector_layers(sd: dict, prefix: str) -> int | None:
    max_i = -1
    needle = prefix + ".transformer_1d_blocks."
    for k in sd:
        if k.startswith(needle):
            try:
                max_i = max(max_i, int(k[len(needle):].split(".", 1)[0]))
            except ValueError:
                pass
    return max_i + 1 if max_i >= 0 else None


def ltx2_detection_metadata(sd: dict, persisted: dict | None = None) -> dict | None:
    """Recover the LTX-2 (`ltxav`) transformer config that ComfyUI's `detect_unet_config`
    normally reads from the diffusers `metadata['config']['transformer']` — but which the
    `.fni8` quantization dropped (its meta is only kind/arch/native_dtype). Returns the
    `metadata=` blob `comfy.sd.load_diffusion_model_state_dict` expects
    (a `{'config': <json str>}`), or None for any DiT that isn't an LTX-2 bundle.

    Layered, most-authoritative last:

    1. SHAPE-DETERMINING config, inferred straight from the checkpoint's OWN tensor
       shapes / key set (no external file needed to make the model *build*):
         * `cross_attention_adaln`  — True when a block's `scale_shift_table` is 9 rows
           (prompt-conditioned adaLN) not the 6-row default. Wrong ⇒ size mismatch on all
           48 blocks.
         * connector head geometry — the video/audio `Embeddings1DConnector` channel dim
           (= `learnable_registers`'s last dim) is `num_heads * head_dim`; `to_gate_logits`
           (present under gated attention) outputs exactly num_heads, disambiguating the
           factoring. LTX-2.3: video 4096 = 32*128, audio 2048 = 32*64 — vs comfy's
           30*128=3840 default ⇒ mismatch on every connector Linear.
         * `connector_num_layers`, `connector_apply_gated_attention`.

    2. FORWARD scalars that don't change any shape but that the FORWARD needs
       (`_LTXAV_FORWARD_SCALARS`: `causal_temporal_positioning`, `use_middle_indices_grid`,
       `av_ca_timestep_scale_multiplier`, `rope_type`, `frequencies_precision`) —
       LTX-2.3 constants comfy's ctor defaults get wrong, injected so the forward is
       faithful, not merely finite. Plus two flags
       that ARE key-detectable: `apply_gated_attention` (the main blocks carry
       `attn*.to_gate_logits`) and `caption_proj_before_connector` (the caption projection
       is identity — no `*caption_projection*` Linear weights — which only happens in the
       before-connector layout).

    3. `persisted` — the real diffusers `transformer` config a converter recorded in the
       `.fni8` meta (see `ltx2_persisted_transformer_config`). When present it OVERRIDES
       everything above (verbatim, authoritative), so a re-converted `.fni8` needs no
       inference at all. This is the durable fix; layers 1–2 are the honest fallback for
       the `.fni8`s that predate the converter recording it."""
    import json

    if not any(k.endswith("audio_adaln_single.linear.weight") for k in sd):
        return None  # not an LTX-2 (ltxav) bundle

    def _dim(key):
        v = sd.get(key)
        return int(v.shape[-1]) if v is not None and getattr(v, "ndim", 0) >= 1 else None

    def _connector_heads(prefix):
        """A connector's channel dim (`learnable_registers` last dim) can factor as
        num_heads * head_dim two ways; `to_gate_logits` (present under gated attention)
        outputs exactly num_heads, disambiguating it — the audio connector uses head_dim
        64 (32*64=2048), the video connector 128 (32*128=4096). Returns
        (num_heads, head_dim) or (None, None)."""
        cdim = _dim(prefix + ".learnable_registers")
        gate = _dim_rows(prefix + ".transformer_1d_blocks.0.attn1.to_gate_logits.weight")
        if cdim and gate:
            return gate, cdim // gate
        if cdim:                                    # not gated: assume the 128 head_dim
            return cdim // _CONNECTOR_HEAD_DIM, _CONNECTOR_HEAD_DIM
        return None, None

    def _dim_rows(key):
        v = sd.get(key)
        return int(v.shape[0]) if v is not None and getattr(v, "ndim", 0) >= 1 else None

    tf: dict = {}
    ss = sd.get("transformer_blocks.0.scale_shift_table")
    if ss is not None and int(ss.shape[0]) == _ADALN_CROSS_ATTN_ROWS:
        tf["cross_attention_adaln"] = True

    vheads, vhead_dim = _connector_heads("video_embeddings_connector")
    if vheads:
        tf["connector_num_attention_heads"] = vheads
        tf["connector_attention_head_dim"] = vhead_dim
    aheads, ahead_dim = _connector_heads("audio_embeddings_connector")
    if aheads:
        tf["audio_connector_num_attention_heads"] = aheads
        tf["audio_connector_attention_head_dim"] = ahead_dim
    nlayers = _count_connector_layers(sd, "video_embeddings_connector")
    if nlayers:
        tf["connector_num_layers"] = nlayers
    if any(k.startswith(("video_embeddings_connector.", "audio_embeddings_connector."))
           and k.endswith("to_gate_logits.weight") for k in sd):
        tf["connector_apply_gated_attention"] = True
    # MAIN transformer blocks' gated attention (`apply_gated_attention`, default False in
    # comfy.ldm.lightricks.av_model): LTX-2.3 gates every block's attention output by a
    # sigmoid of `to_gate_logits`. Without this flag comfy builds UNGATED blocks, DROPS
    # the checkpoint's `to_gate_logits` weights ("unexpected key"), and runs a
    # gated-trained model ungated — the attention output is no longer scaled down, so
    # activations compound across the 48 blocks and overflow bf16 to NaN. Infer it from
    # the presence of a block's gate weights (same signal comfy uses to build the MLP).
    if any(k.startswith("transformer_blocks.") and k.endswith(".attn1.to_gate_logits.weight")
           for k in sd):
        tf["apply_gated_attention"] = True

    # Caption/text projection (renamed from the archive's top-level `text_embedding_
    # projection.*_aggregate_embed.*` by `_LTX2_CAPTION_REMAP` at load). LTX-2.3 uses the
    # SINGLE-linear `NormSingleLinearTextProjection`, which comfy only builds when
    # `caption_proj_before_connector=True` (its default False builds a 2-linear
    # PixArtAlphaTextProjection whose keys won't match). And `caption_channels` is NOT
    # shape-recoverable by `detect_unet_config` (it defaults to 3840) — inject it from the
    # projection's input width (188160), else comfy builds `linear_1` as [inner, 3840] and
    # the renamed [inner, 188160] weight fails to load -> random-init conditioning -> NaN.
    cap_w = sd.get("caption_projection.linear_1.weight")
    if cap_w is not None and getattr(cap_w, "ndim", 0) == 2:
        tf["caption_proj_before_connector"] = True
        tf["caption_channels"] = int(cap_w.shape[-1])

    # ---- layer 2: forward scalars (family constants + key-detectable flags) ----
    tf.update(_LTXAV_FORWARD_SCALARS)
    # `apply_gated_attention` on the MAIN transformer blocks: each block grows an
    # `attn1.to_gate_logits` / `attn2.to_gate_logits` Linear when on. Key-detectable.
    if any(re.search(r"transformer_blocks\.\d+\..*to_gate_logits", k) for k in sd):
        tf["apply_gated_attention"] = True
    # `caption_proj_before_connector`: when True with the shipped `first_linear=False`,
    # the caption projection is `lambda a: a` (identity, NO weights); when False the model
    # carries a `(audio_)caption_projection.*` Linear (PixArtAlphaTextProjection). So the
    # ABSENCE of any caption-projection Linear on a connector-bearing LTX-2 bundle marks
    # the before-connector (identity) layout — LTX-2.3's. (A future `first_linear=True`
    # before-connector variant would carry weights and defeat this heuristic; that case is
    # exactly what `persisted` covers.)
    if any(k.startswith(("video_embeddings_connector.", "audio_embeddings_connector."))
           for k in sd) and not any("caption_projection" in k for k in sd):
        tf["caption_proj_before_connector"] = True

    # ---- layer 3: persisted diffusers config wins verbatim ----
    if persisted:
        tf.update(persisted)

    return {"config": json.dumps({"transformer": tf})} if tf else None


# ---------------------------------------------------------------------------
# diffusers -> ComfyUI-native key remaps for video DiTs (issue #30)
#
# ComfyUI's `comfy.model_detection.detect_unet_config` recognizes a DiT by its NATIVE
# state-dict key names. The published video `.fni8`s were quantized straight from the
# HuggingFace **diffusers** checkpoints (`*-Diffusers`, `Lightricks/LTX-Video`), whose
# layout differs from ComfyUI's native `comfy.ldm.wan.model.WanModel` /
# `comfy.ldm.lightricks.model` — so detection returns `None` and `UnetLoaderFNI8` fails
# before any int8 dp4a code runs (ComfyUI ships a diffusers->native remap only for a
# handful of *image* MMDiTs via `convert_diffusers_mmdit`, nothing for Wan/LTX).
#
# Each rule below rewrites ONLY the key STRING: diffusers already stores q/k/v/out as
# four separate Linears (`to_q`/`to_k`/`to_v`/`to_out.0`) exactly like the native model,
# so no tensor split/merge is ever needed — a pure rename suffices, and it works
# identically on a raw fp tensor or an int8 `FNI8Tensor` value.
# ---------------------------------------------------------------------------

# Wan 2.1 / 2.2 (diffusers `WanTransformer3DModel` -> native `WanModel`).
# Verified against the shipped `Wan2.2-TI2V-5B-Diffusers.dit.b8.fni8` key inventory and
# ComfyUI's `comfy/ldm/wan/model.py`. The load-bearing subtlety is the **norm2->norm3
# swap**: diffusers names the (affine) cross-attention norm `norm2`, native names it
# `norm3`; native's own `norm1`/`norm2` are affine=False (no params), so the only
# per-block norm tensor present must land on `norm3` or cross-attention runs
# un-normalized. `attn1`->self_attn, `attn2`->cross_attn.
_WAN_RULES: tuple[tuple[str, str], ...] = (
    (r"^condition_embedder\.time_embedder\.linear_1", "time_embedding.0"),
    (r"^condition_embedder\.time_embedder\.linear_2", "time_embedding.2"),
    (r"^condition_embedder\.text_embedder\.linear_1", "text_embedding.0"),
    (r"^condition_embedder\.text_embedder\.linear_2", "text_embedding.2"),
    (r"^condition_embedder\.time_proj", "time_projection.1"),
    (r"^condition_embedder\.image_embedder\.norm1", "img_emb.proj.0"),        # I2V
    (r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj", "img_emb.proj.1"),
    (r"^condition_embedder\.image_embedder\.ff\.net\.2", "img_emb.proj.3"),
    (r"^condition_embedder\.image_embedder\.norm2", "img_emb.proj.4"),
    (r"(blocks\.\d+)\.attn1\.to_out\.0", r"\1.self_attn.o"),
    (r"(blocks\.\d+)\.attn1\.to_q", r"\1.self_attn.q"),
    (r"(blocks\.\d+)\.attn1\.to_k", r"\1.self_attn.k"),
    (r"(blocks\.\d+)\.attn1\.to_v", r"\1.self_attn.v"),
    (r"(blocks\.\d+)\.attn1\.norm_q", r"\1.self_attn.norm_q"),
    (r"(blocks\.\d+)\.attn1\.norm_k", r"\1.self_attn.norm_k"),
    (r"(blocks\.\d+)\.attn2\.to_out\.0", r"\1.cross_attn.o"),
    (r"(blocks\.\d+)\.attn2\.to_q", r"\1.cross_attn.q"),
    (r"(blocks\.\d+)\.attn2\.to_k", r"\1.cross_attn.k"),
    (r"(blocks\.\d+)\.attn2\.to_v", r"\1.cross_attn.v"),
    (r"(blocks\.\d+)\.attn2\.add_k_proj", r"\1.cross_attn.k_img"),            # I2V
    (r"(blocks\.\d+)\.attn2\.add_v_proj", r"\1.cross_attn.v_img"),
    (r"(blocks\.\d+)\.attn2\.norm_added_k", r"\1.cross_attn.norm_k_img"),
    (r"(blocks\.\d+)\.attn2\.norm_q", r"\1.cross_attn.norm_q"),
    (r"(blocks\.\d+)\.attn2\.norm_k", r"\1.cross_attn.norm_k"),
    (r"(blocks\.\d+)\.ffn\.net\.0\.proj", r"\1.ffn.0"),
    (r"(blocks\.\d+)\.ffn\.net\.2", r"\1.ffn.2"),
    (r"(blocks\.\d+)\.norm2\.", r"\1.norm3."),        # affine cross-attn norm: norm2->norm3
    (r"(blocks\.\d+)\.scale_shift_table$", r"\1.modulation"),
    (r"^proj_out", "head.head"),
    (r"^scale_shift_table$", "head.modulation"),
    # patch_embedding.{weight,bias} is identical on both sides -- no rule.
)

# LTX-Video (diffusers `LTXVideoTransformer3DModel` -> native `comfy.ldm.lightricks`).
# Verified against `LTX-Video.dit.b8.fni8` + `comfy/ldm/lightricks/model.py`. attn
# to_q/to_k/to_v/to_out.0, ff.net.*, caption_projection.linear_*, per-block and top-level
# scale_shift_table are already native-identical; only the three renames below differ.
_LTX_RULES: tuple[tuple[str, str], ...] = (
    (r"^proj_in", "patchify_proj"),
    (r"^time_embed\.emb\.timestep_embedder", "adaln_single.emb.timestep_embedder"),
    (r"^time_embed\.linear", "adaln_single.linear"),
    (r"(transformer_blocks\.\d+)\.attn1\.norm_q", r"\1.attn1.q_norm"),
    (r"(transformer_blocks\.\d+)\.attn1\.norm_k", r"\1.attn1.k_norm"),
    (r"(transformer_blocks\.\d+)\.attn2\.norm_q", r"\1.attn2.q_norm"),
    (r"(transformer_blocks\.\d+)\.attn2\.norm_k", r"\1.attn2.k_norm"),
)

_REMAPS: dict[str, tuple[tuple[str, str], ...]] = {
    "wan21": _WAN_RULES, "wan22": _WAN_RULES, "ltx_video": _LTX_RULES,
}
# Cheap signature substrings that only appear in the diffusers layout, so the remap is a
# guaranteed no-op on an already-native state dict (idempotent / safe to apply blindly).
_DIFFUSERS_SIGNATURE: dict[str, tuple[str, ...]] = {
    "wan21": ("condition_embedder.", ".attn1.to_q", "scale_shift_table"),
    "wan22": ("condition_embedder.", ".attn1.to_q", "scale_shift_table"),
    "ltx_video": ("time_embed.emb.", "proj_in.weight"),
}


def _compiled(rules):
    return tuple((re.compile(p), r) for p, r in rules)


def _remap_key(key: str, compiled) -> str:
    for pat, repl in compiled:
        new, n = pat.subn(repl, key)
        if n:
            return new
    return key


def remap_diffusers_to_native(sd: dict, arch: str | None) -> dict:
    """Rewrite a video DiT's diffusers-layout state dict to ComfyUI's native key names
    so `detect_unet_config` recognizes it. No-op (returns `sd` unchanged) when `arch`
    has no remap or the state dict is already native, so callers may apply it
    unconditionally to any loaded `.fni8` DiT state dict."""
    rules = _REMAPS.get((arch or "").lower())
    if rules is None:
        return sd
    sig = _DIFFUSERS_SIGNATURE.get(arch.lower(), ())
    if sig and not any(any(s in k for s in sig) for k in sd):
        return sd                                  # already native -> leave untouched
    compiled = _compiled(rules)
    return {_remap_key(k, compiled): v for k, v in sd.items()}


def fuse_attention_qkv_int8(sd: dict, *, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Re-fuse an arch's separately-stored int8 attention projections into the SINGLE
    fused `qkv` int8 weight its ComfyUI model expects, LOSSLESSLY, and return them keyed
    to their NATIVE names so `assign_int8_weights` can re-attach them after the build.

    Background (docs/zimage-full-pipeline-findings.md): the published Z-Image `.fni8`
    stores attention in diffusers layout — separate per-row-int8 `to_q`/`to_k`/`to_v`
    (each `[dim,dim]` with its own per-row fp32 scale `[dim]`). ComfyUI's Z-Image (S3-DiT)
    model instead uses ONE fused `attention.qkv.weight` `[3*dim,dim]`, and its
    diffusers->native convert concatenates the three along dim 0
    (offsets `q=[0:dim]`, `k=[dim:2dim]`, `v=[2dim:3dim]`). #103 kept the projections fp
    because a naive int8-CODE concat sharing one scale is corrupt.

    The key insight this function exploits: `per_row_i8` carries ONE fp32 scale PER OUTPUT
    ROW, so concatenating q/k/v along the output dim is just concatenating their code rows
    AND their scale vectors — no shared scale, no requantization, ZERO added error. The
    fused `qkv` is bit-identical to the three shipped tensors, so int8 attention runs at
    exactly the FFN's int8 quality (which #103 measured coherent at 31 dB). `to_out.0` is
    already a 1:1 `[dim,dim]` linear -> just re-keyed to native `attention.out`.

    Side effect: the diffusers attention projections left in `sd` are DEQUANTIZED to fp
    (`dtype`), so ComfyUI's convert fuses/renames them on the proven fp path and the model
    still BUILDS correctly; the fp params it produces are then overwritten by the int8
    fused weights returned here (exactly how the int8 FFN is re-attached today).

    Returns `{native_key: FNI8Tensor}` (fused `attention.qkv.weight` +
    renamed `attention.out.weight`, per block). Empty if `sd` has no fused-qkv attention.
    """
    from .superl8_tensor import FNI8Tensor

    suffix = ".attention.to_q.weight"
    prefixes = [k[: -len(".to_q.weight")] for k in sd if k.endswith(suffix)]
    fused: dict = {}
    for attn in prefixes:                                   # attn == "<block>.attention"
        q, k, v = (sd.get(f"{attn}.to_{p}.weight") for p in ("q", "k", "v"))
        out = sd.get(f"{attn}.to_out.0.weight")
        if not all(isinstance(t, FNI8Tensor) for t in (q, k, v)):
            continue                                        # already fp (dequant_fp) -> skip
        codes = torch.cat([t.int8_data() for t in (q, k, v)], dim=0).contiguous()
        scales = torch.cat([t.q_scale for t in (q, k, v)], dim=0).contiguous()
        fused[f"{attn}.qkv.weight"] = FNI8Tensor(codes, scales)
        if isinstance(out, FNI8Tensor):
            fused[f"{attn}.out.weight"] = FNI8Tensor(out.int8_data().contiguous(), out.q_scale)
        # Dequantize the diffusers projections in-place so comfy's convert runs on fp
        # (the built fp params are overwritten by the int8 fused weights above).
        for key in (f"{attn}.to_q.weight", f"{attn}.to_k.weight", f"{attn}.to_v.weight",
                    f"{attn}.to_out.0.weight"):
            t = sd.get(key)
            if isinstance(t, FNI8Tensor):
                sd[key] = _dequant_qtensor_tensor(t, dtype)
    return fused


def _dequant_qtensor_tensor(t, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a per-row-int8 FNI8Tensor to a plain fp tensor (int8 [out,in] * per-row
    scale [out]). Mirrors `_dequant_qtensor` but for the FNI8Tensor wrapper."""
    return (t.int8_data().float() * t.q_scale.unsqueeze(1)).to(dtype)


def assign_int8_weights(diffusion_model, sd: dict) -> int:
    """Re-attach the int8 `FNI8Tensor` weights from `sd` onto an already-built model,
    keeping the tensor subclass so `FNI8Ops` engages the dp4a path.

    ComfyUI's `comfy.sd.load_diffusion_model_state_dict` loads weights with
    `assign=False`, i.e. `param.copy_(input)` — which casts our int8 codes into the
    module's fp Parameter (dropping the per-row scale entirely: garbage weights AND no
    int8). `assign=True` isn't reachable through that entry point, and even it would
    wrap the value in `nn.Parameter(...)`, stripping the `FNI8Tensor` subclass. So we
    assign into `module._parameters` directly, which preserves the subclass + scale.
    Returns the number of weights re-attached. Assumes `sd` keys are already in the
    model's native layout (run `remap_diffusers_to_native` first)."""
    from .superl8_tensor import FNI8Tensor

    names = dict(diffusion_model.named_parameters())
    n = 0
    for key, val in sd.items():
        if not isinstance(val, FNI8Tensor) or key not in names:
            continue
        modpath, _, leaf = key.rpartition(".")
        sub = diffusion_model.get_submodule(modpath) if modpath else diffusion_model
        sub._parameters[leaf] = val
        n += 1
    return n
