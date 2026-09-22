# SPDX-License-Identifier: Apache-2.0
"""Supported DiT architecture registry.

Because we PATCH ComfyUI's existing DiT ops (not reimplement the model), supporting a
family is mostly a data entry describing: how its self-attention is reached
(`attn1_replace` for BasicTransformerBlock-style models, or `object_patch` on named
attention submodules for Flux/SD3-style DiTs that call optimized_attention directly),
the state-dict key prefix to strip when mapping a `.fni8`, the typical head dim (for
the kernel-dim gate), and whether weights are openly available. Compute is the same
dp4a int8 GEMM + bidirectional FA everywhere.

Every entry also documents the text-encoder + VAE ComfyUI expects to complete the
graph. The DiT is always int8 (see `loader.py`'s `.fni8` container); the TE and VAE
are loaded through ComfyUI's own state-dict path (see `FNI8ComponentLoader` in
`nodes.py`) and default to fp there, since both are numerically load-bearing. The VAE
(convolutional, not a transformer) always stays fp; the TE — the same transformer
architectures (T5/Qwen/CLIP/umT5) the DiT int8 dp4a ops already accelerate — can
optionally run int8 dp4a too via `FNI8ComponentLoader`'s `clip_precision` input.
`clip_type` is a best-effort hint matching
ComfyUI's `comfy.sd.CLIPType` member name (the same string `CLIPLoader`/
`DualCLIPLoader`'s `type` dropdown expects) as of this pack's latest ComfyUI target;
it can drift as ComfyUI adds/renames types upstream, and is left blank for archs
without a confirmed native `CLIPType` — `FNI8ComponentLoader` fails loudly rather than
guessing when that happens.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiTArch:
    name: str
    kind: str                    # "image" | "video" | "audio"
    attn_seam: str               # "attn1_replace" | "object_patch"
    key_prefix: str              # state-dict prefix to strip (usually diffusion_model.)
    head_dim: int                # typical (runtime gate still checks the allowlist)
    open_weights: bool
    aliases: tuple = ()
    notes: str = ""
    text_encoder: str = ""       # TE family this arch expects (always fp, never quantized)
    clip_type: str = ""          # comfy.sd.CLIPType member name; "" if unconfirmed
    vae: str = ""                # VAE family this arch expects (always fp, never quantized)
    # Extra weight-name substrings whose int8 codes must be DEQUANTIZED back to fp at
    # load time (on top of loader._HEAD_FP_DEQUANT). Use for genuinely-fp linears an arch
    # needs kept off int8 (rare — the runtime SQNR gate usually handles per-op outliers).
    fp_dequant: tuple = ()
    # True when this arch's ComfyUI model FUSES separately-stored diffusers attention
    # projections (`to_q`/`to_k`/`to_v`) into ONE `attention.qkv.weight`. ComfyUI's
    # convert concatenates the three along the output dim; a naive int8-CODE concat that
    # shares one scale is corrupt (why #103 kept these fp). With this flag the loader
    # instead re-fuses them LOSSLESSLY — `per_row_i8` carries one scale PER OUTPUT ROW, so
    # concatenating q/k/v is just concatenating their code rows AND scale vectors (zero
    # added error), producing a single fused int8 `qkv` that runs dp4a at the FFN's int8
    # quality. Z-Image (S3-DiT) fuses qkv -> True. Video DiTs (Wan/LTX) keep SEPARATE
    # q/k/v natively (loader._WAN_RULES/_LTX_RULES) -> False. See
    # loader.fuse_attention_qkv_int8 and docs/zimage-full-pipeline-findings.md.
    fuse_attn_qkv: bool = False


_ARCHS: dict[str, DiTArch] = {}


def register(a: DiTArch):
    _ARCHS[a.name] = a
    return a


def get(name: str) -> DiTArch | None:
    n = name.lower()
    if n in _ARCHS:
        return _ARCHS[n]
    for a in _ARCHS.values():
        if n == a.name or n in a.aliases:
            return a
    return None


def families() -> list[str]:
    return sorted(_ARCHS)


# ---- image DiTs ----
register(DiTArch("flux1", "image", "object_patch", "diffusion_model.", 128, True,
                 ("flux", "flux.1", "flux1-dev", "flux1-schnell"),
                 "BFL MMDiT (double+single stream); joint text+image attention",
                 text_encoder="CLIP-L + T5-XXL (dual)", clip_type="flux",
                 vae="Flux 16-ch AE"))
register(DiTArch("flux2", "image", "object_patch", "diffusion_model.", 128, True,
                 ("flux.2",), "BFL Flux.2 — next-gen MMDiT (Klein 9B: Q6_K GGUF + Qwen3-8B TE)",
                 text_encoder="Qwen3-8B (single TE, fp8mixed)", clip_type="flux2",
                 vae="Flux.2 32-ch AE"))
register(DiTArch("sd3", "image", "object_patch", "diffusion_model.", 64, True,
                 ("sd3_medium", "mmdit"), "Stable Diffusion 3 MMDiT",
                 text_encoder="CLIP-L + CLIP-G + T5-XXL (triple)", clip_type="sd3",
                 vae="SD3 16-ch VAE"))
register(DiTArch("sd35", "image", "object_patch", "diffusion_model.", 64, True,
                 ("sd3.5", "sd3_5_large"), "Stable Diffusion 3.5",
                 text_encoder="CLIP-L + CLIP-G + T5-XXL (triple)", clip_type="sd3",
                 vae="SD3 16-ch VAE"))
register(DiTArch("qwen_image", "image", "object_patch", "diffusion_model.", 128, True,
                 ("qwenimage",), "Qwen-Image MMDiT (Qwen2.5-VL text encoder)",
                 text_encoder="Qwen2.5-VL-7B", clip_type="qwen_image",
                 vae="Qwen-Image 16-ch VAE"))
register(DiTArch("qwen_image_edit", "image", "object_patch", "diffusion_model.", 128, True,
                 ("qwenimageedit",), "Qwen-Image-Edit (dual-path: VL semantic + VAE latent)",
                 text_encoder="Qwen2.5-VL-7B", clip_type="qwen_image",
                 vae="Qwen-Image 16-ch VAE"))
register(DiTArch("zimage", "image", "object_patch", "diffusion_model.", 128, True,
                 ("z-image", "zimage_turbo", "s3dit"),
                 "Z-Image S3-DiT ~6B single-stream, Qwen3-4B encoder; Turbo=8-step. FIRST target",
                 text_encoder="Qwen3-4B",
                 clip_type="",  # too new to confirm a stable comfy.sd.CLIPType member;
                                # verify against your ComfyUI's CLIPLoader/DualCLIPLoader
                                # `type` dropdown before relying on FNI8ComponentLoader
                 vae="Z-Image VAE (dedicated, ships with the DiT release)",
                 # ComfyUI's Z-Image model fuses attention to_q/to_k/to_v into one qkv
                 # weight. The loader re-fuses the separate per-row-int8 projections
                 # LOSSLESSLY (concat codes + per-row scales) into a single int8
                 # `attention.qkv.weight`, so attention runs dp4a too (not just the FFN).
                 # See loader.fuse_attention_qkv_int8 + docs/zimage-full-pipeline-findings.md.
                 fuse_attn_qkv=True))
register(DiTArch("ideogram", "image", "object_patch", "diffusion_model.", 128, False,
                 (), "Ideogram DiT (arch documents the fused-int8-GEMM thesis); weights CLOSED",
                 text_encoder="unknown (closed)", clip_type="", vae="unknown (closed)"))
register(DiTArch("pixart", "image", "attn1_replace", "", 72, True,
                 ("pixart_sigma", "pixart_alpha"), "PixArt DiT",
                 text_encoder="T5-XXL", clip_type="pixart", vae="SDXL VAE (4-ch)"))
register(DiTArch("sana", "image", "object_patch", "diffusion_model.", 32, True,
                 (), "NVIDIA Sana — linear-attention efficient DiT (some layers)",
                 text_encoder="Gemma-2 (2B/9B, decoder-only)", clip_type="",
                 vae="DC-AE (deep-compression AE, 32x)"))
register(DiTArch("chroma", "image", "object_patch", "diffusion_model.", 128, True,
                 (), "Chroma — Flux-derived DiT",
                 text_encoder="T5-XXL", clip_type="chroma", vae="Flux 16-ch AE (shared)"))
register(DiTArch("hidream", "image", "object_patch", "diffusion_model.", 128, True,
                 (), "HiDream MMDiT-style",
                 text_encoder="CLIP-L + CLIP-G + T5-XXL + Llama-3.1-8B (quad)",
                 clip_type="hidream", vae="Flux-style 16-ch VAE"))
register(DiTArch("lumina2", "image", "object_patch", "diffusion_model.", 128, True,
                 ("lumina", "next_dit"), "Lumina-Next / Lumina-2 Next-DiT",
                 text_encoder="Gemma-2-2B", clip_type="lumina2",
                 vae="Flux-style 16-ch VAE"))
register(DiTArch("auraflow", "image", "object_patch", "diffusion_model.", 256, True,
                 (), "AuraFlow MMDiT",
                 text_encoder="Pile-T5-XL", clip_type="", vae="SDXL-style 4-ch VAE"))
register(DiTArch("hunyuan_dit", "image", "attn1_replace", "", 88, True,
                 ("hunyuandit",), "Tencent Hunyuan-DiT",
                 text_encoder="bilingual CLIP + T5", clip_type="hunyuan_dit",
                 vae="SDXL VAE (4-ch)"))

# ---- video DiTs ----
register(DiTArch("wan21", "video", "object_patch", "diffusion_model.", 128, True,
                 ("wan", "wan2.1"), "Alibaba Wan 2.1 video DiT; umT5-XXL encoder",
                 text_encoder="umT5-XXL", clip_type="wan",
                 vae="Wan 2.1 causal 3D VAE"))
register(DiTArch("wan22", "video", "object_patch", "diffusion_model.", 128, True,
                 ("wan2.2",), "Wan 2.2 — TWO 14B DiTs hard-switched by SNR (transformer/transformer_2)",
                 text_encoder="umT5-XXL", clip_type="wan",
                 vae="Wan 2.2 causal 3D VAE"))
register(DiTArch("ltx_video", "video", "object_patch", "diffusion_model.", 128, True,
                 ("ltx", "ltxv", "ltx2"),
                 "Lightricks LTX-Video (v1: diffusers LTXVideoTransformer3DModel, remapped "
                 "to native ltxv) AND LTX-2/2.3 (native ltxav audio+video, loaded from the "
                 "bundled full checkpoint's model.diffusion_model.* sub-tree). 3D causal VAE. "
                 "clip_type='ltxv' covers both: comfy.sd.CLIPType.LTXV auto-routes T5-XXL for "
                 "v1 and Gemma-3-12B for LTX-2 by inspecting the loaded TE state dict.",
                 text_encoder="T5-XXL (LTX-2: Gemma-3-12B)", clip_type="ltxv",
                 vae="LTX 3D causal VAE (LTX-2 also bundles an audio VAE + vocoder)",
                 # adaLN-single modulation tables: the block INDEXES them (`table[slice]`)
                 # and uses the rows as fp scale/shift/gate — the indexed-then-used path on
                 # a quantized FNI8Tensor is broken (and int8/int4 error on every token's
                 # modulation violates the "never quantize adaLN" rule). LTX-2.3 `.fni8`s
                 # bake these tiny [n_ada, dim] tables quantized (per_row_i8 in b8,
                 # per_group_i4 in b4/conn-i4), so dequant them back to fp at load. Covers
                 # all six variants (`scale_shift_table`, `prompt_*`, `audio_*`,
                 # `*_a2v_ca_audio/video`) by substring.
                 fp_dequant=("scale_shift_table",)))
register(DiTArch("hunyuan_video", "video", "object_patch", "diffusion_model.", 128, True,
                 ("hunyuanvideo",), "Tencent HunyuanVideo DiT",
                 text_encoder="Llava-Llama-3 (MLLM) + CLIP-L", clip_type="hunyuan_video",
                 vae="HunyuanVideo causal 3D VAE"))
register(DiTArch("mochi", "video", "object_patch", "diffusion_model.", 128, True,
                 (), "Genmo Mochi video DiT",
                 text_encoder="T5-XXL", clip_type="mochi",
                 vae="Mochi causal 3D VAE (AsymmDiT)"))
register(DiTArch("cogvideox", "video", "object_patch", "diffusion_model.", 64, True,
                 ("cogvideo",), "Zhipu CogVideoX DiT",
                 text_encoder="T5-XXL", clip_type="",
                 vae="CogVideoX causal 3D VAE"))
register(DiTArch("cosmos", "video", "object_patch", "diffusion_model.", 128, True,
                 ("cosmos_predict2",), "NVIDIA Cosmos world-model DiT",
                 text_encoder="T5-XXL", clip_type="cosmos",
                 vae="Cosmos causal tokenizer VAE"))
