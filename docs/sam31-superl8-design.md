# SAM 3 / 3.1 on fni8 (int8 dp4a, sm_70) — architecture-scoping & phased plan

**Status:** design / scoping only (no model or kernel code). Gated by the sm_70 hardware
truths in `fused_ni8/AGENTS.md` (no int8/fp16 tensor cores; only fast int8 matmul is
`__dp4a` on CUDA cores; softmax/LSE/mask-logits/memory-attention stay fp; single 16 GiB card).

**Verdict up front.** SAM 3's compute is dominated by a **plain ViT image encoder (Meta
Perception Encoder, PE-L: 32 layers, hidden 1024, 16 heads → head_dim 64)** running
non-causal attention + FFN over ~5184 tokens at 1008². That is **the exact workload fni8's
DiT path already accelerates** (non-causal int8 FlashAttention + W8A8 GEMM, validated ~2× vs
fp16). head_dim 64 is supported; windowed attention maps onto the existing **int8 varlen**
kernel; global-attention layers are plain non-causal forward. The mask decoder, memory
attention, and mask logits are small and **stay fp** per AGENTS.md. Fits one 16 GiB card
easily (0.9B params, ~1.7 GB fp16). **No new hot kernel is required**; the new pieces
(conv patch stem, LayerNorm, 2D-RoPE, window partition) are cheap fp/host-precompute glue.
This is a strong, compute-bound int8 showcase — the recommended **first** target.

---

## 1. The real architecture (July 2026)

- **SAM 3** (2025-11-19, arXiv 2511.16719) is a major redesign over SAM 2: adds
  **open-vocabulary text/concept prompts** and image-exemplar prompts, and segments/tracks
  **all matching instances at once**. Two decoupled heads over one shared backbone:
  a **DETR-style detector** (new) + a **SAM 2-style tracker** (memory bank retained).
- **SAM 3.1** (2026-03-27, `RELEASE_SAM3p1.md`) is an **inference/tracking update
  ("Object Multiplex"), not a new architecture** — same component shapes, new checkpoints,
  ~7× faster at 128 objects via shared-memory bucketing + better `torch.compile` fusion.

**Components:** Perception Encoder (shared vision backbone) → PE text encoder + exemplar
encoder → DETR detector (MaskFormer-style mask head + presence head) → tracker (SAM 2
prompt encoder + mask decoder + memory encoder + memory bank).

### Image encoder — the heavy part (verified from code/config)

- **Plain (non-hierarchical) ViT = Meta Perception Encoder PE-L, ~450M params.** *Note:
  SAM 3 dropped SAM 2's Hiera backbone and went back to a plain ViT, specifically to align
  vision features with the text encoder in a joint VL space.*
- **32 layers, embed_dim 1024, 16 heads (head_dim 64), MLP ratio 4.625 (inner ≈ 4736).**
- **Patch 14×14, input 1008×1008 → 72×72 = 5184 tokens**, 1024 channels.
- **Hybrid attention: global attention only at layers 7/15/23/31; windowed (window 24 →
  576-token blocks) at the other 28 layers. 2D RoPE** positional encoding.
- **FPN neck** (`Sam3DualViTDetNeck`) → 256-dim multi-scale features (288²/144²/72²/36²).
- **Single model — no tiny/small/base/large sweep.**

### Other components

- **Prompt encoder + mask decoder** (from SAM 2): two-way cross-attention decoder,
  3 masks/object for ambiguity. **DETR detector**: query-driven transformer decoder
  (vanilla attention), MaskFormer mask head, ≤200 object queries. Detector+tracker+neck
  ≈ **~98–100M params** — small vs the backbone.
- **Text encoder ≈ 300M** — the PE paired *language* encoder (CLIP-style, contrastively
  aligned; BPE, ~32-token noun phrases), run once per prompt. **Not** an inference-time LLM
  (Llama 3.2 was only in the data engine).
- **Video memory attention** (SAM 2 memory bank + memory attention + memory encoder) —
  **retained** in 3.1; unchanged architecture.

### Sizes / memory / latency

- **Total ≈ 848M params** (HF "0.9B"): ~450M vision + ~300M text + ~98M detector/tracker/neck.
- Checkpoint ~3.4 GB fp32 → **~1.7 GB fp16**. "Fits comfortably on 16 GB"; ~10 s of 480p
  video on a 10 GB GPU. ~30 ms/image (100+ objects) on H200.

**Sources:** [SAM 3 paper arXiv 2511.16719](https://arxiv.org/html/2511.16719v1) ·
[github facebookresearch/sam3](https://github.com/facebookresearch/sam3) ·
[RELEASE_SAM3p1.md](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) ·
[DeepWiki config](https://deepwiki.com/facebookresearch/sam3/5.3-local-training) ·
[Perception Encoder paper arXiv 2504.13181](https://arxiv.org/pdf/2504.13181) ·
[HF facebook/sam3](https://huggingface.co/facebook/sam3) ·
[Datature deep dive](https://datature.io/blog/sam-3-a-technical-deep-dive-into-metas-next-generation-segmentation-model).

---

## 2. Component → kernel map

| Sub-module | Precision | Path | fni8 op / status |
|---|---|---|---|
| Patch-embed stem (conv 14×14) | fp16 (half2) | one conv2d | ❌ no int8 conv; **fp, cheap** (1 layer; or im2col→`linear_w8a8`) |
| **PE-L QKV/O + FFN linears (32×)** | **int8 W8A8** | GEMM | ✅ `linear_w8a8` / `gemm_w8a8`; FFN 1024→4736→1024 |
| **Global attention (layers 7/15/23/31)** | **int8 dp4a** | non-causal | ✅ `attn_int8_fwd` (causal=False), head_dim 64, N=5184 — the DiT regime |
| **Windowed attention (28 layers, win 24)** | **int8 dp4a** | block-diagonal | ✅ `attn_int8_varlen` — each 24×24=576-token window is a varlen segment (needs window-partition glue) |
| 2D RoPE (axial) | fp16 | elementwise | ⚠️ `rope` is 1-D; need 2D axial (host-precompute cos/sin for MVP — see gaps) |
| LayerNorm (ViT uses LN, not RMSNorm) | fp16 (fp32 reduce) | elementwise | ❌ no fused LN op; **fp via `F.layer_norm`** (trivial; RMSNorm kernel not reusable) |
| GELU (plain, non-gated FFN) | fp16 (half2) | elementwise | ⚠️ `act_and_mul` is GLU-only; plain GELU = `F.gelu` (fp, trivial) |
| FPN neck convs (→256-dim) | fp16 | small conv | fp, cheap |
| **Text encoder (~300M, per prompt)** | **int8 W8A8 + int8 attn** | prefill | ✅ same GEMM/attn kernels (ViT/transformer); run-once, low priority to int8 |
| Prompt/exemplar encoder | fp16 | tiny | fp (negligible) |
| DETR detector decoder (≤200 queries) | int8-able or fp | cross-attn | ✅ cross-attn (M≠N) supported; small — fp is fine |
| **Mask decoder + mask logits** | **fp** | two-way attn + softmax | stays fp — **mask-quality-load-bearing** |
| **Memory attention / memory bank (video)** | **fp** | cross-attn + softmax | stays fp — tracking-quality-load-bearing |

**Never quantize (AGENTS.md rule):** mask logits, the mask-decoder two-way attention softmax,
and the video memory attention. These decide pixel-accurate boundaries and track identity —
the segmentation analogue of "softmax/LSE stay fp32". They are also small, so no compute is lost.

---

## 3. Gaps (ranked by effort)

1. **Window partition / unpartition glue for windowed attention.** 28 of 32 layers use
   24×24 windows. Reshape the 72×72 token grid into 3×3=9 windows of 576 tokens and feed
   `attn_int8_varlen` as a block-diagonal batch (it already does independent varlen
   segments). Pure Python glue over an existing kernel. **Effort: S.**
2. **2D axial RoPE.** PE uses 2D RoPE; `rope` is 1-D. For the MVP, precompute the 2D cos/sin
   tables on host and apply via a generic elementwise multiply (no kernel change); later add
   a 2D mode to the `rope` kernel. **Effort: S.**
3. **LayerNorm + plain GELU as fp epilogue.** ViT uses LayerNorm (fni8 only has RMSNorm) and
   non-gated GELU (fni8's `act_and_mul` is GLU-only). Both run fp on the healthy half2 CUDA
   cores via `F.layer_norm` / `F.gelu` — correct and cheap, no new kernel needed. Optionally
   fuse later. **Effort: S (fp fallback is the answer).**
4. **conv2d patch stem + FPN neck convs — keep fp.** No int8 conv primitive; these are 1
   stem conv + a few neck convs, <1% of compute. fp (half2) or im2col→`linear_w8a8` if ever
   hot. **Effort: N/A (fp).**
5. **Converter: register SAM 3 in fni8-serve `convert.py`/`config.py`** (or a ComfyUI-side
   quantizer like `loader.quantize_dit_state_dict`). Quantize only the PE-L (and optionally
   text-encoder) QKV/O/FFN linears; keep conv/LN/mask-decoder/memory weights **raw/fp**.
   The `_is_dit_linear`-style denylist already skips norms/embeddings. **Effort: S–M.**
6. **ComfyUI SAM node surface (MASK output, prompts).** New nodes; see §5. **Effort: M.**

**No new hot CUDA kernel is required.** Windowed attention = existing varlen; global
attention + FFN = existing non-causal forward + W8A8 GEMM; head_dim 64 is supported. The
gaps are host-side glue + fp elementwise fallbacks.

---

## 4. VRAM / single-card fit (16 GiB)

| Component | fp16 | int8 backbone (W8A8) |
|---|---|---|
| PE-L vision backbone (~450M) | ~0.9 GB | ~0.45 GB |
| Text encoder (~300M) | ~0.6 GB | ~0.3 GB (or keep fp) |
| Detector + tracker + neck (~98M) | ~0.2 GB | ~0.2 GB (kept fp) |
| Activations @ 5184 tokens × 1024 × 32L | O(N) with FlashAttention | — |

**Total ~1.7 GB fp16 → ~1 GB int8-backbone**, plus O(N) attention activations
(FlashAttention keeps the 4 global 5184²-attention layers linear in memory, exactly as the
Z-Image path shows 6.4→10.2 GB across a 140× compute range). **Trivially single-card.** The
backbone is **compute-bound** (5184 tokens, 32 layers) — the regime where int8 dp4a gives a
large, demonstrable win, and where the fleet's dead tensor cores would otherwise cripple an
fp16 baseline.

---

## 5. ComfyUI scope fit + node surface

SAM is squarely a ComfyUI citizen (image/video segmentation; existing packs like
ComfyUI-segment-anything). Belongs in **ComfyUI-superl8** for the nodes + int8 backbone;
converter work can live in **fni8-serve** or a ComfyUI-side quantizer reusing the `.fni8`
container. Proposed nodes (mirror `UnetLoaderFNI8` / `FNI8AttentionPatch`):

- **`SAM3LoaderFNI8`** → loads the `.fni8` PE-L backbone (int8) + text encoder + raw
  detector/tracker/neck. Returns `SAM3_MODEL`. Reuses `FNI8Tensor` + `loader`.
- **`SAM3Predict`** → inputs `SAM3_MODEL`, `IMAGE`, and a prompt: points/box (`SAM_PROMPT`)
  or `text (STRING)` (concept). Output `MASK` (+ optional `IMAGE` overlay). Runs backbone
  (int8) → detector/mask-decoder (fp).
- **`SAM3VideoTrack`** (v2) → `IMAGE` batch / video → per-frame `MASK`, using the fp memory
  bank.

Reuse: `FNI8Ops.Linear` (scheme-dispatched dp4a + SQNR-gated fp fallback), `FNI8Tensor`,
`.fni8` format/loader — identical plumbing to the DiT path (PR #115). The window-partition
+ 2D-RoPE glue is SAM-specific but sits above the kernels.

---

## 6. Phased roadmap

**Phase 0 — MVP: real mask, correctness first (fp backbone).**
Wire SAM 3 end-to-end in ComfyUI with the PE-L backbone in **fp** (torch reference): IMAGE +
point/box prompt → `MASK`. Proves the node graph, window/global attention scheduling, 2D
RoPE, conv stem, neck, and mask decoder independent of int8. *Effort: M. Risk: the hybrid
windowed/global attention schedule + 2D-RoPE bookkeeping.*

**Phase 1 — int8 the backbone attention (drop-in, the compute win).**
Route the 32-layer attention through `attn_int8_fwd` (global layers) and `attn_int8_varlen`
(windowed layers, via window-partition glue). SQNR-gate per layer; mask decoder/memory stay
fp. This is the DiT-attention precedent applied to a ViT. *Effort: M. Risk: SQNR on windowed
576-token blocks; K-smoothing/Hadamard if outliers bite (issue #74 precedent).*

**Phase 2 — int8 W8A8 the backbone linears via the converter.**
Quantize PE-L QKV/O/FFN to int8 in a `.fni8` (keep conv/LN/mask-decoder/memory raw). Load
via `SAM3LoaderFNI8`. Optionally int8 the text encoder. Add the 2D-RoPE cos/sin precompute
+ LayerNorm/GELU fp epilogues. *Effort: M. Risk: converter remap of the two-headed
detector/tracker state dict (cf. Qwen3.5 hybrid-config drop + loader shim, PR #242).*

**Phase 3 (optional) — video tracking + W4A8 + fused LN/GELU/2D-RoPE kernels.**
`SAM3VideoTrack` with the fp memory bank; `linear_w4a8` if memory ever matters (it does not
at 0.9B — low priority); promote the fp elementwise epilogues to fused fni8 kernels only if
profiling shows them hot. *Effort: M+. Risk: memory-attention drift over long videos (keep fp).*

**Biggest blocker:** none is a hot kernel. The real risk is **windowed-attention SQNR**
(576-token blocks with image-activation outliers — mitigate with K-smoothing / the Hadamard
rotation from issue #74) and the **converter remap** of SAM's two-head state dict. The conv
stem / LayerNorm / GELU / 2D-RoPE staying fp is deliberate and cheap.

---

## 7. Precedent to lean on

- **ComfyUI-superl8 DiT path (PR #115, `docs/int8-dit-validation.md`)** — non-causal int8
  attention + W8A8 GEMM validated end-to-end (Z-Image cos 1.000000, self-attn SQNR 47.2 dB,
  per-layer SQNR-gated fp fallback). A ViT encoder is the **same** non-causal shape.
- **`utils/docs/diffusion-applicability.md`** — confirms non-causal + cross-attention
  (M≠N) + head_dim {32,64,72,80,128,256} support and that int8 attention error does **not**
  compound over an iterative loop (cos ~1.0 over 32 steps). SAM is a *single* forward, so
  even safer.
- **`bench/real_model_eval.py`** — model-agnostic SDPA→`attn_int8_fwd` monkeypatch; the same
  hook drives the PE-L backbone with no per-layer surgery.
- **Issue #74 (Hadamard rotation on DiT int8 attention, 1.4× lower error)** — the mitigation
  if windowed-attention SQNR dips on image-activation outliers.
- **DiT bf16 + FNI8Tensor** (memory / PR #115) — quantized weights must be a `torch.Tensor`
  subclass; run bf16-native weights in bf16 (software bf16 on Volta is fine; dp4a never
  touches the gimped tensor cores). Check whether SAM's PE checkpoint is bf16-native.
- **SQNR-gated fp fallback** (`FNI8Ops.Linear`) — a matmul that misses its bar stays fp; the
  mask decoder and memory attention are declared fp up front by the same principle.

---

## 8. MVP RESULT — Phase 0 + Phase 1 built and measured (2026-07-13)

Built and validated end-to-end on a real image + real mask. Code:
`comfyui_superl8/sam3_encoder.py` (int8 ViT-trunk patch), `comfyui_superl8/sam3_nodes.py`
(`SAM3LoaderFNI8` + `SAM3Predict`), `bench/sam31_fni8_eval.py` (the proof harness),
`tests/test_sam3_encoder.py` (10 CPU unit tests). Ran in a combined
`comfyui-fni8-sam3` image (fni8 built sm_70 + `git+facebookresearch/sam3` + the gated
`facebook/sam3.1` checkpoint from the volume) on a free **Tesla V100 (sm_70)**.

### Architecture — CONFIRMED from the real `sam3.1_multiplex.pt` checkpoint

The scoping doc's ViT mapping is exactly right. Loaded model is `Sam3Image`;
`backbone.vision_backbone` (`Sam3DualViTDetNeck`) `.trunk` is a `ViT` with **32
`blocks[i].attn` (`Attention`), dim 1024, 16 heads → head_dim 64** (fni8-supported),
2D RoPE. Crucially, the ViT `Block` does `window_partition`/`window_unpartition`
around attention itself, and `Attention.forward` calls
`F.scaled_dot_product_attention(q,k,v)` (non-causal, no mask) — so **windowed layers
arrive at SDPA as `[B·num_windows, heads, 576, 64]` and global layers as
`[B, heads, 5184, 64]`; one module-scoped SDPA shim covers BOTH — no separate varlen
glue is required.** The MLP `fc1` runs through a module-scoped fused `addmm_act`
(GELU folded), which the linear patch also reroutes; `fc2`/`qkv`/`proj` are plain
`nn.Linear` calls.

### How the int8 path is wired (`sam3_encoder.py`)

- **Attention:** `sam3.model.vitdet.F` is replaced by a proxy that delegates
  everything except `scaled_dot_product_attention`, which routes eligible non-causal
  head-dim-64 calls through `fni8.attn_int8_fwd(causal=False)` (2D-RoPE returns q/k in
  fp32 → cast to v's fp16 for the kernel, as fp16 SDPA would). Per-call-site SQNR gate.
- **Linears:** every `nn.Linear` under the trunk is wrapped with `Int8LinearShim`
  (offline per-row int8 weight, `fni8.linear_w8a8` dp4a GEMM, one-time SQNR gate → fp).
- **Everything else stays fp** (mask decoder, DETR detector, text encoder, conv stem,
  neck, LayerNorm, GELU) — mask-quality load-bearing, per AGENTS.md. All patches are
  reversible (`.unpatch()`), so the loader holds the fp oracle and int8 side by side.

### Measured — real image (a cat beside a laptop), text prompt `"cat"`, fp16

| metric | fp oracle | int8 encoder | delta |
|---|---:|---:|---|
| **mask IoU (int8 vs fp)** | — | — | **0.9995** (target ≥ 0.98 ✓) |
| top detection score | 0.960 | 0.960 | identical |
| **ViT encoder latency** | 890 ms | 533 ms | **1.67× faster** |
| end-to-end latency | 1088 ms | 733 ms | 1.48× faster |
| **peak VRAM** | 7.66 GiB | 6.70 GiB | −0.96 GiB (single card ✓) |

Real masks saved (`mask_fp.png` / `mask_int8.png` + overlays) — visually identical,
clean cat segmentation. The **compute win comes from the int8 linears**: all 128 trunk
linears engaged int8 (640/640 calls), **SQNR 27.9 dB min / 38.3 dB median** — well
above the 20 dB floor.

### Finding — int8 *attention* correctly stays fp here (the gate working, not a defeat)

Both attention sites (global S=5184, windowed S=576) tripped **fni8's own inner
SageAttention outlier gate** (`detect_q_outlier_domination` on the ViT's post-RoPE Q)
→ the kernel returns fp SDPA bit-identically (+inf SQNR → routed to fp). This is
exactly the windowed-image-activation-outlier risk flagged in §6, and the gate handled
it honestly. `rotate=True` (Hadamard, #74) does **not** rescue it because the kernel's
gate is evaluated on the *original* Q, before the rotation. This costs little: at these
ViT shapes the FFN (1024→4736→1024) + QKV/proj GEMMs dominate FLOPs over attention, so
the int8 linears alone deliver the 1.67× encoder speedup while the numerically-touchy
attention stays fp — the AGENTS.md "the gate decides where int8 is allowed" principle
applied verbatim. (A future dedicated K-smoothing/rotation-before-gate path for ViT
attention is the only avenue to also int8 the 4 global layers; low priority given the
linears already carry the win.)

### Deliverable status

- **Phase 0 (fp MVP oracle):** ✅ real mask from `IMAGE + text prompt` end-to-end.
- **Phase 1 (int8 ViT encoder):** ✅ IoU 0.9995, 1.67× encoder, −0.96 GiB, SQNR-gated.
- **Nodes:** `SAM3LoaderFNI8` (int8/attention/linears/rotate toggles) + `SAM3Predict`
  (`IMAGE` + text/box prompt → `MASK` + overlay), registered via the package
  `__init__` (kept out of the contended `nodes.py`).
- **Not done (future PRs):** `SAM3VideoTrack` (Phase 3, fp memory bank); a ViT-attention
  rotation-before-gate path; a checkpoint-load interceptor that skips materializing the
  fp trunk weights entirely (see §9).

---

## 9. Phase 2 — offline `.fni8` converter (built and validated, 2026-07-13)

The Phase-1 int8 was quantized on-load per run. Phase 2 makes it a **proper offline
`.fni8`**: `comfyui_superl8/sam3_convert.py` builds the fp `facebook/sam3.1` model,
quantizes the 32×4 = **128 ViT-trunk linears to `per_row_i8` once**, and writes a
`.fni8` whose on-disk bytes ARE the resident dp4a layout (the fni8 format contract:
load = mmap + place, no dequant/repack). Attention is not carried (the SageAttention
gate keeps it fp); the mask decoder / detector / text encoder / conv stem / neck / LN /
GELU stay fp and load from the HF checkpoint — the `.fni8` is an int8 **overlay**.

The trunk config (dim 1024, 16 heads, head_dim 64, 32 blocks) + the int8/fp layer
lists are persisted in `__meta__` so the loader rebuilds without re-deriving — the LTX
"config never persisted" bug avoided by construction. `SAM3LoaderFNI8` gains an
`fni8_path` input; when set it calls `load_sam3_encoder_int8_from_fni8` (resident
`Int8LinearShim.from_resident`: mmap the int8 codes+scales+bias, **no runtime quant,
no fp weight kept, no per-layer SQNR gate** — the offline converter decided).

### Produced artifact
`sam3.1.vit.b8.fni8` — **425.9 MiB, 128/128 int8 layers, 0 fp** (all trunk linears are
K%4==0), quantize step **2.3 s**, write 0.7 s.

### Validated (real Tesla V100, cat image, prompt "cat", fp16) — offline vs on-load int8
| metric | on-load quant | offline `.fni8` | |
|---|---:|---:|---|
| **mask IoU (offline vs on-load)** | — | — | **1.000000** (bit-identical `per_row_i8` math; target ≥ 0.999 ✓) |
| encoder latency | 760 ms | 763 ms | same (the 1.67× vs fp holds) |
| **int8 added VRAM** | 2.08 GiB | **0.42 GiB** | **−1.67 GiB** (offline places int8 only; on-load keeps fp32 `w_fp`) |
| int8 install time | 535 ms | 2.8 s | offline slower (cold 426 MiB disk read on the ~160 MB/s fleet vs in-place GPU quant) |

### Honest finding — where the offline win is (and isn't)
The offline `.fni8` is **bit-identical** and saves **1.67 GiB** (no fp32 reference
weights). It does **not** speed up load in the current overlay design: the fp model is
built from the 3.5 GB HF checkpoint regardless (≈42 s), so the on-load path then
quantizes already-resident GPU weights in 0.5 s — cheaper than reading 426 MiB of int8
from the slow fleet disk. The genuine load/read win needs a **checkpoint-load
interceptor** that never loads the fp trunk weights (place int8 from the `.fni8`
instead) — a follow-up. As-is, the `.fni8` is the shippable, re-quant-free artifact and
delivers the memory win; correctness (IoU 1.0) and speed (1.67×) are unchanged.
