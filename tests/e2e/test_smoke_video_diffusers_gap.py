# SPDX-License-Identifier: Apache-2.0
"""Per-arch e2e matrix (issue #12): Wan2.2 (TI2V-5B, T2V-A14B) and LTX-Video/LTX-2.3.

`wan21`/`wan22`/`ltx_video` are already registered in `comfyui_superl8/arch.py` (this
pack's attn-seam/key-prefix/text-encoder/VAE metadata) — that part is not the gap. The
gap is upstream: ComfyUI's own `comfy.model_detection.detect_unet_config` doesn't
recognize these checkpoints' **diffusers-format** state-dict key layout (Wan:
`blocks.N.attn1.to_q` / `attn2.*` / `ffn.net.0.proj` / `scale_shift_table`; LTX:
`caption_projection.*` / `proj_in` / ...). ComfyUI has a diffusers->native key remap
for a handful of *image* DiTs (`comfy.model_detection.convert_diffusers_mmdit`:
auraflow/pixart/z_image/flux/mmdit) but nothing for Wan or LTX, so
`comfy.sd.load_diffusion_model_state_dict` couldn't build a `MODEL` from these `.fni8`
DiTs — `UnetLoaderFNI8` failed before any int8 dp4a code ran.

Issue #30 closes that gap for Wan + LTX-**Video v1** (not upstream):
`loader.remap_diffusers_to_native` rewrites the diffusers keys to ComfyUI's native
`WanModel` / lightricks layout inside `UnetLoaderFNI8.load`, so the DiT is detected and
built.

**LTX-2.3 is a different case** (still exercised by this matrix): it is NOT a diffusers
layout needing a key remap — its DiT is already in native ComfyUI `ltxav` (audio+video)
layout, but nested under a `model.diffusion_model.` prefix inside a *bundled full
checkpoint* (DiT + VAE + audio-VAE + vocoder), and its quantization dropped the diffusers
`transformer` config `detect_unet_config` needs to size the model. The loader now
(a) extracts the DiT sub-tree (`dit_bundle_prefix` + `keep_only_prefixed`) and (b) infers
the shape-determining config from tensor shapes (`ltx2_detection_metadata`). So for
LTX-2.3, "undetected natively" holds for a different reason (bundle prefix + missing
config, not a diffusers layout), and the load test checks the bundle path.

Each checkpoint below gets two tests: one that pins ComfyUI STILL can't detect the raw
keys on its own (cheap — reads key names only), and one that the pack's loader now builds
a native MODEL with the int8 dp4a weights engaged. `test_smoke_wan.py` runs a full
denoise step on the small Wan (LTX-2.3's 23.5 GB int8 DiT exceeds a 16 GB card, so its
test stops at load + int8-resident)."""
from __future__ import annotations

import pytest

pytest.importorskip("comfy.sd")
pytest.importorskip("superl8")

import comfy.model_detection
import folder_paths
from superl8 import FQReader

from comfyui_superl8.superl8_tensor import FNI8Tensor
from comfyui_superl8.gate import is_sm70
from comfyui_superl8.nodes import UnetLoaderFNI8

from ._common import load_dit_or_skip

pytestmark = pytest.mark.comfy_e2e

VIDEO_DIFFUSERS_CHECKPOINTS = [
    pytest.param(("Wan-AI__Wan2.2-TI2V-5B-Diffusers.dit.b8.fni8", "wan22"), id="wan22_ti2v_5b"),
    pytest.param(("Wan-AI__Wan2.2-T2V-A14B-Diffusers.dit.b8.fni8", "wan22"), id="wan22_t2v_a14b"),
    pytest.param(("Lightricks__LTX-Video.dit.b8.fni8", "ltx_video"), id="ltx_video"),
    pytest.param(("Lightricks__LTX-2.3.dit.b8.fni8", "ltx_video"), id="ltx_2.3_distilled"),
]


@pytest.fixture(params=VIDEO_DIFFUSERS_CHECKPOINTS)
def video_checkpoint(request):
    unet_name, arch_name = request.param
    if not is_sm70():
        pytest.skip("needs a Volta/CMP (sm_70) GPU")
    if unet_name not in folder_paths.get_filename_list("diffusion_models"):
        pytest.skip(f"{unet_name} not found on the diffusion_models search path")
    return unet_name, arch_name


def test_diffusers_layout_is_undetected_by_comfy_natively(video_checkpoint):
    """The raw `.fni8` keys are still the diffusers layout ComfyUI cannot detect on its
    own — which is WHY `loader.remap_diffusers_to_native` (exercised by the load test
    below) exists. If this ever starts failing, ComfyUI grew native Wan/LTX diffusers
    detection and the pack's remap can be reconsidered."""
    unet_name, _ = video_checkpoint
    path = folder_paths.get_full_path("diffusion_models", unet_name)
    with FQReader(path) as r:
        sd_keys = {name: None for name in r.names}
    assert comfy.model_detection.detect_unet_config(sd_keys, "") is None, (
        f"{unet_name}: comfy.model_detection now recognizes this diffusers-format "
        "state dict natively — the pack's remap may no longer be needed"
    )


def test_video_diffusers_checkpoint_loads_native_with_int8(video_checkpoint):
    """The gap issue #30 closes: after `remap_diffusers_to_native`, `UnetLoaderFNI8`
    builds a real ComfyUI MODEL from the diffusers-layout video `.fni8` AND the int8
    dp4a weights are engaged. Covers all four published video checkpoints at the
    load/build level; `test_smoke_wan.py` runs a full denoise step on the small Wan.
    Big DiTs (14B) skip on a 16GB card via `load_dit_or_skip` (see issue #12)."""
    unet_name, arch_name = video_checkpoint
    (model,) = load_dit_or_skip(lambda: UnetLoaderFNI8().load(unet_name, arch=arch_name))
    dit = model.model.diffusion_model
    n_int8 = sum(
        1 for m in dit.modules()
        if getattr(m, "weight", None) is not None and isinstance(m.weight, FNI8Tensor)
    )
    assert n_int8 > 0, (
        f"{unet_name}: built as {type(dit).__name__} but no int8 FNI8Tensor weights are "
        "resident — the dp4a path is not engaged"
    )
