# SPDX-License-Identifier: Apache-2.0
"""Canonical Z-Image Turbo production profile for the content-factory fleet."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

CANONICAL_MODEL_ROOT = os.environ.get("SUPERL8_MODEL_ROOT", "")
if not CANONICAL_MODEL_ROOT:
    raise RuntimeError(
        "SUPERL8_MODEL_ROOT environment variable is not set. "
        "Set it to the path containing your model weights."
    )

DIT_RELATIVE_PATH = "unet/Tongyi-MAI__Z-Image-Turbo.dit.b8.fni8"
VAE_RELATIVE_PATH = "vae/ae.safetensors"
TEXT_ENCODER_RELATIVE_PATH = "text_encoders/qwen_3_4b.safetensors"

DIT_SHA256 = "1051c94040c0b831c5f1a307d66d51d19a812da45a7742e814782b2027a0355b"
VAE_SHA256 = "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"
TEXT_ENCODER_SHA256 = "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a"

STEPS = 8
CFG = 1.0
FLOW_SHIFT = 3.0
SAMPLER = "res_multistep"
SCHEDULER = "simple"
VAE_STRIDE = 8

# These counts are properties of the exact checksummed production artifact.
EXPECTED_FNI8_TENSORS = 172
EXPECTED_INT8_LINEARS = 170
EXPECTED_INT8_QKV = 34
EXPECTED_INT8_ATTN_OUT = 34
EXPECTED_INT8_PAD_TOKENS = frozenset({"x_pad_token", "cap_pad_token"})

# Retained for experiments only. It is not selected by this profile or service.
GGUF_CANDIDATE_RELATIVE_PATH = "unet/z_image_turbo-Q5_K_M.gguf"


@dataclass(frozen=True)
class ZImageFiles:
    dit: str
    vae: str
    text_encoder: str


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_files(model_root: str | None = None, *, verify: bool = True) -> ZImageFiles:
    """Resolve only the canonical NVMe layout and fail closed on any drift."""
    root = os.path.realpath(model_root or CANONICAL_MODEL_ROOT)
    canonical_root = os.path.realpath(CANONICAL_MODEL_ROOT)
    if root != canonical_root:
        raise ValueError(
            f"Z-Image production root must be NVMe {canonical_root}, got {root}"
        )

    files = ZImageFiles(
        dit=os.path.join(root, DIT_RELATIVE_PATH),
        vae=os.path.join(root, VAE_RELATIVE_PATH),
        text_encoder=os.path.join(root, TEXT_ENCODER_RELATIVE_PATH),
    )
    expected = {
        files.dit: DIT_SHA256,
        files.vae: VAE_SHA256,
        files.text_encoder: TEXT_ENCODER_SHA256,
    }
    for path, checksum in expected.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"canonical Z-Image artifact is missing: {path}")
        if os.path.islink(path):
            raise ValueError(f"canonical Z-Image artifact must not be a symlink: {path}")
        if verify:
            actual = file_sha256(path)
            if actual != checksum:
                raise ValueError(
                    f"canonical Z-Image checksum mismatch for {path}: "
                    f"sha256={actual}, expected={checksum}"
                )
    return files


def validate_geometry(width: int, height: int) -> tuple[int, int]:
    if width % VAE_STRIDE or height % VAE_STRIDE:
        raise ValueError(
            f"width and height must be multiples of {VAE_STRIDE}; got {width}x{height}"
        )
    return height // VAE_STRIDE, width // VAE_STRIDE


def int8_weight_inventory(diffusion_model) -> dict[str, object]:
    from .superl8_tensor import FNI8Tensor

    tensor_names: set[str] = set()
    cuda_tensor_names: set[str] = set()
    for name, param in diffusion_model.named_parameters():
        if not isinstance(param, FNI8Tensor):
            continue
        tensor_names.add(name)
        if param.device.type == "cuda" and param.q_scale.device.type == "cuda":
            cuda_tensor_names.add(name)

    linear_names = {
        f"{name}.weight" if name else "weight"
        for name, module in diffusion_model.named_modules()
        if isinstance(getattr(module, "weight", None), FNI8Tensor)
    }
    pad_tokens = tensor_names - linear_names
    return {
        "tensors_total": len(tensor_names),
        "cuda_tensors": len(cuda_tensor_names),
        "linears": len(linear_names),
        "cuda_linears": len(linear_names & cuda_tensor_names),
        "qkv": sum(name.endswith("attention.qkv.weight") for name in linear_names),
        "out": sum(name.endswith("attention.out.weight") for name in linear_names),
        "pad_tokens": sorted(pad_tokens),
        "cuda_pad_tokens": sorted(pad_tokens & cuda_tensor_names),
    }


def assert_int8_inventory(diffusion_model, *, require_cuda: bool) -> dict[str, object]:
    inventory = int8_weight_inventory(diffusion_model)
    expected = {
        "tensors_total": EXPECTED_FNI8_TENSORS,
        "linears": EXPECTED_INT8_LINEARS,
        "qkv": EXPECTED_INT8_QKV,
        "out": EXPECTED_INT8_ATTN_OUT,
    }
    for key, value in expected.items():
        if inventory[key] != value:
            raise RuntimeError(
                f"Z-Image int8 engagement mismatch: {key}={inventory[key]}, expected={value}"
            )
    pad_tokens = frozenset(inventory["pad_tokens"])
    if pad_tokens != EXPECTED_INT8_PAD_TOKENS:
        raise RuntimeError(
            "Z-Image non-Linear FNI8Tensor set mismatch: "
            f"{sorted(pad_tokens)}, expected={sorted(EXPECTED_INT8_PAD_TOKENS)}"
        )
    if require_cuda and inventory["cuda_tensors"] != inventory["tensors_total"]:
        raise RuntimeError(
            "Z-Image DiT is not fully resident: "
            f"{inventory['cuda_tensors']}/{inventory['tensors_total']} int8 tensors "
            "are on CUDA"
        )
    return inventory


def move_superl8_tensors_(diffusion_model, device) -> dict[str, object]:
    """Move every direct FNI8Tensor parameter, including sidecar scales, in place.

    ComfyUI's model manager moves ordinary parameter storage but cannot discover
    ``FNI8Tensor.q_scale`` because it is Python-side metadata. Walking each
    module's direct ``_parameters`` avoids recursive duplicates and covers both
    Linear weights and the two non-Linear pad embeddings.
    """
    from .superl8_tensor import FNI8Tensor

    moved: list[str] = []
    for module_name, module in diffusion_model.named_modules():
        for parameter_name, parameter in list(module._parameters.items()):
            if not isinstance(parameter, FNI8Tensor):
                continue
            qualified_name = (
                f"{module_name}.{parameter_name}" if module_name else parameter_name
            )
            metadata = (
                parameter.q_scheme,
                parameter.q_group_size,
                parameter.q_codebook,
            )
            replacement = parameter.to(device=device)
            if not isinstance(replacement, FNI8Tensor):
                raise RuntimeError(f"moving {qualified_name} dropped FNI8Tensor type")
            if replacement.q_scale is None:
                raise RuntimeError(f"moving {qualified_name} dropped q_scale")
            if (
                replacement.q_scheme,
                replacement.q_group_size,
                replacement.q_codebook,
            ) != metadata:
                raise RuntimeError(f"moving {qualified_name} changed quantization metadata")
            module._parameters[parameter_name] = replacement
            moved.append(qualified_name)

    if len(moved) != EXPECTED_FNI8_TENSORS:
        raise RuntimeError(
            f"moved {len(moved)} Z-Image FNI8Tensor parameters, "
            f"expected {EXPECTED_FNI8_TENSORS}"
        )
    return assert_int8_inventory(
        diffusion_model, require_cuda=str(device).startswith("cuda")
    )


def runtime_int8_inventory(diffusion_model) -> dict[str, object]:
    from .superl8_tensor import FNI8Tensor

    result: dict[str, object] = {
        "total": 0,
        "int8": 0,
        "fallback": 0,
        "pending": 0,
        "fallback_layers": [],
        "invalid_fallback_layers": [],
        "pending_layers": [],
    }
    fallback_layers: list[dict[str, object]] = result["fallback_layers"]
    invalid_fallback_layers: list[str] = result["invalid_fallback_layers"]
    pending_layers: list[str] = result["pending_layers"]
    for name, module in diffusion_model.named_modules():
        if not isinstance(getattr(module, "weight", None), FNI8Tensor):
            continue
        result["total"] += 1
        verdict = getattr(module, "_fni8_sqnr_pass", None)
        if verdict is True:
            result["int8"] += 1
        elif verdict is False:
            result["fallback"] += 1
            calibrator = getattr(module, "_fni8_calib", None)
            samples = getattr(calibrator, "n", None)
            required_samples = getattr(calibrator, "calib_samples", None)
            worst_cos = getattr(calibrator, "worst_cos", None)
            cos_bar = getattr(calibrator, "cos_bar", None)
            valid = (
                getattr(calibrator, "passed", None) is False
                and isinstance(samples, int)
                and isinstance(required_samples, int)
                and samples >= required_samples >= 1
                and isinstance(worst_cos, (int, float))
                and isinstance(cos_bar, (int, float))
                and math.isfinite(float(worst_cos))
                and math.isfinite(float(cos_bar))
                and float(worst_cos) < float(cos_bar)
            )
            if not valid:
                invalid_fallback_layers.append(name)
            fallback_layers.append(
                {
                    "name": name,
                    "samples": samples,
                    "required_samples": required_samples,
                    "worst_cos": worst_cos,
                    "cos_bar": cos_bar,
                    "quality_gate_valid": valid,
                }
            )
        else:
            result["pending"] += 1
            pending_layers.append(name)
    return result


def assert_runtime_int8_engagement(diffusion_model) -> dict[str, object]:
    """Require every production Linear to have a finalized, evidenced verdict."""
    inventory = runtime_int8_inventory(diffusion_model)
    engaged = inventory["int8"] + inventory["fallback"]
    if (
        inventory["total"] != EXPECTED_INT8_LINEARS
        or engaged != EXPECTED_INT8_LINEARS
        or inventory["pending"] != 0
        or inventory["invalid_fallback_layers"]
    ):
        raise RuntimeError(
            f"Z-Image int8 runtime engagement is incomplete or unproven: {inventory}"
        )
    return inventory


def assert_model_sampling_shift(model, expected: float = FLOW_SHIFT) -> float:
    actual = getattr(getattr(model.model, "model_sampling", None), "shift", None)
    if actual is None or abs(float(actual) - expected) > 1e-6:
        raise RuntimeError(
            f"loaded Z-Image model_sampling.shift={actual!r}; expected {expected}"
        )
    return float(actual)


def load_int8_model(model_root: str | None = None):
    import folder_paths

    from .nodes import UnetLoaderFNI8

    root = os.path.realpath(model_root or CANONICAL_MODEL_ROOT)
    folder_paths.add_model_folder_path(
        "diffusion_models", os.path.join(root, os.path.dirname(DIT_RELATIVE_PATH))
    )
    (model,) = UnetLoaderFNI8().load(os.path.basename(DIT_RELATIVE_PATH), arch="zimage")
    assert_int8_inventory(model.model.diffusion_model, require_cuda=False)
    assert_model_sampling_shift(model)
    return model


def denoise(
    model,
    cond,
    empty,
    in_channels: int,
    *,
    latent_h: int,
    latent_w: int,
    steps: int = STEPS,
    seed: int = 0,
    abort_check=None,
):
    import comfy.sample
    import torch

    latent = torch.zeros((1, in_channels, latent_h, latent_w), dtype=torch.float32)
    noise = comfy.sample.prepare_noise(latent, seed=seed)
    callback = None
    if abort_check is not None:

        def callback(_step, _denoised, _x, _total_steps):
            abort_check()

    samples = comfy.sample.sample(
        model,
        noise,
        steps=steps,
        cfg=CFG,
        sampler_name=SAMPLER,
        scheduler=SCHEDULER,
        positive=cond,
        negative=empty,
        latent_image=latent,
        denoise=1.0,
        callback=callback,
    )
    return samples.detach()
