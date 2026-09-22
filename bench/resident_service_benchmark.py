#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-weight, safe-GPU-elected benchmark for the Z-Image resident service (#181).

Root-review blocker #5: the 83 unit tests are contract/mocked tests run in a
GPU-enabled container -- they are not a real inference, safe-election,
residency, repeat-request, timeout, or 896x1152 proof. This script IS that
proof. It is a real-weight, real-GPU harness meant to be run by a human with
fleet access; it is intentionally NOT invoked by the fast/non-GPU test lane or
CI, and this session does not execute it (no GPU here; heavy real-weight runs
are opt-in per AGENTS.md).

What it measures, end to end, against a live HTTP resident service it spawns
itself on a safely-elected card:

  - 512x512 and 896x1152 (issue #181's acceptance-criteria shapes): cold
    (first) request + three warm repeats each.
  - Per-stage timing (TE/DiT/VAE/PNG) and total wall-clock, read straight off
    the service's own response envelope (not re-derived/estimated).
  - Step rate (steps / dit_s).
  - Peak HBM at each stage and overall, read from the response's `hbm` block.
  - SHA-256 checksums of every returned PNG (byte-identity across warm
    fixed-seed repeats is itself a determinism check).
  - Latent/image oracle metrics: PSNR/SSIM of the service's decoded PNG against
    an independently-run fp reference (same prompt/seed/size), using the same
    `bench.quality` oracle `bench/full_pipeline_zimage.py` already validated
    int8-vs-fp with -- not a "visually plausible" eyeball check.
  - Repeat-request residency: `total_generations` advances by exactly the
    number of successful requests, and warm-repeat `dit_s` does not carry a
    DiT-reload spike relative to other warm repeats (a cold DiT load would show
    up as an outlier well beyond the rest of the warm spread).
  - Timeout overrun: sends a request with a timeout_s no honest pipeline can
    meet, and measures the REAL wall-clock overrun past that timeout_s --
    proving (or disproving) the bound documented in
    `comfyui_superl8/resident_service.py`'s module docstring.
  - Same-card baseline: runs `bench.full_pipeline_zimage`'s own proven harness
    directly (no HTTP layer, no sequential TE/VAE eviction) on the exact same
    elected card, for a bare wall-clock reference point.

GPU binding: this script performs its OWN safe election (reusing
`comfyui_superl8/gpu_election.py`, loaded by file path so this process never
imports `comfyui_superl8`/`comfy`/`torch` before electing) and sets
`CUDA_VISIBLE_DEVICES` for itself BEFORE importing torch, exactly like
`scripts/launch_resident_service.py`. It then spawns the resident service as a
child process inheriting that binding (satisfying
`ZImageResidentService._verify_gpu_election`'s fail-closed check) -- so the
service and this script's own same-card baseline/oracle runs are provably on
the identical physical card.

Run with the canonical NVMe profile:

    python3 bench/resident_service_benchmark.py \\
        --outdir /tmp/zimage_resident_bench

Never use the pinned live-server GPU (AGENTS.md); FNI8_PINNED_GPU_UUIDS must
name it, exactly as required by `comfyui_superl8.gpu_election.filter_eligible`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROMPT = (
    "a photograph of a red fox sitting in a field of autumn leaves, "
    "warm golden hour light, sharp focus, highly detailed"
)
SEED = 0
SIZES = [(512, 512), (896, 1152)]
WARM_REPEATS = 3
_LOCKED_STEPS = 8


# ---------------------------------------------------------------------------
# GPU election (stdlib only until CUDA_VISIBLE_DEVICES is set) --------------
# ---------------------------------------------------------------------------


def _load_gpu_election():
    """Load `comfyui_superl8/gpu_election.py` by file path -- not `from
    comfyui_superl8 import gpu_election` -- so this process never imports the
    `comfyui_superl8` package (and therefore never touches torch/comfy) before it
    has elected and bound a card. Mirrors scripts/launch_resident_service.py."""
    path = os.path.join(_REPO_ROOT, "comfyui_superl8", "gpu_election.py")
    spec = importlib.util.spec_from_file_location("fni8_gpu_election", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _nvidia_smi(query: str) -> str:
    query_flag = "--query-compute-apps" if "gpu_uuid" in query else "--query-gpu"
    result = subprocess.run(
        ["nvidia-smi", f"{query_flag}={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout


def elect_gpu(lock_dir: str):
    ge = _load_gpu_election()
    pinned_raw = os.environ.get("FNI8_PINNED_GPU_UUIDS", "")
    pinned = {u.strip() for u in pinned_raw.split(",") if u.strip()}
    gpus = ge.parse_gpu_csv(_nvidia_smi("index,uuid,name,compute_cap,memory.total,memory.free"))
    resident = ge.parse_compute_apps_csv(_nvidia_smi("gpu_uuid,pid"))
    return ge.elect_and_lock(gpus, resident, pinned, lock_dir)


# ---------------------------------------------------------------------------
# HTTP client (stdlib only) ---------------------------------------------------
# ---------------------------------------------------------------------------


def _http(method: str, url: str, body: dict | None = None, timeout: float = 650.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(data))
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        code = e.code
        payload = json.loads(e.read())
    wall_s = time.monotonic() - t0
    return code, payload, wall_s


def wait_ready(base_url: str, timeout_s: float = 900.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last_status = None
    while time.monotonic() < deadline:
        try:
            code, payload, _ = _http("GET", base_url + "/ready", timeout=10)
            last_status = payload
            if code == 200:
                return payload
        except Exception as e:  # connection refused while the server is starting
            last_status = {"error": str(e)}
        time.sleep(2)
    raise TimeoutError(f"service never became ready: {last_status}")


def run_generate(
    base_url: str, width: int, height: int, seed: int = SEED, prompt: str = PROMPT,
    timeout_s: float = 580.0,
):
    body = {
        "width": width, "height": height, "prompt": prompt, "seed": seed,
        "timeout_s": timeout_s,
    }
    return _http("POST", base_url + "/generate", body, timeout=timeout_s + 30.0)


# ---------------------------------------------------------------------------
# Measurements ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def measure_size(base_url: str, width: int, height: int, outdir: str, label: str) -> list[dict]:
    """Cold (first) request + WARM_REPEATS repeats. Returns one record per
    request with the service's own reported timing/hbm/provenance plus a
    SHA-256 checksum of the returned PNG."""
    records = []
    for i in range(1 + WARM_REPEATS):
        kind = "cold" if i == 0 else f"warm{i}"
        code, payload, wall_s = run_generate(base_url, width, height)
        if code != 200:
            records.append({"kind": kind, "code": code, "error": payload})
            print(f"  [{label}/{kind}] FAILED code={code} {payload}")
            continue
        png_b64 = payload.pop("image_png", None)
        checksum = hashlib.sha256(base64.b64decode(png_b64)).hexdigest() if png_b64 else None
        if png_b64:
            with open(os.path.join(outdir, f"{label}_{kind}.png"), "wb") as f:
                f.write(base64.b64decode(png_b64))
        dit_s = payload.get("timing", {}).get("dit_s", 0.0)
        step_rate = _LOCKED_STEPS / dit_s if dit_s else None
        records.append(
            {
                "kind": kind,
                "wall_s": wall_s,
                "timing": payload.get("timing"),
                "hbm": payload.get("hbm"),
                "checksum": checksum,
                "step_rate_sps": step_rate,
                "provenance": payload.get("provenance"),
            }
        )
        print(
            f"  [{label}/{kind}] wall={wall_s:.2f}s timing={payload.get('timing')} "
            f"step_rate={step_rate} hbm={payload.get('hbm')} checksum={checksum}"
        )
    return records


def measure_repeat_residency(records: list[dict], ready_before: dict, ready_after: dict) -> dict:
    """Prove the DiT stayed resident across repeats without trusting the
    service's self-report alone: `total_generations` must advance by exactly
    the count of successful requests observed here, AND no warm repeat's
    `dit_s` may be an outlier vs. the others (a cold DiT reload dominates
    `dit_s` -- multiple seconds to load ~5 GiB of int8 weights -- so a reload
    hiding inside a "warm" repeat shows up as a large spread)."""
    warm = [r for r in records if r["kind"].startswith("warm") and "timing" in r]
    ok = [r for r in records if "timing" in r]
    counters_advanced = None
    if isinstance(ready_before, dict) and isinstance(ready_after, dict):
        before = ready_before.get("total_generations")
        after = ready_after.get("total_generations")
        if before is not None and after is not None:
            counters_advanced = (after - before) == len(ok)
    if len(warm) < 2:
        return {
            "proved": False,
            "reason": "fewer than 2 warm repeats succeeded",
            "counters_advanced": counters_advanced,
        }
    dit_times = [r["timing"]["dit_s"] for r in warm]
    spread = (max(dit_times) - min(dit_times)) / max(dit_times)
    return {
        # 0.5 is generous headroom over step-to-step jitter; a genuine DiT
        # reload costs seconds on top of an 8-step denoise and blows well past it.
        "proved": spread < 0.5 and (counters_advanced is not False),
        "dit_s_spread_fraction": spread,
        "dit_s_values": dit_times,
        "counters_advanced": counters_advanced,
        "total_generations_before": ready_before.get("total_generations")
        if isinstance(ready_before, dict) else None,
        "total_generations_after": ready_after.get("total_generations")
        if isinstance(ready_after, dict) else None,
    }


def measure_timeout_overrun(base_url: str, width: int, height: int) -> dict:
    """Send a request with an unmeetable timeout_s and measure the REAL
    wall-clock overrun past it, proving (or disproving) the bound documented in
    comfyui_superl8/resident_service.py's module docstring: bounded by the
    slowest of one TE forward pass / one DiT step / one VAE tile, not by the
    whole remaining stage."""
    tiny_timeout = 0.5
    code, payload, wall_s = run_generate(base_url, width, height, timeout_s=tiny_timeout)
    return {
        "timeout_s": tiny_timeout,
        "wall_s": wall_s,
        "overrun_s": wall_s - tiny_timeout,
        "code": code,
        "payload": payload,
    }


def _encode_prompt_direct(weights_dir: str, prompt: str):
    """Same TE-load-encode-free sequence as resident_service._encode_prompt,
    but standalone (no HTTP service) for the same-card baseline / oracle runs
    below. Deliberately built from `weights_dir`, not `bench.full_pipeline_zimage`'s
    hardcoded `WEIGHTS_DIR`/`TE_DIR` module constants, so this script measures
    the operator's actual weights, not a hardcoded default path."""
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils
    import torch

    from comfyui_superl8.zimage_profile import resolve_files

    files = resolve_files(weights_dir)
    state = comfy.utils.load_torch_file(files.text_encoder, safe_load=True)
    clip = comfy.sd.load_text_encoder_state_dicts([state], clip_type=None)
    tokens = clip.tokenize(prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)
    empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    mm.unload_all_models()
    del clip, state
    torch.cuda.empty_cache()
    return cond, empty


def _load_vae_direct(weights_dir: str):
    import comfy.sd
    import comfy.utils

    from comfyui_superl8.zimage_profile import resolve_files

    files = resolve_files(weights_dir)
    return comfy.sd.VAE(sd=comfy.utils.load_torch_file(files.vae, safe_load=True))


def same_card_baseline(width: int, height: int, weights_dir: str) -> dict:
    """Direct `bench.full_pipeline_zimage` denoise/decode on THIS elected card
    (no HTTP layer, no sequential TE/VAE eviction) -- a bare wall-clock
    reference to compare the resident service's own stage timings against, on
    the identical physical card (see module docstring)."""
    import torch

    from bench import full_pipeline_zimage as fp

    t0 = time.monotonic()
    cond, empty = _encode_prompt_direct(weights_dir, PROMPT)
    t_te = time.monotonic()

    model = fp.load_int8_model(weights_dir)
    in_ch = model.model.diffusion_model.in_channels
    lat = fp.denoise(
        model, cond, empty, in_ch, steps=_LOCKED_STEPS, seed=SEED,
        latent_h=height // 8, latent_w=width // 8,
    )
    t_dit = time.monotonic()

    vae = _load_vae_direct(weights_dir)
    fp.vae_decode_tiled(vae, lat, tile_size=512, overlap=64)
    t_vae = time.monotonic()

    return {
        "te_s": t_te - t0,
        "dit_s": t_dit - t_te,
        "vae_s": t_vae - t_dit,
        "total_s": t_vae - t0,
        "peak_hbm_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }


def oracle_metrics(png_bytes: bytes, width: int, height: int, weights_dir: str) -> dict:
    """PSNR/SSIM of the resident service's decoded PNG against an independently
    run fp reference at the same prompt/seed/size, using the same oracle
    `bench.quality` that `bench/full_pipeline_zimage.py` already validated
    int8-vs-fp with -- not a "visually plausible image" eyeball check."""
    import numpy as np
    import torch
    from PIL import Image

    from bench.quality import psnr, ssim
    from bench import full_pipeline_zimage as fp

    from comfyui_superl8.zimage_profile import resolve_files

    path = resolve_files(weights_dir).dit

    img_int8 = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB")).astype("float32") / 255.0
    img_int8_t = torch.from_numpy(img_int8).movedim(-1, 0).unsqueeze(0)

    cond, empty = _encode_prompt_direct(weights_dir, PROMPT)
    m_fp = fp.load_fp_model(path)
    in_ch = m_fp.model.diffusion_model.in_channels
    lat_fp = fp.denoise(
        m_fp, cond, empty, in_ch, steps=_LOCKED_STEPS, seed=SEED,
        latent_h=height // 8, latent_w=width // 8,
    )
    vae = _load_vae_direct(weights_dir)
    img_fp = fp.vae_decode_tiled(vae, lat_fp, tile_size=512, overlap=64).cpu()

    if img_int8_t.shape != img_fp.shape:
        # Tolerate tile-boundary pixel-rounding differences between the two
        # independent tiled-decode runs; a real quality regression is not
        # rounded away by a crop of a handful of edge pixels.
        h = min(img_int8_t.shape[-2], img_fp.shape[-2])
        w = min(img_int8_t.shape[-1], img_fp.shape[-1])
        img_int8_t = img_int8_t[..., :h, :w]
        img_fp = img_fp[..., :h, :w]
    return {"psnr_db": psnr(img_int8_t, img_fp), "ssim_score": ssim(img_int8_t, img_fp)}


# ---------------------------------------------------------------------------
# Main --------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from comfyui_superl8.zimage_profile import CANONICAL_MODEL_ROOT

    ap.add_argument("--weights-dir", default=CANONICAL_MODEL_ROOT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8410)
    ap.add_argument("--lock-dir", default="/tmp/fni8-locks")
    ap.add_argument("--outdir", default="/tmp/zimage_resident_bench")
    ap.add_argument("--service-ready-timeout-s", type=float, default=900.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[1/6] safe GPU election (this benchmark session)")
    try:
        gpu, lock_handle = elect_gpu(args.lock_dir)
    except Exception as e:
        print(f"::error::GPU election failed: {e}", file=sys.stderr)
        return 1
    print(f"      elected GPU {gpu.index} ({gpu.uuid}, {gpu.name})")

    # Bind THIS process before any torch import: the same-card baseline and
    # oracle metrics run in-process later and must see only the elected card,
    # exactly like scripts/launch_resident_service.py does for its child.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu.uuid

    child_env = dict(os.environ)
    child_env["FNI8_GPU_UUID"] = gpu.uuid
    child_env["FNI8_GPU_ELECTED"] = "1"
    child_env["FNI8_GPU"] = str(gpu.index)

    base_url = f"http://{args.host}:{args.port}"
    print(f"[2/6] spawning resident service on the elected card ({base_url})")
    service = subprocess.Popen(
        [
            sys.executable, "-m", "comfyui_superl8.resident_service",
            "--host", args.host, "--port", str(args.port),
            "--weights-dir", args.weights_dir,
        ],
        env=child_env,
        cwd=_REPO_ROOT,
    )

    results = {"gpu": {"index": gpu.index, "uuid": gpu.uuid, "name": gpu.name}, "sizes": {}}
    try:
        t_load0 = time.monotonic()
        ready = wait_ready(base_url, timeout_s=args.service_ready_timeout_s)
        results["cold_load_s"] = time.monotonic() - t_load0
        print(f"      ready after {results['cold_load_s']:.1f}s: {ready}")

        for width, height in SIZES:
            label = f"{width}x{height}"
            print(f"[3/6] {label}: cold + {WARM_REPEATS} warm repeats")
            ready_before = wait_ready(base_url, timeout_s=30)
            records = measure_size(base_url, width, height, args.outdir, label)
            ready_after = wait_ready(base_url, timeout_s=30)

            print(f"[4/6] {label}: repeat-request residency proof")
            residency = measure_repeat_residency(records, ready_before, ready_after)
            print(f"      {residency}")

            print(f"[5/6] {label}: same-card direct-harness baseline")
            baseline = same_card_baseline(width, height, args.weights_dir)
            print(f"      {baseline}")

            oracle = None
            first_ok = next((r for r in records if r.get("checksum")), None)
            if first_ok is not None:
                png_path = os.path.join(args.outdir, f"{label}_{first_ok['kind']}.png")
                with open(png_path, "rb") as f:
                    png_bytes = f.read()
                print(f"[6/6] {label}: latent/image oracle metrics vs fp reference")
                try:
                    oracle = oracle_metrics(png_bytes, width, height, args.weights_dir)
                    print(f"      {oracle}")
                except Exception as e:
                    oracle = {"error": str(e)}
                    print(f"      oracle metrics failed: {e}")

            results["sizes"][label] = {
                "records": records,
                "residency": residency,
                "same_card_baseline_s": baseline,
                "oracle": oracle,
            }

        print("[timeout overrun] tiny timeout_s at 512x512")
        results["timeout_overrun"] = measure_timeout_overrun(base_url, 512, 512)
        print(f"      {results['timeout_overrun']}")

    finally:
        service.terminate()
        try:
            service.wait(timeout=30)
        except subprocess.TimeoutExpired:
            service.kill()
        lock_handle.close()

    out_path = os.path.join(args.outdir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
