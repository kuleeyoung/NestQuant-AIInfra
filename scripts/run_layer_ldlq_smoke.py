"""Compare ordinary E8 quantization and block-LDLQ on one calibrated Llama weight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.beta_search import build_candidate_tables, select_beta_subset
from nestquant.core.codec import generate_e8_dither
from nestquant.core.ldlq import hessian_proxy_loss
from nestquant.core.quantizer import quantize_rows, quantize_rows_ldlq
from nestquant.core.rotation import rotate_covariance, signed_hadamard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--hessian", type=Path, required=True)
    parser.add_argument("--hessian-key", default=None)
    parser.add_argument("--relative-damp", type=float, default=0.01)
    parser.add_argument("--rotation-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from huggingface_hub import get_token
    from transformers import AutoModelForCausalLM

    project = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = project["llm_ppl_experiment"]
    q = project["initial_algorithm_config"]["q"]
    beta_count = project["initial_algorithm_config"]["selected_beta_count"]
    coefficients = tuple(experiment["beta_coefficients_project_scale"])
    candidate_scales = tuple(value / q for value in coefficients)
    seed = project["initial_algorithm_config"]["seed"]
    model_id, revision = experiment["model_id"], experiment["model_revision"]
    device = torch.device(args.device)
    token = get_token()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=args.offline,
    ).to(device).eval()
    modules = dict(model.named_modules())
    if args.module not in modules or not isinstance(modules[args.module], torch.nn.Linear):
        raise ValueError("--module must be the exact name of an nn.Linear")
    original_weight = modules[args.module].weight.detach().float()
    stored = torch.load(args.hessian, map_location="cpu", weights_only=True)
    key = args.hessian_key or args.module
    if not isinstance(stored, dict) or key not in stored:
        raise ValueError(f"Hessian file does not contain key {key!r}")
    hessian = stored[key].to(device=device, dtype=torch.float32)
    if hessian.shape != (original_weight.shape[1], original_weight.shape[1]):
        raise ValueError("Hessian shape does not match module in_features")

    weight = original_weight
    signs = None
    if args.rotation_seed is not None:
        if weight.shape[1] & (weight.shape[1] - 1):
            raise ValueError("signed Hadamard requires power-of-two in_features")
        rotation_generator = torch.Generator(device="cpu").manual_seed(args.rotation_seed)
        signs = (torch.randint(0, 2, (weight.shape[1],), generator=rotation_generator) * 2 - 1).to(
            device=device, dtype=weight.dtype
        )
        weight = signed_hadamard(weight, signs)
        hessian_for_ldlq = rotate_covariance(hessian, signs)
    else:
        hessian_for_ldlq = hessian

    rows, features = weight.shape
    row_norms = torch.linalg.vector_norm(weight, dim=-1)
    safe_norms = row_norms.clamp_min(torch.finfo(weight.dtype).tiny)
    normalized = weight * (math.sqrt(features) / safe_norms).unsqueeze(-1)
    normalized_blocks = normalized.reshape(rows, features // 8, 8)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dither = generate_e8_dither(
        rows * (features // 8), generator, device="cpu", dtype=torch.float32
    ).reshape(rows, features // 8, 8).to(device)
    candidate_mse, overloaded = build_candidate_tables(
        normalized_blocks.reshape(-1, 8), q, candidate_scales,
        dither=dither.reshape(-1, 8),
    )
    selected = select_beta_subset(candidate_scales, candidate_mse, overloaded, beta_count)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    ordinary = quantize_rows(weight, q, selected.beta_values, dither=dither).dequantize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ordinary_seconds = time.perf_counter() - started
    started = time.perf_counter()
    ldlq, ldlq_metadata = quantize_rows_ldlq(
        weight, hessian_for_ldlq, q, selected.beta_values, dither=dither,
        relative_damp=args.relative_damp,
    )
    ldlq_weight = ldlq.dequantize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ldlq_seconds = time.perf_counter() - started
    if signs is not None:
        ordinary_for_evaluation = signed_hadamard(ordinary, signs, inverse=True)
        ldlq_for_evaluation = signed_hadamard(ldlq_weight, signs, inverse=True)
    else:
        ordinary_for_evaluation = ordinary
        ldlq_for_evaluation = ldlq_weight
    ordinary_proxy = float(hessian_proxy_loss(
        original_weight, ordinary_for_evaluation, hessian
    ).item())
    ldlq_proxy = float(hessian_proxy_loss(
        original_weight, ldlq_for_evaluation, hessian
    ).item())
    ordinary_mse = float((original_weight - ordinary_for_evaluation).square().mean().item())
    ldlq_mse = float((original_weight - ldlq_for_evaluation).square().mean().item())
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_completed",
        "model_id": model_id,
        "model_revision": revision,
        "module": args.module,
        "weight_shape": list(original_weight.shape),
        "hessian_path": str(args.hessian.resolve()),
        "hessian_sha256": hashlib.sha256(args.hessian.read_bytes()).hexdigest(),
        "hessian_shape": list(hessian.shape),
        "q": q,
        "selected_beta_coefficients": [value * q for value in selected.beta_values],
        "selected_beta_codec_scales": list(selected.beta_values),
        "dither_seed": seed,
        "rotation_seed": args.rotation_seed,
        "weight_rotation": "none" if signs is None else "signed_hadamard_inverse_weight",
        "ldlq_hessian_coordinates": (
            "original_input" if signs is None else "signed_hadamard_input"
        ),
        "relative_damp": args.relative_damp,
        "ldlq": ldlq_metadata,
        "ordinary": {
            "hessian_proxy_loss": ordinary_proxy,
            "weight_mse": ordinary_mse,
            "quantization_seconds": ordinary_seconds,
        },
        "block_ldlq": {
            "hessian_proxy_loss": ldlq_proxy,
            "weight_mse": ldlq_mse,
            "quantization_seconds": ldlq_seconds,
        },
        "comparison": {
            "proxy_loss_ratio_ldlq_over_ordinary": ldlq_proxy / ordinary_proxy,
            "proxy_loss_reduction_fraction": 1.0 - ldlq_proxy / ordinary_proxy,
            "weight_mse_ratio_ldlq_over_ordinary": ldlq_mse / ordinary_mse,
        },
        "peak_cuda_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "limitations": [
            "One layer and one calibration Hessian smoke; no model PPL conclusion.",
            "Rotated mode transforms the stored covariance exactly as R.T @ H @ R; it does not recollect duplicate activations.",
            "Beta subset is selected before LDLQ, so the paper's post-LDLQ beta optimization is pending.",
            "This run uses the project's subtractive dither protocol; the public author LLM code disables dither.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
