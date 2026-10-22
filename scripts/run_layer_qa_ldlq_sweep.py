"""Run a bounded QA-LDLQ epsilon-scale sweep on representative Llama layers."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.beta_search import build_candidate_tables, select_beta_subset
from nestquant.core.codec import generate_e8_dither
from nestquant.core.ldlq import hessian_proxy_loss, qa_ldlq_adjustment
from nestquant.core.quantizer import quantize_rows_ldlq
from nestquant.models.hessian_store import LlamaHessianStore
from nestquant.models.qa_calibration import ActivationVarianceStore


DEFAULT_MODULES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.1.mlp.down_proj",
    "model.layers.8.self_attn.q_proj",
    "model.layers.15.mlp.down_proj",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    parser.add_argument(
        "--constants", nargs="+", type=float,
        default=(155.0, 158720.0, 317440.0, 634880.0),
    )
    parser.add_argument("--hessian-dir", type=Path, default=Path("results/raw/calibration"))
    parser.add_argument("--hessian-chunks", type=int, default=64)
    parser.add_argument(
        "--variance-file", type=Path,
        default=Path("results/raw/calibration/activation_variance_train20.json"),
    )
    parser.add_argument("--variance-chunks", type=int, default=20)
    parser.add_argument("--relative-damp", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/qa_ldlq_layer_sweep.json"))
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    constants = tuple(float(value) for value in args.constants)
    if not constants or any(not math.isfinite(value) or value <= 0 for value in constants):
        raise ValueError("constants must be positive and finite")

    from huggingface_hub import get_token
    from transformers import AutoModelForCausalLM

    project = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = project["llm_ppl_experiment"]
    algorithm = project["initial_algorithm_config"]
    model_id, revision = experiment["model_id"], experiment["model_revision"]
    q = int(algorithm["q"])
    beta_count = int(algorithm["selected_beta_count"])
    coefficients = tuple(float(value) for value in experiment["beta_coefficients_project_scale"])
    candidate_scales = tuple(value / q for value in coefficients)
    seed = int(algorithm["seed"])
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, token=get_token(), dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=args.offline,
    ).to(device).eval()
    modules = dict(model.named_modules())
    for name in args.modules:
        if name not in modules or not isinstance(modules[name], torch.nn.Linear):
            raise ValueError(f"unsupported Linear module: {name}")

    hessians = LlamaHessianStore(
        args.hessian_dir, calibration_chunks=args.hessian_chunks,
        expected_model_id=model_id, expected_model_revision=revision,
    )
    variances = ActivationVarianceStore(
        args.variance_file, expected_model_id=model_id,
        expected_model_revision=revision, expected_chunks=args.variance_chunks,
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model_id": model_id,
        "model_revision": revision,
        "modules": list(args.modules),
        "epsilon_constants": list(constants),
        "ordinary_control": "epsilon=0",
        "q": q,
        "beta_coefficients_project_scale": list(coefficients),
        "selected_beta_count": beta_count,
        "relative_damp": args.relative_damp,
        "dither_seed_per_module": seed,
        "hessian_chunks": args.hessian_chunks,
        "variance_chunks": args.variance_chunks,
        "variance_sha256": hashlib.sha256(args.variance_file.read_bytes()).hexdigest(),
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name in args.modules:
            print(f"Auditing {name}...", flush=True)
            original = modules[name].weight.detach().float()
            hessian_key, hessian_cpu = hessians.resolve(name)
            hessian = hessian_cpu.to(device=device, dtype=torch.float32)
            variance_key, x_var = variances.resolve(name)  # type: ignore[misc]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            rows, features = original.shape
            dither = generate_e8_dither(
                rows * (features // 8), generator, device="cpu", dtype=torch.float32,
            ).reshape(rows, features // 8, 8).to(device)
            layer = {
                "weight_shape": list(original.shape),
                "hessian_source": hessian_key,
                "variance_source": variance_key,
                "x_var": x_var,
                "hessian_diagonal_mean": float(hessian.diagonal().mean().item()),
                "candidates": [],
            }
            settings: list[tuple[str, float | None]] = [("ordinary", None)] + [
                (f"C={value:g}", value) for value in constants
            ]
            for label, constant in settings:
                started = time.perf_counter()
                if constant is None:
                    adjusted = original
                    effective_hessian = hessian
                    epsilon = 0.0
                else:
                    adjusted, effective_hessian, epsilon = qa_ldlq_adjustment(
                        original, hessian, activation_variance=x_var,
                        epsilon_constant=constant,
                    )
                norms = torch.linalg.vector_norm(adjusted, dim=-1)
                safe = norms.clamp_min(torch.finfo(adjusted.dtype).tiny)
                normalized = adjusted * (math.sqrt(features) / safe).unsqueeze(-1)
                tables = build_candidate_tables(
                    normalized.reshape(-1, 8), q, candidate_scales,
                    dither=dither.reshape(-1, 8),
                )
                selected = select_beta_subset(candidate_scales, *tables, beta_count)
                quantized, metadata = quantize_rows_ldlq(
                    adjusted, effective_hessian, q, selected.beta_values,
                    dither=dither, relative_damp=args.relative_damp,
                )
                restored = quantized.dequantize()
                error = original - restored
                shift = original - adjusted
                record = {
                    "label": label,
                    "epsilon_constant": constant,
                    "epsilon": epsilon,
                    "epsilon_squared_over_hessian_diag_mean": (
                        epsilon * epsilon / layer["hessian_diagonal_mean"]
                    ),
                    "relative_weight_adjustment": float(
                        torch.linalg.vector_norm(shift).item()
                        / torch.linalg.vector_norm(original).item()
                    ),
                    "selected_beta_coefficients": [value * q for value in selected.beta_values],
                    "maximum_beta_coefficient": max(selected.beta_values) * q,
                    "hessian_proxy_loss": float(hessian_proxy_loss(original, restored, hessian).item()),
                    "weight_mse": float(error.square().mean().item()),
                    "max_abs_weight_error": float(error.abs().max().item()),
                    "ldlq_absolute_damping": metadata["absolute_damping_added"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                layer["candidates"].append(record)
                print(
                    f"  {label}: proxy={record['hessian_proxy_loss']:.6g} "
                    f"shift={record['relative_weight_adjustment']:.3e} "
                    f"betas={record['selected_beta_coefficients']}", flush=True,
                )
                del adjusted, effective_hessian, quantized, restored, error, shift, tables
            ordinary_loss = layer["candidates"][0]["hessian_proxy_loss"]
            for record in layer["candidates"]:
                record["proxy_loss_ratio_over_ordinary"] = record["hessian_proxy_loss"] / ordinary_loss
            payload["results"][name] = layer
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            del original, hessian, dither
        payload["status"] = "completed"
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        hessians.close()
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
