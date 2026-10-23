"""Evaluate BF16 and saved NestQuant checkpoints on lm-eval multiple-choice tasks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.resources
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import torch

from .llm_perplexity import (
    UPSTREAM_ACTIVATION_BETA_COEFFICIENTS,
    _load_kv_beta_config,
    _load_nqmc_into_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "project.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    from lm_eval.utils import handle_non_serializable

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=handle_non_serializable) + "\n",
        encoding="utf-8",
    )


def _enable_fresh_quantized_cache(model, cache_factory) -> None:
    original_forward = model.forward

    def forward_with_quantized_cache(self, *args, **kwargs):
        if kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = cache_factory()
        kwargs["use_cache"] = True
        return original_forward(*args, **kwargs)

    model.forward = MethodType(forward_with_quantized_cache, model)


def _task_provenance(tasks: list[str]) -> dict:
    from datasets import load_dataset

    specifications = {
        "piqa": ("baber/piqa", "validation", "piqa/README.md", "piqa/piqa.yaml"),
        "hellaswag": (
            "Rowan/hellaswag", "validation", "hellaswag/README.md",
            "hellaswag/hellaswag.yaml",
        ),
    }
    task_root = importlib.resources.files("lm_eval.tasks")
    provenance = {}
    for task in tasks:
        if task not in specifications:
            provenance[task] = {"status": "unrecorded_task"}
            continue
        dataset_path, split, readme_name, yaml_name = specifications[task]
        dataset = load_dataset(dataset_path, split=split)
        readme = Path(str(task_root.joinpath(readme_name)))
        yaml = Path(str(task_root.joinpath(yaml_name)))
        provenance[task] = {
            "dataset_path": dataset_path,
            "split": split,
            "row_count": len(dataset),
            "dataset_fingerprint": dataset._fingerprint,
            "task_yaml_sha256": _sha256(yaml),
            "task_readme_sha256": _sha256(readme),
        }
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bf16", "w4", "w4a4kv4"), required=True)
    parser.add_argument("--tasks", nargs="+", default=("piqa", "hellaswag"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--kv-beta-config", type=Path,
                        default=Path("configs/kv_betas_upstream_defaults_q14.json"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--limit", type=float, default=None,
                        help="lm-eval per-task document limit; omit for full evaluation")
    parser.add_argument("--offline-model", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method != "bf16" and args.checkpoint is None:
        parser.error("W4 methods require --checkpoint")
    if args.method == "bf16" and args.checkpoint is not None:
        parser.error("BF16 does not accept --checkpoint")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["llm_ppl_experiment"]
    model_id = experiment["model_id"]
    revision = experiment["model_revision"]
    q = config["initial_algorithm_config"]["q"]
    beta_count = config["initial_algorithm_config"]["selected_beta_count"]
    seed = config["initial_algorithm_config"]["seed"]
    beta_coefficients = tuple(experiment["beta_coefficients_project_scale"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    payload = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "tasks": list(args.tasks),
        "limit_per_task": args.limit,
        "zero_shot": True,
        "batch_size": args.batch_size,
        "model": {"id": model_id, "revision": revision, "dtype": "bfloat16"},
        "checkpoint": None if args.checkpoint is None else {
            "path": str(args.checkpoint.resolve()), "sha256": _sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "lm_eval": importlib.metadata.version("lm_eval"),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "evidence_boundary": (
            "W4 checkpoint is dequantized into BF16 Linear execution; A4/KV4 are fake quant. "
            "These results measure task quality, not low-bit kernel speed."
        ),
    }
    _write_json(args.output, payload)
    started = time.perf_counter()
    activation_handles = []
    try:
        from huggingface_hub import get_token
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = get_token()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, token=token,
            local_files_only=args.offline_model, use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, token=token, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=args.offline_model,
        ).to(device).eval()

        runtime_stats = {}
        if args.method != "bf16":
            runtime_stats["checkpoint_round_trip"] = _load_nqmc_into_model(
                args.checkpoint, model, device, model_id, revision, q, seed,
                beta_coefficients, beta_count,
            )
        if args.method == "w4a4kv4":
            from ..evaluation.activation_quantization import (
                ActivationQuantizationStats, enable_activation_quantization,
            )
            from ..evaluation.kv_cache import (
                KVQuantizationStats, NestQuantDynamicCache, enable_kv_attention_rotation,
            )

            activation_betas = tuple(
                value / q for value in UPSTREAM_ACTIVATION_BETA_COEFFICIENTS
            )
            activation_stats = ActivationQuantizationStats()
            activation_stats, activation_handles = enable_activation_quantization(
                model, q=q, betas=activation_betas, rows_per_chunk=4096,
                stats=activation_stats,
            )
            layer_betas, beta_metadata = _load_kv_beta_config(
                args.kv_beta_config, model_id=model_id, model_revision=revision,
                q=q, selected_beta_count=beta_count,
            )
            default_betas = tuple(value / q for value in beta_coefficients[:beta_count])
            kv_stats = KVQuantizationStats()
            enable_kv_attention_rotation(model)

            def cache_factory():
                return NestQuantDynamicCache(
                    config=model.config, q=q, default_betas=default_betas,
                    layer_betas=layer_betas, rows_per_chunk=4096, stats=kv_stats,
                )

            _enable_fresh_quantized_cache(model, cache_factory)
            runtime_stats["activation_beta_values"] = list(activation_betas)
            runtime_stats["kv_beta_config"] = beta_metadata

        harness_model = HFLM(
            pretrained=model, tokenizer=tokenizer, backend="causal", device=str(device),
            dtype=torch.bfloat16, batch_size=args.batch_size, max_length=2048,
        )
        results = simple_evaluate(
            model=harness_model, tasks=list(args.tasks), num_fewshot=0,
            batch_size=args.batch_size, device=str(device), limit=args.limit,
            bootstrap_iters=0, log_samples=True, random_seed=58,
            numpy_random_seed=58, torch_random_seed=58, fewshot_random_seed=58,
        )
        payload["task_provenance"] = _task_provenance(list(args.tasks))
        if args.method == "w4a4kv4":
            runtime_stats["activation_quantization"] = activation_stats.summary()
            runtime_stats["kv_cache_quantization"] = kv_stats.summary()
        payload["runtime_quantization"] = runtime_stats
        payload["lm_eval_results"] = results
        payload["status"] = "completed"
        payload["elapsed_seconds"] = time.perf_counter() - started
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(args.output, payload)
    except Exception as error:
        payload["status"] = "failed"
        payload["elapsed_seconds"] = time.perf_counter() - started
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(args.output, payload)
        raise
    finally:
        for handle in activation_handles:
            handle.remove()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
