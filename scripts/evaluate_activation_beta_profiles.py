"""Compare A4 beta profiles on disjoint held-out WikiText-2 train windows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.evaluation.activation_beta_config import load_activation_beta_config
from nestquant.evaluation.activation_quantization import (
    ActivationQuantizationStats,
    enable_activation_quantization,
)
from nestquant.evaluation.calibration_manifest import (
    load_calibration_manifest,
    validate_calibration_manifest,
)
from nestquant.evaluation.perplexity import evaluate_perplexity
from nestquant.experiments.activation_beta_validation import select_holdout_windows
from nestquant.experiments.llm_perplexity import (
    UPSTREAM_ACTIVATION_BETA_COEFFICIENTS,
    _load_nqmc_into_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/project.json")
    parser.add_argument(
        "--calibration-manifest", type=Path,
        default=PROJECT_ROOT / "configs/calibration/llama32_1b_wikitext2_train_seed58_n128.json",
    )
    parser.add_argument(
        "--beta-config", type=Path,
        default=PROJECT_ROOT / "results/raw/calibration/a4_betas_train8.json",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq.nqmc",
    )
    parser.add_argument("--calibration-window-count", type=int, default=8)
    parser.add_argument("--holdout-start", type=int, default=8)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--beta-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rows-per-chunk", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results/raw/calibration/a4_betas_train8_holdout8.json",
    )
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.q < 2 or args.beta_count < 1 or args.batch_size < 1 or args.rows_per_chunk < 1:
        raise ValueError("invalid q, beta-count, batch-size, or rows-per-chunk")

    project = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = project["llm_ppl_experiment"]
    model_id, model_revision = experiment["model_id"], experiment["model_revision"]
    dataset_id, dataset_config = experiment["dataset_id"], experiment["dataset_config"]
    dataset_revision = experiment["dataset_revision"]
    context_length = project["reproduction_target"]["context_length"]
    seed = project["initial_algorithm_config"]["seed"]
    weight_coefficients = tuple(experiment["beta_coefficients_project_scale"])
    device = torch.device(args.device)
    payload = {
        "schema_version": 1,
        "kind": "nestquant_activation_beta_heldout_validation",
        "status": "preparing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "profile_selection_on_disjoint_train_windows_not_test_set",
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "split": "train",
        "context_length": context_length,
        "q": args.q,
        "selected_beta_count": args.beta_count,
        "checkpoint": file_record(args.checkpoint),
        "beta_config": file_record(args.beta_config),
        "calibration_manifest": file_record(args.calibration_manifest),
        "source": {"git_commit": git_commit(), "working_tree_dirty": git_dirty()},
        "profiles": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    handles = []
    try:
        from datasets import load_dataset
        from huggingface_hub import get_token
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = get_token()
        dataset = load_dataset(
            dataset_id, dataset_config, split="train", revision=dataset_revision,
            download_mode="reuse_dataset_if_exists",
        )
        texts = [text if isinstance(text, str) else "" for text in dataset["text"]]
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=model_revision, token=token,
            local_files_only=args.offline, use_fast=True,
        )
        joined = "\n\n".join(texts)
        if isinstance(getattr(tokenizer, "model_max_length", None), int):
            tokenizer.model_max_length = max(tokenizer.model_max_length, len(joined))
        corpus_ids = tuple(int(value) for value in tokenizer(
            joined, add_special_tokens=False, return_attention_mask=False
        )["input_ids"])
        manifest = load_calibration_manifest(args.calibration_manifest)
        all_windows = validate_calibration_manifest(
            manifest, corpus_ids, model_id=model_id, model_revision=model_revision,
            dataset_id=dataset_id, dataset_config=dataset_config,
            dataset_revision=dataset_revision,
        )
        windows, holdout = select_holdout_windows(
            all_windows,
            calibration_window_count=args.calibration_window_count,
            holdout_start=args.holdout_start,
            holdout_count=args.holdout_count,
        )
        token_ids = tuple(value for window in windows for value in window)
        payload["holdout"] = holdout
        del tokenizer, dataset, texts, corpus_ids, all_windows, windows

        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, token=token, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=args.offline,
        ).to(device).eval()
        payload["checkpoint_load"] = _load_nqmc_into_model(
            args.checkpoint, model, device, model_id, model_revision,
            args.q, seed, weight_coefficients, args.beta_count,
        )
        default_betas = tuple(value / args.q for value in UPSTREAM_ACTIVATION_BETA_COEFFICIENTS)
        specs = [("upstream_default", None, {
            "profile": "upstream_default_global",
            "coefficient_values": list(UPSTREAM_ACTIVATION_BETA_COEFFICIENTS),
            "beta_values": list(default_betas),
        })]
        for profile in ("global", "projection_type", "module"):
            resolved, metadata = load_activation_beta_config(
                args.beta_config, model_id=model_id, model_revision=model_revision,
                q=args.q, selected_beta_count=args.beta_count, profile=profile,
            )
            specs.append((profile, resolved, metadata))

        payload["status"] = "evaluating"
        write_json(args.output, payload)
        for name, module_betas, metadata in specs:
            started = time.perf_counter()
            stats = ActivationQuantizationStats()
            stats, handles = enable_activation_quantization(
                model, q=args.q, betas=default_betas, module_betas=module_betas,
                rows_per_chunk=args.rows_per_chunk, stats=stats,
            )

            def progress(done: int, total: int, *, profile_name=name) -> None:
                print(f"{profile_name}: window {done}/{total}", flush=True)

            try:
                evaluation = evaluate_perplexity(
                    model, token_ids, context_length=context_length, device=device,
                    batch_size=args.batch_size, progress_callback=progress,
                )
            finally:
                for handle in handles:
                    handle.remove()
                handles = []
            payload["profiles"][name] = {
                "beta_profile": metadata,
                "evaluation": evaluation,
                "activation_quantization": stats.summary(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            write_json(args.output, payload)
            print(f"Completed {name}: PPL={evaluation['perplexity']:.6f}", flush=True)

        payload["status"] = "completed"
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, payload)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as error:
        for handle in handles:
            handle.remove()
        payload["status"] = "failed"
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, payload)
        raise
    print(f"Saved held-out activation-beta validation: {args.output}")


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
