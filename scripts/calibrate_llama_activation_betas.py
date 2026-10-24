"""Calibrate global, projection-type, and per-module A4 beta subsets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.evaluation.activation_beta_config import KIND, SCHEMA_VERSION
from nestquant.evaluation.activation_quantization import (
    ActivationBlockSampler,
    attach_activation_calibration,
)
from nestquant.evaluation.calibration_manifest import (
    load_calibration_manifest,
    validate_calibration_manifest,
)
from nestquant.experiments.llm_perplexity import _load_nqmc_into_model


DEFAULT_CANDIDATE_COEFFICIENTS = (
    1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0,
    7.0, 8.0, 10.0, 14.0, 18.0, 24.0, 32.0, 40.0, 50.0, 100.0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/project.json")
    parser.add_argument(
        "--calibration-manifest", type=Path,
        default=PROJECT_ROOT / "configs/calibration/llama32_1b_wikitext2_train_seed58_n128.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-chunks", type=int, default=8)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--beta-count", type=int, default=4)
    parser.add_argument("--candidate-coefficients", type=float, nargs="+",
                        default=DEFAULT_CANDIDATE_COEFFICIENTS)
    parser.add_argument("--sample-limit-per-module", type=int, default=4096)
    parser.add_argument("--search-rows-per-chunk", type=int, default=8192)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results/raw/calibration/a4_betas_train8.json",
    )
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.calibration_chunks < 1 or args.sample_limit_per_module < 1:
        raise ValueError("calibration chunks and sample limit must be positive")
    if args.q < 2 or args.beta_count < 1 or args.beta_count > len(args.candidate_coefficients):
        raise ValueError("invalid q or beta count")
    if any(left >= right for left, right in zip(
            args.candidate_coefficients, args.candidate_coefficients[1:])):
        raise ValueError("candidate coefficients must be strictly increasing")

    project = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = project["llm_ppl_experiment"]
    model_id = experiment["model_id"]
    model_revision = experiment["model_revision"]
    dataset_id = experiment["dataset_id"]
    dataset_config = experiment["dataset_config"]
    dataset_revision = experiment["dataset_revision"]
    context_length = project["reproduction_target"]["context_length"]
    seed = project["initial_algorithm_config"]["seed"]
    weight_coefficients = tuple(experiment["beta_coefficients_project_scale"])
    candidate_betas = tuple(value / args.q for value in args.candidate_coefficients)
    device = torch.device(args.device)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "preparing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "split": "train",
        "context_length": context_length,
        "calibration_chunks": args.calibration_chunks,
        "q": args.q,
        "selected_beta_count": args.beta_count,
        "candidate_coefficients_project_scale": list(args.candidate_coefficients),
        "candidate_beta_values": list(candidate_betas),
        "sample_limit_per_module": args.sample_limit_per_module,
        "sampling": "evenly_spaced_bounded_normalized_e8_blocks_per_window",
        "quantization_coordinates": "local_hadamard_input_when_checkpoint_uses_local_hadamard",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "profiles": {},
        "source": {"git_commit": git_commit(), "working_tree_dirty": git_dirty()},
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
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and len(joined) > tokenizer_limit:
            tokenizer.model_max_length = len(joined)
        corpus_token_ids = tuple(int(value) for value in tokenizer(
            joined, add_special_tokens=False, return_attention_mask=False
        )["input_ids"])
        manifest = load_calibration_manifest(args.calibration_manifest)
        windows = validate_calibration_manifest(
            manifest, corpus_token_ids,
            model_id=model_id, model_revision=model_revision,
            dataset_id=dataset_id, dataset_config=dataset_config,
            dataset_revision=dataset_revision,
        )
        if args.calibration_chunks > len(windows):
            raise ValueError("calibration manifest has too few windows")
        windows = windows[:args.calibration_chunks]
        payload["calibration_manifest"] = str(args.calibration_manifest.resolve())
        payload["calibration_manifest_sha256"] = hashlib.sha256(
            args.calibration_manifest.read_bytes()
        ).hexdigest()
        payload["calibration_token_ids_sha256"] = token_ids_sha256(windows)
        payload["calibration_token_count"] = sum(len(window) for window in windows)
        del tokenizer, dataset, texts

        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, token=token, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=args.offline,
        ).to(device).eval()
        payload["checkpoint_load"] = _load_nqmc_into_model(
            args.checkpoint, model, device, model_id, model_revision,
            args.q, seed, weight_coefficients, args.beta_count,
        )
        sampler = ActivationBlockSampler(
            calibration_chunks=args.calibration_chunks,
            sample_limit_per_module=args.sample_limit_per_module,
        )
        handles = attach_activation_calibration(model, sampler)
        payload["status"] = "collecting"
        write_json(args.output, payload)
        with torch.inference_mode():
            for index, window in enumerate(windows, 1):
                input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
                output = model(input_ids=input_ids, use_cache=False)
                del output, input_ids
                payload["collected_chunks"] = index
                write_json(args.output, payload)
                print(f"Collected A4 calibration window {index}/{len(windows)}", flush=True)
        for handle in handles:
            handle.remove()
        handles = []
        payload["status"] = "searching"
        write_json(args.output, payload)

        def report(record: dict) -> None:
            print(
                f"Selected {record['completed']}/{record['total']} "
                f"{record['scope']}={record['key']}: {record['beta_values']}",
                flush=True,
            )

        payload["profiles"] = sampler.search_profiles(
            q=args.q,
            candidate_betas=candidate_betas,
            selected_beta_count=args.beta_count,
            device=device,
            rows_per_chunk=args.search_rows_per_chunk,
            progress_callback=report,
        )
        payload["status"] = "completed"
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, payload)
        del model, sampler
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
    print(f"Saved calibrated activation beta profiles: {args.output}")


def token_ids_sha256(windows) -> str:
    from nestquant.evaluation.calibration_manifest import token_ids_sha256 as stable_hash
    return stable_hash([value for window in windows for value in window])


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


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
