"""Collect audited WikiText-2 train input covariances for selected Llama Linear modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.evaluation.calibration_manifest import (
    load_calibration_manifest, token_ids_sha256, validate_calibration_manifest,
)
from nestquant.evaluation.perplexity import tokenize_corpus
from nestquant.models.calibration import LinearInputCovarianceCollector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", nargs="+", required=True, help="exact nn.Linear module names")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--accumulator-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--calibration-manifest", type=Path,
                        help="shared versioned calibration-window manifest")
    parser.add_argument("--output", type=Path, required=True, help="torch file; JSON manifest is written beside it")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    args = parser.parse_args()
    if args.chunks < 1 or args.context_length < 2:
        raise ValueError("chunks must be positive and context length must be at least 2")
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from datasets import load_dataset
    from huggingface_hub import get_token
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["llm_ppl_experiment"]
    model_id, model_revision = experiment["model_id"], experiment["model_revision"]
    dataset_revision = experiment["dataset_revision"]
    device = torch.device(args.device)
    accumulator_device = torch.device(args.accumulator_device)
    token = get_token()
    dataset = load_dataset(
        experiment["dataset_id"], experiment["dataset_config"], split="train",
        revision=dataset_revision, download_mode="reuse_dataset_if_exists",
    )
    texts = [value if isinstance(value, str) else "" for value in dataset["text"]]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=model_revision, token=token, local_files_only=args.offline, use_fast=True
    )
    shared_manifest = None
    shared_manifest_sha256 = None
    if args.calibration_manifest is not None:
        shared_manifest = load_calibration_manifest(args.calibration_manifest)
        shared_manifest_sha256 = hashlib.sha256(
            args.calibration_manifest.read_bytes()
        ).hexdigest()
        train_text = "\n\n".join(texts)
        old_limit = tokenizer.model_max_length
        tokenizer.model_max_length = max(old_limit, len(train_text))
        train_ids = tokenizer(
            train_text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        windows = validate_calibration_manifest(
            shared_manifest,
            train_ids,
            model_id=model_id,
            model_revision=model_revision,
            dataset_id=experiment["dataset_id"],
            dataset_config=experiment["dataset_config"],
            dataset_revision=dataset_revision,
        )
        selection = shared_manifest["selection"]
        if selection["context_length"] != args.context_length:
            raise ValueError("calibration manifest context length disagrees with CLI")
        if args.chunks > len(windows):
            raise ValueError("--chunks exceeds calibration manifest sample count")
        selected = torch.tensor(windows[:args.chunks], dtype=torch.long)
        selected_token_hash = token_ids_sha256(selected.reshape(-1).tolist())
        train_corpus_hash = shared_manifest["train_corpus"]["token_ids_sha256"]
        available_chunk_count = shared_manifest["selection"]["sample_count"]
    else:
        corpus = tokenize_corpus(texts, tokenizer, args.context_length)
        if args.chunks > corpus.chunk_count:
            raise ValueError(f"requested {args.chunks} chunks but train split has {corpus.chunk_count}")
        selected = torch.tensor(
            corpus.token_ids[: args.chunks * args.context_length], dtype=torch.long
        ).reshape(args.chunks, args.context_length)
        selected_token_hash = token_ids_sha256(selected.reshape(-1).tolist())
        train_corpus_hash = corpus.token_ids_sha256
        available_chunk_count = corpus.chunk_count
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=args.offline,
    ).to(device).eval()

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with LinearInputCovarianceCollector(
        model, args.modules, accumulator_device=accumulator_device, accumulator_dtype=torch.float32
    ) as collector, torch.inference_mode():
        for index, row in enumerate(selected, 1):
            model(input_ids=row.unsqueeze(0).to(device), use_cache=False)
            print(f"calibration chunk {index}/{args.chunks}", flush=True)
        metadata = collector.metadata()
        covariances = collector.covariances(device="cpu")
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(covariances, args.output)
    blob_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_completed" if args.chunks < available_chunk_count else "completed",
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset": "WikiText-2 train",
        "dataset_revision": dataset_revision,
        "context_length": args.context_length,
        "chunk_count": args.chunks,
        "activation_rows_per_module": args.chunks * args.context_length,
        "train_token_ids_sha256": selected_token_hash,
        "calibration_token_ids_sha256": selected_token_hash,
        "train_corpus_token_ids_sha256": train_corpus_hash,
        "calibration_manifest": None if shared_manifest is None else {
            "path": str(args.calibration_manifest.resolve()),
            "sha256": shared_manifest_sha256,
            "schema_version": shared_manifest["schema_version"],
            "selection_method": shared_manifest["selection"]["method"],
            "available_samples": shared_manifest["selection"]["sample_count"],
            "used_samples": args.chunks,
            "start_indices": shared_manifest["selection"]["start_indices"][:args.chunks],
            "full_selected_token_ids_sha256": shared_manifest[
                "selected_windows"
            ]["token_ids_sha256"],
        },
        "modules": metadata,
        "accumulation": "sum(x.T @ x) / activation_rows",
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": blob_hash,
        "limitations": [
            "Calibration smoke unless all train chunks are used.",
            "Dense FP32 covariance is an offline quantization artifact, not part of the packed model.",
            "This script only collects explicitly named modules and does not quantize weights.",
        ],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
