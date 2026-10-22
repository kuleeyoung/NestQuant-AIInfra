"""Calibrate per-layer NestQuant beta subsets for post-RoPE Llama K/V."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.evaluation.kv_cache import KVBlockSampler, KVCalibrationCache
from nestquant.evaluation.perplexity import tokenize_corpus


DEFAULT_CANDIDATE_COEFFICIENTS = (
    1.0, 2.0, 3.5, 4.5, 6.0, 8.0, 10.0,
    14.5, 20.0, 25.0, 32.0, 40.0, 50.0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/project.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-chunks", type=int, default=6)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--beta-count", type=int, default=4)
    parser.add_argument("--candidate-coefficients", type=float, nargs="+",
                        default=DEFAULT_CANDIDATE_COEFFICIENTS)
    parser.add_argument("--sample-limit-per-tensor", type=int, default=65536)
    parser.add_argument("--search-rows-per-chunk", type=int, default=8192)
    parser.add_argument("--overload-margin-coefficient", type=float, default=4.0,
                        help="paper Appendix G KV safety margin, divided by q")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results/raw/calibration/kv_betas_train6.json")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.calibration_chunks < 1 or args.context_length < 2:
        raise ValueError("invalid calibration size")
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
    candidate_betas = tuple(value / args.q for value in args.candidate_coefficients)
    device = torch.device(args.device)

    payload = {
        "schema_version": 1,
        "status": "preparing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset": dataset_id,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "split": "train",
        "context_length": args.context_length,
        "calibration_chunks": args.calibration_chunks,
        "q": args.q,
        "selected_beta_count": args.beta_count,
        "candidate_coefficients_project_scale": list(args.candidate_coefficients),
        "candidate_beta_values": list(candidate_betas),
        "sample_limit_per_tensor": args.sample_limit_per_tensor,
        "search_rows_per_chunk": args.search_rows_per_chunk,
        "sampling": "evenly_spaced_bounded_samples_per_context_window",
        "overload_margin_coefficient": args.overload_margin_coefficient,
        "overload_margin": args.overload_margin_coefficient / args.q,
        "quantization_location": "post_rope_before_kv_cache_insertion",
        "vector_layout": "one_row_per_attention_head",
        "key_rotation": "normalized_hadamard_after_rope",
        "value_rotation": "none",
        "layers": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)

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
        corpus = tokenize_corpus(texts, tokenizer, args.context_length)
        selected_tokens = corpus.token_ids[:args.calibration_chunks * args.context_length]
        if len(selected_tokens) < args.calibration_chunks * args.context_length:
            raise ValueError("training corpus has too few full calibration chunks")
        payload["train_corpus_token_ids_sha256"] = corpus.token_ids_sha256
        payload["token_ids_sha256"] = token_hash(selected_tokens)
        payload["calibration_token_count"] = len(selected_tokens)
        write_json(args.output, payload)
        del tokenizer, dataset, texts

        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, token=token, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=args.offline,
        ).to(device).eval()
        sampler = KVBlockSampler(
            calibration_chunks=args.calibration_chunks,
            sample_limit_per_tensor=args.sample_limit_per_tensor,
        )
        tokens = torch.tensor(selected_tokens, dtype=torch.long).reshape(
            args.calibration_chunks, args.context_length
        )
        payload["status"] = "collecting"
        with torch.inference_mode():
            for index, row in enumerate(tokens, 1):
                cache = KVCalibrationCache(config=model.config, sampler=sampler)
                output = model.model(
                    input_ids=row.unsqueeze(0).to(device),
                    use_cache=True,
                    past_key_values=cache,
                )
                del output, cache
                payload["collected_chunks"] = index
                write_json(args.output, payload)
                print(f"Collected post-RoPE K/V for chunk {index}/{args.calibration_chunks}", flush=True)

        payload["status"] = "searching"
        write_json(args.output, payload)

        def report(record: dict) -> None:
            print(
                f"Selected {record['completed']}/{record['total']} "
                f"layer={record['layer_idx']} {record['kind']}: {record['beta_values']}",
                flush=True,
            )

        payload["layers"] = sampler.search(
            q=args.q,
            candidate_betas=candidate_betas,
            selected_beta_count=args.beta_count,
            device=device,
            rows_per_chunk=args.search_rows_per_chunk,
            overload_margin=args.overload_margin_coefficient / args.q,
            progress_callback=report,
        )
        payload["status"] = "completed"
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, payload)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, payload)
        raise
    print(f"Saved calibrated K/V beta subsets: {args.output}")


def token_hash(token_ids) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<q", int(token_id)))
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
