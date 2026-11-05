"""Evaluate a packed W4A4 runtime on the full WikiText-2 test set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

# Keep dataset/model resolution offline and deterministic. These variables must
# be set before importing Hugging Face packages.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.evaluation.perplexity import tokenize_corpus
from nestquant.models.fused_linear import FusedNQRTLinear, convert_linears_to_fused_nqrt_


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = PROJECT_ROOT / "configs/project.json"
PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
ACTIVATION_BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
EXPECTED_TOKEN_SHA256 = "54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567"


def _sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    parser.add_argument(
        "--compute-backend",
        choices=("packed_w4a4_tiled", "packed_w4a4_cuda_mma"),
        default="packed_w4a4_tiled",
    )
    parser.add_argument(
        "--tiled-fallback-projections", nargs="+", default=(),
        choices=PROJECTIONS,
        help="Projection suffixes forced to tiled Triton for sensitivity ablation.",
    )
    parser.add_argument(
        "--tiled-fallback-names", nargs="+", default=(),
        help="Exact Linear module names forced to tiled Triton.",
    )
    args = parser.parse_args()
    if args.limit_windows is not None and args.limit_windows < 1:
        parser.error("--limit-windows must be positive")
    if args.context_length < 2 or args.logit_chunk_size < 1:
        parser.error("context length must be >= 2 and logit chunk size positive")
    if args.progress is None:
        args.progress = args.output.with_suffix(".progress.json")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    config = json.loads(PROJECT_CONFIG.read_text(encoding="utf-8"))["llm_ppl_experiment"]
    checkpoint = args.checkpoint.resolve()
    model_path = args.model.resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    run_signature = {
        "protocol": f"{args.compute_backend}-wikitext2-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "model_path": str(model_path),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "dataset_id": config["dataset_id"],
        "dataset_config": config["dataset_config"],
        "dataset_revision": config["dataset_revision"],
        "context_length": args.context_length,
        "logit_chunk_size": args.logit_chunk_size,
        "compute_backend": args.compute_backend,
        "activation_q": 14,
        "activation_betas": ACTIVATION_BETAS,
        "tiled_fallback_projections": list(args.tiled_fallback_projections),
        "tiled_fallback_names": list(args.tiled_fallback_names),
    }
    signature_sha256 = hashlib.sha256(
        json.dumps(run_signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    print("Loading cached WikiText-2 test split...", flush=True)
    dataset = load_dataset(
        config["dataset_id"], config["dataset_config"], split="test",
        revision=config["dataset_revision"], download_mode="reuse_dataset_if_exists",
    )
    texts = [text if isinstance(text, str) else "" for text in dataset["text"]]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    corpus = tokenize_corpus(texts, tokenizer, args.context_length)
    if corpus.token_ids_sha256 != EXPECTED_TOKEN_SHA256:
        raise RuntimeError(
            "WikiText-2 token hash mismatch: "
            f"expected {EXPECTED_TOKEN_SHA256}, got {corpus.token_ids_sha256}"
        )
    total_windows = corpus.chunk_count
    target_windows = min(args.limit_windows or total_windows, total_windows)
    print(
        f"Protocol verified: {total_windows} windows, "
        f"{corpus.predicted_token_count} full-corpus predicted tokens; "
        f"running {target_windows} windows.", flush=True,
    )

    completed = 0
    total_nll = 0.0
    predicted_tokens = 0
    elapsed_seconds_prior = 0.0
    if args.resume and args.progress.exists():
        saved = json.loads(args.progress.read_text(encoding="utf-8"))
        if saved.get("signature_sha256") != signature_sha256:
            raise RuntimeError("progress file does not match this evaluation signature")
        if saved.get("token_ids_sha256") != corpus.token_ids_sha256:
            raise RuntimeError("progress file token hash mismatch")
        completed = int(saved["completed_windows"])
        total_nll = float(saved["total_nll"])
        predicted_tokens = int(saved["predicted_tokens"])
        elapsed_seconds_prior = float(saved.get("elapsed_seconds", 0.0))
        if completed > target_windows:
            raise RuntimeError("progress contains more windows than this run requests")
        print(f"Resuming after {completed}/{target_windows} windows.", flush=True)

    print("Loading BF16 base model and installing 112 packed W4A4 projections...", flush=True)
    reader = NQMCFileReader(checkpoint, verify_checksum=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    names = tuple(
        f"model.layers.{layer}.{projection}"
        for layer in range(model.config.num_hidden_layers)
        for projection in PROJECTIONS
    )
    fallback_names = tuple(dict.fromkeys([
        name for name in names
        if any(name.endswith(suffix) for suffix in args.tiled_fallback_projections)
    ] + list(args.tiled_fallback_names)))
    unknown_fallback_names = sorted(set(fallback_names) - set(names))
    if unknown_fallback_names:
        raise ValueError(
            "unknown --tiled-fallback-names: " + ", ".join(unknown_fallback_names)
        )
    conversion = convert_linears_to_fused_nqrt_(
        model, reader, names, output_dtype=torch.bfloat16,
        compute_backend=args.compute_backend, activation_q=14,
        activation_betas=ACTIVATION_BETAS,
        w4a4_tiled_fallback_names=fallback_names,
    )
    if len(conversion.replaced_linear_names) != len(names):
        raise RuntimeError("not all requested projections were converted")
    fused_modules = [module for module in model.modules() if isinstance(module, FusedNQRTLinear)]
    if len(fused_modules) != len(names):
        raise RuntimeError("unexpected packed W4A4 module count")

    token_tensor = torch.tensor(
        corpus.token_ids[:total_windows * args.context_length], dtype=torch.long
    ).reshape(total_windows, args.context_length)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for window_index in range(completed, target_windows):
            window_started = time.perf_counter()
            input_ids = token_tensor[window_index:window_index + 1].to("cuda")
            output = model(input_ids=input_ids, use_cache=False)
            logits = output.logits
            window_nll = torch.zeros((), dtype=torch.float64, device="cuda")
            for start in range(0, args.context_length - 1, args.logit_chunk_size):
                end = min(start + args.logit_chunk_size, args.context_length - 1)
                logits_slice = logits[:, start:end, :].float().reshape(-1, logits.shape[-1])
                labels = input_ids[:, start + 1:end + 1].reshape(-1)
                window_nll += F.cross_entropy(logits_slice, labels, reduction="sum").double()
            total_nll += float(window_nll.item())
            predicted_tokens += args.context_length - 1
            completed = window_index + 1
            elapsed = elapsed_seconds_prior + time.perf_counter() - started
            progress = {
                "signature_sha256": signature_sha256,
                "token_ids_sha256": corpus.token_ids_sha256,
                "completed_windows": completed,
                "target_windows": target_windows,
                "total_nll": total_nll,
                "predicted_tokens": predicted_tokens,
                "elapsed_seconds": elapsed,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(args.progress, progress)
            mean = total_nll / predicted_tokens
            print(
                f"[{completed:03d}/{target_windows:03d}] "
                f"window={time.perf_counter() - window_started:.2f}s "
                f"running_ppl={math.exp(mean):.6f}", flush=True,
            )
            del output, logits, input_ids, window_nll

    elapsed_seconds = elapsed_seconds_prior + time.perf_counter() - started
    mean_nll = total_nll / predicted_tokens
    payload = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_signature": run_signature,
        "signature_sha256": signature_sha256,
        "data": {
            "text_sha256": corpus.text_sha256,
            "token_ids_sha256": corpus.token_ids_sha256,
            "token_count_with_boundary_tokens": corpus.token_count,
            "full_corpus_chunk_count": corpus.chunk_count,
            "evaluated_chunk_count": completed,
            "predicted_token_count": predicted_tokens,
            "full_corpus_predicted_token_count": corpus.predicted_token_count,
            "dropped_tail_tokens": corpus.dropped_tail_tokens,
            "dataset_rows": len(texts),
            "bos_token_id": corpus.bos_token_id,
            "eos_token_id": corpus.eos_token_id,
        },
        "conversion": {
            "requested_projection_count": len(names),
            "replaced_projection_count": len(conversion.replaced_linear_names),
            "live_fused_module_count": len(fused_modules),
            "tiled_fallback_projection_count": len(
                conversion.w4a4_tiled_fallback_names
            ),
            "tiled_fallback_names": list(conversion.w4a4_tiled_fallback_names),
        },
        "result": {
            "total_nll": total_nll,
            "mean_nll_per_predicted_token": mean_nll,
            "perplexity": math.exp(mean_nll),
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "scope": (
            "Formal full-corpus packed W4A4 quality evaluation. use_cache=False, so this "
            "run does not exercise the packed KV4 cache path."
        ),
    }
    _write_json(args.output, payload)
    print(json.dumps(payload["result"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
