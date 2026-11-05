"""Check local prerequisites without loading the full model or running inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.evaluation.perplexity import tokenize_corpus


EXPECTED_TOKEN_SHA256 = "54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--verify-checksum", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    reader = NQMCFileReader(checkpoint, verify_checksum=args.verify_checksum)
    model_id = reader.model_metadata["model_id"]
    model_revision = reader.model_metadata["model_revision"]
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))[
        "llm_ppl_experiment"
    ]
    dataset = load_dataset(
        config["dataset_id"], config["dataset_config"], split="test",
        revision=config["dataset_revision"], download_mode="reuse_dataset_if_exists",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=model_revision, local_files_only=True, use_fast=True
    )
    corpus = tokenize_corpus(
        [text if isinstance(text, str) else "" for text in dataset["text"]],
        tokenizer,
        2048,
    )
    if corpus.token_ids_sha256 != EXPECTED_TOKEN_SHA256:
        raise RuntimeError(
            "WikiText-2 token hash mismatch: "
            f"expected {EXPECTED_TOKEN_SHA256}, got {corpus.token_ids_sha256}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    paths = [path.resolve() for path in (args.output, args.progress) if path]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(checkpoint.parent)
    report = {
        "status": "passed",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_tensor_count": reader.tensor_count,
        "checkpoint_checksum_verified": args.verify_checksum,
        "dataset": f"{config['dataset_id']}/{config['dataset_config']}:test",
        "dataset_revision": config["dataset_revision"],
        "dataset_rows": len(dataset),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_vocab_size": len(tokenizer),
        "token_count": len(corpus.token_ids),
        "token_ids_sha256": corpus.token_ids_sha256,
        "free_disk_gib": disk.free / 2**30,
        "output_paths": [
            {"path": str(path), "exists": path.exists()} for path in paths
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
