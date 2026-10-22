"""Create the shared Llama-3.2-1B WikiText-2 train calibration manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.evaluation.calibration_manifest import build_calibration_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/calibration/llama32_1b_wikitext2_train_seed58_n128.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration manifest: {args.output}")
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from datasets import load_dataset
    from huggingface_hub import get_token
    from transformers import AutoTokenizer

    project = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = project["llm_ppl_experiment"]
    dataset = load_dataset(
        experiment["dataset_id"],
        experiment["dataset_config"],
        split="train",
        revision=experiment["dataset_revision"],
        download_mode="reuse_dataset_if_exists",
    )
    texts = [value if isinstance(value, str) else "" for value in dataset["text"]]
    train_text = "\n\n".join(texts)
    tokenizer = AutoTokenizer.from_pretrained(
        experiment["model_id"],
        revision=experiment["model_revision"],
        token=get_token(),
        local_files_only=args.offline,
        use_fast=True,
    )
    old_limit = tokenizer.model_max_length
    tokenizer.model_max_length = max(old_limit, len(train_text))
    token_ids = tokenizer(
        train_text, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    manifest = build_calibration_manifest(
        token_ids,
        model_id=experiment["model_id"],
        model_revision=experiment["model_revision"],
        dataset_id=experiment["dataset_id"],
        dataset_config=experiment["dataset_config"],
        dataset_revision=experiment["dataset_revision"],
        context_length=args.context_length,
        sample_count=args.samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **manifest["selected_windows"]}, indent=2))


if __name__ == "__main__":
    main()

