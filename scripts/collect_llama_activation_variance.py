"""Collect the 20-sequence per-projection activation variances used by QA-LDLQ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.evaluation.perplexity import tokenize_corpus


def _hash_tokens(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/project.json")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/raw/calibration/activation_variance_train20.json",
    )
    args = parser.parse_args()
    if args.chunks < 1 or args.context_length < 2:
        raise ValueError("chunks and context length must be positive")
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from datasets import load_dataset
    from huggingface_hub import get_token
    from transformers import AutoModelForCausalLM, AutoTokenizer

    project = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = project["llm_ppl_experiment"]
    token = get_token()
    dataset = load_dataset(
        experiment["dataset_id"], experiment["dataset_config"], split="train",
        revision=experiment["dataset_revision"], download_mode="reuse_dataset_if_exists",
    )
    texts = [value if isinstance(value, str) else "" for value in dataset["text"]]
    tokenizer = AutoTokenizer.from_pretrained(
        experiment["model_id"], revision=experiment["model_revision"], token=token,
        local_files_only=args.offline, use_fast=True,
    )
    corpus = tokenize_corpus(texts, tokenizer, args.context_length)
    selected_ids = corpus.token_ids[:args.chunks * args.context_length]
    if len(selected_ids) != args.chunks * args.context_length:
        raise ValueError("train corpus does not contain enough full chunks")
    rows = torch.tensor(selected_ids, dtype=torch.long).reshape(args.chunks, args.context_length)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        experiment["model_id"], revision=experiment["model_revision"], token=token,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=args.offline,
    ).to(device).eval()

    modules = dict(model.named_modules())
    canonical_names = []
    aliases = {}
    for layer in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        groups = {
            f"{prefix}.self_attn.q_proj": (
                f"{prefix}.self_attn.q_proj", f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
            ),
            f"{prefix}.self_attn.o_proj": (f"{prefix}.self_attn.o_proj",),
            f"{prefix}.mlp.gate_proj": (
                f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj",
            ),
            f"{prefix}.mlp.down_proj": (f"{prefix}.mlp.down_proj",),
        }
        for source, targets in groups.items():
            canonical_names.append(source)
            for target in targets:
                aliases[target] = source

    accumulators = {
        name: {"sum2": 0.0, "items": 0}
        for name in canonical_names
    }
    handles = []
    for name in canonical_names:
        if name not in modules or not isinstance(modules[name], torch.nn.Linear):
            raise ValueError(f"missing expected Llama Linear module: {name}")

        def hook(_module, inputs, *, module_name=name):
            values = inputs[0].detach().float()
            accumulators[module_name]["sum2"] += float(values.square().sum().item())
            accumulators[module_name]["items"] += values.numel()

        handles.append(modules[name].register_forward_pre_hook(hook))

    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            model(input_ids=row.unsqueeze(0).to(device), use_cache=False)
            print(f"activation variance chunk {index}/{args.chunks}", flush=True)
    for handle in handles:
        handle.remove()

    records = {}
    for target, source in aliases.items():
        accumulator = accumulators[source]
        if accumulator["items"] == 0:
            raise RuntimeError(f"no activation elements collected for {source}")
        records[target] = {
            "source_module": source,
            "x_var": accumulator["sum2"] / accumulator["items"],
            "element_count": accumulator["items"],
        }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": experiment["model_id"],
        "model_revision": experiment["model_revision"],
        "dataset": experiment["dataset_id"],
        "dataset_config": experiment["dataset_config"],
        "dataset_revision": experiment["dataset_revision"],
        "split": "train",
        "context_length": args.context_length,
        "calibration_chunks": args.chunks,
        "calibration_token_count": len(selected_ids),
        "calibration_token_ids_sha256": _hash_tokens(selected_ids),
        "train_corpus_token_ids_sha256": corpus.token_ids_sha256,
        "statistic": "sum(x**2) / activation_element_count",
        "upstream_protocol": "parallel/err.py default samples=20",
        "modules": records,
        "excluded": {
            "lm_head": "public weight quantizer only applies QA-LDLQ to transformer layer weights"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"Saved activation variance artifact: {args.output}")


if __name__ == "__main__":
    main()
