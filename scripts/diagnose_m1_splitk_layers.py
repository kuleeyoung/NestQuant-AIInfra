"""Sweep one split-K MLP down-projection at a time in the NQKV Graph runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from scripts.benchmark_full_nqrt_model import ACTIVATION_BETAS, _load_model, _projection_names
from scripts.diagnose_m1_splitk_logits import (
    compare_traces,
    nqkv_graph_trace,
    select_m1_down_splitk_modules,
)


def layer_index(name: str) -> int:
    parts = name.split(".")
    try:
        return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"cannot parse layer index from {name}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--prompt",
        default=(
            "The capital of France is Paris. The capital of Germany is Berlin. "
            "The capital of Italy is Rome. The capital of Spain is Madrid."
        ),
    )
    parser.add_argument(
        "--prompt-suite-name",
        help="Use an exact prompt from diagnose_m1_splitk_prompt_suite.py.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/m1_splitk_layer_sweep_v1.json"),
    )
    args = parser.parse_args()
    if args.prompt_suite_name is not None:
        from scripts.diagnose_m1_splitk_prompt_suite import PROMPT_SUITE

        suite = dict(PROMPT_SUITE)
        if args.prompt_suite_name not in suite:
            parser.error(
                "unknown --prompt-suite-name; choose from " + ", ".join(suite)
            )
        args.prompt = suite[args.prompt_suite_name]
    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least 2 to observe the known divergence")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model_id = reader.model_metadata["model_id"]
    revision = reader.model_metadata["model_revision"]
    model = _load_model(model_id, revision, args.offline)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=args.offline, use_fast=True
    )
    convert_linears_to_fused_nqrt_(
        model,
        reader,
        _projection_names(model, reader),
        output_dtype=torch.bfloat16,
        group_qkv=False,
        compute_backend="packed_w4a4_cuda_mma",
        activation_q=14,
        activation_betas=ACTIVATION_BETAS,
        enable_m1_down_splitk=False,
    )
    prompt_ids = tuple(tokenizer(args.prompt, return_tensors="pt")["input_ids"][0].tolist())
    eligible = select_m1_down_splitk_modules(model, set())
    baseline = nqkv_graph_trace(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        capacity=args.capacity,
        max_cache_len=args.max_cache_len,
    )

    trials = []
    for name in eligible:
        select_m1_down_splitk_modules(model, {name})
        candidate = nqkv_graph_trace(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            capacity=args.capacity,
            max_cache_len=args.max_cache_len,
        )
        comparison = compare_traces(baseline, candidate)
        decision = comparison["steps"][1]
        divergence_step = comparison["first_token_divergence_step"]
        divergence = (
            None if divergence_step is None else comparison["steps"][divergence_step]
        )
        trial = {
            "module": name,
            "layer": layer_index(name),
            "candidate_token_ids": [item["selected_token_id"] for item in candidate],
            "first_token_divergence_step": comparison["first_token_divergence_step"],
            "step1_argmax_equal": decision["argmax_equal"],
            "step1_candidate_token_id": decision["candidate_token_id"],
            "step1_candidate_margin": decision["candidate_top1_top2_margin"],
            "step1_max_abs_logit_error": decision["max_abs_logit_error"],
            "step1_mean_abs_logit_error": decision["mean_abs_logit_error"],
            "step1_relative_l2_logit_error": decision["relative_l2_logit_error"],
            "first_divergence": divergence,
        }
        trials.append(trial)
        print(json.dumps(trial), flush=True)

    select_m1_down_splitk_modules(model, set(eligible))
    all_layers = nqkv_graph_trace(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        capacity=args.capacity,
        max_cache_len=args.max_cache_len,
    )
    all_comparison = compare_traces(baseline, all_layers)
    payload = {
        "schema_version": 1,
        "protocol": "single-layer M1 split-K NQKV Graph sensitivity sweep",
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "capacity": args.capacity,
        "max_cache_len": args.max_cache_len,
        "max_new_tokens": args.max_new_tokens,
        "baseline_token_ids": [item["selected_token_id"] for item in baseline],
        "eligible_modules": eligible,
        "single_layer_trials": trials,
        "single_layer_divergent_modules": [
            item["module"] for item in trials
            if item["first_token_divergence_step"] is not None
        ],
        "all_layers_token_ids": [item["selected_token_id"] for item in all_layers],
        "all_layers_first_divergence_step": all_comparison["first_token_divergence_step"],
        "claim_boundary": (
            "One deterministic prompt and one module enabled per candidate run. "
            "A non-divergent single layer does not prove arbitrary layer combinations safe."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_token_ids": payload["baseline_token_ids"],
        "single_layer_divergent_modules": payload["single_layer_divergent_modules"],
        "all_layers_token_ids": payload["all_layers_token_ids"],
    }, indent=2))


if __name__ == "__main__":
    main()
