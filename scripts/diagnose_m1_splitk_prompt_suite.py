"""Evaluate all-layer and layer-10-fallback split-K across diverse prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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


PROMPT_SUITE = (
    ("short_greeting", "Hello"),
    ("short_factual", "The largest planet in the solar system is"),
    ("capital_list", (
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome. The capital of Spain is Madrid."
    )),
    ("arithmetic", "If a train travels 60 miles in 2 hours, its average speed is"),
    ("python_code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"),
    ("reasoning", (
        "All roses are flowers. Some flowers fade quickly. Therefore, it is "
        "not necessarily true that all roses"
    )),
    ("long_context", (
        "A small research team measured latency, throughput, memory usage, and "
        "model quality under identical inputs. They repeated every experiment, "
        "stored the raw measurements, and compared the optimized implementation "
        "with the baseline. Their most important conclusion was"
    )),
    ("chinese", "人工智能基础设施中，模型量化的主要目标是"),
)


def policy_summary(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    comparison = compare_traces(baseline, candidate)
    same_input_steps = [item for item in comparison["steps"] if item["same_input"]]
    return {
        "token_ids": [item["selected_token_id"] for item in candidate],
        "tokens_equal": comparison["all_tokens_equal"],
        "first_divergence_step": comparison["first_token_divergence_step"],
        "max_abs_logit_error_on_same_input": max(
            (item["max_abs_logit_error"] for item in same_input_steps), default=0.0
        ),
        "max_relative_l2_error_on_same_input": max(
            (item["relative_l2_logit_error"] for item in same_input_steps), default=0.0
        ),
        "steps": comparison["steps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--fallback-module",
        action="append",
        default=None,
        help="Module kept unsplit by the hybrid policy; may be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/m1_splitk_prompt_suite_v1.json"),
    )
    args = parser.parse_args()
    fallback_modules = set(args.fallback_module or ["model.layers.10.mlp.down_proj"])
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

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
    eligible = select_m1_down_splitk_modules(model, set())
    unknown_fallbacks = fallback_modules - set(eligible)
    if unknown_fallbacks:
        parser.error(f"fallback modules are not eligible: {sorted(unknown_fallbacks)}")
    hybrid_modules = set(eligible) - fallback_modules

    prompt_results = []
    for prompt_name, prompt in PROMPT_SUITE:
        prompt_ids = tuple(tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist())
        select_m1_down_splitk_modules(model, set())
        baseline = nqkv_graph_trace(
            model, prompt_ids, max_new_tokens=args.max_new_tokens,
            capacity=args.capacity, max_cache_len=args.max_cache_len,
        )
        select_m1_down_splitk_modules(model, set(eligible))
        all_layers = nqkv_graph_trace(
            model, prompt_ids, max_new_tokens=args.max_new_tokens,
            capacity=args.capacity, max_cache_len=args.max_cache_len,
        )
        select_m1_down_splitk_modules(model, hybrid_modules)
        hybrid = nqkv_graph_trace(
            model, prompt_ids, max_new_tokens=args.max_new_tokens,
            capacity=args.capacity, max_cache_len=args.max_cache_len,
        )
        result = {
            "name": prompt_name,
            "prompt": prompt,
            "prompt_token_count": len(prompt_ids),
            "baseline_token_ids": [item["selected_token_id"] for item in baseline],
            "all_layers": policy_summary(baseline, all_layers),
            "hybrid": policy_summary(baseline, hybrid),
        }
        prompt_results.append(result)
        print(json.dumps({
            "name": prompt_name,
            "prompt_token_count": len(prompt_ids),
            "baseline": result["baseline_token_ids"],
            "all_layers": result["all_layers"]["token_ids"],
            "hybrid": result["hybrid"]["token_ids"],
        }), flush=True)

    all_failures = [item["name"] for item in prompt_results if not item["all_layers"]["tokens_equal"]]
    hybrid_failures = [item["name"] for item in prompt_results if not item["hybrid"]["tokens_equal"]]
    payload = {
        "schema_version": 1,
        "protocol": "multi-prompt NQKV Graph split-K policy correctness screen",
        "checkpoint": str(args.checkpoint),
        "capacity": args.capacity,
        "max_cache_len": args.max_cache_len,
        "max_new_tokens": args.max_new_tokens,
        "fallback_modules": sorted(fallback_modules),
        "prompt_count": len(prompt_results),
        "all_layers_token_equal_count": len(prompt_results) - len(all_failures),
        "hybrid_token_equal_count": len(prompt_results) - len(hybrid_failures),
        "all_layers_failure_prompts": all_failures,
        "hybrid_failure_prompts": hybrid_failures,
        "prompts": prompt_results,
        "claim_boundary": (
            "Eight deterministic hand-selected prompts and three greedy tokens each. "
            "This is a sensitivity screen, not a statistical quality benchmark or proof "
            "of universal token equivalence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_layers_token_equal_count": payload["all_layers_token_equal_count"],
        "hybrid_token_equal_count": payload["hybrid_token_equal_count"],
        "all_layers_failure_prompts": all_failures,
        "hybrid_failure_prompts": hybrid_failures,
    }, indent=2))


if __name__ == "__main__":
    main()
