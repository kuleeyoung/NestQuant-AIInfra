"""Screen split-K policies on deterministic synthetic token-ID prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from scripts.benchmark_full_nqrt_model import ACTIVATION_BETAS, _load_model, _projection_names
from scripts.diagnose_m1_splitk_logits import (
    nqkv_graph_trace,
    select_m1_down_splitk_modules,
)
from scripts.diagnose_m1_splitk_prompt_suite import policy_summary


def generate_token_prompts(
    *,
    vocab_size: int,
    lengths: tuple[int, ...],
    samples_per_length: int,
    seed: int,
) -> list[tuple[str, tuple[int, ...]]]:
    if vocab_size <= 2048:
        raise ValueError("vocab_size must exceed 2048")
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("lengths must be positive")
    if samples_per_length < 1:
        raise ValueError("samples_per_length must be positive")
    generator = random.Random(seed)
    prompts = []
    for length in lengths:
        for sample in range(samples_per_length):
            token_ids = tuple(
                generator.randrange(1000, vocab_size - 1) for _ in range(length)
            )
            prompts.append((f"len{length}_sample{sample}", token_ids))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=(2, 4, 8, 16, 32, 64))
    parser.add_argument("--samples-per-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--fallback-module", action="append", default=None,
        help="Module kept unsplit by the hybrid policy; may be repeated.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/raw/m1_splitk_token_screen_v1.json"),
    )
    args = parser.parse_args()
    fallback_modules = set(args.fallback_module or [
        "model.layers.1.mlp.down_proj",
        "model.layers.10.mlp.down_proj",
        "model.layers.11.mlp.down_proj",
        "model.layers.14.mlp.down_proj",
    ])
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = _load_model(
        reader.model_metadata["model_id"],
        reader.model_metadata["model_revision"],
        args.offline,
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
    prompts = generate_token_prompts(
        vocab_size=int(model.config.vocab_size),
        lengths=tuple(args.lengths),
        samples_per_length=args.samples_per_length,
        seed=args.seed,
    )

    results = []
    for name, prompt_ids in prompts:
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
            "name": name,
            "prompt_length": len(prompt_ids),
            "prompt_token_ids": prompt_ids,
            "baseline_token_ids": [item["selected_token_id"] for item in baseline],
            "all_layers": policy_summary(baseline, all_layers),
            "hybrid": policy_summary(baseline, hybrid),
        }
        results.append(result)
        print(json.dumps({
            "name": name,
            "baseline": result["baseline_token_ids"],
            "all_equal": result["all_layers"]["tokens_equal"],
            "hybrid_equal": result["hybrid"]["tokens_equal"],
        }), flush=True)

    all_failures = [item["name"] for item in results if not item["all_layers"]["tokens_equal"]]
    hybrid_failures = [item["name"] for item in results if not item["hybrid"]["tokens_equal"]]
    by_length = {}
    for length in args.lengths:
        group = [item for item in results if item["prompt_length"] == length]
        by_length[str(length)] = {
            "count": len(group),
            "all_layers_equal": sum(item["all_layers"]["tokens_equal"] for item in group),
            "hybrid_equal": sum(item["hybrid"]["tokens_equal"] for item in group),
        }
    payload = {
        "schema_version": 1,
        "protocol": "deterministic synthetic token-ID NQKV Graph split-K screen",
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "lengths": args.lengths,
        "samples_per_length": args.samples_per_length,
        "max_new_tokens": args.max_new_tokens,
        "fallback_modules": sorted(fallback_modules),
        "prompt_count": len(results),
        "all_layers_token_equal_count": len(results) - len(all_failures),
        "hybrid_token_equal_count": len(results) - len(hybrid_failures),
        "all_layers_failure_prompts": all_failures,
        "hybrid_failure_prompts": hybrid_failures,
        "by_length": by_length,
        "prompts": results,
        "claim_boundary": (
            "Deterministic synthetic token IDs broaden activation and length coverage but "
            "are not a natural-language quality benchmark or universal equivalence proof."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "prompt_count": payload["prompt_count"],
        "all_layers_token_equal_count": payload["all_layers_token_equal_count"],
        "hybrid_token_equal_count": payload["hybrid_token_equal_count"],
        "hybrid_failure_prompts": hybrid_failures,
        "by_length": by_length,
    }, indent=2))


if __name__ == "__main__":
    main()
