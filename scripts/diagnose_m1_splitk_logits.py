"""Diagnose the first autoregressive logit divergence caused by M1 split-K."""

from __future__ import annotations

import argparse
import gc
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
from nestquant.models.fused_linear import FusedNQRTLinear, convert_linears_to_fused_nqrt_
from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.inference_engine import AdmissionPolicy, ContinuousBatchInferenceEngine
from nestquant.runtime.nqkv_batched_hf_backend import (
    BatchedNQKVLlamaCache,
    install_batched_nqkv_llama_backend,
)
from nestquant.runtime.nqkv_batched_runner import BatchedNQKVLlamaRunner
from scripts.benchmark_full_nqrt_model import ACTIVATION_BETAS, _load_model, _projection_names
from scripts.benchmark_service_workloads import KEY_BETAS, VALUE_BETAS


def set_m1_down_splitk(model: torch.nn.Module, enabled: bool) -> list[str]:
    """Set the experimental dispatch flag and return the affected module names."""
    names = []
    for name, module in model.named_modules():
        if (
            isinstance(module, FusedNQRTLinear)
            and module.in_features == 8192
            and module.out_features == 2048
        ):
            module.enable_m1_down_splitk = enabled
            names.append(name)
    return names


def select_m1_down_splitk_modules(
    model: torch.nn.Module, selected_names: set[str]
) -> list[str]:
    """Enable split-K only for selected eligible module names."""
    eligible = []
    for name, module in model.named_modules():
        if (
            isinstance(module, FusedNQRTLinear)
            and module.in_features == 8192
            and module.out_features == 2048
        ):
            eligible.append(name)
            module.enable_m1_down_splitk = name in selected_names
    unknown = selected_names - set(eligible)
    if unknown:
        raise ValueError(f"unknown split-K module names: {sorted(unknown)}")
    return eligible


def greedy_trace(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run eager greedy decoding and retain CPU FP32 logits for each decision."""
    trace: list[dict[str, Any]] = []
    cache = None
    current_ids = input_ids
    current_mask = attention_mask
    with torch.inference_mode():
        for step in range(max_new_tokens):
            output = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=cache,
                use_cache=True,
            )
            logits = output.logits[:, -1, :].float()
            values, indices = torch.topk(logits, k=2, dim=-1)
            token = indices[:, :1].to(dtype=torch.long)
            trace.append({
                "step": step,
                "input_token_ids": current_ids.detach().cpu().tolist(),
                "selected_token_id": int(token.item()),
                "top1_logit": float(values[0, 0].item()),
                "top2_token_id": int(indices[0, 1].item()),
                "top2_logit": float(values[0, 1].item()),
                "top1_top2_margin": float((values[0, 0] - values[0, 1]).item()),
                "logits": logits.detach().cpu(),
            })
            cache = output.past_key_values
            current_ids = token
            current_mask = torch.cat(
                (current_mask, torch.ones_like(token, device=current_mask.device)), dim=1
            )
    return trace


class TraceSampler:
    """Greedy sampler that preserves each scheduler decision's logits."""

    def __init__(self) -> None:
        self.logits: list[torch.Tensor] = []

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        self.logits.append(logits.detach().float().cpu())
        return logits.argmax(dim=-1)


def nqkv_graph_trace(
    model: torch.nn.Module,
    prompt_token_ids: tuple[int, ...],
    *,
    max_new_tokens: int,
    capacity: int,
    max_cache_len: int,
) -> list[dict[str, Any]]:
    """Run the production NQKV Graph runner without HTTP or request concurrency."""
    head_dim = int(getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    ))
    cache = BatchedNQKVLlamaCache(
        capacity=capacity,
        num_layers=int(model.config.num_hidden_layers),
        num_query_heads=int(model.config.num_attention_heads),
        num_kv_heads=int(model.config.num_key_value_heads),
        head_dim=head_dim,
        max_cache_len=max_cache_len,
        page_size=16,
        q=14,
        key_betas=KEY_BETAS,
        value_betas=VALUE_BETAS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    install_batched_nqkv_llama_backend(model, cache)
    runner = BatchedNQKVLlamaRunner(model, cache, execution_policy="full")
    sampler = TraceSampler()
    engine = ContinuousBatchInferenceEngine(
        runner,
        sampler=sampler,
        admission_policy=AdmissionPolicy(1, 0),
    )
    engine.submit(DecodeRequest("diagnostic", prompt_token_ids, max_new_tokens))
    tokens: list[int] = []
    while engine.has_work():
        result = engine.step()
        if "diagnostic" in result.generated_tokens:
            tokens.append(int(result.generated_tokens["diagnostic"]))
    if len(tokens) != max_new_tokens or len(sampler.logits) != max_new_tokens:
        raise RuntimeError("runner trace did not produce the requested decisions")
    trace = []
    prior_tokens: list[int] = []
    for step, (token, logits) in enumerate(zip(tokens, sampler.logits)):
        if logits.shape[0] != 1:
            raise RuntimeError("diagnostic expected one sampled request")
        values, indices = torch.topk(logits, k=2, dim=-1)
        trace.append({
            "step": step,
            "input_token_ids": (
                [list(prompt_token_ids)] if step == 0 else [[prior_tokens[-1]]]
            ),
            "selected_token_id": token,
            "top1_logit": float(values[0, 0].item()),
            "top2_token_id": int(indices[0, 1].item()),
            "top2_logit": float(values[0, 1].item()),
            "top1_top2_margin": float((values[0, 0] - values[0, 1]).item()),
            "logits": logits,
        })
        prior_tokens.append(token)
    runner.clear_slots(torch.ones(capacity, dtype=torch.bool, device="cuda"))
    del engine, runner, cache, sampler
    gc.collect()
    torch.cuda.empty_cache()
    return trace


def compare_traces(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise ValueError("trace lengths differ")
    steps = []
    first_token_divergence = None
    for left, right in zip(baseline, candidate):
        if left["step"] != right["step"]:
            raise ValueError("trace step numbers differ")
        left_logits = left["logits"].float()
        right_logits = right["logits"].float()
        difference = right_logits - left_logits
        left_norm = torch.linalg.vector_norm(left_logits)
        same_input = left["input_token_ids"] == right["input_token_ids"]
        same_token = left["selected_token_id"] == right["selected_token_id"]
        if not same_token and first_token_divergence is None:
            first_token_divergence = int(left["step"])
        steps.append({
            "step": int(left["step"]),
            "same_input": same_input,
            "baseline_token_id": int(left["selected_token_id"]),
            "candidate_token_id": int(right["selected_token_id"]),
            "argmax_equal": same_token,
            "baseline_top1_top2_margin": float(left["top1_top2_margin"]),
            "candidate_top1_top2_margin": float(right["top1_top2_margin"]),
            "max_abs_logit_error": float(difference.abs().max().item()),
            "mean_abs_logit_error": float(difference.abs().mean().item()),
            "relative_l2_logit_error": float(
                (torch.linalg.vector_norm(difference) / left_norm).item()
                if left_norm.item() else 0.0
            ),
        })
    return {
        "first_token_divergence_step": first_token_divergence,
        "all_tokens_equal": first_token_divergence is None,
        "steps": steps,
    }


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
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument(
        "--mode", choices=("eager", "nqkv_graph"), default="eager"
    )
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--max-cache-len", type=int, default=2048)
    parser.add_argument(
        "--exclude-candidate-module",
        action="append",
        default=[],
        help="Eligible module to keep unsplit in the candidate; may be repeated.",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/m1_splitk_logit_diagnosis_v1.json"),
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.capacity < 1 or args.max_cache_len < 1:
        parser.error("token, capacity and cache-length arguments must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model_id = reader.model_metadata["model_id"]
    revision = reader.model_metadata["model_revision"]
    model = _load_model(model_id, revision, args.offline)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=args.offline, use_fast=True
    )
    names = _projection_names(model, reader)
    convert_linears_to_fused_nqrt_(
        model,
        reader,
        names,
        output_dtype=torch.bfloat16,
        group_qkv=False,
        compute_backend="packed_w4a4_cuda_mma",
        activation_q=14,
        activation_betas=ACTIVATION_BETAS,
        enable_m1_down_splitk=False,
    )
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")

    affected = set_m1_down_splitk(model, False)
    excluded = set(args.exclude_candidate_module)
    unknown_excluded = excluded - set(affected)
    if unknown_excluded:
        parser.error("unknown excluded modules: " + ", ".join(sorted(unknown_excluded)))
    if args.mode == "eager":
        baseline = greedy_trace(
            model, input_ids, attention_mask, max_new_tokens=args.max_new_tokens
        )
    else:
        baseline = nqkv_graph_trace(
            model,
            tuple(input_ids[0].tolist()),
            max_new_tokens=args.max_new_tokens,
            capacity=args.capacity,
            max_cache_len=args.max_cache_len,
        )
    torch.cuda.synchronize()
    candidate_modules = set(affected) - excluded
    select_m1_down_splitk_modules(model, candidate_modules)
    if args.mode == "eager":
        candidate = greedy_trace(
            model, input_ids, attention_mask, max_new_tokens=args.max_new_tokens
        )
    else:
        candidate = nqkv_graph_trace(
            model,
            tuple(input_ids[0].tolist()),
            max_new_tokens=args.max_new_tokens,
            capacity=args.capacity,
            max_cache_len=args.max_cache_len,
        )
    torch.cuda.synchronize()

    comparison = compare_traces(baseline, candidate)
    payload = {
        "schema_version": 1,
        "protocol": f"same-model {args.mode} greedy split-k off-on logit diagnosis",
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "prompt_token_ids": input_ids.cpu().tolist(),
        "max_new_tokens": args.max_new_tokens,
        "mode": args.mode,
        "capacity": args.capacity if args.mode == "nqkv_graph" else None,
        "affected_modules": affected,
        "candidate_enabled_modules": sorted(candidate_modules),
        "candidate_excluded_modules": sorted(excluded),
        **comparison,
        "baseline_token_ids": [item["selected_token_id"] for item in baseline],
        "candidate_token_ids": [item["selected_token_id"] for item in candidate],
        "claim_boundary": (
            "Single prompt, same-process off/on diagnostic without HTTP or request "
            "concurrency. Eager mode isolates packed-linear numerics; nqkv_graph mode "
            "adds the production cache, scheduler and CUDA Graph runner."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
