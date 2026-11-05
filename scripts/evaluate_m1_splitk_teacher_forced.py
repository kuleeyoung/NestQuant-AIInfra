"""Evaluate M1 split-K on identical teacher-forced WikiText-2 decode inputs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.evaluation.perplexity import tokenize_corpus
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.inference_engine import AdmissionPolicy, ContinuousBatchInferenceEngine
from nestquant.runtime.nqkv_batched_hf_backend import (
    BatchedNQKVLlamaCache,
    install_batched_nqkv_llama_backend,
)
from nestquant.runtime.nqkv_batched_runner import BatchedNQKVLlamaRunner
from scripts.benchmark_full_nqrt_model import ACTIVATION_BETAS, _load_model, _projection_names
from scripts.benchmark_service_workloads import KEY_BETAS, VALUE_BETAS
from scripts.diagnose_m1_splitk_logits import select_m1_down_splitk_modules


EXPECTED_TOKEN_SHA256 = "54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ForcedTokenSampler:
    """Return prescribed tokens while retaining the logits for each decision."""

    def __init__(self, target_tokens: tuple[int, ...]) -> None:
        self.target_tokens = target_tokens
        self.index = 0
        self.logits: list[torch.Tensor] = []

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[0] != 1 or self.index >= len(self.target_tokens):
            raise RuntimeError("teacher-forced sampler received an unexpected decision")
        self.logits.append(logits.detach().float().cpu())
        token = self.target_tokens[self.index]
        self.index += 1
        return torch.tensor([token], dtype=torch.long, device=logits.device)


def forced_nqkv_logits(
    model: torch.nn.Module,
    prompt_tokens: tuple[int, ...],
    target_tokens: tuple[int, ...],
    *,
    capacity: int,
    max_cache_len: int,
) -> list[torch.Tensor]:
    """Run production NQKV Graph decode while forcing identical target tokens."""
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
    sampler = ForcedTokenSampler(target_tokens)
    engine = ContinuousBatchInferenceEngine(
        runner, sampler=sampler, admission_policy=AdmissionPolicy(1, 0)
    )
    engine.submit(DecodeRequest("teacher_forced", prompt_tokens, len(target_tokens)))
    emitted: list[int] = []
    while engine.has_work():
        result = engine.step()
        if "teacher_forced" in result.generated_tokens:
            emitted.append(int(result.generated_tokens["teacher_forced"]))
    if emitted != list(target_tokens) or len(sampler.logits) != len(target_tokens):
        raise RuntimeError("teacher-forced runner did not preserve the target sequence")
    logits = sampler.logits
    runner.clear_slots(torch.ones(capacity, dtype=torch.bool, device="cuda"))
    del engine, runner, cache, sampler
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def compare_forced_logits(
    baseline: list[torch.Tensor],
    candidate: list[torch.Tensor],
    target_tokens: tuple[int, ...],
) -> dict[str, Any]:
    """Compare paired logits under the same teacher-forced token sequence."""
    if len(baseline) != len(candidate) or len(baseline) != len(target_tokens):
        raise ValueError("paired trace lengths do not match targets")
    rows = []
    for index, (left, right, target) in enumerate(
        zip(baseline, candidate, target_tokens)
    ):
        left = left.reshape(-1)
        right = right.reshape(-1)
        left_logp = F.log_softmax(left, dim=-1)
        right_logp = F.log_softmax(right, dim=-1)
        difference = right - left
        rows.append({
            "step": index,
            "target_token_id": target,
            "baseline_nll": float(-left_logp[target].item()),
            "candidate_nll": float(-right_logp[target].item()),
            "argmax_equal": bool(left.argmax().item() == right.argmax().item()),
            "baseline_argmax": int(left.argmax().item()),
            "candidate_argmax": int(right.argmax().item()),
            "max_abs_logit_error": float(difference.abs().max().item()),
            "mean_abs_logit_error": float(difference.abs().mean().item()),
            "relative_l2_logit_error": float(
                (torch.linalg.vector_norm(difference)
                 / torch.linalg.vector_norm(left)).item()
            ),
            "baseline_to_candidate_kl": float(torch.sum(
                left_logp.exp() * (left_logp - right_logp)
            ).item()),
        })
    return {"steps": rows}


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired teacher-forced samples into quality metrics."""
    steps = [step for sample in samples for step in sample["steps"]]
    baseline_nll = sum(step["baseline_nll"] for step in steps)
    candidate_nll = sum(step["candidate_nll"] for step in steps)
    count = len(steps)
    return {
        "sample_count": len(samples),
        "predicted_token_count": count,
        "baseline_mean_nll": baseline_nll / count,
        "candidate_mean_nll": candidate_nll / count,
        "mean_nll_delta": (candidate_nll - baseline_nll) / count,
        "baseline_perplexity": math.exp(baseline_nll / count),
        "candidate_perplexity": math.exp(candidate_nll / count),
        "relative_perplexity_delta": math.exp((candidate_nll - baseline_nll) / count) - 1.0,
        "argmax_equal_count": sum(step["argmax_equal"] for step in steps),
        "argmax_agreement_rate": sum(step["argmax_equal"] for step in steps) / count,
        "mean_kl": sum(step["baseline_to_candidate_kl"] for step in steps) / count,
        "mean_relative_l2_logit_error": sum(
            step["relative_l2_logit_error"] for step in steps
        ) / count,
        "max_abs_logit_error": max(step["max_abs_logit_error"] for step in steps),
        "reproducibility_max_abs_logit_error": max(
            float(sample.get("reproducibility_max_abs_logit_error", 0.0))
            for sample in samples
        ),
    }


def max_trace_difference(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> float:
    if len(left) != len(right):
        raise ValueError("repeated trace lengths differ")
    return max(
        float((first.float() - second.float()).abs().max().item())
        for first, second in zip(left, right)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--target-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.prompt_tokens, args.target_tokens, args.capacity) < 1:
        parser.error("sample, token and capacity arguments must be positive")
    if args.prompt_tokens + args.target_tokens > args.max_cache_len:
        parser.error("prompt plus target tokens exceed max cache length")
    if args.progress is None:
        args.progress = args.output.with_suffix(".progress.json")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model_id = reader.model_metadata["model_id"]
    revision = reader.model_metadata["model_revision"]
    model = _load_model(model_id, revision, args.offline)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=args.offline, use_fast=True
    )
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))[
        "llm_ppl_experiment"
    ]
    dataset = load_dataset(
        config["dataset_id"], config["dataset_config"], split="test",
        revision=config["dataset_revision"], download_mode="reuse_dataset_if_exists",
    )
    corpus = tokenize_corpus(
        [text if isinstance(text, str) else "" for text in dataset["text"]],
        tokenizer,
        2048,
    )
    if corpus.token_ids_sha256 != EXPECTED_TOKEN_SHA256:
        raise RuntimeError("WikiText-2 token hash mismatch")
    span = args.prompt_tokens + args.target_tokens
    generator = random.Random(args.seed)
    starts = sorted(generator.sample(range(0, len(corpus.token_ids) - span), args.samples))
    sample_plan_sha256 = hashlib.sha256(
        json.dumps(starts, separators=(",", ":")).encode()
    ).hexdigest()
    run_signature = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_id": model_id,
        "model_revision": revision,
        "token_ids_sha256": corpus.token_ids_sha256,
        "sample_plan_sha256": sample_plan_sha256,
        "seed": args.seed,
        "samples": args.samples,
        "prompt_tokens": args.prompt_tokens,
        "target_tokens": args.target_tokens,
        "capacity": args.capacity,
        "max_cache_len": args.max_cache_len,
    }
    signature_sha256 = hashlib.sha256(
        json.dumps(run_signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    convert_linears_to_fused_nqrt_(
        model, reader, _projection_names(model, reader),
        output_dtype=torch.bfloat16, group_qkv=False,
        compute_backend="packed_w4a4_cuda_mma", activation_q=14,
        activation_betas=ACTIVATION_BETAS, enable_m1_down_splitk=False,
    )
    eligible = select_m1_down_splitk_modules(model, set())
    samples: list[dict[str, Any]] = []
    elapsed_seconds_prior = 0.0
    if args.resume and args.progress.exists():
        saved = json.loads(args.progress.read_text(encoding="utf-8"))
        if saved.get("signature_sha256") != signature_sha256:
            raise RuntimeError("progress file does not match this evaluation signature")
        samples = list(saved.get("samples", []))
        elapsed_seconds_prior = float(saved.get("elapsed_seconds", 0.0))
        if len(samples) > args.samples:
            raise RuntimeError("progress contains too many samples")
        print(f"Resuming after {len(samples)}/{args.samples} contexts.", flush=True)
    started = time.perf_counter()
    for sample_index in range(len(samples), args.samples):
        start = starts[sample_index]
        tokens = corpus.token_ids[start:start + span]
        prompt = tuple(tokens[:args.prompt_tokens])
        targets = tuple(tokens[args.prompt_tokens:])
        order = "AB" if sample_index % 2 == 0 else "BA"
        traces: dict[str, list[torch.Tensor]] = {}
        for label in order:
            selected = set() if label == "A" else set(eligible)
            select_m1_down_splitk_modules(model, selected)
            traces[label] = forced_nqkv_logits(
                model, prompt, targets, capacity=args.capacity,
                max_cache_len=args.max_cache_len,
            )
        reproducibility_error = 0.0
        if sample_index < 2:
            repeated_label = order[0]
            selected = set() if repeated_label == "A" else set(eligible)
            select_m1_down_splitk_modules(model, selected)
            repeated = forced_nqkv_logits(
                model, prompt, targets, capacity=args.capacity,
                max_cache_len=args.max_cache_len,
            )
            reproducibility_error = max_trace_difference(
                traces[repeated_label], repeated
            )
        baseline, candidate = traces["A"], traces["B"]
        comparison = compare_forced_logits(baseline, candidate, targets)
        comparison.update({
            "sample": sample_index,
            "token_start": start,
            "execution_order": order,
            "reproducibility_max_abs_logit_error": reproducibility_error,
        })
        samples.append(comparison)
        running = summarize(samples)
        elapsed_seconds = elapsed_seconds_prior + time.perf_counter() - started
        write_json_atomic(args.progress, {
            "schema_version": 1,
            "status": "in_progress",
            "signature_sha256": signature_sha256,
            "run_signature": run_signature,
            "completed_samples": len(samples),
            "target_samples": args.samples,
            "elapsed_seconds": elapsed_seconds,
            "summary": running,
            "samples": samples,
        })
        print(
            f"[{sample_index + 1}/{args.samples}] "
            f"delta_ppl={running['relative_perplexity_delta']:+.6%} "
            f"argmax={running['argmax_agreement_rate']:.3%}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "paired teacher-forced NQKV Graph M1 split-K quality v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_id": model_id,
        "model_revision": revision,
        "token_ids_sha256": corpus.token_ids_sha256,
        "sample_plan_sha256": sample_plan_sha256,
        "signature_sha256": signature_sha256,
        "run_signature": run_signature,
        "seed": args.seed,
        "prompt_tokens": args.prompt_tokens,
        "target_tokens": args.target_tokens,
        "capacity": args.capacity,
        "max_cache_len": args.max_cache_len,
        "eligible_splitk_modules": eligible,
        "summary": summarize(samples),
        "elapsed_seconds": elapsed_seconds_prior + time.perf_counter() - started,
        "samples": samples,
        "claim_boundary": (
            "Deterministic sampled WikiText-2 teacher-forced decode on the production "
            "NQKV Graph runner with alternating AB/BA execution order and early ABA/BAB "
            "repeatability checks. It covers identical M1 inputs and reports paired NLL, "
            "but is not full-corpus perplexity or a downstream-task accuracy result."
        ),
    }
    write_json_atomic(args.output, payload)
    write_json_atomic(args.progress, {
        **payload,
        "status": "complete",
        "completed_samples": len(samples),
        "target_samples": args.samples,
    })
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
