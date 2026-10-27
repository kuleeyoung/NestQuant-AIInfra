"""Long-running mixed-request stability test for the packed continuous-batching engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.inference_engine import (
    AdmissionPolicy,
    ContinuousBatchInferenceEngine,
    GreedyHostSampler,
)
from nestquant.runtime.ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


def current_rss_bytes() -> int:
    with open("/proc/self/statm", encoding="utf-8") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


class PeriodicFiniteSampler(GreedyHostSampler):
    def __init__(self, interval: int = 50) -> None:
        self.interval = interval
        self.calls = 0
        self.checks = 0

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls % self.interval == 0:
            self.checks += 1
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("non-finite logits detected during soak")
        return super().__call__(logits)


@torch.inference_mode()
def greedy_full(model, request: DecodeRequest) -> tuple[list[int], list[float]]:
    sequence = torch.tensor(
        [request.prompt_token_ids], dtype=torch.long, device="cuda"
    )
    generated = []
    margins = []
    for _ in range(request.max_new_tokens):
        logits = model(input_ids=sequence).logits[:, -1].float()
        top2 = logits.topk(2, dim=-1)
        token = int(top2.indices[0, 0].item())
        margins.append(float((top2.values[0, 0] - top2.values[0, 1]).item()))
        generated.append(token)
        sequence = torch.cat((
            sequence,
            torch.tensor([[token]], dtype=torch.long, device="cuda"),
        ), dim=1)
    return generated, margins


@torch.inference_mode()
def actual_token_gaps(model, request: DecodeRequest) -> list[float]:
    """Return max-logit minus chosen-logit under each actual generated prefix."""
    sequence = torch.tensor(
        [request.prompt_token_ids], dtype=torch.long, device="cuda"
    )
    gaps = []
    for token in request.generated_token_ids:
        logits = model(input_ids=sequence).logits[:, -1].float()
        gaps.append(float((logits.max() - logits[0, token]).item()))
        sequence = torch.cat((
            sequence,
            torch.tensor([[token]], dtype=torch.long, device="cuda"),
        ), dim=1)
    return gaps


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--backlog", type=int, default=16)
    parser.add_argument("--sample-interval-seconds", type=float, default=30.0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = _load_model(reader.model_metadata["model_id"],
                        reader.model_metadata["model_revision"], args.offline)
    names = _projection_names(model, reader)
    convert_linears_to_fused_nqrt_(
        model, reader, names, output_dtype=torch.bfloat16,
        group_qkv=False, compute_backend="cuda_mma_j2",
    )
    runner = NQRTRaggedGraphDecodeRunner(model, batch_size=args.capacity, max_cache_len=64)
    sampler = PeriodicFiniteSampler()
    engine = ContinuousBatchInferenceEngine(
        runner,
        sampler=sampler,
        admission_policy=AdmissionPolicy(min_prefill_batch=8, max_wait_steps=4),
    )
    rng = random.Random(20261027)
    token_generator = torch.Generator(device="cuda").manual_seed(20261027)

    # Compile/capture representative shapes before resetting peak statistics.
    for index in range(args.capacity):
        length = 4 + index * 3
        prompt = tuple(torch.randint(
            1, model.config.vocab_size, (length,), device="cuda", generator=token_generator
        ).tolist())
        engine.submit(DecodeRequest(f"warmup-{index}", prompt, 3))
    while engine.has_work():
        result = engine.step()
        for request_id in result.finished_request_ids:
            engine.forget_terminal(request_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    submitted = 0
    finished = 0
    generated_tokens = 0
    steps = 0
    sampled_requests: list[DecodeRequest] = []
    memory_samples = []
    started = time.perf_counter()
    deadline = started + args.duration_seconds
    next_sample = started

    def submit_until_backlog() -> None:
        nonlocal submitted
        current = len(engine.scheduler.running) + engine.scheduler.pending_count()
        while current < args.backlog:
            prompt_length = rng.randint(4, 32)
            output_length = rng.randint(2, 8)
            prompt = tuple(torch.randint(
                1, model.config.vocab_size, (prompt_length,),
                device="cuda", generator=token_generator,
            ).tolist())
            engine.submit(DecodeRequest(f"soak-{submitted}", prompt, output_length))
            submitted += 1
            current += 1

    while time.perf_counter() < deadline:
        submit_until_backlog()
        result = engine.step()
        steps += 1
        generated_tokens += len(result.generated_tokens)
        for request_id in result.finished_request_ids:
            request = engine.forget_terminal(request_id)
            finished += 1
            if len(sampled_requests) < 3:
                sampled_requests.append(request)
        now = time.perf_counter()
        if now >= next_sample:
            torch.cuda.synchronize()
            sample = {
                "elapsed_seconds": now - started,
                "submitted": submitted,
                "finished": finished,
                "generated_tokens": generated_tokens,
                "registered_requests": engine.scheduler.registered_count(),
                "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                "cpu_rss_bytes": current_rss_bytes(),
            }
            memory_samples.append(sample)
            print(json.dumps(sample), flush=True)
            next_sample = now + args.sample_interval_seconds

    # Stop arrivals and drain the bounded backlog.
    while engine.has_work():
        result = engine.step()
        steps += 1
        generated_tokens += len(result.generated_tokens)
        for request_id in result.finished_request_ids:
            request = engine.forget_terminal(request_id)
            finished += 1
            if len(sampled_requests) < 3:
                sampled_requests.append(request)
    torch.cuda.synchronize()
    ended = time.perf_counter()
    final_sample = {
        "elapsed_seconds": ended - started,
        "submitted": submitted,
        "finished": finished,
        "generated_tokens": generated_tokens,
        "registered_requests": engine.scheduler.registered_count(),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "cpu_rss_bytes": current_rss_bytes(),
    }
    memory_samples.append(final_sample)
    print(json.dumps(final_sample), flush=True)

    correctness = []
    for request in sampled_requests:
        reference, reference_margins = greedy_full(model, request)
        chosen_gaps = actual_token_gaps(model, request)
        correctness.append({
            "request_id": request.request_id,
            "prompt_token_ids": request.prompt_token_ids,
            "actual": request.generated_token_ids,
            "reference": reference,
            "reference_top1_top2_margins": reference_margins,
            "chosen_token_max_logit_gaps": chosen_gaps,
            "exact_sequence_equal": request.generated_token_ids == reference,
            "all_tokens_are_argmax": all(gap <= 1e-6 for gap in chosen_gaps),
        })
    allocated_values = [sample["cuda_allocated_bytes"] for sample in memory_samples]
    reserved_values = [sample["cuda_reserved_bytes"] for sample in memory_samples]
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(),
        "requested_duration_seconds": args.duration_seconds,
        "elapsed_with_drain_seconds": ended - started,
        "capacity": args.capacity,
        "target_backlog": args.backlog,
        "submitted_requests": submitted,
        "finished_requests": finished,
        "generated_tokens": generated_tokens,
        "steps": steps,
        "tokens_per_second": generated_tokens / (ended - started),
        "finite_logit_checks": sampler.checks,
        "final_registered_requests": engine.scheduler.registered_count(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_allocated_range_bytes": max(allocated_values) - min(allocated_values),
        "cuda_reserved_range_bytes": max(reserved_values) - min(reserved_values),
        "memory_samples": memory_samples,
        "sampled_correctness": correctness,
        "claim_boundary": (
            "Single-process greedy engine soak without tokenizer, network I/O, cancellation, "
            "random sampling, or multiple GPUs."
        ),
    }
    if (submitted != finished
            or engine.scheduler.registered_count() != 0
            or not all(item["all_tokens_are_argmax"] for item in correctness)
            or report["cuda_allocated_range_bytes"] > 256 * 1024 * 1024
            or report["cuda_reserved_range_bytes"] > 256 * 1024 * 1024):
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "memory_samples"}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
