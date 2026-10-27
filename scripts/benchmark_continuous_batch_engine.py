"""Run a bounded mixed-request benchmark on the packed continuous-batching engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.inference_engine import AdmissionPolicy, ContinuousBatchInferenceEngine
from nestquant.runtime.ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--min-prefill-batch", type=int, default=4)
    parser.add_argument("--max-wait-steps", type=int, default=2)
    parser.add_argument("--max-cache-len", type=int, default=64)
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
    runner = NQRTRaggedGraphDecodeRunner(
        model, batch_size=args.capacity, max_cache_len=args.max_cache_len
    )
    policy = AdmissionPolicy(
        min_prefill_batch=args.min_prefill_batch,
        max_wait_steps=args.max_wait_steps,
    )
    engine = ContinuousBatchInferenceEngine(runner, admission_policy=policy)
    rng = random.Random(20261026)
    token_generator = torch.Generator(device="cuda").manual_seed(20261026)
    engine.initialize()
    for index in range(args.warmup_requests):
        prompt_length = 4 + index % 8
        prompt = torch.randint(
            1, model.config.vocab_size, (prompt_length,),
            device="cuda", generator=token_generator,
        ).tolist()
        engine.submit(DecodeRequest(f"warmup-{index}", tuple(prompt), 2))
    while engine.has_work():
        engine.step()
    torch.cuda.synchronize()

    requests = []
    for index in range(args.requests):
        prompt_length = rng.randint(4, 16)
        output_length = rng.randint(2, 6)
        prompt = torch.randint(
            1, model.config.vocab_size, (prompt_length,),
            device="cuda", generator=token_generator,
        ).tolist()
        requests.append(DecodeRequest(
            f"request-{index}", tuple(prompt), output_length
        ))
    for request in requests:
        engine.submit(request)

    torch.cuda.synchronize()
    started = time.perf_counter()
    first_token_at = {}
    finished_at = {}
    step_latencies_ms = []
    admission_step_latencies_ms = []
    decode_only_step_latencies_ms = []
    step_count = 0
    while engine.has_work():
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        result = engine.step()
        torch.cuda.synchronize()
        now = time.perf_counter()
        step_latency_ms = (now - step_started) * 1000
        step_latencies_ms.append(step_latency_ms)
        if result.admitted_request_ids:
            admission_step_latencies_ms.append(step_latency_ms)
        else:
            decode_only_step_latencies_ms.append(step_latency_ms)
        step_count += 1
        for request_id in result.generated_tokens:
            first_token_at.setdefault(request_id, now)
        for request_id in result.finished_request_ids:
            finished_at[request_id] = now
        if step_count > args.requests * 10:
            raise RuntimeError("bounded workload did not drain")
    torch.cuda.synchronize()
    ended = time.perf_counter()

    immediate_engine = ContinuousBatchInferenceEngine(
        runner, admission_policy=AdmissionPolicy(min_prefill_batch=1, max_wait_steps=0)
    )
    immediate_engine.initialize()
    immediate_requests = [
        DecodeRequest(
            f"immediate-{index}", request.prompt_token_ids, request.max_new_tokens
        )
        for index, request in enumerate(requests)
    ]
    for request in immediate_requests:
        immediate_engine.submit(request)
    torch.cuda.synchronize()
    immediate_started = time.perf_counter()
    while immediate_engine.has_work():
        immediate_engine.step()
    torch.cuda.synchronize()
    immediate_ended = time.perf_counter()

    # Static-wave baseline: same backend and requests, but the next group is
    # admitted only after every request in the current capacity-sized wave ends.
    static_engine = ContinuousBatchInferenceEngine(runner, admission_policy=policy)
    static_engine.initialize()
    static_requests = [
        DecodeRequest(
            f"static-{index}", request.prompt_token_ids, request.max_new_tokens
        )
        for index, request in enumerate(requests)
    ]
    torch.cuda.synchronize()
    static_started = time.perf_counter()
    for offset in range(0, len(static_requests), args.capacity):
        for request in static_requests[offset:offset + args.capacity]:
            static_engine.submit(request)
        while static_engine.has_work():
            static_engine.step()
    torch.cuda.synchronize()
    static_ended = time.perf_counter()

    total_tokens = sum(len(request.generated_token_ids) for request in requests)
    ttft_ms = [(first_token_at[request.request_id] - started) * 1000 for request in requests]
    e2e_ms = [(finished_at[request.request_id] - started) * 1000 for request in requests]
    report = {
        "status": "completed",
        "device": torch.cuda.get_device_name(),
        "scope": "closed-loop host-driven packed continuous batching",
        "capacity": args.capacity,
        "request_count": args.requests,
        "warmup_request_count": args.warmup_requests,
        "admission_policy": {
            "min_prefill_batch": args.min_prefill_batch,
            "max_wait_steps": args.max_wait_steps,
        },
        "step_count": step_count,
        "total_generated_tokens": total_tokens,
        "elapsed_seconds": ended - started,
        "aggregate_tokens_per_second": total_tokens / (ended - started),
        "immediate_admission_baseline": {
            "elapsed_seconds": immediate_ended - immediate_started,
            "aggregate_tokens_per_second": total_tokens / (immediate_ended - immediate_started),
            "microbatch_speedup": (immediate_ended - immediate_started) / (ended - started),
        },
        "static_wave_baseline": {
            "elapsed_seconds": static_ended - static_started,
            "aggregate_tokens_per_second": total_tokens / (static_ended - static_started),
            "continuous_speedup": (static_ended - static_started) / (ended - started),
        },
        "step_latency_ms": {
            "p50": percentile(step_latencies_ms, 0.50),
            "p95": percentile(step_latencies_ms, 0.95),
            "p99": percentile(step_latencies_ms, 0.99),
            "mean": statistics.fmean(step_latencies_ms),
        },
        "admission_step_latency_ms": {
            "count": len(admission_step_latencies_ms),
            "p50": percentile(admission_step_latencies_ms, 0.50),
            "p95": percentile(admission_step_latencies_ms, 0.95),
            "mean": statistics.fmean(admission_step_latencies_ms),
        },
        "decode_only_step_latency_ms": {
            "count": len(decode_only_step_latencies_ms),
            "p50": percentile(decode_only_step_latencies_ms, 0.50),
            "p95": percentile(decode_only_step_latencies_ms, 0.95),
            "mean": statistics.fmean(decode_only_step_latencies_ms),
        },
        "logical_ttft_ms": {
            "p50": percentile(ttft_ms, 0.50),
            "p95": percentile(ttft_ms, 0.95),
            "p99": percentile(ttft_ms, 0.99),
        },
        "logical_e2e_ms": {
            "p50": percentile(e2e_ms, 0.50),
            "p95": percentile(e2e_ms, 0.95),
            "p99": percentile(e2e_ms, 0.99),
        },
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "claim_boundary": (
            "All requests were queued before timing, so TTFT is closed-loop queueing latency. "
            "This is not an open-loop arrival-rate benchmark or a network-service result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
