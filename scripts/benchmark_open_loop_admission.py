"""Compare admission policies under fixed wall-clock request arrivals."""

from __future__ import annotations

import argparse
import json
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
from nestquant.runtime.inference_engine import AdmissionPolicy, ContinuousBatchInferenceEngine
from nestquant.runtime.ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


POLICIES = ((1, 0), (4, 2), (8, 4))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--arrival-interval-ms", type=float, default=20.0)
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
    rng = random.Random(20261026)
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    specs = []
    for _ in range(args.requests):
        prompt_length = rng.randint(4, 16)
        output_length = rng.randint(2, 6)
        prompt = tuple(torch.randint(
            1, model.config.vocab_size, (prompt_length,), device="cuda", generator=generator
        ).tolist())
        specs.append((prompt, output_length))

    warmup = ContinuousBatchInferenceEngine(runner, admission_policy=AdmissionPolicy(1, 0))
    for index in range(args.capacity):
        prompt, _ = specs[index]
        warmup.submit(DecodeRequest(f"warmup-{index}", prompt, 2))
    while warmup.has_work():
        warmup.step()
    torch.cuda.synchronize()

    results = []
    interval_seconds = args.arrival_interval_ms / 1000
    for min_batch, max_wait in POLICIES:
        engine = ContinuousBatchInferenceEngine(
            runner, admission_policy=AdmissionPolicy(min_batch, max_wait)
        )
        engine.initialize()
        requests = [
            DecodeRequest(f"open-{min_batch}-{max_wait}-{i}", prompt, output_length)
            for i, (prompt, output_length) in enumerate(specs)
        ]
        submitted_at = {}
        first_token_at = {}
        finished_at = {}
        next_index = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        while next_index < len(requests) or engine.has_work():
            now = time.perf_counter()
            while (next_index < len(requests)
                   and now - started >= next_index * interval_seconds):
                request = requests[next_index]
                engine.submit(request)
                submitted_at[request.request_id] = started + next_index * interval_seconds
                next_index += 1
            if engine.has_work():
                step = engine.step()
                torch.cuda.synchronize()
                completed_at = time.perf_counter()
                for request_id in step.generated_tokens:
                    first_token_at.setdefault(request_id, completed_at)
                for request_id in step.finished_request_ids:
                    finished_at[request_id] = completed_at
            elif next_index < len(requests):
                delay = started + next_index * interval_seconds - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        ended = time.perf_counter()
        ttft = [
            (first_token_at[request.request_id] - submitted_at[request.request_id]) * 1000
            for request in requests
        ]
        e2e = [
            (finished_at[request.request_id] - submitted_at[request.request_id]) * 1000
            for request in requests
        ]
        token_count = sum(len(request.generated_token_ids) for request in requests)
        item = {
            "min_prefill_batch": min_batch,
            "max_wait_steps": max_wait,
            "elapsed_seconds": ended - started,
            "tokens_per_second": token_count / (ended - started),
            "ttft_p50_ms": percentile(ttft, 0.50),
            "ttft_p95_ms": percentile(ttft, 0.95),
            "ttft_p99_ms": percentile(ttft, 0.99),
            "e2e_p95_ms": percentile(e2e, 0.95),
        }
        results.append(item)
        print(json.dumps(item), flush=True)

    report = {
        "status": "completed",
        "device": torch.cuda.get_device_name(),
        "scope": "wall-clock fixed-interval open-loop arrivals",
        "capacity": args.capacity,
        "request_count": args.requests,
        "arrival_interval_ms": args.arrival_interval_ms,
        "offered_request_rate_per_second": 1000 / args.arrival_interval_ms,
        "results": results,
        "claim_boundary": (
            "Single-process synthetic fixed-interval arrivals without tokenizer or network I/O. "
            "This is an engine benchmark, not production service capacity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
