"""Sweep admission microbatch policies on one warmed packed Llama runtime."""

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


POLICIES = ((1, 0), (2, 1), (4, 1), (4, 2), (6, 2), (8, 4))


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
    specifications = []
    for _ in range(args.requests):
        prompt_length = rng.randint(4, 16)
        output_length = rng.randint(2, 6)
        prompt = tuple(torch.randint(
            1, model.config.vocab_size, (prompt_length,), device="cuda", generator=generator
        ).tolist())
        specifications.append((prompt, output_length))

    warmup = ContinuousBatchInferenceEngine(
        runner, admission_policy=AdmissionPolicy(1, 0)
    )
    for index in range(args.capacity):
        prompt, _ = specifications[index]
        warmup.submit(DecodeRequest(f"warmup-{index}", prompt, 2))
    while warmup.has_work():
        warmup.step()
    torch.cuda.synchronize()

    results = []
    for min_batch, max_wait in POLICIES:
        engine = ContinuousBatchInferenceEngine(
            runner,
            admission_policy=AdmissionPolicy(min_batch, max_wait),
        )
        engine.initialize()
        requests = [
            DecodeRequest(f"p{min_batch}-w{max_wait}-{index}", prompt, output_length)
            for index, (prompt, output_length) in enumerate(specifications)
        ]
        for request in requests:
            engine.submit(request)
        torch.cuda.synchronize()
        started = time.perf_counter()
        first_token = {}
        admission_steps = 0
        steps = 0
        while engine.has_work():
            step = engine.step()
            torch.cuda.synchronize()
            now = time.perf_counter()
            admission_steps += bool(step.admitted_request_ids)
            steps += 1
            for request_id in step.generated_tokens:
                first_token.setdefault(request_id, now)
        ended = time.perf_counter()
        token_count = sum(len(request.generated_token_ids) for request in requests)
        ttft = [(first_token[request.request_id] - started) * 1000 for request in requests]
        item = {
            "min_prefill_batch": min_batch,
            "max_wait_steps": max_wait,
            "steps": steps,
            "admission_steps": admission_steps,
            "elapsed_seconds": ended - started,
            "tokens_per_second": token_count / (ended - started),
            "logical_ttft_p50_ms": percentile(ttft, 0.50),
            "logical_ttft_p95_ms": percentile(ttft, 0.95),
        }
        results.append(item)
        print(json.dumps(item), flush=True)

    report = {
        "status": "completed",
        "device": torch.cuda.get_device_name(),
        "scope": "closed-loop admission-policy sweep",
        "capacity": args.capacity,
        "request_count": args.requests,
        "results": results,
        "claim_boundary": (
            "All requests are queued before timing. Logical TTFT includes queue order and cannot "
            "be interpreted as open-loop online-service latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
