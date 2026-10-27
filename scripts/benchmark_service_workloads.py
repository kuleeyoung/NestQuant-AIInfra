"""Benchmark packed continuous batching under steady, burst and mixed arrivals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.inference_engine import AdmissionPolicy, ContinuousBatchInferenceEngine
from nestquant.runtime.ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from nestquant.runtime.service_benchmark import (
    distribution,
    generate_workload,
    summarize_repetitions,
    validate_benchmark_config,
    workload_seed,
)
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


def flatten_distribution(prefix: str, values: list[float]) -> dict[str, float]:
    stats = distribution(values)
    return {
        f"{prefix}_{name}": value
        for name, value in stats.items()
        if name != "count"
    }


@torch.inference_mode()
def run_once(
    runner: NQRTRaggedGraphDecodeRunner,
    specs,
    prompts: list[tuple[int, ...]],
    policy: dict,
    *,
    run_id: str,
) -> dict:
    engine = ContinuousBatchInferenceEngine(
        runner,
        admission_policy=AdmissionPolicy(
            policy["min_prefill_batch"], policy["max_wait_steps"]
        ),
    )
    engine.initialize()
    requests = [
        DecodeRequest(f"{run_id}-{index}", prompts[index], spec.output_length)
        for index, spec in enumerate(specs)
    ]
    scheduled_at: dict[str, float] = {}
    submitted_at: dict[str, float] = {}
    token_times: dict[str, list[float]] = {request.request_id: [] for request in requests}
    finished_at: dict[str, float] = {}
    rejected: dict[str, str] = {}
    queue_depths: list[float] = []
    step_records: list[dict] = []
    next_index = 0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    while next_index < len(requests) or engine.has_work():
        now = time.perf_counter()
        while (next_index < len(requests)
               and now - started >= specs[next_index].arrival_seconds):
            request = requests[next_index]
            scheduled = started + specs[next_index].arrival_seconds
            scheduled_at[request.request_id] = scheduled
            submitted_at[request.request_id] = now
            try:
                engine.submit(request)
            except Exception as error:  # preserve a per-request failure record
                rejected[request.request_id] = f"{type(error).__name__}: {error}"
            next_index += 1

        queue_depths.append(float(
            engine.scheduler.pending_count() + len(engine.scheduler.running)
        ))
        if engine.has_work():
            step_started = time.perf_counter()
            result = engine.step()
            torch.cuda.synchronize()
            completed_at = time.perf_counter()
            admitted_prompts = [
                len(engine.scheduler.get(request_id).prompt_token_ids)
                for request_id in result.admitted_request_ids
            ]
            step_records.append({
                "latency_ms": (completed_at - step_started) * 1000,
                "admitted_count": len(result.admitted_request_ids),
                "max_admitted_prompt_length": max(admitted_prompts, default=0),
                "generated_count": len(result.generated_tokens),
                "queue_depth_before_step": queue_depths[-1],
            })
            for request_id in result.generated_tokens:
                token_times[request_id].append(completed_at)
            for request_id in result.finished_request_ids:
                finished_at[request_id] = completed_at
                engine.forget_terminal(request_id)
        elif next_index < len(requests):
            wake_at = started + specs[next_index].arrival_seconds
            delay = wake_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    torch.cuda.synchronize()
    ended = time.perf_counter()
    completed = [request for request in requests if request.request_id in finished_at]
    ttft_ms = [
        (token_times[request.request_id][0] - scheduled_at[request.request_id]) * 1000
        for request in completed
    ]
    e2e_ms = [
        (finished_at[request.request_id] - scheduled_at[request.request_id]) * 1000
        for request in completed
    ]
    tpot_ms = [
        (times[index] - times[index - 1]) * 1000
        for request in completed
        for times in [token_times[request.request_id]]
        for index in range(1, len(times))
    ]
    arrival_lag_ms = [
        (submitted_at[request.request_id] - scheduled_at[request.request_id]) * 1000
        for request in requests
        if request.request_id in submitted_at
    ]
    generated_tokens = sum(len(request.generated_token_ids) for request in completed)
    elapsed = ended - started
    if not completed or not tpot_ms:
        raise RuntimeError("workload produced insufficient completed multi-token requests")
    item = {
        "run_id": run_id,
        "policy": policy["name"],
        "elapsed_seconds": elapsed,
        "scheduled_requests": len(requests),
        "completed_requests": len(completed),
        "failed_requests": len(requests) - len(completed),
        "rejected_requests": rejected,
        "completion_rate": len(completed) / len(requests),
        "generated_tokens": generated_tokens,
        "request_throughput_per_second": len(completed) / elapsed,
        "token_throughput_per_second": generated_tokens / elapsed,
        **flatten_distribution("ttft_ms", ttft_ms),
        **flatten_distribution("tpot_ms", tpot_ms),
        **flatten_distribution("e2e_ms", e2e_ms),
        **flatten_distribution("arrival_lag_ms", arrival_lag_ms),
        **flatten_distribution("queue_depth", queue_depths),
        **flatten_distribution(
            "step_latency_ms", [record["latency_ms"] for record in step_records]
        ),
        "slowest_steps": sorted(
            step_records, key=lambda record: record["latency_ms"], reverse=True
        )[:10],
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cuda_allocated_bytes_at_end": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes_at_end": torch.cuda.memory_reserved(),
        "final_registered_requests": engine.scheduler.registered_count(),
    }
    if item["failed_requests"] or item["final_registered_requests"]:
        raise RuntimeError(f"incomplete service benchmark run: {item}")
    return item


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/service_benchmark.v1.json"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.profile:
        requested = set(args.profile)
        config["profiles"] = [
            profile for profile in config["profiles"] if profile["name"] in requested
        ]
        missing = requested - {profile["name"] for profile in config["profiles"]}
        if missing:
            raise ValueError(f"unknown profiles: {sorted(missing)}")
    if args.repetitions is not None:
        if args.repetitions < 1:
            raise ValueError("--repetitions must be positive")
        config["repetitions"] = args.repetitions
    validate_benchmark_config(config)

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = _load_model(
        reader.model_metadata["model_id"], reader.model_metadata["model_revision"], args.offline
    )
    names = _projection_names(model, reader)
    convert_linears_to_fused_nqrt_(
        model, reader, names, output_dtype=torch.bfloat16,
        group_qkv=False, compute_backend="cuda_mma_j2",
    )
    runner = NQRTRaggedGraphDecodeRunner(
        model,
        batch_size=int(config["capacity"]),
        max_cache_len=int(config["max_cache_len"]),
    )

    # Capture and allocator warmup before any reported run.
    warmup = ContinuousBatchInferenceEngine(
        runner, admission_policy=AdmissionPolicy(1, 0)
    )
    generator = torch.Generator(device="cuda").manual_seed(int(config["seed"]))
    for index in range(int(config["warmup_requests"])):
        length = 4 + index
        prompt = tuple(torch.randint(
            1, model.config.vocab_size, (length,), device="cuda", generator=generator
        ).tolist())
        warmup.submit(DecodeRequest(f"warmup-{index}", prompt, 3))
    while warmup.has_work():
        result = warmup.step()
        for request_id in result.finished_request_ids:
            warmup.forget_terminal(request_id)

    warmup_prompt_lengths = []
    if config.get("warmup_all_prompt_lengths", False):
        bounds = []
        for profile in config["profiles"]:
            if profile.get("length_mode", "uniform") == "uniform":
                bounds.append(profile["prompt_length"])
            else:
                bounds.extend((profile["short"]["prompt_length"], profile["long"]["prompt_length"]))
        minimum = min(bound[0] for bound in bounds)
        maximum = max(bound[1] for bound in bounds)
        warmup_prompt_lengths = list(range(minimum, maximum + 1))
        for length in warmup_prompt_lengths:
            for slot in range(int(config["capacity"])):
                prompt = tuple(torch.randint(
                    1, model.config.vocab_size, (length,),
                    device="cuda", generator=generator,
                ).tolist())
                warmup.submit(DecodeRequest(f"shape-{length}-{slot}", prompt, 1))
            while warmup.has_work():
                result = warmup.step()
                for request_id in result.finished_request_ids:
                    warmup.forget_terminal(request_id)
    torch.cuda.synchronize()

    all_runs = []
    summaries = []
    repetitions = int(config["repetitions"])
    for profile_index, profile in enumerate(config["profiles"]):
        profile_runs = []
        for repetition in range(repetitions):
            seed = workload_seed(
                config["seed"], profile_index, repetition,
                couple_profiles=config.get("couple_profile_seeds", False),
            )
            specs = generate_workload(profile, seed=seed)
            prompt_generator = torch.Generator(device="cuda").manual_seed(seed)
            prompts = [
                tuple(torch.randint(
                    1, model.config.vocab_size, (spec.prompt_length,),
                    device="cuda", generator=prompt_generator,
                ).tolist())
                for spec in specs
            ]
            # Rotate execution order to reduce systematic warm-cache bias.
            policies = list(config["policies"])
            if repetition % 2:
                policies.reverse()
            for policy in policies:
                run_id = f"{profile['name']}-{policy['name']}-r{repetition}"
                item = run_once(runner, specs, prompts, policy, run_id=run_id)
                item["profile"] = profile["name"]
                item["repetition"] = repetition
                profile_runs.append(item)
                all_runs.append(item)
                print(json.dumps({
                    "run_id": run_id,
                    "tokens_per_second": item["token_throughput_per_second"],
                    "ttft_p95_ms": item["ttft_ms_p95"],
                    "tpot_p95_ms": item["tpot_ms_p95"],
                    "e2e_p95_ms": item["e2e_ms_p95"],
                }), flush=True)
        for policy in config["policies"]:
            selected = [item for item in profile_runs if item["policy"] == policy["name"]]
            # Normalize flattened run keys to the stable summary schema.
            normalized = [{
                "request_throughput_per_second": item["request_throughput_per_second"],
                "token_throughput_per_second": item["token_throughput_per_second"],
                "completion_rate": item["completion_rate"],
                "ttft_p50_ms": item["ttft_ms_p50"],
                "ttft_p95_ms": item["ttft_ms_p95"],
                "ttft_p99_ms": item["ttft_ms_p99"],
                "tpot_p50_ms": item["tpot_ms_p50"],
                "tpot_p95_ms": item["tpot_ms_p95"],
                "tpot_p99_ms": item["tpot_ms_p99"],
                "e2e_p50_ms": item["e2e_ms_p50"],
                "e2e_p95_ms": item["e2e_ms_p95"],
                "e2e_p99_ms": item["e2e_ms_p99"],
                "arrival_lag_p95_ms": item["arrival_lag_ms_p95"],
                "queue_depth_p95": item["queue_depth_p95"],
                "cuda_peak_allocated_bytes": item["cuda_peak_allocated_bytes"],
                "cuda_peak_reserved_bytes": item["cuda_peak_reserved_bytes"],
            } for item in selected]
            summaries.append({
                "profile": profile["name"],
                "policy": policy["name"],
                "repetitions": repetitions,
                "metrics": summarize_repetitions(normalized),
            })

    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "capacity": config["capacity"],
        "max_cache_len": config["max_cache_len"],
        "warmup_prompt_lengths": warmup_prompt_lengths,
        "profiles": config["profiles"],
        "policies": config["policies"],
        "runs": all_runs,
        "summaries": summaries,
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "runs": len(all_runs),
        "summaries": len(summaries),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
