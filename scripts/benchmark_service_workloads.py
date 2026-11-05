"""Benchmark packed continuous batching under steady, burst and mixed arrivals."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
from nestquant.runtime.nqkv_batched_hf_backend import (
    BatchedNQKVLlamaCache,
    install_batched_nqkv_llama_backend,
)
from nestquant.runtime.nqkv_batched_runner import BatchedNQKVLlamaRunner
from nestquant.runtime.ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from nestquant.runtime.service_benchmark import (
    distribution,
    generate_workload,
    summarize_repetitions,
    validate_benchmark_config,
    workload_seed,
)


KEY_BETAS = (0.25, 0.3271428571428571, 0.46214285714285713, 1.2185714285714284)
VALUE_BETAS = (0.2521428571428571, 0.3992857142857143, 0.687142857142857, 2.073571428571429)
from scripts.benchmark_full_nqrt_model import (
    ACTIVATION_BETAS,
    _load_model,
    _projection_names,
    _w4a4_dispatch_counts,
)


def flatten_distribution(prefix: str, values: list[float]) -> dict[str, float]:
    stats = distribution(values)
    return {
        f"{prefix}_{name}": value
        for name, value in stats.items()
        if name != "count"
    }


def _identity_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.inference_mode()
def run_once(
    runner,
    specs,
    prompts: list[tuple[int, ...]],
    policy: dict,
    *,
    run_id: str,
) -> dict:
    dispatch_before = _w4a4_dispatch_counts(runner.model)
    boundary_fallbacks_before = int(
        getattr(runner, "graph_boundary_fallback_count", 0)
    )
    boundary_replays_before = int(
        getattr(runner, "graph_boundary_replay_count", 0)
    )
    dense_prefill_launches_before = int(
        getattr(getattr(runner, "cache", None), "dense_prefill_layer_launch_count", 0)
    )
    ragged_prefill_launches_before = int(
        getattr(getattr(runner, "cache", None), "ragged_subpage_layer_launch_count", 0)
    )
    ragged_full_page_packs_before = int(
        getattr(getattr(runner, "cache", None), "ragged_full_page_layer_pack_count", 0)
    )
    ragged_cross_page_launches_before = int(
        getattr(getattr(runner, "cache", None), "ragged_cross_page_layer_launch_count", 0)
    )
    ragged_multi_page_rounds_before = int(
        getattr(getattr(runner, "cache", None), "ragged_multi_page_layer_round_count", 0)
    )
    batched_pack_before = int(
        getattr(getattr(runner, "cache", None), "multi_page_batched_pack_count", 0)
    )
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
    linear_m_forward_histogram: Counter[int] = Counter()
    prefill_m_forward_histogram: Counter[int] = Counter()
    decode_m_forward_histogram: Counter[int] = Counter()
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
            running_before_step = len(engine.scheduler.running)
            boundary_replays_before_step = int(
                getattr(runner, "graph_boundary_replay_count", 0)
            )
            boundary_fallbacks_before_step = int(
                getattr(runner, "graph_boundary_fallback_count", 0)
            )
            step_started = time.perf_counter()
            result = engine.step()
            torch.cuda.synchronize()
            completed_at = time.perf_counter()
            admitted_prompts = [
                len(engine.scheduler.get(request_id).prompt_token_ids)
                for request_id in result.admitted_request_ids
            ]
            max_admitted_prompt_length = max(admitted_prompts, default=0)
            prefill_m = (
                runner.batch_size * max_admitted_prompt_length
                if admitted_prompts else 0
            )
            decode_m = runner.batch_size if running_before_step else 0
            if prefill_m:
                prefill_m_forward_histogram[prefill_m] += 1
                linear_m_forward_histogram[prefill_m] += 1
            if decode_m:
                decode_m_forward_histogram[decode_m] += 1
                linear_m_forward_histogram[decode_m] += 1
            if prefill_m and decode_m:
                step_kind = "mixed_prefill_decode"
            elif prefill_m:
                step_kind = "prefill_only"
            elif decode_m:
                step_kind = "decode_only"
            else:
                step_kind = "scheduler_only"
            step_records.append({
                "kind": step_kind,
                "latency_ms": (completed_at - step_started) * 1000,
                "admitted_count": len(result.admitted_request_ids),
                "max_admitted_prompt_length": max_admitted_prompt_length,
                "prefill_linear_m": prefill_m,
                "decode_linear_m": decode_m,
                "running_before_step": running_before_step,
                "generated_count": len(result.generated_tokens),
                "queue_depth_before_step": queue_depths[-1],
                "nqkv_boundary_replays": int(
                    getattr(runner, "graph_boundary_replay_count", 0)
                ) - boundary_replays_before_step,
                "nqkv_boundary_fallbacks": int(
                    getattr(runner, "graph_boundary_fallback_count", 0)
                ) - boundary_fallbacks_before_step,
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
    workload_identity_sha256 = _identity_sha256([
        {
            "arrival_seconds": spec.arrival_seconds,
            "prompt_token_ids": list(request.prompt_token_ids),
            "output_length": request.max_new_tokens,
        }
        for spec, request in zip(specs, requests)
    ])
    output_identity_sha256 = _identity_sha256([
        {
            "request_index": index,
            "generated_token_ids": list(request.generated_token_ids),
        }
        for index, request in enumerate(requests)
    ])
    elapsed = ended - started
    if not completed or not tpot_ms:
        raise RuntimeError("workload produced insufficient completed multi-token requests")
    step_kind_metrics = {}
    for step_kind in (
        "prefill_only", "decode_only", "mixed_prefill_decode", "scheduler_only"
    ):
        latencies = [
            record["latency_ms"] for record in step_records
            if record["kind"] == step_kind
        ]
        if latencies:
            step_kind_metrics[step_kind] = distribution(latencies)
    boundary_latencies = [
        record["latency_ms"] for record in step_records
        if record["nqkv_boundary_replays"]
    ]
    if boundary_latencies:
        step_kind_metrics["nqkv_boundary"] = distribution(boundary_latencies)
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
        "workload_identity_sha256": workload_identity_sha256,
        "output_identity_sha256": output_identity_sha256,
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
        "step_records": step_records,
        "step_kind_metrics": step_kind_metrics,
        "linear_m_forward_histogram": {
            str(key): value for key, value in sorted(linear_m_forward_histogram.items())
        },
        "prefill_m_forward_histogram": {
            str(key): value for key, value in sorted(prefill_m_forward_histogram.items())
        },
        "decode_m_forward_histogram": {
            str(key): value for key, value in sorted(decode_m_forward_histogram.items())
        },
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cuda_allocated_bytes_at_end": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes_at_end": torch.cuda.memory_reserved(),
        "final_registered_requests": engine.scheduler.registered_count(),
        "final_backend_cache_requests": (
            runner.cache.manager.active_requests
            if isinstance(runner, BatchedNQKVLlamaRunner) else 0
        ),
        "nqkv_graph_boundary_fallbacks": (
            int(getattr(runner, "graph_boundary_fallback_count", 0))
            - boundary_fallbacks_before
        ),
        "nqkv_graph_boundary_replays": (
            int(getattr(runner, "graph_boundary_replay_count", 0))
            - boundary_replays_before
        ),
        "nqkv_dense_prefill_layer_launches": int(
            getattr(getattr(runner, "cache", None), "dense_prefill_layer_launch_count", 0)
        ) - dense_prefill_launches_before,
        "nqkv_ragged_subpage_layer_launches": int(
            getattr(getattr(runner, "cache", None), "ragged_subpage_layer_launch_count", 0)
        ) - ragged_prefill_launches_before,
        "nqkv_ragged_full_page_layer_packs": int(
            getattr(getattr(runner, "cache", None), "ragged_full_page_layer_pack_count", 0)
        ) - ragged_full_page_packs_before,
        "nqkv_ragged_cross_page_layer_launches": int(
            getattr(getattr(runner, "cache", None), "ragged_cross_page_layer_launch_count", 0)
        ) - ragged_cross_page_launches_before,
        "nqkv_ragged_multi_page_layer_rounds": int(
            getattr(getattr(runner, "cache", None), "ragged_multi_page_layer_round_count", 0)
        ) - ragged_multi_page_rounds_before,
        "nqkv_multi_page_batched_packs": int(
            getattr(getattr(runner, "cache", None), "multi_page_batched_pack_count", 0)
        ) - batched_pack_before,
    }
    dispatch_after = _w4a4_dispatch_counts(runner.model)
    item["w4a4_dispatch_count_delta"] = {
        key: dispatch_after[key] - dispatch_before[key]
        for key in dispatch_after
    }
    if (item["failed_requests"] or item["final_registered_requests"]
            or item["final_backend_cache_requests"]):
        raise RuntimeError(f"incomplete service benchmark run: {item}")
    return item


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/service_benchmark.v1.json"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--repetitions", type=int)
    parser.add_argument(
        "--compute-backend",
        choices=("cuda_mma_j2", "packed_w4a4_tiled", "packed_w4a4_cuda_mma"),
        default="cuda_mma_j2",
    )
    parser.add_argument(
        "--kv-backend",
        choices=("bf16_static", "nqkv4"),
        default="bf16_static",
        help="BF16 StaticCache or packed NQKV v2 paged KV4.",
    )
    parser.add_argument("--nqkv-page-size", type=int, default=16)
    parser.add_argument(
        "--tiled-fallback-names", nargs="+", default=(),
        help="Exact Linear module names forced to tiled Triton for W4A4 modes.",
    )
    splitk_group = parser.add_mutually_exclusive_group()
    splitk_group.add_argument(
        "--enable-m1-down-splitk", dest="enable_m1_down_splitk", action="store_true"
    )
    splitk_group.add_argument(
        "--disable-m1-down-splitk", dest="enable_m1_down_splitk", action="store_false"
    )
    parser.set_defaults(enable_m1_down_splitk=False)
    parser.add_argument(
        "--m1-down-splitk-fallback-name", action="append", default=[]
    )
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
    if args.policy:
        requested_policies = set(args.policy)
        config["policies"] = [
            policy for policy in config["policies"]
            if policy["name"] in requested_policies
        ]
        missing_policies = requested_policies - {
            policy["name"] for policy in config["policies"]
        }
        if missing_policies:
            raise ValueError(f"unknown policies: {sorted(missing_policies)}")
    if args.repetitions is not None:
        if args.repetitions < 1:
            raise ValueError("--repetitions must be positive")
        config["repetitions"] = args.repetitions
    if args.m1_down_splitk_fallback_name and not args.enable_m1_down_splitk:
        parser.error("split-K fallback names require --enable-m1-down-splitk")
    validate_benchmark_config(config)

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = _load_model(
        reader.model_metadata["model_id"], reader.model_metadata["model_revision"], args.offline
    )
    names = _projection_names(model, reader)
    fallback_names = tuple(dict.fromkeys(args.tiled_fallback_names))
    unknown_fallback_names = sorted(set(fallback_names) - set(names))
    if unknown_fallback_names:
        parser.error(
            "unknown --tiled-fallback-names: " + ", ".join(unknown_fallback_names)
        )
    conversion = convert_linears_to_fused_nqrt_(
        model, reader, names, output_dtype=torch.bfloat16,
        group_qkv=False, compute_backend=args.compute_backend,
        activation_q=14,
        activation_betas=(
            ACTIVATION_BETAS
            if args.compute_backend.startswith("packed_w4a4") else None
        ),
        w4a4_tiled_fallback_names=(
            fallback_names
            if args.compute_backend.startswith("packed_w4a4") else ()
        ),
        enable_m1_down_splitk=args.enable_m1_down_splitk,
        m1_down_splitk_fallback_names=args.m1_down_splitk_fallback_name,
    )
    if args.kv_backend == "nqkv4":
        head_dim = int(getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ))
        nqkv_cache = BatchedNQKVLlamaCache(
            capacity=int(config["capacity"]),
            num_layers=int(model.config.num_hidden_layers),
            num_query_heads=int(model.config.num_attention_heads),
            num_kv_heads=int(model.config.num_key_value_heads),
            head_dim=head_dim,
            max_cache_len=int(config["max_cache_len"]),
            page_size=args.nqkv_page_size,
            q=14,
            key_betas=KEY_BETAS,
            value_betas=VALUE_BETAS,
            dtype=torch.bfloat16,
            device="cuda",
        )
        install_batched_nqkv_llama_backend(model, nqkv_cache)
        runner = BatchedNQKVLlamaRunner(model, nqkv_cache)
    else:
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

    warmup_prompt_lengths = [
        int(length) for length in config.get("warmup_prompt_lengths", [])
    ]
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
    if warmup_prompt_lengths:
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
    if (
        warmup_prompt_lengths
        and config.get("warmup_each_policy", False)
        and isinstance(runner, BatchedNQKVLlamaRunner)
    ):
        for policy_index, policy in enumerate(config["policies"]):
            runner.cache.ragged_subpage_enabled = bool(
                policy.get("nqkv_ragged_subpage_enabled", True)
            )
            runner.cache.ragged_full_page_pack_enabled = bool(
                policy.get("nqkv_ragged_full_page_pack_enabled", True)
            )
            runner.cache.ragged_cross_page_enabled = bool(
                policy.get("nqkv_ragged_cross_page_enabled", True)
            )
            runner.cache.ragged_multi_page_enabled = bool(
                policy.get("nqkv_ragged_multi_page_enabled", True)
            )
            runner.cache.multi_page_batched_pack_enabled = bool(
                policy.get("nqkv_multi_page_batched_pack_enabled", True)
            )
            for length in warmup_prompt_lengths:
                for slot in range(int(config["capacity"])):
                    prompt = tuple(torch.randint(
                        1, model.config.vocab_size, (length,),
                        device="cuda", generator=generator,
                    ).tolist())
                    warmup.submit(DecodeRequest(
                        f"policy-shape-{policy_index}-{length}-{slot}", prompt, 1
                    ))
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
                if isinstance(runner, BatchedNQKVLlamaRunner):
                    runner.cache.ragged_subpage_enabled = bool(
                        policy.get("nqkv_ragged_subpage_enabled", True)
                    )
                    runner.cache.ragged_full_page_pack_enabled = bool(
                        policy.get("nqkv_ragged_full_page_pack_enabled", True)
                    )
                    runner.cache.ragged_cross_page_enabled = bool(
                        policy.get("nqkv_ragged_cross_page_enabled", True)
                    )
                    runner.cache.ragged_multi_page_enabled = bool(
                        policy.get("nqkv_ragged_multi_page_enabled", True)
                    )
                    runner.cache.multi_page_batched_pack_enabled = bool(
                        policy.get("nqkv_multi_page_batched_pack_enabled", True)
                    )
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
        "compute_backend": args.compute_backend,
        "kv_backend": args.kv_backend,
        "enable_m1_down_splitk": args.enable_m1_down_splitk,
        "m1_down_splitk_fallback_names": args.m1_down_splitk_fallback_name,
        "decode_execution": (
            "dual_cuda_graph_with_host_page_reservation"
            if args.kv_backend == "nqkv4"
            else "cuda_graph"
        ),
        "nqkv_page_size": (
            args.nqkv_page_size if args.kv_backend == "nqkv4" else None
        ),
        "tiled_fallback_names": list(conversion.w4a4_tiled_fallback_names),
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
