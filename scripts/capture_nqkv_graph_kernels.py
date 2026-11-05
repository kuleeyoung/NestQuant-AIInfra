"""Capture one normal and one boundary NQKV full-model Graph with Nsight Systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.nqkv_batched_hf_backend import (
    BatchedNQKVLlamaCache,
    install_batched_nqkv_llama_backend,
)
from nestquant.runtime.nqkv_batched_runner import BatchedNQKVLlamaRunner
from scripts.benchmark_full_nqrt_model import (
    ACTIVATION_BETAS,
    _load_model,
    _projection_names,
)
from scripts.benchmark_service_workloads import KEY_BETAS, VALUE_BETAS


@torch.inference_mode()
def _prepare_cycle(
    runner: BatchedNQKVLlamaRunner,
    model: torch.nn.Module,
    *,
    capacity: int,
    page_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    prompt_width = page_size - 2
    prompt = torch.randint(
        1,
        model.config.vocab_size,
        (capacity, prompt_width),
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(
        prompt_width, device="cuda", dtype=torch.long
    ).expand(capacity, -1)
    write_mask = torch.ones_like(prompt, dtype=torch.bool)
    prompt_lengths = torch.full(
        (capacity,), prompt_width, device="cuda", dtype=torch.long
    )
    runner.prefill_slots(
        prompt,
        positions=positions,
        write_mask=write_mask,
        valid_lengths=prompt_lengths,
    )
    active = torch.ones(
        (capacity, 1), device="cuda", dtype=torch.bool
    )
    normal_token = torch.randint(
        1,
        model.config.vocab_size,
        (capacity, 1),
        device="cuda",
        generator=generator,
    )
    normal_positions = torch.full_like(normal_token, prompt_width)
    normal_lengths = torch.full(
        (capacity,), prompt_width + 1, device="cuda", dtype=torch.long
    )
    boundary_token = torch.randint(
        1,
        model.config.vocab_size,
        (capacity, 1),
        device="cuda",
        generator=generator,
    )
    boundary_positions = torch.full_like(
        boundary_token, prompt_width + 1
    )
    boundary_lengths = torch.full(
        (capacity,), page_size, device="cuda", dtype=torch.long
    )
    return (
        normal_token,
        normal_positions,
        active,
        normal_lengths,
        boundary_token,
        boundary_positions,
        boundary_lengths,
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-cache-len", type=int, default=32)
    parser.add_argument("--warmup-cycles", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cuda-profiler-api", action="store_true")
    parser.add_argument("--torch-profiler-output", type=Path)
    parser.add_argument("--torch-profiler-summary", type=Path)
    parser.add_argument(
        "--compute-backend",
        choices=("packed_w4a4_tiled", "packed_w4a4_cuda_mma"),
        default="packed_w4a4_cuda_mma",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.capacity, args.warmup_cycles + 1) < 1:
        parser.error("capacity must be positive and warmup-cycles nonnegative")
    if args.page_size < 2 or args.max_cache_len < args.page_size:
        parser.error("page-size must be >=2 and fit inside max-cache-len")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = _load_model(
        reader.model_metadata["model_id"],
        reader.model_metadata["model_revision"],
        args.offline,
    )
    names = _projection_names(model, reader)
    convert_linears_to_fused_nqrt_(
        model,
        reader,
        names,
        output_dtype=torch.bfloat16,
        group_qkv=False,
        compute_backend=args.compute_backend,
        activation_q=14,
        activation_betas=ACTIVATION_BETAS,
    )
    head_dim = int(getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    ))
    cache = BatchedNQKVLlamaCache(
        capacity=args.capacity,
        num_layers=int(model.config.num_hidden_layers),
        num_query_heads=int(model.config.num_attention_heads),
        num_kv_heads=int(model.config.num_key_value_heads),
        head_dim=head_dim,
        max_cache_len=args.max_cache_len,
        page_size=args.page_size,
        q=14,
        key_betas=KEY_BETAS,
        value_betas=VALUE_BETAS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    install_batched_nqkv_llama_backend(model, cache)
    runner = BatchedNQKVLlamaRunner(model, cache)
    sample = torch.zeros((args.capacity, 1), device="cuda", dtype=torch.long)
    sample_positions = torch.zeros_like(sample)
    sample_active = torch.ones_like(sample, dtype=torch.bool)
    sample_lengths = torch.ones(args.capacity, device="cuda", dtype=torch.long)
    runner.prefill(sample)
    runner.capture(
        sample,
        positions=sample_positions,
        active_mask=sample_active,
        valid_lengths=sample_lengths,
    )
    clear = torch.ones(args.capacity, device="cuda", dtype=torch.bool)
    runner.clear_slots(clear)
    generator = torch.Generator(device="cuda").manual_seed(20261102)

    for _ in range(args.warmup_cycles):
        values = _prepare_cycle(
            runner,
            model,
            capacity=args.capacity,
            page_size=args.page_size,
            generator=generator,
        )
        runner.decode(
            values[0], positions=values[1], active_mask=values[2],
            valid_lengths=values[3],
        )
        runner.decode(
            values[4], positions=values[5], active_mask=values[2],
            valid_lengths=values[6],
        )
        runner.clear_slots(clear)
    torch.cuda.synchronize()

    values = _prepare_cycle(
        runner,
        model,
        capacity=args.capacity,
        page_size=args.page_size,
        generator=generator,
    )
    def run_profiled_decodes() -> None:
        torch.cuda.nvtx.range_push("nqkv_normal_decode_graph")
        with torch.profiler.record_function("nqkv_normal_decode_graph"):
            runner.decode(
                values[0], positions=values[1], active_mask=values[2],
                valid_lengths=values[3],
            )
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("nqkv_boundary_decode_graph")
        with torch.profiler.record_function("nqkv_boundary_decode_graph"):
            runner.decode(
                values[4], positions=values[5], active_mask=values[2],
                valid_lengths=values[6],
            )
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    profiler_summary = None
    if args.torch_profiler_output is not None:
        if args.torch_profiler_summary is None:
            parser.error(
                "--torch-profiler-summary is required with "
                "--torch-profiler-output"
            )
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            run_profiled_decodes()
        args.torch_profiler_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.torch_profiler_output))
        events = []
        for event in profiler.key_averages():
            device_total = float(getattr(event, "device_time_total", 0.0))
            self_device = float(getattr(event, "self_device_time_total", 0.0))
            if device_total <= 0.0 and self_device <= 0.0:
                continue
            events.append({
                "name": event.key,
                "count": int(event.count),
                "device_time_total_us": device_total,
                "self_device_time_total_us": self_device,
                "cpu_time_total_us": float(event.cpu_time_total),
            })
        events.sort(
            key=lambda item: item["self_device_time_total_us"], reverse=True
        )
        profiler_summary = {
            "status": "passed",
            "events": events,
            "event_count": len(events),
        }
        args.torch_profiler_summary.parent.mkdir(parents=True, exist_ok=True)
        args.torch_profiler_summary.write_text(
            json.dumps(profiler_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        if args.cuda_profiler_api:
            torch.cuda.cudart().cudaProfilerStart()
        run_profiled_decodes()
        if args.cuda_profiler_api:
            torch.cuda.cudart().cudaProfilerStop()
    runner.clear_slots(clear)

    payload = {
        "status": "passed",
        "protocol": "nsys_cuda_profiler_api_normal_and_boundary_graph",
        "model_id": reader.model_metadata["model_id"],
        "model_revision": reader.model_metadata["model_revision"],
        "checkpoint": str(args.checkpoint.resolve()),
        "compute_backend": args.compute_backend,
        "capacity": args.capacity,
        "page_size": args.page_size,
        "warmup_cycles": args.warmup_cycles,
        "torch_profiler_event_count": (
            profiler_summary["event_count"]
            if profiler_summary is not None else None
        ),
        "normal_nvtx_range": "nqkv_normal_decode_graph",
        "boundary_nvtx_range": "nqkv_boundary_decode_graph",
        "boundary_replays": runner.graph_boundary_replay_count,
        "boundary_fallbacks": runner.graph_boundary_fallback_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
