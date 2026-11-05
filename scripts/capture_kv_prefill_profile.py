"""Capture a fixed-shape packed-W4A4 prefill for BF16-KV or NQKV4."""

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
from nestquant.runtime.ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from scripts.benchmark_full_nqrt_model import (
    ACTIVATION_BETAS,
    _load_model,
    _projection_names,
)
from scripts.benchmark_service_workloads import KEY_BETAS, VALUE_BETAS


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--kv-backend", choices=("nqkv4", "bf16_static"), required=True)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=15)
    parser.add_argument("--max-cache-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

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
        compute_backend="packed_w4a4_cuda_mma",
        activation_q=14,
        activation_betas=ACTIVATION_BETAS,
    )
    if args.kv_backend == "nqkv4":
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
            page_size=16,
            q=14,
            key_betas=KEY_BETAS,
            value_betas=VALUE_BETAS,
            dtype=torch.bfloat16,
            device="cuda",
        )
        install_batched_nqkv_llama_backend(model, cache)
        runner = BatchedNQKVLlamaRunner(model, cache)
    else:
        runner = NQRTRaggedGraphDecodeRunner(
            model,
            batch_size=args.capacity,
            max_cache_len=args.max_cache_len,
        )

    sample = torch.ones((args.capacity, 1), dtype=torch.long, device="cuda")
    runner.prefill(sample)
    runner.clear_slots(torch.ones(args.capacity, dtype=torch.bool, device="cuda"))
    generator = torch.Generator(device="cuda").manual_seed(20261102)
    input_ids = torch.randint(
        1,
        model.config.vocab_size,
        (args.capacity, args.prompt_length),
        generator=generator,
        device="cuda",
    )
    positions = torch.arange(
        args.prompt_length, device="cuda", dtype=torch.long
    ).expand(args.capacity, -1)
    write_mask = torch.ones_like(input_ids, dtype=torch.bool)
    valid_lengths = torch.full(
        (args.capacity,), args.prompt_length, dtype=torch.long, device="cuda"
    )
    clear = torch.ones(args.capacity, dtype=torch.bool, device="cuda")

    def run_prefill() -> None:
        runner.prefill_slots(
            input_ids,
            positions=positions,
            write_mask=write_mask,
            valid_lengths=valid_lengths,
        )

    for _ in range(args.warmup):
        run_prefill()
        torch.cuda.synchronize()
        runner.clear_slots(clear)
        torch.cuda.synchronize()
    # Keep cache clearing and other warmup work outside the profiling window.
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=False,
        profile_memory=False,
    ) as profiler:
        with torch.profiler.record_function(
            f"{args.kv_backend}_prefill_b{args.capacity}_p{args.prompt_length}"
        ):
            run_prefill()
            torch.cuda.synchronize()
    runner.clear_slots(clear)

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace))
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
    events.sort(key=lambda item: item["self_device_time_total_us"], reverse=True)
    payload = {
        "status": "passed",
        "kv_backend": args.kv_backend,
        "capacity": args.capacity,
        "prompt_length": args.prompt_length,
        "events": events,
        "event_count": len(events),
        "claim_boundary": "Single profiled prefill after warmup; profiler timing is diagnostic.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "events"}, indent=2))


if __name__ == "__main__":
    main()
