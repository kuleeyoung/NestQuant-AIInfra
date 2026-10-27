"""Profile steady one-token decode for BF16 and packed CUDA MMA backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_full_nqrt_model import (
    _load_model,
    _projection_names,
)
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warm-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=4)
    parser.add_argument("--mode", choices=("bf16", "cuda_mma_j2"), default="cuda_mma_j2")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.warm_steps, args.profile_steps) < 1:
        parser.error("batch, sequence length, warm steps and profile steps must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    metadata = reader.model_metadata
    model = _load_model(metadata["model_id"], metadata["model_revision"], args.offline)
    names = _projection_names(model, reader)
    if args.mode != "bf16":
        convert_linears_to_fused_nqrt_(model, reader, names,
                                       output_dtype=torch.bfloat16,
                                       group_qkv=False, compute_backend=args.mode)
    generator = torch.Generator(device="cuda").manual_seed(20261025)
    input_ids = torch.randint(0, model.config.vocab_size,
                              (args.batch_size, args.sequence_length),
                              generator=generator, device="cuda")

    with torch.inference_mode():
        result = model(input_ids=input_ids, use_cache=True)
        cache = result.past_key_values
        token = result.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warm_steps):
            result = model(input_ids=token, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            token = result.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False,
                     with_stack=False) as prof:
            for _ in range(args.profile_steps):
                result = model(input_ids=token, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                token = result.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()

    events = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
    rows = [
        {"name": e.key, "calls": e.count,
         "self_gpu_us": round(e.self_device_time_total, 3),
         "self_cpu_us": round(e.self_cpu_time_total, 3)}
        for e in events if e.self_device_time_total > 0 or e.self_cpu_time_total > 0
    ]
    kernels = [r for r in rows if r["self_gpu_us"] > 0
               and not r["name"].startswith(("aten::", "cuda", "cu"))]
    kernel_gpu_us = sum(r["self_gpu_us"] for r in kernels)
    partial_gpu_us = sum(r["self_gpu_us"] for r in kernels
                         if r["name"].startswith("nqrt4_mma_j2_"))
    reduction_gpu_us = sum(r["self_gpu_us"] for r in kernels
                           if r["name"].startswith("nqrt4_splitk_reduce"))
    output = {
        "mode": args.mode, "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "profile_steps": args.profile_steps,
        "device": torch.cuda.get_device_name(),
        "top_events": rows[:80],
        "kernel_gpu_us_per_step": round(kernel_gpu_us / args.profile_steps, 3),
        "packed_partial_gpu_us_per_step": round(partial_gpu_us / args.profile_steps, 3),
        "splitk_reduction_gpu_us_per_step": round(reduction_gpu_us / args.profile_steps, 3),
        "packed_partial_fraction_of_kernel_time": round(partial_gpu_us / kernel_gpu_us, 4),
        "note": "Kernel times exclude aten wrappers. Sum is GPU busy time, not end-to-end latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
