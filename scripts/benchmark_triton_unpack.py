"""Benchmark reference versus Triton unpack on one real NQMC tensor record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_unpack import unpack_indices_triton


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _measure(operation, repeats: int, device: torch.device):
    operation()
    torch.cuda.synchronize(device)
    times = []
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    last = None
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        last = operation()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - started)
        del last
    return _stats(times), max(0, torch.cuda.max_memory_allocated(device) - baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weight_name")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    device = torch.device(args.device)
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    entry, record = reader.read_named_record(args.weight_name)
    if not isinstance(record, PackedNestQuantTensor):
        raise ValueError("weight must be a packed NestQuant record")

    expected_codes, expected_betas = record.indices.unpack(device=device)
    actual_codes, actual_betas = unpack_indices_triton(
        record.indices, device=device, output_dtype=torch.int64
    )
    torch.cuda.synchronize(device)
    if not torch.equal(actual_codes, expected_codes) or not torch.equal(
        actual_betas, expected_betas
    ):
        raise RuntimeError("Triton unpack disagrees with the reference")
    del expected_codes, expected_betas, actual_codes, actual_betas

    reference_stats, reference_peak = _measure(
        lambda: record.indices.unpack(device=device), args.repeats, device
    )
    triton_i64_stats, triton_i64_peak = _measure(
        lambda: unpack_indices_triton(
            record.indices, device=device, output_dtype=torch.int64
        ),
        args.repeats, device,
    )
    triton_i16_stats, triton_i16_peak = _measure(
        lambda: unpack_indices_triton(
            record.indices, device=device, output_dtype=torch.int16
        ),
        args.repeats, device,
    )
    code_count = record.indices.shape[0] * record.indices.shape[1] * 8
    beta_count = record.indices.shape[0] * record.indices.shape[1]
    report = {
        "scope": "real NQPK full-stream unpack including packed-payload H2D upload",
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_name": args.weight_name,
        "shape": list(entry.shape),
        "q": record.indices.q,
        "code_bit_width": record.indices.code_bit_width,
        "beta_count": record.indices.beta_count,
        "beta_bit_width": record.indices.beta_index_bit_width,
        "packed_payload_bytes": (
            len(record.indices.code_payload) + len(record.indices.beta_payload)
        ),
        "expanded_value_count": code_count + beta_count,
        "repeats": args.repeats,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "reference_int64": {
            "seconds": reference_stats,
            "cuda_peak_increment_bytes": reference_peak,
        },
        "triton_int64": {
            "seconds": triton_i64_stats,
            "cuda_peak_increment_bytes": triton_i64_peak,
            "speedup_vs_reference_mean": (
                reference_stats["mean"] / triton_i64_stats["mean"]
            ),
        },
        "triton_int16": {
            "seconds": triton_i16_stats,
            "cuda_peak_increment_bytes": triton_i16_peak,
            "speedup_vs_reference_mean": (
                reference_stats["mean"] / triton_i16_stats["mean"]
            ),
        },
        "correctness": "Triton int64 codes and beta IDs are bitwise equal to reference",
        "claim_boundary": (
            "This measures fixed-width unpack and payload upload only. It excludes "
            "dither replay, E8 reconstruction, row scaling, BF16 output, and GEMM."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
