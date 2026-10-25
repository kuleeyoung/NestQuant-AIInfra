"""Compare reference chunked decode with Triton-unpack hybrid decode."""

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
from nestquant.runtime.triton_decode import dequantize_with_triton_unpack


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values), "median": statistics.median(values),
        "min": min(values), "max": max(values),
    }


def _measure(operation, repeats: int, device: torch.device):
    warmup = operation()
    torch.cuda.synchronize(device)
    del warmup
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = operation()
        torch.cuda.synchronize(device)
        values.append(time.perf_counter() - started)
        del output
    return _stats(values), max(0, torch.cuda.max_memory_allocated(device) - baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weight_name", nargs="+")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    results = []
    for name in args.weight_name:
        entry, record = reader.read_named_record(name)
        if not isinstance(record, PackedNestQuantTensor):
            raise ValueError(f"not a packed record: {name}")
        reference = record.dequantize_chunked(
            device=device, output_dtype=torch.bfloat16,
            rows_per_chunk=args.rows_per_chunk,
        )
        hybrid = dequantize_with_triton_unpack(
            record, device=device, output_dtype=torch.bfloat16,
            rows_per_chunk=args.rows_per_chunk,
        )
        torch.cuda.synchronize(device)
        equal = bool(torch.equal(reference, hybrid))
        max_error = float((reference.float() - hybrid.float()).abs().max().item())
        del reference, hybrid
        reference_stats, reference_peak = _measure(
            lambda: record.dequantize_chunked(
                device=device, output_dtype=torch.bfloat16,
                rows_per_chunk=args.rows_per_chunk,
            ), args.repeats, device,
        )
        hybrid_stats, hybrid_peak = _measure(
            lambda: dequantize_with_triton_unpack(
                record, device=device, output_dtype=torch.bfloat16,
                rows_per_chunk=args.rows_per_chunk,
            ), args.repeats, device,
        )
        phase_events = []
        profiled = dequantize_with_triton_unpack(
            record, device=device, output_dtype=torch.bfloat16,
            rows_per_chunk=args.rows_per_chunk,
            profile_callback=phase_events.append,
        )
        torch.cuda.synchronize(device)
        del profiled
        results.append({
            "weight_name": name,
            "shape": list(entry.shape),
            "reference_seconds": reference_stats,
            "hybrid_seconds": hybrid_stats,
            "speedup": reference_stats["mean"] / hybrid_stats["mean"],
            "reference_cuda_peak_increment_bytes": reference_peak,
            "hybrid_cuda_peak_increment_bytes": hybrid_peak,
            "bitwise_equal": equal,
            "max_abs_error": max_error,
            "hybrid_phase_seconds": {
                event["phase"]: event["seconds"] for event in phase_events
            },
        })
    report = {
        "scope": "end-to-end BF16 decode with Triton unpack and reference E8 reconstruction",
        "checkpoint": str(args.checkpoint.resolve()),
        "rows_per_chunk": args.rows_per_chunk,
        "repeats": args.repeats,
        "device_name": torch.cuda.get_device_name(device),
        "tensors": results,
        "claim_boundary": (
            "Only fixed-width unpack is compiled. Dither generation, E8 nearest-point "
            "reconstruction, row scaling, and BF16 materialization remain PyTorch."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
