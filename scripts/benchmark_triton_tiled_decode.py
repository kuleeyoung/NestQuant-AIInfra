"""Benchmark reference, hybrid, and fused tiled NQRT v4 tensor decode."""

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
from nestquant.runtime.triton_tiled_decode import dequantize_tiled_triton


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
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
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--tile-blocks", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
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
        tiled = dequantize_tiled_triton(
            record, device=device, output_dtype=torch.bfloat16,
            tile_blocks=args.tile_blocks,
        )
        torch.cuda.synchronize(device)
        equal = bool(torch.equal(reference, tiled))
        max_error = float((reference.float() - tiled.float()).abs().max().item())
        del reference, tiled
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
        tiled_stats, tiled_peak = _measure(
            lambda: dequantize_tiled_triton(
                record, device=device, output_dtype=torch.bfloat16,
                tile_blocks=args.tile_blocks,
            ), args.repeats, device,
        )
        results.append({
            "weight_name": name,
            "shape": list(entry.shape),
            "reference": {
                "seconds": reference_stats,
                "cuda_peak_increment_bytes": reference_peak,
            },
            "hybrid": {
                "seconds": hybrid_stats,
                "cuda_peak_increment_bytes": hybrid_peak,
                "speedup_vs_reference": reference_stats["mean"] / hybrid_stats["mean"],
            },
            "tiled": {
                "seconds": tiled_stats,
                "cuda_peak_increment_bytes": tiled_peak,
                "speedup_vs_reference": reference_stats["mean"] / tiled_stats["mean"],
                "speedup_vs_hybrid": hybrid_stats["mean"] / tiled_stats["mean"],
            },
            "bitwise_equal": equal,
            "max_abs_error": max_error,
        })
    report = {
        "scope": "NQRT v4 BF16 tensor materialization; GEMM excluded",
        "checkpoint": str(args.checkpoint.resolve()),
        "rows_per_chunk": args.rows_per_chunk,
        "tile_blocks": args.tile_blocks,
        "repeats": args.repeats,
        "device_name": torch.cuda.get_device_name(device),
        "tensors": results,
        "claim_boundary": (
            "The tiled kernel fuses packed-stream reads, stateless dither, E8 "
            "reconstruction, beta selection, and row scaling. It still materializes "
            "the final BF16 weight and does not fuse GEMM."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
