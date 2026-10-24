"""Benchmark one lazy NQMC-backed Linear, including decode and BF16 GEMM."""

from __future__ import annotations

import argparse
import gc
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
from nestquant.core.rotation import hadamard
from nestquant.models.lazy_linear import LazyNQMCLinear


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weight_name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-rows", type=int, default=128)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cache-policy", choices=("none", "persistent"), default="none")
    parser.add_argument("--input-hadamard", action="store_true")
    parser.add_argument("--verify-checksum", action="store_true")
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input_rows < 1 or args.rows_per_chunk < 1 or args.repeats < 1:
        parser.error("input rows, chunk rows, and repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    reader_started = time.perf_counter()
    reader = NQMCFileReader(args.checkpoint, verify_checksum=args.verify_checksum)
    reader_seconds = time.perf_counter() - reader_started
    entry = reader.get_entry(args.weight_name)
    if len(entry.shape) != 2:
        raise ValueError("weight must be two-dimensional")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(
        args.input_rows, entry.shape[1], generator=generator, dtype=torch.bfloat16
    ).to(device)
    layer = LazyNQMCLinear(
        reader,
        args.weight_name,
        input_hadamard=args.input_hadamard,
        compute_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        rows_per_chunk=args.rows_per_chunk,
        cache_policy=args.cache_policy,
    ).to(device)

    # Initialize the CUDA context before measuring the first lazy forward.
    _sync(device)
    if device.type == "cuda":
        probe = torch.empty(1, device=device)
        _sync(device)
        del probe
        torch.cuda.empty_cache()

    reference = None
    if args.verify_reference:
        full_weight = reader.decode_tensor(
            args.weight_name, device=device, output_dtype=torch.bfloat16,
            rows_per_chunk=None,
        )
        transformed = hadamard(inputs) if args.input_hadamard else inputs
        reference = torch.nn.functional.linear(transformed, full_weight)
        _sync(device)
        del full_weight, transformed
        torch.cuda.empty_cache() if device.type == "cuda" else None

    latencies = []
    peak_increments = []
    equality = None
    max_abs_error = None
    checksum = None
    for iteration in range(args.repeats + 1):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            baseline_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = None
        _sync(device)
        started = time.perf_counter()
        output = layer(inputs)
        _sync(device)
        elapsed = time.perf_counter() - started
        if iteration == 0:
            cold_seconds = elapsed
            cold_peak_increment = (
                max(0, torch.cuda.max_memory_allocated(device) - baseline_allocated)
                if device.type == "cuda" else None
            )
            if reference is not None:
                equality = bool(torch.equal(output, reference))
                max_abs_error = float((output.float() - reference.float()).abs().max().item())
        else:
            latencies.append(elapsed)
            if device.type == "cuda":
                peak_increments.append(max(
                    0, torch.cuda.max_memory_allocated(device) - baseline_allocated
                ))
        checksum = float(output.float().sum().item())
        del output

    report = {
        "scope": "single-layer lazy NQMC decode plus PyTorch BF16 Linear",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": reader.file_bytes,
        "weight_name": args.weight_name,
        "weight_shape": list(entry.shape),
        "record_bytes": entry.length,
        "input_shape": [args.input_rows, entry.shape[1]],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "compute_dtype": "bfloat16",
        "rows_per_chunk": args.rows_per_chunk,
        "cache_policy": args.cache_policy,
        "input_hadamard": args.input_hadamard,
        "reader_initialization_seconds": reader_seconds,
        "payload_checksum_verified": args.verify_checksum,
        "cold_forward_seconds": cold_seconds,
        "cold_cuda_peak_increment_bytes": cold_peak_increment,
        "repeated_forward_seconds": _stats(latencies),
        "repeated_cuda_peak_increment_bytes": (
            _stats([float(value) for value in peak_increments])
            if peak_increments else None
        ),
        "decoded_weight_cached_after_run": layer.cached,
        "decoded_weight_cache_bytes": layer.cached_weight_bytes,
        "output_checksum": checksum,
        "reference_verified": args.verify_reference,
        "reference_bitwise_equal": equality,
        "reference_max_abs_error": max_abs_error,
        "claim_boundary": (
            "Includes file read, unpack/dequantization, optional online Hadamard, and an "
            "ordinary PyTorch BF16 Linear according to cache policy. It is not packed "
            "low-bit GEMM, full-model latency, throughput, or end-to-end generation."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
