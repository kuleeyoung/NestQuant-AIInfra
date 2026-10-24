"""Benchmark named, single-tensor NQMC reads and selective dequantization."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable; run this benchmark on Linux/WSL2")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tensor", nargs="+", help="state-dict tensor name or tied alias")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rows-per-chunk", type=int)
    parser.add_argument(
        "--output-dtype", choices=("record", "float16", "bfloat16", "float32", "float64"),
        default="record",
    )
    parser.add_argument("--verify-checksum", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats must be positive")
    if args.rows_per_chunk is not None and args.rows_per_chunk < 1:
        parser.error("--rows-per-chunk must be positive")
    device = torch.device(args.device)
    output_dtype = (
        None if args.output_dtype == "record" else getattr(torch, args.output_dtype)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    validation_started = time.perf_counter()
    reader = NQMCFileReader(args.checkpoint, verify_checksum=args.verify_checksum)
    validation_seconds = time.perf_counter() - validation_started
    cuda_context_warmup_seconds = None
    if device.type == "cuda":
        cuda_started = time.perf_counter()
        probe = torch.empty(1, device=device)
        torch.cuda.synchronize(device)
        del probe
        torch.cuda.empty_cache()
        cuda_context_warmup_seconds = time.perf_counter() - cuda_started
    results = []
    for requested_name in args.tensor:
        entry = reader.get_entry(requested_name)
        canonical_name = reader.canonical_name(requested_name)
        read_times = []
        decode_times = []
        output_bytes = None
        decoded_dtype = None
        baseline_rss = _rss_bytes()
        peak_rss = baseline_rss
        stop_sampling = threading.Event()

        def sample_rss() -> None:
            nonlocal peak_rss
            while not stop_sampling.wait(0.005):
                peak_rss = max(peak_rss, _rss_bytes())

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(args.warmup + args.repeats):
            gc.collect()
            read_started = time.perf_counter()
            _loaded_entry, record = reader.read_named_record(requested_name)
            read_seconds = time.perf_counter() - read_started
            _sync(device)
            decode_started = time.perf_counter()
            if isinstance(record, PackedNestQuantTensor):
                value = (
                    record.dequantize(device=device)
                    if args.rows_per_chunk is None else
                    record.dequantize_chunked(
                        device=device, output_dtype=output_dtype,
                        rows_per_chunk=args.rows_per_chunk,
                    )
                )
                if args.rows_per_chunk is None and output_dtype is not None:
                    value = value.to(output_dtype)
            else:
                value = record.to(device=device, dtype=output_dtype or record.dtype)
            _sync(device)
            decode_seconds = time.perf_counter() - decode_started
            output_bytes = value.numel() * value.element_size()
            decoded_dtype = str(value.dtype).removeprefix("torch.")
            peak_rss = max(peak_rss, _rss_bytes())
            if iteration >= args.warmup:
                read_times.append(read_seconds)
                decode_times.append(decode_seconds)
            del value, record
        stop_sampling.set()
        sampler.join()
        peak_rss = max(peak_rss, _rss_bytes())
        results.append({
            "requested_name": requested_name,
            "canonical_name": canonical_name,
            "resolved_alias": requested_name != canonical_name,
            "kind": entry.kind,
            "shape": list(entry.shape),
            "dtype": str(entry.dtype).removeprefix("torch."),
            "decoded_dtype": decoded_dtype,
            "record_bytes": entry.length,
            "decoded_tensor_bytes": output_bytes,
            "read_seconds": _stats(read_times),
            "decode_seconds": _stats(decode_times),
            "rss_peak_increment_bytes": max(0, peak_rss - baseline_rss),
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        })
    report = {
        "scope": "named single-tensor NQMC read and selective decode benchmark",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": reader.file_bytes,
        "payload_checksum_verified": args.verify_checksum,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows_per_chunk": args.rows_per_chunk,
        "requested_output_dtype": args.output_dtype,
        "reader_initialization_seconds": validation_seconds,
        "cuda_context_warmup_seconds_excluded_from_tensor_measurements": (
            cuda_context_warmup_seconds
        ),
        "tensors": results,
        "claim_boundary": (
            "Measures seek/read and standalone tensor dequantization only; it does not "
            "measure GEMM, end-to-end inference, lazy-layer scheduling, or packed execution."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
