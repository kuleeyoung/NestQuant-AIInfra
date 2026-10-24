"""Benchmark a capacity-bounded LRU across multiple lazy NQMC Linear layers."""

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
from nestquant.core.rotation import hadamard
from nestquant.models.lazy_linear import LazyNQMCLinear, LazyNQMCWeightCache


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
    parser.add_argument("weight_names", nargs="+")
    parser.add_argument("--sequence", required=True,
                        help="comma-separated zero-based indices into weight_names")
    parser.add_argument("--capacity-mib", type=float, required=True)
    parser.add_argument("--input-rows", type=int, default=128)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-hadamard", action="store_true")
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.capacity_mib < 0 or args.input_rows < 1 or args.rows_per_chunk < 1:
        parser.error("capacity must be nonnegative and row counts must be positive")
    try:
        sequence = [int(value) for value in args.sequence.split(",")]
    except ValueError as error:
        parser.error(f"sequence must contain integers: {error}")
    if not sequence or any(index < 0 or index >= len(args.weight_names) for index in sequence):
        parser.error("sequence indices must address weight_names")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    entries = [reader.get_entry(name) for name in args.weight_names]
    in_features = entries[0].shape[1]
    if any(len(entry.shape) != 2 or entry.shape[1] != in_features for entry in entries):
        raise ValueError("all benchmark weights must be 2D with the same in_features")

    capacity_bytes = round(args.capacity_mib * 1024 * 1024)
    cache = LazyNQMCWeightCache(capacity_bytes)
    layers = [
        LazyNQMCLinear(
            reader, name, input_hadamard=args.input_hadamard,
            compute_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
            rows_per_chunk=args.rows_per_chunk, cache_policy="lru",
            weight_cache=cache,
        ).to(device)
        for name in args.weight_names
    ]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(
        args.input_rows, in_features, generator=generator, dtype=torch.bfloat16
    ).to(device)
    if device.type == "cuda":
        probe = torch.empty(1, device=device)
        _sync(device)
        del probe
        torch.cuda.empty_cache()

    # Warm up the complete decode/Hadamard/GEMM path, then remove its weight and
    # accounting so CUDA first-use initialization is not attributed to an LRU miss.
    warmup_started = time.perf_counter()
    warmup_output = layers[sequence[0]](inputs)
    _sync(device)
    warmup_seconds = time.perf_counter() - warmup_started
    del warmup_output
    cache.clear(reset_statistics=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    events = []
    retained_outputs: dict[int, torch.Tensor] = {}
    for step, index in enumerate(sequence):
        before = cache.snapshot()
        if device.type == "cuda":
            baseline_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = None
        _sync(device)
        started = time.perf_counter()
        output = layers[index](inputs)
        _sync(device)
        elapsed = time.perf_counter() - started
        after = cache.snapshot()
        retained_outputs.setdefault(index, output.detach().clone())
        events.append({
            "step": step,
            "layer_index": index,
            "weight_name": args.weight_names[index],
            "cache_hit": after["hits"] > before["hits"],
            "seconds": elapsed,
            "resident_bytes_after": after["resident_bytes"],
            "cumulative_evictions": after["evictions"],
            "cuda_peak_increment_bytes": (
                max(0, torch.cuda.max_memory_allocated(device) - baseline_allocated)
                if device.type == "cuda" else None
            ),
        })
        del output

    correctness = None
    if args.verify_reference:
        checks = []
        transformed = hadamard(inputs) if args.input_hadamard else inputs
        for index, actual in retained_outputs.items():
            weight = reader.decode_tensor(
                args.weight_names[index], device=device,
                output_dtype=torch.bfloat16, rows_per_chunk=None,
            )
            expected = torch.nn.functional.linear(transformed, weight)
            checks.append({
                "layer_index": index,
                "bitwise_equal": bool(torch.equal(actual, expected)),
                "max_abs_error": float((actual.float() - expected.float()).abs().max().item()),
            })
            del weight, expected
        correctness = checks

    hit_times = [event["seconds"] for event in events if event["cache_hit"]]
    miss_times = [event["seconds"] for event in events if not event["cache_hit"]]
    report = {
        "scope": "multi-layer capacity-bounded LRU over decoded NQMC BF16 weights",
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_names": args.weight_names,
        "sequence": sequence,
        "capacity_bytes": capacity_bytes,
        "input_shape": [args.input_rows, in_features],
        "rows_per_chunk": args.rows_per_chunk,
        "input_hadamard": args.input_hadamard,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "warmup_seconds_excluded": warmup_seconds,
        "events": events,
        "hit_seconds": _stats(hit_times) if hit_times else None,
        "miss_seconds": _stats(miss_times) if miss_times else None,
        "cache": cache.snapshot(),
        "correctness": correctness,
        "claim_boundary": (
            "Measures scheduling of materialized BF16 weights for selected layers. It is "
            "not full-model execution, host offload overlap, or packed low-bit GEMM."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
