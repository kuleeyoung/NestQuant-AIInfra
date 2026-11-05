"""Compare production and packed-word direct W4A4 M1 CUDA decoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import (
    _load_extension,
    linear_packed_w4a4_gemv_m1,
    linear_packed_w4a4_gemv_m1_legacy,
)
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import quantize_activation_cuda


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
DEFAULT_WEIGHTS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
)


def _time(operation, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--weights", nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument(
        "--nvtx-iterations", type=int, default=0,
        help="emit labelled production/probe loops for nsys after timing",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.groups) < 1:
        parser.error("warmup, iterations, and groups must be positive")
    if args.nvtx_iterations < 0:
        parser.error("nvtx-iterations must be non-negative")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    generator = torch.Generator(device="cuda").manual_seed(20261030)
    rows = []
    nvtx_operations = []
    for name in args.weights:
        _, record = reader.read_named_record(name)
        if not isinstance(record, PackedNestQuantTensor):
            raise TypeError(f"{name} is not packed")
        weight = prepare_fused_linear(record)
        inputs = torch.randn(
            1, record.feature_count, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        production = lambda: linear_packed_w4a4_gemv_m1(activation, weight)
        legacy = lambda: linear_packed_w4a4_gemv_m1_legacy(activation, weight)
        expected, actual = legacy(), production()
        error = (actual.float() - expected.float()).abs()
        if not torch.equal(actual, expected):
            raise RuntimeError(
                f"packed-word probe changed {name}: max={error.max().item()}"
            )
        production_samples, probe_samples = [], []
        for group in range(args.groups):
            operations = (legacy, production) if group % 2 == 0 else (production, legacy)
            first = _time(operations[0], args.warmup, args.iterations)
            second = _time(operations[1], args.warmup, args.iterations)
            if group % 2 == 0:
                production_samples.append(first); probe_samples.append(second)
            else:
                probe_samples.append(first); production_samples.append(second)
        production_median = statistics.median(production_samples)
        probe_median = statistics.median(probe_samples)
        rows.append({
            "weight": name,
            "shape": [1, record.feature_count, record.indices.shape[0]],
            "legacy_us": production_median,
            "production_packed_word_us": probe_median,
            "speedup_vs_legacy": production_median / probe_median,
            "legacy_us_samples": production_samples,
            "production_packed_word_us_samples": probe_samples,
            "max_abs_error_vs_legacy": float(error.max().item()),
            "mean_abs_error_vs_legacy": float(error.mean().item()),
        })
        nvtx_operations.append((name, legacy, production))

    if args.nvtx_iterations:
        torch.cuda.synchronize()
        for name, legacy, production in nvtx_operations:
            safe_name = name.replace(".", "_")
            torch.cuda.nvtx.range_push(f"legacy_{safe_name}")
            for _ in range(args.nvtx_iterations):
                legacy()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"production_packed_word_{safe_name}")
            for _ in range(args.nvtx_iterations):
                production()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    extension_path = str(Path(_load_extension().__file__).resolve())
    report = {
        "scope": "promoted packed-word M1 versus retained legacy M1",
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "extension_path": extension_path,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "groups": args.groups,
        "rows": rows,
        "claim_boundary": (
            "Kernel-only A/B; packed-word M1 is connected to production "
            "runtime dispatch and legacy M1 is retained only as a baseline."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
