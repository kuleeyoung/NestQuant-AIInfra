"""Benchmark real packed W4A4 projection shapes from one NQRT4 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import (
    FusedNQRTLinear,
    convert_linears_to_fused_nqrt_,
)
from nestquant.core.rotation import hadamard
from nestquant.runtime.cuda_w4a4_mma import (
    can_use_w4a4_gemv_m1,
    can_use_w4a4_mma_large_m_staged,
    can_use_w4a4_mma_m16,
    can_use_w4a4_mma_small_m,
    linear_packed_w4a4_gemv_m1,
    linear_packed_w4a4_mma_large_m_staged,
    linear_packed_w4a4_mma_m16,
    linear_packed_w4a4_mma_small_m,
)
from nestquant.runtime.triton_w4a4_linear import quantize_activation_cuda
from scripts.benchmark_full_nqrt_model import (
    ACTIVATION_BETAS,
    _load_model,
    _projection_names,
)


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _clear_input_caches(module: FusedNQRTLinear) -> None:
    if module.input_transform_cache is not None:
        module.input_transform_cache.clear()
    if module.packed_activation_cache is not None:
        module.packed_activation_cache.clear()


@torch.inference_mode()
def _benchmark(
    module: FusedNQRTLinear,
    *,
    rows: int,
    warmups: int,
    iterations: int,
) -> dict[str, object]:
    inputs = torch.randn(
        (rows, module.in_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for _ in range(warmups):
        _clear_input_caches(module)
        module(inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        _clear_input_caches(module)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = module(inputs)
        end.record()
        end.synchronize()
        if not torch.isfinite(output).all():
            raise RuntimeError("projection produced non-finite output")
        samples.append(float(start.elapsed_time(end)))
    return {
        "rows": rows,
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


@torch.inference_mode()
def _benchmark_kernel_only(
    module: FusedNQRTLinear,
    *,
    rows: int,
    warmups: int,
    iterations: int,
) -> dict[str, object]:
    inputs = torch.randn(
        (rows, module.in_features), device="cuda", dtype=torch.bfloat16
    )
    transformed = hadamard(inputs) if module.input_hadamard else inputs
    packed = quantize_activation_cuda(
        transformed,
        q=module.activation_q,
        betas=module.activation_betas,
        beta_table=module.activation_beta_table,
    )
    prepared = module._prepared_for(inputs.device)
    if can_use_w4a4_mma_large_m_staged(packed, prepared):
        backend = "large_m_staged"
        launch = lambda: linear_packed_w4a4_mma_large_m_staged(packed, prepared)
    elif can_use_w4a4_mma_m16(packed, prepared):
        backend = "m16"
        launch = lambda: linear_packed_w4a4_mma_m16(packed, prepared)
    elif can_use_w4a4_mma_small_m(packed, prepared):
        backend = "small_m"
        launch = lambda: linear_packed_w4a4_mma_small_m(packed, prepared)
    elif can_use_w4a4_gemv_m1(packed, prepared):
        backend = "gemv_m1"
        launch = lambda: linear_packed_w4a4_gemv_m1(packed, prepared)
    else:
        raise RuntimeError("projection shape did not select a CUDA W4A4 backend")
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = launch()
        end.record()
        end.synchronize()
        if not torch.isfinite(output).all():
            raise RuntimeError("projection kernel produced non-finite output")
        samples.append(float(start.elapsed_time(end)))
    return {
        "rows": rows,
        "backend": backend,
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 2, 128))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if min((*args.rows, args.warmups, args.iterations)) < 1:
        parser.error("rows, warmups, and iterations must be positive")

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
    selected: dict[str, tuple[str, FusedNQRTLinear]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, FusedNQRTLinear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix in PROJECTIONS and suffix not in selected:
            selected[suffix] = (name, module)
    missing = set(PROJECTIONS) - set(selected)
    if missing:
        raise RuntimeError(f"missing fused projections: {sorted(missing)}")

    records = []
    for suffix in PROJECTIONS:
        name, module = selected[suffix]
        measurements = [
            _benchmark(
                module,
                rows=rows,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            for rows in args.rows
        ]
        kernel_measurements = [
            _benchmark_kernel_only(
                module,
                rows=rows,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            for rows in args.rows
        ]
        records.append({
            "projection": suffix,
            "module_name": name,
            "weight_name": module.weight_name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "weight_elements": module.in_features * module.out_features,
            "measurements": measurements,
            "kernel_measurements": kernel_measurements,
        })
        print(
            suffix,
            f"[{module.out_features}, {module.in_features}]",
            ", ".join(
                f"M{item['rows']}={item['median_ms']:.3f}ms"
                for item in measurements
            ),
            "kernel",
            ", ".join(
                f"M{item['rows']}={item['median_ms']:.3f}ms/{item['backend']}"
                for item in kernel_measurements
            ),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "status": "passed",
        "protocol": "isolated_real_checkpoint_projection_cold_input_caches",
        "checkpoint": str(args.checkpoint),
        "warmups": args.warmups,
        "iterations": args.iterations,
        "rows": args.rows,
        "records": records,
        "claim_boundary": (
            "One first-layer projection per suffix with Hadamard/A4 input caches "
            "cleared before every timed call; use for shape ranking, not service latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
