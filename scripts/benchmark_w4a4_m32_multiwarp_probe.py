"""Benchmark the fixed M32 four-warp packed W4A4 Tensor Core probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from nestquant.runtime.cuda_w4a4_cute import (
    _load_cute_extension,
    linear_packed_w4a4_cute_large_m_probe,
)

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import (
    _load_extension,
    linear_packed_w4a4_mma_large_m_multiwarp_probe,
    linear_packed_w4a4_mma_large_m_staged,
    linear_packed_w4a4_mma_chunked_staged_probe,
    linear_packed_w4a4_mma_m128_staged_probe,
    linear_packed_w4a4_mma_m128_pipeline_probe,
)
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import (
    linear_packed_w4a4_tiled_triton,
    quantize_activation_cuda,
    select_w4a4_tiled_config,
)


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
DEFAULT_WEIGHT = "model.layers.0.self_attn.q_proj.weight"


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
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument(
        "--implementation",
        choices=("unstaged", "staged", "pipeline", "cute", "chunked"),
        default="unstaged"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument(
        "--compare-staged", action="store_true",
        help="For the CuTe probe, add a same-process staged CUDA baseline.",
    )
    parser.add_argument(
        "--compare-existing", action="store_true",
        help="For the CuTe probe, add the matching existing CUDA baseline.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.groups) < 1:
        parser.error("warmup, iterations, and groups must be positive")
    if args.compare_staged and args.implementation != "cute":
        parser.error("--compare-staged requires --implementation cute")
    if args.compare_staged and args.rows != 128:
        parser.error("--compare-staged requires --rows 128")
    if args.compare_existing and args.implementation != "cute":
        parser.error("--compare-existing requires --implementation cute")
    if args.compare_existing and args.compare_staged:
        parser.error("select only one same-process CUDA baseline")
    if args.rows < 1:
        parser.error("--rows must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("selected weight must be packed")
    shape = [args.rows, record.feature_count, record.indices.shape[0]]
    if shape[1] % 16 or shape[2] % 16:
        raise ValueError(f"probe requires K and N divisible by 16, got {shape}")
    weight = prepare_fused_linear(record)
    generator = torch.Generator(device="cuda").manual_seed(20261101)
    inputs = torch.randn(
        args.rows, record.feature_count, generator=generator, device="cuda",
        dtype=torch.bfloat16,
    )
    activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
    if args.implementation in {"staged", "pipeline", "cute", "chunked"}:
        if args.implementation == "pipeline" and args.rows != 128:
            parser.error("pipeline implementation requires --rows 128")
        if args.implementation in {"staged", "cute"} and args.rows not in (32, 64, 128):
            parser.error("staged/cute implementations require M32/M64/M128")
        operations = {
            "staged": linear_packed_w4a4_mma_large_m_staged,
            "pipeline": linear_packed_w4a4_mma_m128_pipeline_probe,
            "cute": linear_packed_w4a4_cute_large_m_probe,
            "chunked": linear_packed_w4a4_mma_chunked_staged_probe,
        }
        operation = operations[args.implementation]
        probe = lambda: operation(activation, weight)
    else:
        probe = lambda: linear_packed_w4a4_mma_large_m_multiwarp_probe(
            activation, weight
        )
    triton_config = select_w4a4_tiled_config(args.rows, shape[2], shape[1])
    triton = lambda: linear_packed_w4a4_tiled_triton(
        activation, weight, **triton_config
    )
    actual, reference = probe(), triton()
    error = (actual.float() - reference.float()).abs()
    if not torch.allclose(actual.float(), reference.float(), atol=0.04, rtol=0.01):
        raise RuntimeError(
            f"M32 probe failed correctness: max={error.max().item()} "
            f"mean={error.mean().item()}"
        )

    probe_samples, triton_samples, staged_samples = [], [], []
    if args.compare_staged or args.compare_existing:
        baseline_operation = (
            linear_packed_w4a4_mma_m128_staged_probe
            if args.rows == 128 else
            linear_packed_w4a4_mma_large_m_multiwarp_probe
        )
        staged = lambda: baseline_operation(activation, weight)
        staged_actual = staged()
        if not torch.equal(actual, staged_actual):
            raise RuntimeError("CuTe and existing CUDA outputs are not bit-exact")
        labeled = (("probe", probe), ("staged", staged), ("triton", triton))
        sample_map = {
            "probe": probe_samples,
            "staged": staged_samples,
            "triton": triton_samples,
        }
        for group in range(args.groups):
            ordered = labeled[group % 3:] + labeled[:group % 3]
            for label, timed_operation in ordered:
                sample_map[label].append(
                    _time(timed_operation, args.warmup, args.iterations)
                )
    else:
        for group in range(args.groups):
            operations = (probe, triton) if group % 2 == 0 else (triton, probe)
            first = _time(operations[0], args.warmup, args.iterations)
            second = _time(operations[1], args.warmup, args.iterations)
            if group % 2 == 0:
                probe_samples.append(first); triton_samples.append(second)
            else:
                triton_samples.append(first); probe_samples.append(second)
    probe_us = statistics.median(probe_samples)
    triton_us = statistics.median(triton_samples)
    report = {
        "scope": f"fixed-M{args.rows} multi-warp structure probe",
        "implementation": args.implementation,
        "weight": args.weight,
        "shape_mkn": shape,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "extension": str(Path(
            (_load_cute_extension() if args.implementation == "cute" else
             _load_extension()).__file__
        ).resolve()),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "groups": args.groups,
        "triton_config": triton_config,
        "probe_us": statistics.median(probe_samples),
        "triton_us": statistics.median(triton_samples),
        "speedup_vs_triton": triton_us / probe_us,
        "probe_us_samples": probe_samples,
        "triton_us_samples": triton_samples,
        "staged_us": (
            statistics.median(staged_samples) if staged_samples else None
        ),
        "speedup_vs_staged": (
            statistics.median(staged_samples) / probe_us
            if staged_samples else None
        ),
        "staged_us_samples": staged_samples,
        "cuda_baseline": (
            "staged_m128" if staged_samples and args.rows == 128 else
            "unstaged_large_m" if staged_samples else None
        ),
        "max_abs_error_vs_triton": float(error.max().item()),
        "mean_abs_error_vs_triton": float(error.mean().item()),
        "finite": bool(torch.isfinite(actual).all()),
        "claim_boundary": (
            f"Production-dispatched M{args.rows} decoded-weight shared-memory "
            "staging result; two full-CTA barriers per K=16 tile and no "
            "double buffering."
            if args.implementation == "staged" else
            "Isolated M128 double-buffered warp-specialized pipeline result; "
            "no CuTe dependency and no production dispatch."
            if args.implementation == "pipeline" else
            "Isolated M128 CUTLASS CuTe layout result; no production dispatch."
            if args.implementation == "cute" else
            "Arbitrary-M chunk-plus-padding staged CUDA prototype; no production "
            "dispatch and no claim until measured against tiled Triton."
            if args.implementation == "chunked" else
            "Isolated large-M structure result. Weight decode/read is still "
            "repeated per warp; no shared-memory pipeline and no production dispatch."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
