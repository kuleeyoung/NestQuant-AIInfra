"""Benchmark deterministic M2 split-K against the production small-M kernel."""

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
    _linear_packed_w4a4_mma_small_m_unsplit,
    linear_packed_w4a4_mma_small_m_splitk_probe,
)
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import quantize_activation_cuda


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
DEFAULT_WEIGHT = "model.layers.0.mlp.down_proj.weight"
SPLITS = (2, 4, 8)


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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.groups) < 1:
        parser.error("warmup, iterations, and groups must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("selected weight must be packed")
    shape = [2, record.feature_count, record.indices.shape[0]]
    if shape[1] % (16 * max(SPLITS)) or shape[2] % 16:
        raise ValueError(f"probe shape is not split-K compatible: {shape}")

    weight = prepare_fused_linear(record)
    generator = torch.Generator(device="cuda").manual_seed(20261102)
    inputs = torch.randn(
        2,
        record.feature_count,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
    operations = {
        "baseline": lambda: _linear_packed_w4a4_mma_small_m_unsplit(
            activation, weight
        ),
        **{
            f"split_{split}": (
                lambda split=split: linear_packed_w4a4_mma_small_m_splitk_probe(
                    activation, weight, split_k=split
                )
            )
            for split in SPLITS
        },
    }

    reference = operations["baseline"]()
    correctness = {}
    for split in SPLITS:
        name = f"split_{split}"
        actual = operations[name]()
        error = (actual.float() - reference.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), reference.float().flatten(), dim=0
        )
        correctness[name] = {
            "max_abs_error": float(error.max().item()),
            "mean_abs_error": float(error.mean().item()),
            "cosine_similarity": float(cosine.item()),
            "allclose_atol_0_05_rtol_0_01": bool(
                torch.allclose(actual.float(), reference.float(), atol=0.05, rtol=0.01)
            ),
            "finite": bool(torch.isfinite(actual).all()),
        }
        if not correctness[name]["allclose_atol_0_05_rtol_0_01"]:
            raise RuntimeError(f"{name} failed correctness: {correctness[name]}")

    names = list(operations)
    samples = {name: [] for name in names}
    for group in range(args.groups):
        offset = group % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            samples[name].append(_time(operations[name], args.warmup, args.iterations))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    baseline_us = medians["baseline"]
    speedups = {
        name: baseline_us / latency
        for name, latency in medians.items()
        if name != "baseline"
    }
    best_name = max(speedups, key=speedups.get)
    best_speedup = speedups[best_name]
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = operations[best_name]()
    graph.replay()
    torch.cuda.synchronize()
    graph_error = (graph_output.float() - reference.float()).abs()
    graph_samples = [
        _time(graph.replay, args.warmup, args.iterations)
        for _ in range(args.groups)
    ]
    graph_us = statistics.median(graph_samples)
    report = {
        "scope": "isolated deterministic M2 split-K down-projection probe",
        "weight": args.weight,
        "shape_mkn": shape,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "extension": str(Path(_load_extension().__file__).resolve()),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "groups": args.groups,
        "latency_us_median": medians,
        "latency_us_samples": samples,
        "speedup_vs_baseline": speedups,
        "best_candidate": best_name,
        "best_speedup": best_speedup,
        "cuda_graph": {
            "capture_succeeded": True,
            "latency_us_median": graph_us,
            "latency_us_samples": graph_samples,
            "speedup_vs_baseline": baseline_us / graph_us,
            "max_abs_error_vs_baseline": float(graph_error.max().item()),
            "mean_abs_error_vs_baseline": float(graph_error.mean().item()),
            "allclose_atol_0_05_rtol_0_01": bool(
                torch.allclose(
                    graph_output.float(), reference.float(), atol=0.05, rtol=0.01
                )
            ),
        },
        "decision": (
            "continue_to_graph_workspace_probe"
            if best_speedup >= 1.15
            else "stop_split_k_route"
        ),
        "correctness_vs_baseline": correctness,
        "claim_boundary": (
            "Includes temporary partial-buffer allocation and reduction, but is an "
            "isolated layer-shape benchmark; no production dispatch or end-to-end claim."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
