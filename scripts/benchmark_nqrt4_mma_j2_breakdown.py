"""Single-variable timing breakdown of the J2 partial kernel on real weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from scripts.benchmark_nqrt4_mma_j2_checkpoint import DEFAULT_CHECKPOINT, DEFAULT_WEIGHTS, validate_record

MODES = {
    "full_decode": 0,
    "packed_affine_proxy": 1,
    "mma_floor": 2,
    "skip_first_e8": 3,
    "skip_second_e8": 4,
}


def measure(operation, warmup: int, iterations: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}


def auto_splits(k_size: int, n_size: int) -> int:
    splits = 64 if k_size >= 8192 and n_size <= 2048 else (8 if n_size >= 8192 else 16)
    return min(splits, k_size // 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokens", type=int, choices=(1, 16), default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("invalid timing counts")

    extension = load(
        name="nqrt4_mma_j2_breakdown_ext",
        sources=[str(ROOT / "nestquant/runtime/nqrt4_mma_j2_breakdown_binding.cpp"),
                 str(ROOT / "nestquant/runtime/nqrt4_mma_j2_breakdown.cu")],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas=-v"], verbose=False,
    )
    production = load(
        name="nqrt4_mma_j2_ext",
        sources=[str(ROOT / "nestquant/runtime/nqrt4_mma_j2_binding.cpp"),
                 str(ROOT / "nestquant/runtime/nqrt4_mma_j2.cu")],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False,
    )
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    results = []
    for weight_name in DEFAULT_WEIGHTS:
        _, record = reader.read_named_record(weight_name)
        if not isinstance(record, PackedNestQuantTensor):
            raise TypeError(weight_name)
        validate_record(record)
        prepared = prepare_fused_linear(record)
        n_size, _, _ = record.indices.shape
        k_size = record.feature_count
        inputs = torch.randn((args.tokens, k_size), device="cuda", dtype=torch.bfloat16,
                             generator=generator)
        splits = auto_splits(k_size, n_size)
        diagnostic_full = extension.run(
            inputs, prepared.code_payload, prepared.beta_payload, prepared.row_norms,
            prepared.beta_table, 0, record.dither_seed, splits, MODES["full_decode"])
        production_run = production.run_m1_splitk if args.tokens == 1 else production.run_m16_splitk
        production_full = production_run(
            inputs, prepared.code_payload, prepared.beta_payload, prepared.row_norms,
            prepared.beta_table, 0, record.dither_seed, splits)
        # The diagnostic stores the same split partials. PyTorch's tree sum can
        # round differently from the production kernel's ascending FP32 loop.
        diagnostic_reduced = diagnostic_full.sum(dim=0)
        validation_max_abs = float((diagnostic_reduced - production_full).abs().max().item())
        timings = {}
        for label, mode in MODES.items():
            timings[label] = measure(
                lambda mode=mode: extension.run(
                    inputs, prepared.code_payload, prepared.beta_payload,
                    prepared.row_norms, prepared.beta_table, 0, record.dither_seed,
                    splits, mode),
                args.warmup, args.iterations, args.repeats,
            )
        full = timings["full_decode"]["median_us"]
        proxy = timings["packed_affine_proxy"]["median_us"]
        floor = timings["mma_floor"]["median_us"]
        results.append({
            "weight_name": weight_name, "shape": [n_size, k_size], "splits": splits,
            "full_variant_max_abs_vs_production_after_reduction": validation_max_abs,
            "timings": timings,
            "derived": {
                "hash_and_e8_upper_bound_us": full - proxy,
                "packed_reads_scale_pack_upper_bound_us": proxy - floor,
                "mma_floor_us": floor,
                "full_over_mma_floor": full / floor,
                "first_e8_upper_bound_us": full - timings["skip_first_e8"]["median_us"],
                "second_e8_upper_bound_us": full - timings["skip_second_e8"]["median_us"],
            },
        })
        print(f"{weight_name} full={full:.3f}us proxy={proxy:.3f}us floor={floor:.3f}us", flush=True)
    report = {
        "scope": "J2 partial-kernel diagnostic variants on real packed projection records",
        "device": torch.cuda.get_device_name(), "tokens": args.tokens,
        "warmup": args.warmup, "iterations": args.iterations, "repeats": args.repeats,
        "records": results,
        "claim_boundary": (
            "Diagnostic kernels change the computation. Differences are upper-bound indicators, "
            "not independently additive cycle attribution or model-quality results."
        ),
    }
    output = args.output or ROOT / f"results/raw/llama32_1b_nqrt4_mma_j2_breakdown_m{args.tokens}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
