"""Benchmark seven packed fused projections inside one real Llama block."""

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
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.models.fused_linear import (
    FusedNQRTLinear,
    GroupedQKVProjection,
    convert_linears_to_fused_nqrt_,
)
from nestquant.models.rotated_linear import HadamardRotatedLinear


PROJECTION_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _measure(operation, repeats: int) -> tuple[dict[str, float], int]:
    output = operation()
    torch.cuda.synchronize()
    del output
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = operation()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
        del output
    return _stats(values), max(0, torch.cuda.max_memory_allocated() - baseline)


def _profile_cuda(operation, repeats: int) -> list[dict[str, object]]:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    operation()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
    ) as profiler:
        for _ in range(repeats):
            operation()
        torch.cuda.synchronize()
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        device_us = float(getattr(event, "self_device_time_total", 0.0))
        if device_us <= 0.0:
            device_us = float(getattr(event, "self_cuda_time_total", 0.0))
        if device_us <= 0.0:
            continue
        rows.append({
            "name": event.key,
            "count": event.count,
            "input_shapes": event.input_shapes,
            "self_device_total_us": device_us,
            "self_device_mean_us": device_us / event.count,
        })
    rows.sort(key=lambda item: item["self_device_total_us"], reverse=True)
    return rows


def _module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    return dict(model.named_modules())[name]


def _replace(model: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    modules = dict(model.named_modules())
    parent_name, _, child_name = name.rpartition(".")
    parent = model if not parent_name else modules[parent_name]
    setattr(parent, child_name, module)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-repeats", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--group-qkv", action="store_true")
    parser.add_argument(
        "--compute-backend",
        choices=("tensor_core_dot", "scalar", "cuda_mma_j2"),
        default="tensor_core_dot",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.layer_index < 0 or args.repeats < 1 or args.profile_repeats < 0:
        parser.error(
            "layer-index and profile-repeats must be nonnegative; repeats must be positive"
        )
    if any(length < 1 for length in args.sequence_length):
        parser.error("sequence lengths must be positive")

    from transformers import AutoModelForCausalLM

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model_id = reader.model_metadata.get("model_id")
    revision = reader.model_metadata.get("model_revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("checkpoint must identify its base model and revision")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        local_files_only=args.offline,
    ).eval().to("cuda")
    layer_name = f"model.layers.{args.layer_index}"
    layer = _module(model, layer_name)
    names = tuple(f"{layer_name}.{suffix}" for suffix in PROJECTION_SUFFIXES)

    # Capture the exact positional and attention arguments constructed by the
    # installed Transformers version, rather than duplicating private helpers.
    calls: dict[int, tuple[tuple[object, ...], dict[str, object]]] = {}
    generator = torch.Generator(device="cuda").manual_seed(20261025)
    for sequence_length in args.sequence_length:
        captured = []

        def capture(_module, positional, keyword):
            captured.append((positional, keyword))

        handle = layer.register_forward_pre_hook(capture, with_kwargs=True)
        input_ids = torch.randint(
            0, model.config.vocab_size, (1, sequence_length),
            device="cuda", generator=generator,
        )
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
        handle.remove()
        if len(captured) != 1:
            raise RuntimeError("expected exactly one captured decoder-layer call")
        calls[sequence_length] = captured[0]
        del input_ids

    bf16_weight_bytes = 0
    for name in names:
        _, record = reader.read_named_record(f"{name}.weight")
        if not isinstance(record, PackedNestQuantTensor):
            raise TypeError(f"expected packed weight: {name}")
        weight = record.dequantize_chunked(
            device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=128
        )
        source = _module(model, name)
        replacement = HadamardRotatedLinear(
            weight,
            None if source.bias is None else source.bias.detach(),
            output_dtype=torch.bfloat16,
        ).to("cuda", torch.bfloat16).eval()
        bf16_weight_bytes += weight.numel() * weight.element_size()
        _replace(model, name, replacement)
        del weight, record, source, replacement
    layer = _module(model, layer_name)

    references = {}
    reference_results = {}
    with torch.inference_mode():
        for sequence_length, (positional, keyword) in calls.items():
            operation = lambda p=positional, k=keyword: layer(*p, **k)
            reference = operation()
            torch.cuda.synchronize()
            references[sequence_length] = reference.detach().cpu()
            stats, peak = _measure(operation, args.repeats)
            reference_results[sequence_length] = {
                "seconds": stats,
                "peak_increment_bytes": peak,
            }
            if args.profile_repeats:
                reference_results[sequence_length]["profiler_cuda_events"] = (
                    _profile_cuda(operation, args.profile_repeats)
                )

    conversion = convert_linears_to_fused_nqrt_(
        model, reader, names, group_qkv=args.group_qkv,
        compute_backend=args.compute_backend,
    )
    layer = _module(model, layer_name)
    gc.collect()
    torch.cuda.empty_cache()
    fused_results = {}
    cold_prepare_seconds = None
    with torch.inference_mode():
        for sequence_length, (positional, keyword) in calls.items():
            operation = lambda p=positional, k=keyword: layer(*p, **k)
            started = time.perf_counter()
            actual = operation()
            torch.cuda.synchronize()
            cold = time.perf_counter() - started
            if cold_prepare_seconds is None:
                cold_prepare_seconds = cold
            error = actual.float().cpu() - references[sequence_length].float()
            relative_l2 = float(
                (torch.linalg.vector_norm(error)
                 / torch.linalg.vector_norm(references[sequence_length].float())).item()
            )
            stats, peak = _measure(operation, args.repeats)
            fused_results[sequence_length] = {
                "seconds": stats,
                "peak_increment_bytes": peak,
                "max_abs_error": float(error.abs().max().item()),
                "relative_l2_error": relative_l2,
            }
            if args.profile_repeats:
                fused_results[sequence_length]["profiler_cuda_events"] = (
                    _profile_cuda(operation, args.profile_repeats)
                )
            del actual, error

    fused_modules = [_module(model, name) for name in names]
    if not all(isinstance(module, (FusedNQRTLinear, GroupedQKVProjection)) for module in fused_modules):
        raise RuntimeError("fused conversion did not replace every projection")
    packed_device_bytes = sum(
        module.packed_device_bytes
        for module in fused_modules if isinstance(module, FusedNQRTLinear)
    )
    coordinators = {
        id(module.coordinator): module.coordinator
        for module in fused_modules if isinstance(module, GroupedQKVProjection)
    }
    packed_device_bytes += sum(
        coordinator.packed_device_bytes for coordinator in coordinators.values()
    )
    results = []
    for sequence_length in args.sequence_length:
        reference = reference_results[sequence_length]
        fused = fused_results[sequence_length]
        results.append({
            "sequence_length": sequence_length,
            "dequantized_bf16_block": reference,
            "packed_fused_block": fused,
            "median_speedup": (
                reference["seconds"]["median"] / fused["seconds"]["median"]
            ),
        })
    report = {
        "scope": "one real Llama decoder block with all seven Linear projections replaced",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_id": model_id,
        "model_revision": revision,
        "layer_index": args.layer_index,
        "device_name": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "profile_repeats": args.profile_repeats,
        "compute_backend": args.compute_backend,
        "replaced_linear_names": list(conversion.replaced_linear_names),
        "shared_hadamard_groups": [
            list(group) for group in conversion.shared_hadamard_groups
        ],
        "grouped_qkv_groups": [
            list(group) for group in conversion.grouped_qkv_groups
        ],
        "bf16_weight_bytes": bf16_weight_bytes,
        "packed_device_bytes": packed_device_bytes,
        "first_fused_block_call_including_seven_record_reads_and_preparation_seconds": cold_prepare_seconds,
        "results": results,
        "claim_boundary": (
            "This isolates one decoder block after capturing real model inputs. "
            "It is not an end-to-end generation or full-model throughput result."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
