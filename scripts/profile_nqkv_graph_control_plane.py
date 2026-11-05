"""Profile NQKV dual-Graph control-plane phases on a real packed model.

The phase timings deliberately synchronize CUDA between operations so that
asynchronous work is attributed to the phase that launched it. They are an
optimization diagnostic, not an end-to-end serving latency claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, TypeVar

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.nqkv_batched_hf_backend import (
    BatchedNQKVLlamaCache,
    install_batched_nqkv_llama_backend,
)
from nestquant.runtime.nqkv_batched_runner import BatchedNQKVLlamaRunner
from scripts.benchmark_full_nqrt_model import (
    ACTIVATION_BETAS,
    _load_model,
    _projection_names,
)
from scripts.benchmark_service_workloads import KEY_BETAS, VALUE_BETAS


T = TypeVar("T")


def _measure(operation: Callable[[], T]) -> tuple[T, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": percentile(0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _record(store: dict[str, list[float]], name: str, value: float) -> None:
    store.setdefault(name, []).append(value)


@torch.inference_mode()
def _profile_decode(
    runner: BatchedNQKVLlamaRunner,
    token: torch.Tensor,
    positions: torch.Tensor,
    active_mask: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | bool | int]]:
    phases: dict[str, float | bool | int] = {}

    _, phases["validation_ms"] = _measure(
        lambda: runner._validate_common(
            token, positions, active_mask, valid_lengths
        )
    )

    row_active = active_mask[:, 0]

    def detect_boundary():
        active_slots = runner.cache.slot_ids[row_active].long()
        if not active_slots.numel():
            return active_slots, active_slots
        lengths = runner.cache.manager.layer_caches[0].hot_lengths[active_slots]
        boundary_slots_tensor = active_slots[
            lengths >= runner.cache.page_size - 1
        ]
        # Match the production path: materialize the host slot list.
        tuple(int(slot) for slot in boundary_slots_tensor.cpu().tolist())
        return active_slots, boundary_slots_tensor

    (active_slots, boundary_slots_tensor), phases["boundary_detection_ms"] = (
        _measure(detect_boundary)
    )
    boundary_slots = tuple(
        int(slot) for slot in boundary_slots_tensor.cpu().tolist()
    )
    is_boundary = bool(boundary_slots)
    phases["is_boundary"] = is_boundary
    phases["active_slots"] = int(active_slots.numel())
    phases["boundary_slots"] = len(boundary_slots)

    if is_boundary:
        _, phases["page_reservation_ms"] = _measure(
            lambda: runner.cache.manager.reserve_graph_transition_pages(
                boundary_slots
            )
        )

        def prepare_boundary_metadata() -> None:
            runner.cache.boundary_mask.zero_()
            runner.cache.boundary_mask[boundary_slots_tensor] = True
            runner.cache.active_mask.copy_(row_active)

        _, phases["metadata_prepare_ms"] = _measure(
            prepare_boundary_metadata
        )
        output, phases["graph_replay_ms"] = _measure(
            lambda: runner._graphed_boundary_decode(token, positions)
        )
        _, phases["page_commit_ms"] = _measure(
            lambda: runner.cache.manager.commit_graph_transition_pages(
                boundary_slots
            )
        )
        runner.graph_boundary_replay_count += 1
    else:
        _, phases["metadata_prepare_ms"] = _measure(
            lambda: runner.cache.active_mask.copy_(row_active)
        )
        output, phases["graph_replay_ms"] = _measure(
            lambda: runner._graphed_decode(token, positions)
        )

    measured_phase_names = tuple(
        name for name in phases if name.endswith("_ms")
    )
    phases["profiled_total_ms"] = sum(
        float(phases[name]) for name in measured_phase_names
    )
    phases["control_plane_ms"] = sum(
        float(phases[name]) for name in measured_phase_names
        if name != "graph_replay_ms"
    )
    return output, phases


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--max-cache-len", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--compute-backend",
        choices=("packed_w4a4_tiled", "packed_w4a4_cuda_mma"),
        default="packed_w4a4_cuda_mma",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.capacity < 1:
        parser.error("iterations and capacity must be positive")
    if args.page_size < 2 or args.max_cache_len < args.page_size:
        parser.error("page-size must be >=2 and fit inside max-cache-len")

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
        compute_backend=args.compute_backend,
        activation_q=14,
        activation_betas=ACTIVATION_BETAS,
    )
    head_dim = int(getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    ))
    cache = BatchedNQKVLlamaCache(
        capacity=args.capacity,
        num_layers=int(model.config.num_hidden_layers),
        num_query_heads=int(model.config.num_attention_heads),
        num_kv_heads=int(model.config.num_key_value_heads),
        head_dim=head_dim,
        max_cache_len=args.max_cache_len,
        page_size=args.page_size,
        q=14,
        key_betas=KEY_BETAS,
        value_betas=VALUE_BETAS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    install_batched_nqkv_llama_backend(model, cache)
    runner = BatchedNQKVLlamaRunner(model, cache)

    sample_token = torch.zeros(
        (args.capacity, 1), device="cuda", dtype=torch.long
    )
    sample_positions = torch.zeros_like(sample_token)
    sample_active = torch.ones_like(sample_token, dtype=torch.bool)
    sample_lengths = torch.ones(
        args.capacity, device="cuda", dtype=torch.long
    )
    runner.prefill(sample_token)
    runner.capture(
        sample_token,
        positions=sample_positions,
        active_mask=sample_active,
        valid_lengths=sample_lengths,
    )
    clear = torch.ones(args.capacity, device="cuda", dtype=torch.bool)
    runner.clear_slots(clear)

    generator = torch.Generator(device="cuda").manual_seed(20261102)
    prompt_width = args.page_size - 2
    positions = torch.arange(
        prompt_width, device="cuda", dtype=torch.long
    ).expand(args.capacity, -1)
    write_mask = torch.ones(
        (args.capacity, prompt_width), device="cuda", dtype=torch.bool
    )
    prompt_lengths = torch.full(
        (args.capacity,), prompt_width, device="cuda", dtype=torch.long
    )
    decode_active = torch.ones(
        (args.capacity, 1), device="cuda", dtype=torch.bool
    )
    samples: dict[str, dict[str, list[float]]] = {
        "normal": {}, "boundary": {}
    }

    for iteration in range(args.iterations):
        prompt = torch.randint(
            1,
            model.config.vocab_size,
            (args.capacity, prompt_width),
            device="cuda",
            generator=generator,
        )
        runner.prefill_slots(
            prompt,
            positions=positions,
            write_mask=write_mask,
            valid_lengths=prompt_lengths,
        )

        normal_token = torch.randint(
            1, model.config.vocab_size, (args.capacity, 1),
            device="cuda", generator=generator,
        )
        normal_position = torch.full_like(
            normal_token, prompt_width
        )
        normal_lengths = torch.full(
            (args.capacity,), prompt_width + 1,
            device="cuda", dtype=torch.long,
        )
        _, normal = _profile_decode(
            runner,
            normal_token,
            normal_position,
            decode_active,
            normal_lengths,
        )

        boundary_token = torch.randint(
            1, model.config.vocab_size, (args.capacity, 1),
            device="cuda", generator=generator,
        )
        boundary_position = torch.full_like(
            boundary_token, prompt_width + 1
        )
        boundary_lengths = torch.full(
            (args.capacity,), args.page_size,
            device="cuda", dtype=torch.long,
        )
        _, boundary = _profile_decode(
            runner,
            boundary_token,
            boundary_position,
            decode_active,
            boundary_lengths,
        )
        if normal["is_boundary"] or not boundary["is_boundary"]:
            raise RuntimeError("profile did not exercise the intended Graphs")
        for phase, value in normal.items():
            if phase.endswith("_ms"):
                _record(samples["normal"], phase, float(value))
        for phase, value in boundary.items():
            if phase.endswith("_ms"):
                _record(samples["boundary"], phase, float(value))
        runner.clear_slots(clear)
        print(
            json.dumps({
                "iteration": iteration,
                "normal_total_ms": normal["profiled_total_ms"],
                "boundary_total_ms": boundary["profiled_total_ms"],
            }),
            flush=True,
        )

    summary = {
        kind: {
            phase: _distribution(values)
            for phase, values in phase_samples.items()
        }
        for kind, phase_samples in samples.items()
    }
    for kind in ("normal", "boundary"):
        total = summary[kind]["profiled_total_ms"]["median_ms"]
        control = summary[kind]["control_plane_ms"]["median_ms"]
        summary[kind]["control_plane_share_of_profiled_total"] = (
            control / total if total else 0.0
        )

    payload = {
        "schema_version": 1,
        "status": "passed",
        "protocol": "synchronized_phase_attribution_not_service_latency",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_id": reader.model_metadata["model_id"],
        "model_revision": reader.model_metadata["model_revision"],
        "compute_backend": args.compute_backend,
        "capacity": args.capacity,
        "page_size": args.page_size,
        "iterations": args.iterations,
        "summary": summary,
        "raw_samples_ms": samples,
        "claim_boundary": (
            "Each phase is CUDA-synchronized for attribution. The sum is not "
            "the asynchronous production decode latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
