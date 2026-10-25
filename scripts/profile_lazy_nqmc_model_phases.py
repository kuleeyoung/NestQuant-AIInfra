"""Profile synchronized runtime phases of a real lazy NQMC model forward."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from nestquant.models.lazy_linear import LazyNQMCWeightCache
from nestquant.models.lazy_model import convert_model_to_lazy_nqmc_


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _projection_type(name: str) -> str:
    module_name, _, _field = name.rpartition(".")
    return module_name.rsplit(".", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-mib", type=float, default=256)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--decode-backend", choices=("reference", "triton_tiled"),
        default="reference",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cache_mib < 0 or args.rows_per_chunk < 1 or args.sequence_length < 2:
        parser.error("cache must be nonnegative and row counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    from transformers import AutoModelForCausalLM

    reader = NQMCFileReader(args.checkpoint)
    metadata = reader.model_metadata
    model_id = metadata["model_id"]
    revision = metadata["model_revision"]
    config = metadata["model_config"]
    events: list[dict[str, object]] = []
    cache = LazyNQMCWeightCache(round(args.cache_mib * 1024 * 1024))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16,
        local_files_only=args.offline,
    ).eval()
    conversion = convert_model_to_lazy_nqmc_(
        model, reader, cache,
        compute_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        rows_per_chunk=args.rows_per_chunk,
        profile_callback=events.append,
        synchronize_profile=True,
        decode_backend=args.decode_backend,
    )
    model.to(device)
    generator = torch.Generator(device="cpu").manual_seed(20261024)
    input_ids = torch.randint(
        0, config["vocab_size"], (1, args.sequence_length), generator=generator
    )
    input_ids[0, 0] = int(config.get("bos_token_id") or 0)
    input_ids = input_ids.to(device)

    # Exercise every projection shape once so first-use CUDA work is excluded.
    warmup_started = time.perf_counter()
    with torch.inference_mode():
        warmup_logits = model(input_ids=input_ids, use_cache=False).logits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    del warmup_logits
    events.clear()
    cache.clear(reset_statistics=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started

    by_phase: dict[str, list[float]] = defaultdict(list)
    decode_by_projection: dict[str, list[float]] = defaultdict(list)
    decode_by_weight: dict[str, float] = {}
    for event in events:
        phase = str(event["phase"])
        seconds = float(event["seconds"])
        by_phase[phase].append(seconds)
        if phase == "unpack_dequant":
            name = str(event["weight_name"])
            decode_by_projection[_projection_type(name)].append(seconds)
            decode_by_weight[name] = seconds
    phase_stats = {phase: _stats(values) for phase, values in sorted(by_phase.items())}
    measured_phase_seconds = sum(item["total_seconds"] for item in phase_stats.values())
    for item in phase_stats.values():
        item["percent_of_forward_wall"] = 100 * item["total_seconds"] / forward_seconds
    unaccounted = max(0.0, forward_seconds - measured_phase_seconds)

    report = {
        "scope": "synchronized phase breakdown of one real lazy NQMC Llama forward",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_id": model_id,
        "model_revision": revision,
        "input_shape": list(input_ids.shape),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "rows_per_chunk": args.rows_per_chunk,
        "decode_backend": args.decode_backend,
        "cache_capacity_bytes": cache.capacity_bytes,
        "replaced_linear_count": len(conversion.replaced_linear_names),
        "warmup_seconds_excluded": warmup_seconds,
        "profiled_forward_seconds": forward_seconds,
        "phase_stats": phase_stats,
        "measured_phase_seconds": measured_phase_seconds,
        "unaccounted_model_seconds": unaccounted,
        "unaccounted_percent_of_forward": 100 * unaccounted / forward_seconds,
        "decode_by_projection": {
            name: _stats(values) for name, values in sorted(decode_by_projection.items())
        },
        "slowest_decode_weights": [
            {"weight_name": name, "seconds": seconds}
            for name, seconds in sorted(
                decode_by_weight.items(), key=lambda item: item[1], reverse=True
            )[:10]
        ],
        "cache": cache.snapshot(),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "output_checksum": float(logits.float().sum().item()),
        "methodology": (
            "Every reported GPU phase synchronizes before and after the operation. "
            "This intentionally perturbs normal asynchronous execution but produces "
            "additive diagnostic wall times after a full-path warmup."
        ),
        "claim_boundary": (
            "The profile covers Python/PyTorch BF16 materialization and GEMM, not a "
            "compiled packed low-bit kernel or production serving workload."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
