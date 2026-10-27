"""Profile each Llama projection for packed/BF16 hybrid residency planning."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import FusedNQRTLinear
from nestquant.runtime.hybrid_residency import make_residency_groups
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_measure(operation, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    outputs = []
    for start, end in zip(starts, ends):
        start.record()
        outputs.append(operation())
        end.record()
    torch.cuda.synchronize()
    values = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    del outputs
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def validate_config(config: dict) -> None:
    counts = config.get("profile_token_counts")
    weights = config.get("profile_token_weights")
    if (not isinstance(counts, list) or not counts
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                   for value in counts)):
        raise ValueError("profile_token_counts must contain positive integers")
    if (not isinstance(weights, list) or len(weights) != len(counts)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in weights)
            or sum(weights) <= 0):
        raise ValueError("profile_token_weights must be finite nonnegative values")
    for name in ("warmup", "iterations"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model_id = reader.model_metadata.get("model_id")
    revision = reader.model_metadata.get("model_revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("checkpoint must identify model_id and model_revision")
    model = _load_model(model_id, revision, args.offline)
    names = _projection_names(model, reader)
    modules = dict(model.named_modules())
    rotation = reader.model_metadata.get("hadamard_rotation", {})
    converted = set(rotation.get("converted_linear_names", [])) if isinstance(rotation, dict) else set()

    generator = torch.Generator(device="cuda").manual_seed(int(config["seed"]))
    weights = [float(value) for value in config["profile_token_weights"]]
    weight_sum = sum(weights)
    profiles = []
    dense_bytes: dict[str, int] = {}
    packed_bytes: dict[str, int] = {}
    measured_saved_ms: dict[str, float] = {}

    with torch.inference_mode():
        for index, name in enumerate(names):
            dense = modules[name]
            fused = FusedNQRTLinear(
                reader,
                f"{name}.weight",
                None if dense.bias is None else dense.bias.detach(),
                input_hadamard=name in converted,
                output_dtype=torch.bfloat16,
                compute_backend="cuda_mma_j2",
            ).eval()
            token_results = []
            weighted_saved = 0.0
            for token_count, weight in zip(config["profile_token_counts"], weights):
                inputs = torch.randn(
                    (int(token_count), int(dense.in_features)),
                    device="cuda", dtype=torch.bfloat16, generator=generator,
                )
                packed_result = event_measure(
                    lambda: fused(inputs),
                    warmup=int(config["warmup"]),
                    iterations=int(config["iterations"]),
                )
                bf16_result = event_measure(
                    lambda: dense(inputs),
                    warmup=int(config["warmup"]),
                    iterations=int(config["iterations"]),
                )
                saved = packed_result["median_ms"] - bf16_result["median_ms"]
                weighted_saved += weight * saved
                token_results.append({
                    "token_count": int(token_count),
                    "packed": packed_result,
                    "bf16": bf16_result,
                    "packed_minus_bf16_median_ms": saved,
                })
                del inputs
            dense_bytes[name] = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (dense.weight, dense.bias) if tensor is not None
            )
            packed_bytes[name] = fused.packed_device_bytes
            measured_saved_ms[name] = weighted_saved / weight_sum
            profiles.append({
                "name": name,
                "projection_family": name.rsplit(".", 1)[-1],
                "dense_bytes": dense_bytes[name],
                "packed_device_bytes": packed_bytes[name],
                "incremental_bf16_bytes": max(0, dense_bytes[name] - packed_bytes[name]),
                "weighted_measured_saved_ms": measured_saved_ms[name],
                "token_profiles": token_results,
            })
            del fused
            if (index + 1) % 16 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"profiled {index + 1}/{len(names)} projections", file=sys.stderr)

    family_values: dict[str, list[float]] = {}
    for name, value in measured_saved_ms.items():
        family_values.setdefault(name.rsplit(".", 1)[-1], []).append(value)
    family_medians = {
        family: statistics.median(values)
        for family, values in family_values.items()
    }
    predicted_saved_ms = {
        name: family_medians[name.rsplit(".", 1)[-1]]
        for name in names
    }
    for profile in profiles:
        profile["selection_predicted_saved_ms"] = predicted_saved_ms[profile["name"]]

    groups = make_residency_groups(
        names,
        dense_bytes=dense_bytes,
        packed_bytes=packed_bytes,
        predicted_saved_ms=predicted_saved_ms,
    )
    group_payload = [{
        "name": group.name,
        "projection_names": list(group.projection_names),
        "dense_bytes": group.dense_bytes,
        "packed_device_bytes": group.packed_bytes,
        "incremental_bf16_bytes": group.extra_device_bytes,
        "weighted_predicted_saved_ms": group.predicted_saved_ms,
        "predicted_saved_ms_per_mib": group.saved_ms_per_mib,
    } for group in groups]
    group_payload.sort(key=lambda item: (
        -item["predicted_saved_ms_per_mib"], -item["weighted_predicted_saved_ms"], item["name"]
    ))
    report = {
        "schema_version": 1,
        "status": "passed",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config": config,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "model_id": model_id,
        "model_revision": revision,
        "projection_count": len(names),
        "selection_family_median_saved_ms": family_medians,
        "profiles": profiles,
        "groups_ranked": group_payload,
        "claim_boundary": (
            "Isolated projection timings rank candidates only. Formal Pareto claims must use "
            "measured full-model latency and memory, not predicted savings."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed", "output": str(args.output),
        "projection_count": len(names), "group_count": len(groups),
    }, indent=2))


if __name__ == "__main__":
    main()
