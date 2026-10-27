"""Measure the full-model packed/BF16 residency Pareto curve."""

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
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.hybrid_residency import (
    nondominated_indices,
    residency_groups_from_payload,
    select_residency_plan,
)
from scripts.benchmark_full_nqrt_model import (
    _load_model,
    _measure,
    _packed_bytes,
    _projection_names,
    _run_decode,
    _run_prefill,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("latency samples must be finite and nonempty")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": percentile(0.95) * 1000,
        "min_ms": min(values) * 1000,
        "max_ms": max(values) * 1000,
    }


def validate_config(config: dict) -> None:
    budgets = config.get("budget_mib")
    benchmark = config.get("benchmark")
    if (not isinstance(budgets, list) or not budgets
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError("budget_mib must be sorted unique nonnegative integers")
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be an object")
    for name in ("batch_size", "prompt_length", "decode_tokens", "warmup", "repetitions"):
        value = benchmark.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"benchmark.{name} must be a positive integer")
    if benchmark["decode_tokens"] < 3:
        raise ValueError("decode_tokens must be at least three for measured TPOT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if profile.get("status") != "passed" or profile.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("profile does not belong to the supplied checkpoint")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model_id = reader.model_metadata.get("model_id")
    revision = reader.model_metadata.get("model_revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("checkpoint must identify model_id and model_revision")
    probe = _load_model(model_id, revision, args.offline)
    names = _projection_names(probe, reader)
    vocab_size = int(probe.config.vocab_size)
    del probe
    gc.collect()
    torch.cuda.empty_cache()

    groups = residency_groups_from_payload(profile["groups_ranked"])
    profile_names = {
        name for group in groups for name in group.projection_names
    }
    if profile_names != set(names):
        raise ValueError("profile projection set does not match the model")

    bench = config["benchmark"]
    cpu_generator = torch.Generator().manual_seed(int(config["seed"]))
    cpu_ids = torch.randint(
        0, vocab_size,
        (int(bench["batch_size"]), int(bench["prompt_length"])),
        generator=cpu_generator,
    )
    input_sha256 = hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest()
    results = []
    logits_by_budget: dict[int, torch.Tensor] = {}

    for budget_mib in config["budget_mib"]:
        plan = select_residency_plan(
            groups,
            budget_bytes=int(budget_mib) * 1024 * 1024,
            all_projection_names=names,
        )
        model = _load_model(model_id, revision, args.offline)
        conversion = None
        if plan.packed_projection_names:
            conversion = convert_linears_to_fused_nqrt_(
                model,
                reader,
                plan.packed_projection_names,
                output_dtype=torch.bfloat16,
                group_qkv=False,
                compute_backend="cuda_mma_j2",
            )
        gc.collect()
        input_ids = cpu_ids.to("cuda")
        with torch.inference_mode():
            # Materialize all selected packed streams and one-time kernels.
            for _ in range(int(bench["warmup"])):
                warm = model(input_ids=input_ids, use_cache=False)
                del warm
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            logits = model(input_ids=input_ids, use_cache=False).logits.detach().cpu()
            logits_by_budget[int(budget_mib)] = logits
            prefill = _measure(
                lambda: _run_prefill(model, input_ids, use_cache=False),
                int(bench["repetitions"]),
            )
            ttft_samples = []
            tpot_samples = []
            generated = None
            for _ in range(int(bench["repetitions"])):
                ttft, tpot, token_ids = _run_decode(
                    model, input_ids, int(bench["decode_tokens"])
                )
                ttft_samples.append(ttft)
                tpot_samples.append(tpot)
                if generated is None:
                    generated = token_ids
            torch.cuda.synchronize()
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            peak_allocated = int(torch.cuda.max_memory_allocated())

        result = {
            "budget_mib": int(budget_mib),
            "selected_group_count": len(plan.selected_group_names),
            "resident_projection_count": len(plan.resident_projection_names),
            "packed_projection_count": len(plan.packed_projection_names),
            "selected_group_names": list(plan.selected_group_names),
            "resident_projection_names": list(plan.resident_projection_names),
            "predicted_extra_device_bytes": plan.predicted_extra_device_bytes,
            "predicted_saved_ms": plan.predicted_saved_ms,
            "actual_packed_device_bytes": _packed_bytes(model),
            "cuda_allocated_bytes_after_warmup": allocated,
            "cuda_reserved_bytes_after_warmup": reserved,
            "cuda_peak_allocated_bytes": peak_allocated,
            "steady_prefill_seconds": prefill,
            "ttft": stats(ttft_samples),
            "tpot": stats(tpot_samples),
            "decode_tokens_per_second_per_request": (
                1000.0 / stats(tpot_samples)["mean_ms"]
            ),
            "generated_token_ids": generated,
            "converted_projection_count": (
                0 if conversion is None else len(conversion.replaced_linear_names)
            ),
        }
        results.append(result)
        print(
            f"budget={budget_mib}MiB resident={len(plan.resident_projection_names)} "
            f"allocated={allocated / 2**20:.1f}MiB "
            f"ttft={result['ttft']['median_ms']:.2f}ms "
            f"tpot={result['tpot']['median_ms']:.2f}ms",
            file=sys.stderr,
        )
        del model, input_ids, logits
        gc.collect()
        torch.cuda.empty_cache()

    packed_budget = min(logits_by_budget)
    bf16_budget = max(logits_by_budget)
    packed_logits = logits_by_budget[packed_budget].float()
    bf16_logits = logits_by_budget[bf16_budget].float()
    for result in results:
        logits = logits_by_budget[result["budget_mib"]].float()
        for label, reference in (("packed", packed_logits), ("bf16", bf16_logits)):
            error = logits - reference
            result[f"max_abs_logit_error_vs_{label}"] = float(error.abs().max().item())
            result[f"relative_l2_logit_error_vs_{label}"] = float(
                (torch.linalg.vector_norm(error)
                 / torch.linalg.vector_norm(reference)).item()
            )
    pareto_inputs = [{
        "memory": item["cuda_allocated_bytes_after_warmup"],
        "ttft": item["ttft"]["median_ms"],
        "tpot": item["tpot"]["median_ms"],
    } for item in results]
    pareto_indices = nondominated_indices(
        pareto_inputs, metric_names=("memory", "ttft", "tpot")
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "profile": str(args.profile.resolve()),
        "profile_sha256": sha256_file(args.profile),
        "config": config,
        "config_sha256": sha256_file(args.config),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "model_id": model_id,
        "model_revision": revision,
        "input_ids_sha256": input_sha256,
        "results": results,
        "pareto_budget_mib": [results[index]["budget_mib"] for index in pareto_indices],
        "claim_boundary": (
            "Single-GPU finite-sample full-model benchmark. Ranking uses isolated family-median "
            "timings; only measured nondominated points support Pareto claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed", "output": str(args.output),
        "pareto_budget_mib": report["pareto_budget_mib"],
    }, indent=2))


if __name__ == "__main__":
    main()
