"""Compare full NQMC reload with a model converted to lazy NQMC Linears."""

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
from nestquant.models.lazy_linear import LazyNQMCWeightCache
from nestquant.models.lazy_model import convert_model_to_lazy_nqmc_
from nestquant.models.rotated_linear import restore_hadamard_structure_from_metadata_


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _allocated(device: torch.device) -> int | None:
    return torch.cuda.memory_allocated(device) if device.type == "cuda" else None


def _load_base(model_id: str, revision: str, offline: bool):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        local_files_only=offline,
    ).eval()


def _time_forward(model, input_ids, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits
    _sync(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return logits, elapsed, peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-mib", type=float, default=256)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--lazy-repeats", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cache_mib < 0 or args.rows_per_chunk < 1:
        parser.error("cache size must be nonnegative and chunk rows must be positive")
    if args.sequence_length < 2 or args.lazy_repeats < 1:
        parser.error("sequence length must be at least two and repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    reader_started = time.perf_counter()
    reader = NQMCFileReader(
        args.checkpoint, verify_checksum=not args.skip_checksum
    )
    reader_seconds = time.perf_counter() - reader_started
    metadata = reader.model_metadata
    model_id = metadata.get("model_id")
    revision = metadata.get("model_revision")
    config = metadata.get("model_config")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("checkpoint is missing model identity metadata")
    if not isinstance(config, dict) or not isinstance(config.get("vocab_size"), int):
        raise ValueError("checkpoint is missing model config metadata")
    generator = torch.Generator(device="cpu").manual_seed(20261024)
    input_ids = torch.randint(
        0, config["vocab_size"], (1, args.sequence_length), generator=generator
    )
    input_ids[0, 0] = int(config.get("bos_token_id") or 0)

    # Reference: materialize every quantized record in normal model parameters.
    baseline_load_started = time.perf_counter()
    baseline = _load_base(model_id, revision, args.offline).to(device)
    hadamard_metadata = metadata.get("hadamard_rotation")
    if hadamard_metadata is not None:
        restore_hadamard_structure_from_metadata_(baseline, hadamard_metadata)
        baseline.to(device)
    reader.load_into_model(baseline, device=device)
    baseline_load_seconds = time.perf_counter() - baseline_load_started
    baseline_input = input_ids.to(device)
    _warmup_logits, baseline_warmup_seconds, _warmup_peak = _time_forward(
        baseline, baseline_input, device
    )
    del _warmup_logits
    baseline_logits, baseline_seconds, baseline_peak = _time_forward(
        baseline, baseline_input, device
    )
    reference_logits = baseline_logits.detach().cpu()
    baseline_allocated = _allocated(device)
    del baseline_logits, baseline_input, baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Lazy path: replace packed Linears on CPU, then move only dense state/biases.
    lazy_load_started = time.perf_counter()
    lazy_model = _load_base(model_id, revision, args.offline)
    cache = LazyNQMCWeightCache(round(args.cache_mib * 1024 * 1024))
    conversion = convert_model_to_lazy_nqmc_(
        lazy_model,
        reader,
        cache,
        compute_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        rows_per_chunk=args.rows_per_chunk,
    )
    lazy_model.to(device)
    lazy_load_seconds = time.perf_counter() - lazy_load_started
    lazy_input = input_ids.to(device)
    lazy_events = []
    first_lazy_logits = None
    for iteration in range(args.lazy_repeats):
        before = cache.snapshot()
        logits, elapsed, peak = _time_forward(lazy_model, lazy_input, device)
        after = cache.snapshot()
        if first_lazy_logits is None:
            first_lazy_logits = logits.detach().cpu()
        lazy_events.append({
            "iteration": iteration,
            "seconds": elapsed,
            "cuda_peak_allocated_bytes": peak,
            "cache_hits_delta": after["hits"] - before["hits"],
            "cache_misses_delta": after["misses"] - before["misses"],
            "cache_decodes_delta": after["decodes"] - before["decodes"],
            "cache_evictions_delta": after["evictions"] - before["evictions"],
            "cache_after": after,
        })
        del logits
    lazy_allocated = _allocated(device)

    assert first_lazy_logits is not None
    difference = first_lazy_logits.float() - reference_logits.float()
    exact = bool(torch.equal(first_lazy_logits, reference_logits))
    max_abs_error = float(difference.abs().max().item())
    mean_abs_error = float(difference.abs().mean().item())
    report = {
        "scope": "real Llama forward with automatic lazy NQMC Linear replacement",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": reader.file_bytes,
        "payload_checksum_verified": not args.skip_checksum,
        "reader_initialization_seconds": reader_seconds,
        "model_id": model_id,
        "model_revision": revision,
        "input_shape": list(input_ids.shape),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "baseline": {
            "load_and_full_dequantize_seconds": baseline_load_seconds,
            "warmup_forward_seconds_excluded": baseline_warmup_seconds,
            "forward_seconds": baseline_seconds,
            "cuda_peak_allocated_bytes": baseline_peak,
            "cuda_allocated_after_forward_bytes": baseline_allocated,
        },
        "lazy": {
            "load_convert_and_move_seconds": lazy_load_seconds,
            "replaced_linear_count": len(conversion.replaced_linear_names),
            "dense_tensor_count": len(conversion.dense_tensor_names),
            "alias_count": len(conversion.alias_names),
            "input_hadamard_count": len(conversion.input_hadamard_names),
            "cache_capacity_bytes": cache.capacity_bytes,
            "events": lazy_events,
            "final_cache": cache.snapshot(),
            "cuda_allocated_after_forward_bytes": lazy_allocated,
        },
        "correctness": {
            "logits_bitwise_equal": exact,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "baseline_checksum": float(reference_logits.float().sum().item()),
            "lazy_checksum": float(first_lazy_logits.float().sum().item()),
        },
        "claim_boundary": (
            "The lazy model executes a real Llama forward, but cache misses materialize "
            "BF16 weights and GEMMs remain ordinary PyTorch BF16 operations. This does "
            "not establish packed low-bit execution or production serving throughput."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
