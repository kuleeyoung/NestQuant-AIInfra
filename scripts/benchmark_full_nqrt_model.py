"""End-to-end BF16/scalar-packed/Tensor-Core-packed Llama benchmark."""

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
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_


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


def _measure(operation, repeats: int) -> dict[str, float]:
    operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = operation()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
        del output
    return _stats(values)


def _load_model(model_id: str, revision: str, offline: bool):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        local_files_only=offline,
    ).eval().to("cuda")


def _projection_names(model, reader: NQMCFileReader) -> tuple[str, ...]:
    layer_count = int(model.config.num_hidden_layers)
    modules = dict(model.named_modules())
    names = []
    for layer_index in range(layer_count):
        for suffix in PROJECTION_SUFFIXES:
            name = f"model.layers.{layer_index}.{suffix}"
            if name not in modules:
                raise ValueError(f"base model is missing expected module: {name}")
            entry = reader.get_entry(f"{name}.weight")
            if entry.kind != "nestquant":
                raise ValueError(f"checkpoint is missing packed record: {name}.weight")
            names.append(name)
    return tuple(names)


def _packed_bytes(model) -> int:
    return sum(
        int(getattr(module, "packed_device_bytes", 0))
        for module in model.modules()
    )


def _run_prefill(model, input_ids: torch.Tensor, use_cache: bool):
    return model(input_ids=input_ids, use_cache=use_cache)


def _run_decode(
    model,
    input_ids: torch.Tensor,
    decode_tokens: int,
) -> tuple[float, float, list[list[int]]]:
    with torch.inference_mode():
        # Warm the cache-enabled prefill and one-token decode separately. The
        # first call may compile attention kernels or allocate cache storage;
        # those one-time costs must not be reported as steady TTFT/TPOT.
        warm = model(input_ids=input_ids, use_cache=True)
        warm_next = warm.logits[:, -1:].argmax(dim=-1)
        model(
            input_ids=warm_next,
            past_key_values=warm.past_key_values,
            use_cache=True,
        )
        del warm, warm_next
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - started
        past = output.past_key_values
        next_token = output.logits[:, -1:].argmax(dim=-1)
        generated = [next_token[:, 0].detach().cpu().tolist()]
        decode_values = []
        # Warm one decode transition from the timed prefill state, then measure
        # subsequent transitions. The warm transition is excluded from TPOT.
        if decode_tokens > 1:
            warm_output = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = warm_output.past_key_values
            next_token = warm_output.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token[:, 0].detach().cpu().tolist())
        for _ in range(max(0, decode_tokens - 2)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            torch.cuda.synchronize()
            decode_values.append(time.perf_counter() - started)
            past = output.past_key_values
            next_token = output.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token[:, 0].detach().cpu().tolist())
        tpot = statistics.mean(decode_values) if decode_values else 0.0
    return ttft, tpot, generated


def _benchmark_path(
    checkpoint: Path,
    model_id: str,
    revision: str,
    reader: NQMCFileReader,
    input_ids: torch.Tensor,
    *,
    mode: str,
    repeats: int,
    decode_tokens: int,
    offline: bool,
    names: tuple[str, ...],
) -> dict[str, object]:
    load_started = time.perf_counter()
    model = _load_model(model_id, revision, offline)
    load_seconds = time.perf_counter() - load_started
    conversion = None
    cold_prepare_seconds = None
    if mode != "bf16":
        conversion = convert_linears_to_fused_nqrt_(
            model,
            reader,
            names,
            output_dtype=torch.bfloat16,
            group_qkv=False,
            compute_backend=mode,
        )

    with torch.inference_mode():
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        cold_started = time.perf_counter()
        first = _run_prefill(model, input_ids, use_cache=False)
        torch.cuda.synchronize()
        cold_prepare_seconds = time.perf_counter() - cold_started
        first_logits = first.logits.detach().cpu()
        del first
        steady = _measure(
            lambda: _run_prefill(model, input_ids, use_cache=False), repeats
        )
        resident_allocated = int(torch.cuda.memory_allocated())
        peak_increment = int(
            max(0, torch.cuda.max_memory_allocated() - baseline_allocated)
        )
        ttft, tpot, generated = _run_decode(model, input_ids, decode_tokens)

    result: dict[str, object] = {
        "mode": mode,
        "load_seconds": load_seconds,
        "cold_prefill_seconds": cold_prepare_seconds,
        "steady_prefill_seconds": steady,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "decode_tokens": decode_tokens,
        "decode_tokens_per_second": (
            0.0 if tpot == 0.0 else 1.0 / tpot
        ),
        "generated_token_ids": generated,
        "cuda_peak_allocated_bytes_from_process_start": peak_increment,
        "cuda_model_and_runtime_allocated_bytes_after_warmup": resident_allocated,
        "packed_device_bytes_after_warmup": _packed_bytes(model),
        "replaced_linear_count": (
            0 if conversion is None else len(conversion.replaced_linear_names)
        ),
        "first_logits": first_logits,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequence-length", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1])
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=("bf16", "scalar", "tensor_core_dot"),
                        default=["bf16", "scalar", "tensor_core_dot"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(length < 1 for length in args.sequence_length):
        parser.error("sequence lengths must be positive")
    if any(batch < 1 for batch in args.batch_size):
        parser.error("batch sizes must be positive")
    if args.decode_tokens < 1 or args.repeats < 1:
        parser.error("decode-tokens and repeats must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model_id = reader.model_metadata.get("model_id")
    revision = reader.model_metadata.get("model_revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("checkpoint must identify model_id and model_revision")

    # Load once only to discover the exact layer count and module names.
    probe = _load_model(model_id, revision, args.offline)
    names = _projection_names(probe, reader)
    vocab_size = int(probe.config.vocab_size)
    del probe
    gc.collect()
    torch.cuda.empty_cache()

    generator = torch.Generator(device="cuda").manual_seed(20261025)
    all_results = []
    for batch_size in args.batch_size:
        for sequence_length in args.sequence_length:
            input_ids = torch.randint(
                0,
                vocab_size,
                (batch_size, sequence_length),
                device="cuda",
                generator=generator,
            )
            path_results = []
            reference_logits = None
            for mode in args.modes:
                result = _benchmark_path(
                    args.checkpoint,
                    model_id,
                    revision,
                    reader,
                    input_ids,
                    mode=mode,
                    repeats=args.repeats,
                    decode_tokens=args.decode_tokens,
                    offline=args.offline,
                    names=names,
                )
                logits = result.pop("first_logits")
                if reference_logits is None:
                    reference_logits = logits
                error = logits.float() - reference_logits.float()
                result["max_abs_logit_error_vs_first_mode"] = float(error.abs().max().item())
                result["relative_l2_logit_error_vs_first_mode"] = float(
                    (torch.linalg.vector_norm(error)
                     / torch.linalg.vector_norm(reference_logits.float())).item()
                )
                path_results.append(result)
            all_results.append({
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "input_ids_sha256": __import__("hashlib").sha256(
                    input_ids.cpu().numpy().tobytes()
                ).hexdigest(),
                "results": path_results,
            })
            del input_ids

    report = {
        "scope": "end-to-end Llama-3.2-1B packed NQRT inference benchmark",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_id": model_id,
        "model_revision": revision,
        "device_name": torch.cuda.get_device_name(),
        "sequence_lengths": args.sequence_length,
        "batch_sizes": args.batch_size,
        "decode_tokens": args.decode_tokens,
        "repeats": args.repeats,
        "modes": args.modes,
        "quantized_projection_count": len(names),
        "quantized_projection_names": list(names),
        "results": all_results,
        "claim_boundary": (
            "This is single-GPU end-to-end inference evidence for the tested "
            "Llama revision and measured batch sizes. It is not a multi-user "
            "serving or multi-GPU result."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
