"""Validate and benchmark NQRTGraphDecodeRunner on a real packed model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.cuda_graph_decode import NQRTGraphDecodeRunner
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


def measure(operation, warmup: int, iterations: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize(); started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1e6 / iterations)
    return {"median_us": statistics.median(values), "min_us": min(values), "max_us": max(values)}


@torch.inference_mode()
def eager_greedy(runner: NQRTGraphDecodeRunner, input_ids: torch.Tensor,
                 max_new_tokens: int) -> torch.Tensor:
    logits = runner.prefill(input_ids)
    token = logits.argmax(dim=-1).contiguous()
    generated = [token.clone()]
    for position in range(input_ids.shape[1], input_ids.shape[1] + max_new_tokens - 1):
        logits = runner.decode_eager(token, position=position)
        token = logits[:, -1:, :].argmax(dim=-1).contiguous()
        generated.append(token.clone())
    return torch.cat(generated, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (min(args.batch_size) < 1 or min(args.sequence_length, args.new_tokens,
            args.validation_steps, args.iterations, args.repeats) < 1):
        parser.error("all sizes and counts must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = _load_model(reader.model_metadata["model_id"],
                        reader.model_metadata["model_revision"], args.offline)
    names = _projection_names(model, reader)
    convert_linears_to_fused_nqrt_(
        model, reader, names, output_dtype=torch.bfloat16,
        group_qkv=False, compute_backend="cuda_mma_j2",
    )
    results = []
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    max_cache_len = args.sequence_length + args.new_tokens + 4
    for batch in args.batch_size:
        input_ids = torch.randint(0, model.config.vocab_size,
                                  (batch, args.sequence_length), device="cuda",
                                  generator=generator)
        graph_runner = NQRTGraphDecodeRunner(
            model, batch_size=batch, max_cache_len=max_cache_len)
        eager_runner = NQRTGraphDecodeRunner(
            model, batch_size=batch, max_cache_len=max_cache_len)
        graph_prefill = graph_runner.prefill(input_ids)
        eager_prefill = eager_runner.prefill(input_ids)
        prefill_error = float((graph_prefill - eager_prefill).abs().max().item())
        token = graph_prefill.argmax(dim=-1).contiguous()
        graph_runner.capture(token, position=args.sequence_length)
        step_errors = []
        graph_tokens = [token.clone()]
        eager_tokens = [token.clone()]
        for position in range(args.sequence_length,
                              args.sequence_length + args.validation_steps):
            graph_logits = graph_runner.decode(token, position=position).clone()
            eager_logits = eager_runner.decode_eager(token, position=position).clone()
            torch.cuda.synchronize()
            step_errors.append(float((graph_logits - eager_logits).abs().max().item()))
            graph_token = graph_logits.argmax(dim=-1).contiguous()
            eager_token = eager_logits.argmax(dim=-1).contiguous()
            graph_tokens.append(graph_token.clone())
            eager_tokens.append(eager_token.clone())
            token = graph_token

        fixed_position = args.sequence_length + args.validation_steps
        graph_timing = measure(
            lambda: graph_runner.decode(token, position=fixed_position),
            warmup=5, iterations=args.iterations, repeats=args.repeats)
        eager_timing = measure(
            lambda: eager_runner.decode_eager(token, position=fixed_position),
            warmup=5, iterations=args.iterations, repeats=args.repeats)

        graph_generated = graph_runner.greedy_generate(
            input_ids, max_new_tokens=args.new_tokens)
        eager_generated = eager_greedy(eager_runner, input_ids, args.new_tokens)
        token_match = bool(torch.equal(graph_generated, eager_generated))
        item = {
            "batch_size": batch,
            "prefill_max_abs_error": prefill_error,
            "validation_step_max_abs_errors": step_errors,
            "validation_tokens_equal": bool(torch.equal(
                torch.cat(graph_tokens, dim=1), torch.cat(eager_tokens, dim=1))),
            "generated_tokens_equal": token_match,
            "generated_token_count_per_request": args.new_tokens,
            "graph_decode_wall_us": graph_timing,
            "eager_decode_wall_us": eager_timing,
            "decode_speedup": eager_timing["median_us"] / graph_timing["median_us"],
            "graph_aggregate_tokens_per_second": (
                batch * 1e6 / graph_timing["median_us"]
            ),
        }
        results.append(item)
        print(f"batch={batch} graph={graph_timing['median_us']:.3f}us "
              f"eager={eager_timing['median_us']:.3f}us "
              f"speedup={item['decode_speedup']:.3f}x tokens_equal={token_match}",
              flush=True)

    report = {
        "scope": "reusable NQRTGraphDecodeRunner validation",
        "device": torch.cuda.get_device_name(),
        "sequence_length": args.sequence_length,
        "new_tokens": args.new_tokens,
        "validation_steps": args.validation_steps,
        "results": results,
        "claim_boundary": (
            "Fixed batch size and greedy decoding. Argmax remains outside the CUDA Graph; "
            "request scheduling and mixed batch sizes are not included."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
