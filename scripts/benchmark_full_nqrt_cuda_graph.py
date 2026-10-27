"""Test CUDA Graph capture and replay for one-token full-model decode."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch
from transformers import StaticCache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


def event_measure(operation, warmup: int, iterations: int, repeats: int) -> dict[str, float]:
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


def wall_measure(operation, warmup: int, iterations: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(); started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6 / iterations)
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}


def benchmark_mode(reader, names, mode: str, batch: int, sequence_length: int,
                   warmup: int, iterations: int, repeats: int, offline: bool,
                   no_cache_probe: bool, capture_api: str) -> dict[str, object]:
    model_id = reader.model_metadata["model_id"]
    revision = reader.model_metadata["model_revision"]
    model = _load_model(model_id, revision, offline)
    if mode != "bf16":
        convert_linears_to_fused_nqrt_(
            model, reader, names, output_dtype=torch.bfloat16,
            group_qkv=False, compute_backend=mode,
        )
    generator = torch.Generator(device="cuda").manual_seed(20261026 + batch)
    input_ids = torch.randint(0, model.config.vocab_size, (batch, sequence_length),
                              device="cuda", generator=generator)
    cache = StaticCache(model.config, max_cache_len=sequence_length + 10)
    prefill_positions = torch.arange(sequence_length, device="cuda", dtype=torch.long)
    with torch.inference_mode():
        prefill = model(input_ids=input_ids, past_key_values=cache,
                        cache_position=prefill_positions, use_cache=True)
        token = prefill.logits[:, -1:].argmax(dim=-1).contiguous()
        warm_position = torch.tensor([sequence_length], device="cuda", dtype=torch.long)
        warm = model(input_ids=token, past_key_values=cache,
                     cache_position=warm_position, use_cache=True)
        static_token = warm.logits[:, -1:].argmax(dim=-1).contiguous()
        static_position = torch.tensor([sequence_length + 1], device="cuda", dtype=torch.long)
        position_holder = [static_position]

        # Transformers 5.20 StaticLayer.update creates an arange tensor from a
        # mutable cumulative_length on every call. Replace only the decode-time
        # update with a fixed-address/fixed-position operation suitable for a
        # CUDA Graph. Prefill and the first warm decode above still use the
        # library implementation.
        def fixed_update(layer, key_states, value_states, *unused_args, **unused_kwargs):
            layer.keys.index_copy_(2, position_holder[0], key_states)
            layer.values.index_copy_(2, position_holder[0], value_states)
            return layer.keys, layer.values

        for layer in cache.layers:
            layer.update = types.MethodType(fixed_update, layer)

        def forward_token(token, position):
            position_holder[0] = position
            position_ids = position.view(1, 1).expand(batch, 1)
            if no_cache_probe:
                return model(input_ids=token, position_ids=position_ids,
                             use_cache=False).logits
            return model(input_ids=token, past_key_values=cache,
                         cache_position=position,
                         position_ids=position_ids, use_cache=True).logits

        def eager_step():
            return forward_token(static_token, static_position)

        eager_event = event_measure(eager_step, warmup, iterations, repeats)
        eager_wall = wall_measure(eager_step, warmup, iterations, repeats)

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                eager_step()
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        eager_reference = eager_step().detach().clone()

        if capture_api == "raw":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_logits = eager_step()
            torch.cuda.synchronize()
            capture_logits = graph_logits.detach().clone()
            graph.replay(); torch.cuda.synchronize()
            replay_logits = graph_logits.detach().clone()
            graph.replay(); torch.cuda.synchronize()
            replay_twice_logits = graph_logits.detach().clone()
            graph_step = graph.replay
        else:
            graph = None
            graphed_callable = torch.cuda.make_graphed_callables(
                forward_token, (static_token, static_position), num_warmup_iters=3,
            )
            capture_logits = graphed_callable(static_token, static_position).detach().clone()
            replay_logits = graphed_callable(static_token, static_position).detach().clone()
            replay_twice_logits = graphed_callable(static_token, static_position).detach().clone()
            graph_step = lambda: graphed_callable(static_token, static_position)
        eager_after_graph = eager_step().detach().clone()
        torch.cuda.synchronize()
        correctness = {
            "capture_vs_eager_max_abs": float((capture_logits - eager_reference).abs().max().item()),
            "replay_vs_capture_max_abs": float((replay_logits - capture_logits).abs().max().item()),
            "second_replay_vs_first_max_abs": float((replay_twice_logits - replay_logits).abs().max().item()),
            "eager_after_graph_vs_eager_before_max_abs": float(
                (eager_after_graph - eager_reference).abs().max().item()
            ),
        }
        graph_event = event_measure(graph_step, warmup, iterations, repeats)
        graph_wall = wall_measure(graph_step, warmup, iterations, repeats)
        dynamic_position_errors = []
        if capture_api == "make_graphed" and not no_cache_probe:
            dynamic_token = replay_twice_logits[:, -1:].argmax(dim=-1).contiguous()
            for position_value in range(sequence_length + 2, sequence_length + 6):
                dynamic_position = torch.tensor(
                    [position_value], device="cuda", dtype=torch.long
                )
                graph_dynamic = graphed_callable(
                    dynamic_token, dynamic_position
                ).detach().clone()
                eager_dynamic = forward_token(
                    dynamic_token, dynamic_position
                ).detach().clone()
                torch.cuda.synchronize()
                dynamic_position_errors.append({
                    "position": position_value,
                    "max_abs_graph_vs_eager": float(
                        (graph_dynamic - eager_dynamic).abs().max().item()
                    ),
                })
                dynamic_token = graph_dynamic[:, -1:].argmax(dim=-1).contiguous()

    result = {
        "mode": mode, "batch_size": batch,
        "no_cache_probe": no_cache_probe,
        "capture_api": capture_api,
        "eager_event_us": eager_event, "graph_event_us": graph_event,
        "eager_wall_us": eager_wall, "graph_wall_us": graph_wall,
        "wall_speedup": eager_wall["median_us"] / graph_wall["median_us"],
        "correctness": correctness,
        "dynamic_position_validation": dynamic_position_errors,
        "capture_semantics": (
            "The graph replays one fixed token at one fixed cache position. "
            "It is a launch-overhead feasibility test, not autoregressive generation."
        ),
    }
    del graph, cache, model, input_ids, prefill, warm
    gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--modes", nargs="+", choices=("bf16", "cuda_mma_j2"),
                        default=["cuda_mma_j2"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-cache-probe", action="store_true")
    parser.add_argument("--capture-api", choices=("raw", "make_graphed"),
                        default="make_graphed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (min(args.batch_size) < 1 or args.sequence_length < 1 or args.warmup < 0
            or args.iterations < 1 or args.repeats < 1):
        parser.error("invalid benchmark counts")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    probe = _load_model(reader.model_metadata["model_id"],
                        reader.model_metadata["model_revision"], args.offline)
    names = _projection_names(probe, reader)
    del probe; gc.collect(); torch.cuda.empty_cache()
    results = []
    for batch in args.batch_size:
        for mode in args.modes:
            item = benchmark_mode(reader, names, mode, batch, args.sequence_length,
                                  args.warmup, args.iterations, args.repeats, args.offline,
                                  args.no_cache_probe, args.capture_api)
            results.append(item)
            print(f"mode={mode} batch={batch} eager={item['eager_wall_us']['median_us']:.3f}us "
                  f"graph={item['graph_wall_us']['median_us']:.3f}us "
                  f"speedup={item['wall_speedup']:.3f}x", flush=True)
    report = {
        "scope": "fixed-address one-token CUDA Graph feasibility",
        "device": torch.cuda.get_device_name(), "sequence_length": args.sequence_length,
        "iterations": args.iterations, "repeats": args.repeats,
        "results": results,
        "claim_boundary": (
            "Fixed token and cache position replay only. Dynamic token and cache-position "
            "updates must be validated before integration into generation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
