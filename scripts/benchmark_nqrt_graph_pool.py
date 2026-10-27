"""Validate a multi-capacity NQRT CUDA Graph pool on a packed Llama model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.cuda_graph_decode import NQRTGraphDecodePool
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--active-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7, 8])
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--max-cache-len", type=int, default=64)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = _load_model(reader.model_metadata["model_id"],
                        reader.model_metadata["model_revision"], args.offline)
    names = _projection_names(model, reader)
    convert_linears_to_fused_nqrt_(
        model, reader, names, output_dtype=torch.bfloat16,
        group_qkv=False, compute_backend="cuda_mma_j2",
    )
    pool = NQRTGraphDecodePool(
        model, batch_sizes=args.batch_sizes, max_cache_len=args.max_cache_len)
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    results = []
    captured = set()
    for active_count in args.active_counts:
        inputs = torch.randint(
            0, model.config.vocab_size,
            (active_count, args.sequence_length),
            device="cuda", generator=generator,
        )
        prefill_logits = pool.prefill(inputs)
        token = prefill_logits.argmax(dim=-1).contiguous()
        capacity = pool.capacity
        if capacity not in captured:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            pool.capture(token, position=args.sequence_length)
            torch.cuda.synchronize()
            capture_delta = torch.cuda.memory_allocated() - before
            captured.add(capacity)
        else:
            capture_delta = 0
        graph_logits = pool.decode(token, position=args.sequence_length).clone()
        eager_logits = pool.decode_eager(token, position=args.sequence_length).clone()
        torch.cuda.synchronize()
        error = float((graph_logits - eager_logits).abs().max().item())
        graph_token = graph_logits.argmax(dim=-1)
        eager_token = eager_logits.argmax(dim=-1)
        result = {
            "active_count": active_count,
            "capacity": capacity,
            "padding_slots": capacity - active_count,
            "capture_allocation_delta_bytes": capture_delta,
            "max_abs_error": error,
            "argmax_equal": bool(torch.equal(graph_token, eager_token)),
        }
        if error != 0.0 or not result["argmax_equal"]:
            raise AssertionError(f"graph pool mismatch: {result}")
        results.append(result)
        print(json.dumps(result), flush=True)

    report = {
        "status": "passed",
        "scope": "NQRTGraphDecodePool fixed-position padding validation",
        "device": torch.cuda.get_device_name(),
        "batch_sizes": args.batch_sizes,
        "sequence_length": args.sequence_length,
        "max_cache_len": args.max_cache_len,
        "results": results,
        "final_allocated_bytes": torch.cuda.memory_allocated(),
        "final_reserved_bytes": torch.cuda.memory_reserved(),
        "claim_boundary": (
            "All active requests in a replay share prompt length and cache position. "
            "Per-request positions and continuous batching are not implemented."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
