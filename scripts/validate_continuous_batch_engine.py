"""Validate host-driven continuous batching on the packed Llama-3.2-1B runtime."""

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
from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.inference_engine import ContinuousBatchInferenceEngine
from nestquant.runtime.ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


@torch.inference_mode()
def greedy_full(model, prompt: torch.Tensor, count: int) -> list[int]:
    sequence = prompt.clone()
    generated = []
    for _ in range(count):
        token = int(model(input_ids=sequence).logits[:, -1].argmax(dim=-1).item())
        generated.append(token)
        sequence = torch.cat(
            (sequence, torch.tensor([[token]], dtype=torch.long, device=sequence.device)), dim=1
        )
    return generated


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
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
    runner = NQRTRaggedGraphDecodeRunner(model, batch_size=2, max_cache_len=64)
    engine = ContinuousBatchInferenceEngine(runner, pad_token_id=0)
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    prompts = [
        torch.randint(1, model.config.vocab_size, (1, 7), device="cuda", generator=generator),
        torch.randint(1, model.config.vocab_size, (1, 4), device="cuda", generator=generator),
        torch.randint(1, model.config.vocab_size, (1, 3), device="cuda", generator=generator),
    ]
    limits = [3, 2, 2]
    requests = [
        DecodeRequest(f"request-{index}", tuple(prompt[0].tolist()), limit)
        for index, (prompt, limit) in enumerate(zip(prompts, limits))
    ]
    for request in requests:
        engine.submit(request)

    timeline = []
    while engine.has_work():
        step = engine.step()
        timeline.append({
            "admitted": list(step.admitted_request_ids),
            "generated": step.generated_tokens,
            "finished": list(step.finished_request_ids),
            "active": list(step.active_request_ids),
        })
        if len(timeline) > 16:
            raise RuntimeError("engine failed to drain the bounded workload")

    references = [greedy_full(model, prompt, limit) for prompt, limit in zip(prompts, limits)]
    results = []
    for request, reference in zip(requests, references):
        actual = request.generated_token_ids
        results.append({
            "request_id": request.request_id,
            "prompt_length": len(request.prompt_token_ids),
            "generated": actual,
            "reference": reference,
            "equal": actual == reference,
        })
    report = {
        "status": "passed" if all(item["equal"] for item in results) else "failed",
        "device": torch.cuda.get_device_name(),
        "capacity": 2,
        "request_count": 3,
        "timeline": timeline,
        "results": results,
        "claim_boundary": (
            "Deterministic greedy host-driven validation. Random arrivals, cancellation, "
            "soak duration, latency distributions and network serving are not included."
        ),
    }
    if report["status"] != "passed":
        raise AssertionError(json.dumps(report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
