"""Check exact Llama logits after local paper-style Hadamard Linear conversion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.models.rotated_linear import convert_linears_to_hadamard_coordinates_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--revision", default="4e20de362430cd3b72f300e6b0f18e50e7166e08")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument(
        "--max-linears", type=int, default=None,
        help="convert only the first N eligible Linear modules for accumulation audits",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2")
    if args.max_linears is not None and args.max_linears < 1:
        parser.error("--max-linears must be positive")

    from huggingface_hub import get_token
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    token = get_token()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        token=token,
        torch_dtype=torch.bfloat16,
        local_files_only=args.offline,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    vocab_size = model.config.vocab_size
    generator = torch.Generator(device="cpu").manual_seed(58)
    input_ids = torch.randint(0, vocab_size, (1, args.tokens), generator=generator).to(device)

    with torch.inference_mode():
        reference = model(input_ids=input_ids, use_cache=False).logits.float().cpu()
    include_names = None
    if args.max_linears is not None:
        embedding_weight = model.get_input_embeddings().weight
        eligible = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and module.weight.data_ptr() != embedding_weight.data_ptr()
            and not (module.in_features & (module.in_features - 1))
        ]
        include_names = eligible[:args.max_linears]
    report = convert_linears_to_hadamard_coordinates_(model, include_names=include_names)
    with torch.inference_mode():
        rotated = model(input_ids=input_ids, use_cache=False).logits.float().cpu()
    difference = reference - rotated
    reference_norm = torch.linalg.vector_norm(reference)
    top1_agreement = (reference.argmax(dim=-1) == rotated.argmax(dim=-1)).float().mean()
    payload = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision": args.revision,
        "device": str(device),
        "dtype": "bfloat16",
        "token_count": args.tokens,
        "seed": 58,
        "converted_count": len(report.converted),
        "requested_max_linears": args.max_linears,
        "converted": list(report.converted),
        "skipped_tied": list(report.skipped_tied),
        "skipped_non_power_of_two": list(report.skipped_non_power_of_two),
        "max_abs_logit_error": difference.abs().max().item(),
        "mean_abs_logit_error": difference.abs().mean().item(),
        "relative_l2_logit_error": (
            torch.linalg.vector_norm(difference) / reference_norm
        ).item(),
        "top1_token_agreement": top1_agreement.item(),
        "reference_max_abs_logit": reference.abs().max().item(),
        "evidence_boundary": (
            "Short fixed-token equivalence smoke without quantization; not a PPL or speed result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
