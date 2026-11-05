"""Evaluate scalar or tiled packed W4A4 on one deterministic text window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_


PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
TEXT = (
    "The history of natural language processing began with careful studies of "
    "language, information, and computation. Modern neural networks learn "
    "statistical structure from large collections of text, while efficient "
    "inference systems reduce memory traffic and computation. "
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--kernel", choices=("scalar", "tiled"), required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
    ).eval().to("cuda")
    names = tuple(
        f"model.layers.{layer}.{projection}"
        for layer in range(model.config.num_hidden_layers)
        for projection in PROJECTIONS
    )
    convert_linears_to_fused_nqrt_(
        model, reader, names, output_dtype=torch.bfloat16,
        compute_backend=(
            "packed_w4a4_tiled" if args.kernel == "tiled" else "packed_w4a4"
        ),
        activation_q=14, activation_betas=BETAS,
    )
    repeated = TEXT * math.ceil(args.sequence_length * 6 / len(TEXT.split()))
    token_ids = tokenizer(repeated, return_tensors="pt").input_ids[:, :args.sequence_length]
    if token_ids.shape[1] != args.sequence_length:
        raise RuntimeError("micro-PPL text did not produce enough tokens")
    token_sha256 = hashlib.sha256(
        token_ids.contiguous().numpy().tobytes()
    ).hexdigest()
    token_ids = token_ids.cuda()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        output = model(input_ids=token_ids, labels=token_ids, use_cache=False)
    end.record()
    end.synchronize()
    loss = float(output.loss.item())
    hot_start = torch.cuda.Event(enable_timing=True)
    hot_end = torch.cuda.Event(enable_timing=True)
    hot_start.record()
    with torch.inference_mode():
        hot_output = model(input_ids=token_ids, labels=token_ids, use_cache=False)
    hot_end.record()
    hot_end.synchronize()
    hot_loss = float(hot_output.loss.item())
    payload = {
        "protocol": "packed-w4a4-micro-ppl-v1",
        "kernel": args.kernel,
        "sequence_length": args.sequence_length,
        "predicted_tokens": args.sequence_length - 1,
        "token_ids_sha256": token_sha256,
        "loss": loss,
        "perplexity": math.exp(loss),
        "forward_ms_including_lazy_prepare": start.elapsed_time(end),
        "hot_forward_ms": hot_start.elapsed_time(hot_end),
        "hot_loss": hot_loss,
        "scope": "one deterministic local text window; not WikiText-2",
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
