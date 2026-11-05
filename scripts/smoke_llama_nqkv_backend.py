"""Run two real Llama decode steps through the Transformers NQKV backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM

from nestquant.runtime import NQKVLlamaCache, install_nqkv_llama_backend


KEY_BETAS = (0.25, 0.3271428571428571, 0.46214285714285713, 1.2185714285714284)
VALUE_BETAS = (0.2521428571428571, 0.3992857142857143, 0.687142857142857, 2.073571428571429)


def _measure_step(model, token_ids: list[int], cache: NQKVLlamaCache):
    input_ids = torch.tensor([token_ids], device="cuda", dtype=torch.long)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Local Hugging Face model snapshot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-cache-len", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--prompt-length", type=int, default=17)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
    ).eval().to("cuda")
    load_seconds = time.perf_counter() - load_start
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    cache = NQKVLlamaCache(
        num_layers=config.num_hidden_layers,
        num_query_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        max_cache_len=args.max_cache_len,
        page_size=args.page_size,
        q=14,
        key_betas=KEY_BETAS,
        value_betas=VALUE_BETAS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    backend_name = install_nqkv_llama_backend(model, cache)

    prompt_tokens = [128000] + [42 + index for index in range(args.prompt_length - 1)]
    with torch.inference_mode():
        prompt, prefill_ms = _measure_step(model, prompt_tokens, cache)
        first_decoded, first_decode_ms = _measure_step(model, [42], cache)
        decoded, hot_decode_ms = _measure_step(model, [43], cache)

    lengths = [cache.get_seq_length(index) for index in range(cache.num_layers)]
    logits = decoded.logits
    cache.reset()
    generate_input = torch.tensor([prompt_tokens], device="cuda", dtype=torch.long)
    torch.cuda.synchronize()
    generate_start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=generate_input,
            past_key_values=cache,
            min_new_tokens=3,
            max_new_tokens=3,
            do_sample=False,
        )
    torch.cuda.synchronize()
    generate_ms = (time.perf_counter() - generate_start) * 1000.0
    generate_lengths = [
        cache.get_seq_length(index) for index in range(cache.num_layers)
    ]
    result = {
        "protocol": "llama32-1b-nqkv-hf-backend-smoke-v2",
        "model_path": str(args.model),
        "backend": backend_name,
        "dtype": str(cache.dtype),
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": head_dim,
        "page_size": args.page_size,
        "max_cache_len": args.max_cache_len,
        "prompt_length": args.prompt_length,
        "decode_steps": 2,
        "load_seconds": load_seconds,
        "prefill_ms": prefill_ms,
        "first_decode_ms": first_decode_ms,
        "hot_decode_ms": hot_decode_ms,
        "layer_cache_lengths": lengths,
        "all_layers_expected_length": all(
            length == args.prompt_length + 2 for length in lengths
        ),
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "next_token": int(logits[:, -1].argmax(dim=-1).item()),
        "generate_ms": generate_ms,
        "generated_token_ids": generated[0].tolist(),
        "generate_layer_cache_lengths": generate_lengths,
        "generate_all_layers_expected_length": all(
            length == args.prompt_length + 2 for length in generate_lengths
        ),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "limitations": [
            "batch_size_one",
            "no_beam_reorder",
            "prefill_supports_unpadded_initial_prompts_only",
            "latency_includes_first-call compilation",
        ],
    }
    if (
        not result["all_layers_expected_length"]
        or not result["generate_all_layers_expected_length"]
        or not result["logits_finite"]
    ):
        raise RuntimeError(f"real-model NQKV smoke validation failed: {result}")
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
