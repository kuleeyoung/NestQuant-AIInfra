"""Run a complete Llama with packed NQRT4 weights and packed NQKV cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.models.fused_linear import FusedNQRTLinear
from nestquant.runtime import NQKVLlamaCache, install_nqkv_llama_backend


PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
KEY_BETAS = (0.25, 0.3271428571428571, 0.46214285714285713, 1.2185714285714284)
VALUE_BETAS = (0.2521428571428571, 0.3992857142857143, 0.687142857142857, 2.073571428571429)
ACTIVATION_BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--logits-output", type=Path)
    parser.add_argument("--prompt-length", type=int, default=17)
    parser.add_argument("--activation4", action="store_true")
    parser.add_argument(
        "--a4-kernel", choices=("scalar", "tiled"), default="tiled"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
    ).eval().to("cuda")
    names = tuple(
        f"model.layers.{layer}.{projection}"
        for layer in range(model.config.num_hidden_layers)
        for projection in PROJECTIONS
    )
    conversion = convert_linears_to_fused_nqrt_(
        model, reader, names, output_dtype=torch.bfloat16,
        group_qkv=False,
        compute_backend=(
            f"packed_w4a4_{args.a4_kernel}"
            if args.activation4 and args.a4_kernel == "tiled"
            else "packed_w4a4" if args.activation4 else "cuda_mma_j2"
        ),
        activation_q=14,
        activation_betas=ACTIVATION_BETAS if args.activation4 else None,
    )
    config = model.config
    cache = NQKVLlamaCache(
        num_layers=config.num_hidden_layers,
        num_query_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_cache_len=32,
        page_size=16,
        q=14,
        key_betas=KEY_BETAS,
        value_betas=VALUE_BETAS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    install_nqkv_llama_backend(model, cache)
    tokens = [128000] + [42 + index for index in range(args.prompt_length - 1)]
    input_ids = torch.tensor([tokens], device="cuda", dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            min_new_tokens=3,
            max_new_tokens=3,
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    lengths = [cache.get_seq_length(index) for index in range(cache.num_layers)]
    cache.reset()
    torch.cuda.synchronize()
    hot_started = time.perf_counter()
    with torch.inference_mode():
        hot_generated = model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            min_new_tokens=3,
            max_new_tokens=3,
            do_sample=False,
        )
    torch.cuda.synchronize()
    hot_elapsed_ms = (time.perf_counter() - hot_started) * 1000.0
    hot_lengths = [cache.get_seq_length(index) for index in range(cache.num_layers)]
    if args.logits_output is not None:
        cache.reset()
        with torch.inference_mode():
            comparison_logits = model(
                input_ids=input_ids, past_key_values=cache, use_cache=True
            ).logits[:, -1].detach().cpu()
        args.logits_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(comparison_logits, args.logits_output)
    packed_weight_bytes = sum(
        int(getattr(module, "packed_device_bytes", 0)) for module in model.modules()
    )
    fused_modules = [
        module for module in model.modules() if isinstance(module, FusedNQRTLinear)
    ]
    result = {
        "protocol": (
            "llama32-1b-packed-w4a4kv4-hf-smoke-v1"
            if args.activation4 else "llama32-1b-packed-w4kv4-hf-smoke-v1"
        ),
        "checkpoint": str(args.checkpoint),
        "model": str(args.model),
        "weight_runtime": (
            f"NQRT4 packed direct {args.a4_kernel} W4xA4 Triton"
            if args.activation4 else "NQRT4 packed cuda_mma_j2/tensor_core_dot fallback"
        ),
        "kv_runtime": "NQKV v2 packed K1",
        "replaced_linear_count": len(conversion.replaced_linear_names),
        "prompt_length": args.prompt_length,
        "generated_token_ids": generated[0].tolist(),
        "elapsed_ms_including_lazy_weight_prepare": elapsed_ms,
        "hot_generate_ms": hot_elapsed_ms,
        "hot_generated_token_ids": hot_generated[0].tolist(),
        "layer_cache_lengths": lengths,
        "all_layers_expected_length": all(
            length == args.prompt_length + 2 for length in lengths
        ),
        "hot_all_layers_expected_length": all(
            length == args.prompt_length + 2 for length in hot_lengths
        ),
        "packed_weight_device_bytes": packed_weight_bytes,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "activation_runtime": (
            "NQA4 packed direct W4xA4 Triton"
            if args.activation4 else "BF16"
        ),
        "activation_betas": list(ACTIVATION_BETAS) if args.activation4 else None,
        "a4_module_count": sum(
            module.compute_backend.startswith("packed_w4a4") for module in fused_modules
        ),
        "a4_kernel_call_count": sum(
            module.a4_call_count for module in fused_modules
        ),
        "a4_encode_count": sum(
            module.a4_encode_count for module in fused_modules
        ),
        "a4_cumulative_packed_input_bytes": sum(
            module.a4_packed_input_bytes for module in fused_modules
        ),
        "a4_unique_encoded_input_bytes": sum(
            module.a4_encoded_input_bytes for module in fused_modules
        ),
        "a4_largest_single_module_payload_bytes": max(
            (module.a4_max_resident_bytes for module in fused_modules), default=0
        ),
    }
    if len(conversion.replaced_linear_names) != config.num_hidden_layers * 7:
        raise RuntimeError("not every Llama projection was replaced by packed W4")
    if not result["all_layers_expected_length"] or not result["hot_all_layers_expected_length"]:
        raise RuntimeError("packed W4+KV4 cache length validation failed")
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
