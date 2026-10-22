"""NestQuant A4 quality simulation for Llama projection inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import weakref

import torch

from ..core.rotation import hadamard
from ..core.runtime_quantization import fake_quantize_nestquant_rows


LLAMA_PROJECTION_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
)


class ActivationQuantizationStats:
    """Accumulate projection-input A4 error without retaining activations."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def record(self, module_name: str, stats: dict) -> None:
        current = self._records.setdefault(module_name, {
            "updates": 0,
            "vector_count": 0,
            "entry_count": 0,
            "e8_block_count": 0,
            "squared_error": 0.0,
            "signal_energy": 0.0,
            "max_abs_error": 0.0,
            "overload_count": 0,
            "beta_histogram": [0] * len(stats["betas"]),
            "q": stats["q"],
            "betas": stats["betas"],
            "feature_count": stats["feature_count"],
            "storage_estimate": stats["storage_estimate"],
        })
        if (current["q"] != stats["q"] or current["betas"] != stats["betas"]
                or current["feature_count"] != stats["feature_count"]):
            raise ValueError("inconsistent activation quantization configuration")
        current["updates"] += 1
        for field in ("vector_count", "entry_count", "e8_block_count"):
            current[field] += stats[field]
        for field in ("squared_error", "signal_energy"):
            current[field] += stats[field]
        current["max_abs_error"] = max(current["max_abs_error"], stats["max_abs_error"])
        current["overload_count"] += stats["overload_count"]
        current["beta_histogram"] = [
            left + right
            for left, right in zip(current["beta_histogram"], stats["beta_histogram"])
        ]

    def summary(self) -> dict:
        modules = {}
        total_entries = total_vectors = total_blocks = total_overloads = 0
        total_error = total_signal = 0.0
        estimated_bits = 0
        for name, record in sorted(self._records.items()):
            item = dict(record)
            item["relative_squared_error"] = (
                item["squared_error"] / item["signal_energy"]
                if item["signal_energy"] else 0.0
            )
            item["overload_fraction"] = (
                item["overload_count"] / item["e8_block_count"]
                if item["e8_block_count"] else 0.0
            )
            storage = item["storage_estimate"]
            item_bits = (
                storage["residue_bits_per_entry"] * item["entry_count"]
                + storage["beta_index_bits_per_block"] * item["e8_block_count"]
                + storage["row_norm_bits_per_vector"] * item["vector_count"]
            )
            item["storage_estimate"] = dict(storage) | {
                "estimated_total_bits": item_bits,
                "effective_bits_per_entry": item_bits / item["entry_count"],
            }
            modules[name] = item
            total_entries += item["entry_count"]
            total_vectors += item["vector_count"]
            total_blocks += item["e8_block_count"]
            total_overloads += item["overload_count"]
            total_error += item["squared_error"]
            total_signal += item["signal_energy"]
            estimated_bits += item_bits
        return {
            "status": "fake_quant_quality_simulation",
            "location": "inputs_to_llama_attention_and_mlp_projection_linears",
            "rotation": "normalized_hadamard_before_e8_and_inverse_after_reconstruction",
            "runtime_dtype": "floating_point_reconstruction",
            "module_count": len(modules),
            "entry_count": total_entries,
            "vector_count": total_vectors,
            "e8_block_count": total_blocks,
            "squared_error": total_error,
            "signal_energy": total_signal,
            "relative_squared_error": total_error / total_signal if total_signal else 0.0,
            "overload_count": total_overloads,
            "overload_fraction": total_overloads / total_blocks if total_blocks else 0.0,
            "analytical_rate": {
                "status": "per_module_estimate_only_no_low_bit_activation_kernel",
                "estimated_total_bits": estimated_bits,
                "effective_bits_per_entry": estimated_bits / total_entries if total_entries else None,
            },
            "modules": modules,
        }


def enable_activation_quantization(
    model,
    *,
    q: int,
    betas: Sequence[float],
    rows_per_chunk: int = 4096,
    stats: ActivationQuantizationStats | None = None,
) -> tuple[ActivationQuantizationStats, list[torch.utils.hooks.RemovableHandle]]:
    """Attach A4 pre-hooks to the seven Llama projection types per layer."""
    accumulator = stats if stats is not None else ActivationQuantizationStats()
    handles = []
    matched = []
    # Llama passes the same tensor object to q/k/v and to gate/up.  The
    # quantizer is deterministic, so retaining one reconstruction only while
    # that source tensor is alive removes duplicate work without changing the
    # simulated value.  A weak reference prevents cross-forward reuse when an
    # allocator later recycles the same Python id or storage address.
    cache: dict[int, tuple[weakref.ReferenceType, torch.Tensor, dict]] = {}
    for name, module in model.named_modules():
        if not name.endswith(LLAMA_PROJECTION_SUFFIXES):
            continue
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"expected torch.nn.Linear at {name}")
        matched.append(name)

        def hook(_module, inputs, *, module_name=name):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{module_name} must receive a tensor as its first input")
            source = inputs[0]
            key = id(source)
            cached = cache.get(key)
            if cached is not None and cached[0]() is source:
                restored, quant_stats = cached[1], cached[2]
            else:
                rotated = hadamard(source.float())
                quantized, quant_stats = fake_quantize_nestquant_rows(
                    rotated, q=q, betas=betas, rows_per_chunk=rows_per_chunk,
                )
                restored = hadamard(quantized.float()).to(dtype=source.dtype)

                def clear(reference, *, source_id=key):
                    current = cache.get(source_id)
                    if current is not None and current[0] is reference:
                        cache.pop(source_id, None)

                cache[key] = (weakref.ref(source, clear), restored, quant_stats)
            accumulator.record(module_name, quant_stats)
            return (restored,) + tuple(inputs[1:])

        handles.append(module.register_forward_pre_hook(hook))
    if not matched:
        raise ValueError("no supported Llama attention/MLP projection modules were found")
    return accumulator, handles
