"""Post-RoPE NestQuant KV-cache quality simulation for Transformers models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..core.beta_search import build_candidate_tables, select_beta_subset
from ..core.rotation import hadamard
from ..core.runtime_quantization import fake_quantize_nestquant_rows


_ROTATED_SDPA_NAME = "nestquant_rotated_sdpa"
_BASE_SDPA_ATTENTION = ALL_ATTENTION_FUNCTIONS["sdpa"]


def hadamard_attention_heads(states: torch.Tensor) -> torch.Tensor:
    """Apply the paper's normalized Hadamard independently to each head."""
    return hadamard(states.float()).to(dtype=states.dtype)


def _rotated_sdpa_attention(
    module, query, key, value, attention_mask, **kwargs
):
    return _BASE_SDPA_ATTENTION(
        module, hadamard_attention_heads(query), key, value, attention_mask, **kwargs
    )


def enable_kv_attention_rotation(model) -> str:
    """Route Llama attention through SDPA with a post-RoPE query Hadamard."""
    ALL_ATTENTION_FUNCTIONS.register(_ROTATED_SDPA_NAME, _rotated_sdpa_attention)
    model.config._attn_implementation = _ROTATED_SDPA_NAME
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None:
            config._attn_implementation = _ROTATED_SDPA_NAME
    return _ROTATED_SDPA_NAME


class KVQuantizationStats:
    """Accumulate scalar KV fake-quant measurements across context windows."""

    def __init__(self) -> None:
        self._records: dict[tuple[int, str], dict[str, Any]] = {}

    def record(self, layer_idx: int, kind: str, stats: dict) -> None:
        key = (int(layer_idx), kind)
        current = self._records.setdefault(key, {
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
            raise ValueError("inconsistent KV quantization configuration for one layer")
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
        layers: dict[str, dict] = defaultdict(dict)
        total_entries = 0
        total_vectors = 0
        total_blocks = 0
        total_error = 0.0
        total_signal = 0.0
        estimated_bits = 0
        for (layer_idx, kind), record in sorted(self._records.items()):
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
            layers[str(layer_idx)][kind] = item
            total_entries += item["entry_count"]
            total_vectors += item["vector_count"]
            total_blocks += item["e8_block_count"]
            total_error += item["squared_error"]
            total_signal += item["signal_energy"]
            estimated_bits += item_bits
        return {
            "status": "fake_quant_quality_simulation",
            "location": "post_rope_before_dynamic_cache_update",
            "vector_layout": "one_row_per_attention_head",
            "key_rotation": "normalized_hadamard_after_rope",
            "query_rotation": "normalized_hadamard_after_rope_via_attention_interface",
            "value_rotation": "none",
            "stored_runtime_dtype": "floating_point_reconstruction",
            "layer_count": len(layers),
            "entry_count": total_entries,
            "vector_count": total_vectors,
            "e8_block_count": total_blocks,
            "squared_error": total_error,
            "signal_energy": total_signal,
            "relative_squared_error": total_error / total_signal if total_signal else 0.0,
            "storage_estimate": {
                "status": "estimate_only_no_packed_kv_cache_allocated",
                "estimated_total_bits": estimated_bits,
                "effective_bits_per_entry": estimated_bits / total_entries if total_entries else None,
            },
            "layers": dict(layers),
        }


class NestQuantDynamicCache(DynamicCache):
    """DynamicCache that fake-quantizes each new post-RoPE K/V tensor."""

    def __init__(
        self,
        *,
        config,
        q: int,
        default_betas: Sequence[float],
        layer_betas: Mapping[int, Mapping[str, Sequence[float]]] | None = None,
        rows_per_chunk: int = 4096,
        stats: KVQuantizationStats | None = None,
    ) -> None:
        super().__init__(config=config)
        self.q = q
        self.default_betas = tuple(float(value) for value in default_betas)
        self.layer_betas = layer_betas or {}
        self.rows_per_chunk = rows_per_chunk
        self.quantization_stats = stats if stats is not None else KVQuantizationStats()

    def _betas(self, layer_idx: int, kind: str) -> tuple[float, ...]:
        configured = self.layer_betas.get(layer_idx, {}).get(kind, self.default_betas)
        return tuple(float(value) for value in configured)

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        rotated_key = hadamard_attention_heads(key_states)
        quantized_key, key_stats = fake_quantize_nestquant_rows(
            rotated_key, q=self.q, betas=self._betas(layer_idx, "key"),
            rows_per_chunk=self.rows_per_chunk,
        )
        quantized_value, value_stats = fake_quantize_nestquant_rows(
            value_states, q=self.q, betas=self._betas(layer_idx, "value"),
            rows_per_chunk=self.rows_per_chunk,
        )
        self.quantization_stats.record(layer_idx, "key", key_stats)
        self.quantization_stats.record(layer_idx, "value", value_stats)
        return super().update(
            quantized_key, quantized_value, layer_idx, *args, **kwargs
        )


class KVBlockSampler:
    """Bounded deterministic samples of normalized post-RoPE K/V E8 blocks."""

    def __init__(self, *, calibration_chunks: int, sample_limit_per_tensor: int) -> None:
        if calibration_chunks < 1 or sample_limit_per_tensor < 1:
            raise ValueError("calibration_chunks and sample limit must be positive")
        self.calibration_chunks = calibration_chunks
        self.sample_limit_per_tensor = sample_limit_per_tensor
        self._samples: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(list)
        self._sample_counts: dict[tuple[int, str], int] = defaultdict(int)
        self._seen_blocks: dict[tuple[int, str], int] = defaultdict(int)

    def record(self, layer_idx: int, kind: str, tensor: torch.Tensor) -> None:
        if kind not in {"key", "value"}:
            raise ValueError("kind must be key or value")
        if tensor.shape[-1] % 8:
            raise ValueError("KV head dimension must be divisible by 8")
        rows = tensor.detach().float().reshape(-1, tensor.shape[-1])
        norms = torch.linalg.vector_norm(rows, dim=-1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        normalized = rows * (math.sqrt(rows.shape[-1]) / safe_norms).unsqueeze(-1)
        normalized = torch.where(norms.unsqueeze(-1) > 0, normalized, torch.zeros_like(normalized))
        blocks = normalized.reshape(-1, 8)
        key = (int(layer_idx), kind)
        self._seen_blocks[key] += blocks.shape[0]
        remaining = self.sample_limit_per_tensor - self._sample_counts[key]
        if remaining <= 0:
            return
        per_update = math.ceil(self.sample_limit_per_tensor / self.calibration_chunks)
        take = min(remaining, per_update, blocks.shape[0])
        # Evenly spaced positions avoid storing a giant random permutation and
        # ensure every calibration window contributes to the bounded sample.
        indices = torch.div(
            torch.arange(take, device=blocks.device, dtype=torch.int64) * blocks.shape[0],
            take,
            rounding_mode="floor",
        )
        sampled = blocks.index_select(0, indices).to(device="cpu", dtype=torch.float32)
        self._samples[key].append(sampled)
        self._sample_counts[key] += take

    def search(
        self,
        *,
        q: int,
        candidate_betas: Sequence[float],
        selected_beta_count: int,
        device: torch.device | str,
        rows_per_chunk: int = 8192,
        overload_margin: float = 0.0,
        progress_callback=None,
    ) -> dict[str, dict[str, dict]]:
        if rows_per_chunk < 1:
            raise ValueError("rows_per_chunk must be positive")
        beta_values = tuple(float(value) for value in candidate_betas)
        result: dict[str, dict[str, dict]] = defaultdict(dict)
        keys = sorted(self._samples)
        for completed, key in enumerate(keys, 1):
            layer_idx, kind = key
            samples = torch.cat(self._samples[key], dim=0)
            digest = hashlib.sha256(samples.contiguous().numpy().tobytes()).hexdigest()
            mse_parts, overload_parts = [], []
            for start in range(0, samples.shape[0], rows_per_chunk):
                vectors = samples[start:start + rows_per_chunk].to(device)
                mse, overload = build_candidate_tables(vectors, q, beta_values)
                mse_parts.append(mse.cpu())
                overload_parts.append(overload.cpu())
            mse = torch.cat(mse_parts, dim=0)
            overload = torch.cat(overload_parts, dim=0)
            selected = select_beta_subset(
                beta_values, mse, overload, selected_beta_count
            )
            pre_margin_indices = list(selected.beta_indices)
            final_indices = list(selected.beta_indices)
            if overload_margin:
                target = beta_values[final_indices[-1]] + overload_margin
                margin_index = next(
                    (index for index, beta in enumerate(beta_values) if beta >= target),
                    None,
                )
                if margin_index is None:
                    raise ValueError(
                        f"candidate beta grid cannot provide overload margin for layer "
                        f"{layer_idx} {kind}; target={target}"
                    )
                final_indices[-1] = margin_index
            final_betas = [beta_values[index] for index in final_indices]
            final_overload = overload[:, final_indices]
            if final_overload.all(dim=1).any().item():
                raise RuntimeError("final beta subset leaves calibration samples overloaded")
            first_fitting = (~final_overload).to(torch.int64).argmax(dim=1)
            assignments = torch.tensor(final_indices, dtype=torch.int64)[first_fitting]
            final_first_beta_mse = float(
                mse[torch.arange(mse.shape[0]), assignments].sum().item()
            )
            result[str(layer_idx)][kind] = {
                "beta_indices_before_overload_margin": pre_margin_indices,
                "beta_values_before_overload_margin": list(selected.beta_values),
                "beta_indices": final_indices,
                "beta_values": final_betas,
                "overload_margin": overload_margin,
                "first_beta_total_squared_error": final_first_beta_mse,
                "sample_count": samples.shape[0],
                "seen_block_count": self._seen_blocks[key],
                "sample_sha256": digest,
                "overload_count_by_candidate": overload.sum(dim=0).tolist(),
            }
            if progress_callback is not None:
                progress_callback({
                    "completed": completed, "total": len(keys),
                    "layer_idx": layer_idx, "kind": kind,
                    "beta_values": final_betas,
                })
        return dict(result)


class KVCalibrationCache(DynamicCache):
    """DynamicCache that records post-RoPE K/V samples without quantizing."""

    def __init__(self, *, config, sampler: KVBlockSampler) -> None:
        super().__init__(config=config)
        self.sampler = sampler

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        self.sampler.record(layer_idx, "key", hadamard_attention_heads(key_states))
        self.sampler.record(layer_idx, "value", value_states)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


class RotatedKeyDynamicCache(DynamicCache):
    """Unquantized control cache for checking matched Q/K rotation invariance."""

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        return super().update(
            hadamard_attention_heads(key_states), value_states,
            layer_idx, *args, **kwargs,
        )
