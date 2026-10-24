"""NestQuant A4 quality simulation for Llama projection inputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any
import weakref

import torch

from ..core.rotation import hadamard
from ..core.runtime_quantization import fake_quantize_nestquant_rows
from ..core.beta_search import build_candidate_tables, select_beta_subset
from ..models.rotated_linear import HadamardRotatedLinear


LLAMA_PROJECTION_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
)


def projection_kind(module_name: str) -> str:
    """Return the stable seven-way projection class for a supported module."""
    for suffix in LLAMA_PROJECTION_SUFFIXES:
        if module_name.endswith(suffix):
            return suffix.rsplit(".", 1)[-1]
    raise ValueError(f"unsupported Llama projection name: {module_name}")


class ActivationBlockSampler:
    """Bounded deterministic samples of normalized projection-input E8 blocks."""

    def __init__(self, *, calibration_chunks: int, sample_limit_per_module: int) -> None:
        if calibration_chunks < 1 or sample_limit_per_module < 1:
            raise ValueError("calibration_chunks and sample limit must be positive")
        self.calibration_chunks = calibration_chunks
        self.sample_limit_per_module = sample_limit_per_module
        self._samples: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._sample_counts: dict[str, int] = defaultdict(int)
        self._seen_blocks: dict[str, int] = defaultdict(int)

    def record(self, module_name: str, tensor: torch.Tensor) -> None:
        if not module_name.endswith(LLAMA_PROJECTION_SUFFIXES):
            raise ValueError(f"unsupported Llama projection name: {module_name}")
        if tensor.shape[-1] % 8:
            raise ValueError("activation feature dimension must be divisible by 8")
        rows = tensor.detach().float().reshape(-1, tensor.shape[-1])
        norms = torch.linalg.vector_norm(rows, dim=-1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        normalized = rows * (math.sqrt(rows.shape[-1]) / safe_norms).unsqueeze(-1)
        normalized = torch.where(
            norms.unsqueeze(-1) > 0, normalized, torch.zeros_like(normalized)
        )
        blocks = normalized.reshape(-1, 8)
        self._seen_blocks[module_name] += blocks.shape[0]
        remaining = self.sample_limit_per_module - self._sample_counts[module_name]
        if remaining <= 0:
            return
        per_update = math.ceil(self.sample_limit_per_module / self.calibration_chunks)
        take = min(remaining, per_update, blocks.shape[0])
        indices = torch.div(
            torch.arange(take, device=blocks.device, dtype=torch.int64) * blocks.shape[0],
            take,
            rounding_mode="floor",
        )
        self._samples[module_name].append(
            blocks.index_select(0, indices).to(device="cpu", dtype=torch.float32)
        )
        self._sample_counts[module_name] += take

    def search_profiles(
        self,
        *,
        q: int,
        candidate_betas: Sequence[float],
        selected_beta_count: int,
        device: torch.device | str,
        rows_per_chunk: int = 8192,
        progress_callback=None,
    ) -> dict:
        """Select global, projection-type, and per-module beta subsets."""
        if rows_per_chunk < 1:
            raise ValueError("rows_per_chunk must be positive")
        if not self._samples:
            raise ValueError("no activation samples were recorded")
        beta_values = tuple(float(value) for value in candidate_betas)
        samples = {
            name: torch.cat(parts, dim=0) for name, parts in sorted(self._samples.items())
        }
        expected = set(self._seen_blocks)
        if set(samples) != expected:
            raise RuntimeError("one or more observed modules have no retained samples")

        groups: list[tuple[str, str, list[str]]] = [
            ("global", "global", list(samples)),
        ]
        by_kind: dict[str, list[str]] = defaultdict(list)
        for name in samples:
            by_kind[projection_kind(name)].append(name)
        groups.extend(("projection_type", kind, names) for kind, names in sorted(by_kind.items()))
        groups.extend(("module", name, [name]) for name in samples)

        profiles: dict[str, dict] = {"global": {}, "projection_type": {}, "module": {}}
        for completed, (scope, key, names) in enumerate(groups, 1):
            vectors = torch.cat([samples[name] for name in names], dim=0)
            mse_parts, overload_parts = [], []
            for start in range(0, vectors.shape[0], rows_per_chunk):
                batch = vectors[start:start + rows_per_chunk].to(device)
                mse, overload = build_candidate_tables(batch, q, beta_values)
                mse_parts.append(mse.cpu())
                overload_parts.append(overload.cpu())
            mse = torch.cat(mse_parts)
            overload = torch.cat(overload_parts)
            selected = select_beta_subset(
                beta_values, mse, overload, selected_beta_count
            )
            indices = list(selected.beta_indices)
            selected_overload = overload[:, indices]
            first_fitting = (~selected_overload).to(torch.int64).argmax(dim=1)
            assignments = torch.tensor(indices, dtype=torch.int64)[first_fitting]
            total_error = float(mse[torch.arange(mse.shape[0]), assignments].sum().item())
            record = {
                "beta_indices": indices,
                "beta_values": list(selected.beta_values),
                "sample_count": vectors.shape[0],
                "module_count": len(names),
                "sample_sha256": hashlib.sha256(
                    vectors.contiguous().numpy().tobytes()
                ).hexdigest(),
                "first_beta_total_squared_error": total_error,
                "overload_count_by_candidate": overload.sum(dim=0).tolist(),
            }
            profiles[scope][key] = record
            if progress_callback is not None:
                progress_callback({
                    "completed": completed,
                    "total": len(groups),
                    "scope": scope,
                    "key": key,
                    "beta_values": record["beta_values"],
                })
        profiles["sampling"] = {
            "sample_limit_per_module": self.sample_limit_per_module,
            "module_count": len(samples),
            "sample_count_by_module": {
                name: int(value.shape[0]) for name, value in samples.items()
            },
            "seen_block_count_by_module": dict(sorted(self._seen_blocks.items())),
        }
        return profiles


def attach_activation_calibration(
    model, sampler: ActivationBlockSampler
) -> list[torch.utils.hooks.RemovableHandle]:
    """Observe projection inputs in their A4 coordinates without changing them."""
    handles = []
    matched = []
    for name, module in model.named_modules():
        if not name.endswith(LLAMA_PROJECTION_SUFFIXES):
            continue
        if not isinstance(module, (torch.nn.Linear, HadamardRotatedLinear)):
            raise TypeError(f"expected Linear or HadamardRotatedLinear at {name}")
        matched.append(name)
        if isinstance(module, HadamardRotatedLinear):
            def rotated_hook(_module, _source, rotated, *, module_name=name):
                sampler.record(module_name, rotated)
                return None

            handles.append(module.register_rotated_input_hook(rotated_hook))
        else:
            def hook(_module, inputs, *, module_name=name):
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise TypeError(f"{module_name} must receive a tensor as its first input")
                sampler.record(module_name, hadamard(inputs[0].float()))

            handles.append(module.register_forward_pre_hook(hook))
    if not matched:
        raise ValueError("no supported Llama attention/MLP projection modules were found")
    return handles


class ActivationQuantizationStats:
    """Accumulate projection-input A4 error without retaining activations."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def record(self, module_name: str, stats: dict, *, coordinates: str) -> None:
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
            "coordinates": coordinates,
        })
        if (current["q"] != stats["q"] or current["betas"] != stats["betas"]
                or current["feature_count"] != stats["feature_count"]
                or current["coordinates"] != coordinates):
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
            "coordinate_modes": sorted({item["coordinates"] for item in modules.values()}),
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
    module_betas: Mapping[str, Sequence[float]] | None = None,
    rows_per_chunk: int = 4096,
    stats: ActivationQuantizationStats | None = None,
) -> tuple[ActivationQuantizationStats, list[torch.utils.hooks.RemovableHandle]]:
    """Attach A4 simulation to the seven Llama projection types per layer.

    Ordinary Linear modules retain the upstream-equivalent ``H -> A4 -> H``
    round trip.  ``HadamardRotatedLinear`` modules quantize their already
    rotated ``X H`` input directly, avoiding two redundant transforms while
    producing the same dot product coordinates.
    """
    accumulator = stats if stats is not None else ActivationQuantizationStats()
    configured_betas = {
        str(name): tuple(float(value) for value in values)
        for name, values in (module_betas or {}).items()
    }
    handles = []
    matched = []
    # Llama passes the same tensor object to q/k/v and to gate/up.  The
    # quantizer is deterministic, so retaining one reconstruction only while
    # that source tensor is alive removes duplicate work without changing the
    # simulated value.  A weak reference prevents cross-forward reuse when an
    # allocator later recycles the same Python id or storage address.
    cache: dict[tuple[int, str, tuple[float, ...]], tuple[weakref.ReferenceType, torch.Tensor, dict]] = {}
    for name, module in model.named_modules():
        if not name.endswith(LLAMA_PROJECTION_SUFFIXES):
            continue
        if not isinstance(module, (torch.nn.Linear, HadamardRotatedLinear)):
            raise TypeError(f"expected Linear or HadamardRotatedLinear at {name}")
        matched.append(name)

        if isinstance(module, HadamardRotatedLinear):
            def rotated_hook(_module, source, rotated, *, module_name=name):
                active_betas = configured_betas.get(module_name, tuple(betas))
                key = (id(source), "local_hadamard_input", active_betas)
                cached = cache.get(key)
                if cached is not None and cached[0]() is source:
                    restored, quant_stats = cached[1], cached[2]
                else:
                    restored, quant_stats = fake_quantize_nestquant_rows(
                        rotated.float(), q=q,
                        betas=active_betas,
                        rows_per_chunk=rows_per_chunk,
                    )
                    restored = restored.to(dtype=rotated.dtype)

                    def clear(reference, *, source_id=key):
                        current = cache.get(source_id)
                        if current is not None and current[0] is reference:
                            cache.pop(source_id, None)

                    cache[key] = (weakref.ref(source, clear), restored, quant_stats)
                accumulator.record(
                    module_name, quant_stats, coordinates="local_hadamard_input"
                )
                return restored

            handles.append(module.register_rotated_input_hook(rotated_hook))
            continue

        def hook(_module, inputs, *, module_name=name):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{module_name} must receive a tensor as its first input")
            source = inputs[0]
            active_betas = configured_betas.get(module_name, tuple(betas))
            key = (id(source), "original_input_after_hadamard_round_trip", active_betas)
            cached = cache.get(key)
            if cached is not None and cached[0]() is source:
                restored, quant_stats = cached[1], cached[2]
            else:
                rotated = hadamard(source.float())
                quantized, quant_stats = fake_quantize_nestquant_rows(
                    rotated, q=q, betas=active_betas,
                    rows_per_chunk=rows_per_chunk,
                )
                restored = hadamard(quantized.float()).to(dtype=source.dtype)

                def clear(reference, *, source_id=key):
                    current = cache.get(source_id)
                    if current is not None and current[0] is reference:
                        cache.pop(source_id, None)

                cache[key] = (weakref.ref(source, clear), restored, quant_stats)
            accumulator.record(
                module_name, quant_stats,
                coordinates="original_input_after_hadamard_round_trip",
            )
            return (restored,) + tuple(inputs[1:])

        handles.append(module.register_forward_pre_hook(hook))
    if not matched:
        raise ValueError("no supported Llama attention/MLP projection modules were found")
    if configured_betas and set(configured_betas) != set(matched):
        missing = sorted(set(matched) - set(configured_betas))
        extra = sorted(set(configured_betas) - set(matched))
        raise ValueError(
            f"activation beta module set mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    return accumulator, handles
