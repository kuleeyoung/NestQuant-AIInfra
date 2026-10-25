"""Model-level conversion from an NQMC checkpoint to lazy Linear execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..core.model_checkpoint_file import NQMCFileReader
from .lazy_linear import (
    LazyNQMCLinear,
    LazyNQMCWeightCache,
    RuntimeProfileCallback,
)
from .rotated_linear import PAPER_HADAMARD_FORMAT, HadamardRotatedLinear


@dataclass(frozen=True)
class LazyNQMCConversionReport:
    replaced_linear_names: tuple[str, ...]
    dense_tensor_names: tuple[str, ...]
    alias_names: tuple[str, ...]
    input_hadamard_names: tuple[str, ...]
    cache_capacity_bytes: int
    compute_dtype: torch.dtype
    rows_per_chunk: int | None
    decode_backend: str


def convert_model_to_lazy_nqmc_(
    model: nn.Module,
    reader: NQMCFileReader,
    weight_cache: LazyNQMCWeightCache,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype | None = None,
    rows_per_chunk: int | None = 128,
    profile_callback: RuntimeProfileCallback | None = None,
    synchronize_profile: bool = False,
    decode_backend: str = "reference",
) -> LazyNQMCConversionReport:
    """Load dense state and transactionally replace packed Linear weights.

    The model must still have its original module structure. Every structural
    condition is checked before dense tensors are copied or modules replaced.
    The resulting model can subsequently be moved to the execution device;
    packed weights are decoded only when their lazy modules run.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(reader, NQMCFileReader):
        raise TypeError("reader must be an NQMCFileReader")
    if not isinstance(weight_cache, LazyNQMCWeightCache):
        raise TypeError("weight_cache must be a LazyNQMCWeightCache")
    floating_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if compute_dtype not in floating_dtypes:
        raise ValueError("compute_dtype must be a floating-point dtype")
    if output_dtype is not None and output_dtype not in floating_dtypes:
        raise ValueError("output_dtype must be a floating-point dtype or None")
    if rows_per_chunk is not None and (
        isinstance(rows_per_chunk, bool)
        or not isinstance(rows_per_chunk, int)
        or rows_per_chunk < 1
    ):
        raise ValueError("rows_per_chunk must be a positive integer or None")
    if profile_callback is not None and not callable(profile_callback):
        raise TypeError("profile_callback must be callable")
    if synchronize_profile and profile_callback is None:
        raise ValueError("synchronize_profile requires a profile_callback")
    if decode_backend not in {"reference", "triton_tiled"}:
        raise ValueError("decode_backend must be 'reference' or 'triton_tiled'")

    state = model.state_dict()
    checkpoint_names = {entry.name for entry in reader.entries} | set(reader.aliases)
    if set(state) != checkpoint_names:
        missing = sorted(set(state) - checkpoint_names)
        unexpected = sorted(checkpoint_names - set(state))
        raise ValueError(
            f"state-dict mismatch; missing={missing}, unexpected={unexpected}"
        )
    modules = dict(model.named_modules())
    quantized_entries = [entry for entry in reader.entries if entry.kind == "nestquant"]
    dense_entries = [entry for entry in reader.entries if entry.kind == "dense"]

    hadamard_names = _hadamard_names(reader.model_metadata)
    replacement_names = []
    staged = []
    for entry in quantized_entries:
        module_name, separator, field = entry.name.rpartition(".")
        if not separator or field != "weight":
            raise ValueError(f"quantized record is not a module weight: {entry.name}")
        module = modules.get(module_name)
        if not isinstance(module, (nn.Linear, HadamardRotatedLinear)):
            raise ValueError(f"quantized weight does not resolve to a Linear: {entry.name}")
        if tuple(module.weight.shape) != entry.shape:
            raise ValueError(f"quantized weight shape mismatch: {entry.name}")
        replacement_names.append(module_name)
        staged.append((module_name, module, entry.name))

    if hadamard_names and hadamard_names != set(replacement_names):
        missing_lazy = sorted(hadamard_names - set(replacement_names))
        missing_rotation = sorted(set(replacement_names) - hadamard_names)
        raise ValueError(
            "Hadamard metadata and lazy quantized modules disagree; "
            f"dense_rotated={missing_lazy}, unrotated_quantized={missing_rotation}"
        )
    for entry in dense_entries:
        target = state[entry.name]
        if tuple(target.shape) != entry.shape or target.dtype != entry.dtype:
            raise ValueError(f"dense tensor metadata mismatch: {entry.name}")
    for alias, canonical in reader.aliases.items():
        if tuple(state[alias].shape) != tuple(state[canonical].shape):
            raise ValueError(f"alias tensor shape mismatch: {alias}")
        if state[alias].dtype != state[canonical].dtype:
            raise ValueError(f"alias tensor dtype mismatch: {alias}")

    # Structural validation above is complete. Copy bounded dense records before
    # replacing modules so a bias record is available to the lazy constructor.
    with torch.no_grad():
        for entry in dense_entries:
            value = reader.read_entry(entry)
            assert isinstance(value, torch.Tensor)
            state[entry.name].copy_(value.to(state[entry.name].device))
            del value
        for alias, canonical in reader.aliases.items():
            if state[alias].data_ptr() != state[canonical].data_ptr():
                state[alias].copy_(state[canonical])

    for module_name, module, weight_name in staged:
        lazy = LazyNQMCLinear(
            reader,
            weight_name,
            None if module.bias is None else module.bias.detach(),
            input_hadamard=module_name in hadamard_names,
            compute_dtype=compute_dtype,
            output_dtype=(
                getattr(module, "output_dtype", module.weight.dtype)
                if output_dtype is None else output_dtype
            ),
            rows_per_chunk=rows_per_chunk,
            cache_policy="lru",
            weight_cache=weight_cache,
            profile_callback=profile_callback,
            synchronize_profile=synchronize_profile,
            decode_backend=decode_backend,
        )
        lazy.train(module.training)
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model if not parent_name else modules[parent_name]
        setattr(parent, child_name, lazy)

    return LazyNQMCConversionReport(
        replaced_linear_names=tuple(sorted(replacement_names)),
        dense_tensor_names=tuple(entry.name for entry in dense_entries),
        alias_names=tuple(sorted(reader.aliases)),
        input_hadamard_names=tuple(sorted(hadamard_names)),
        cache_capacity_bytes=weight_cache.capacity_bytes,
        compute_dtype=compute_dtype,
        rows_per_chunk=rows_per_chunk,
        decode_backend=decode_backend,
    )


def _hadamard_names(metadata: dict[str, object]) -> set[str]:
    value = metadata.get("hadamard_rotation")
    if value is None:
        return set()
    if not isinstance(value, dict) or value.get("format") != PAPER_HADAMARD_FORMAT:
        raise ValueError("unsupported or malformed Hadamard checkpoint metadata")
    converted = value.get("converted_linear_names")
    skipped_tied = value.get("skipped_tied_linear_names")
    skipped_non_power = value.get("skipped_non_power_of_two_linear_names")
    if not all(
        isinstance(names, list)
        and all(isinstance(name, str) and name for name in names)
        for names in (converted, skipped_tied, skipped_non_power)
    ):
        raise ValueError("Hadamard checkpoint module lists must contain nonempty strings")
    if len(converted) != len(set(converted)):
        raise ValueError("Hadamard converted module names must be unique")
    return set(converted)
