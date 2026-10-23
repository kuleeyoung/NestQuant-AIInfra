"""Reference Linear module for paper Section 4.3 Hadamard coordinates."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Collection

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

from ..core.rotation import hadamard


PAPER_HADAMARD_FORMAT = "local_hadamard_linear_v1"


@dataclass(frozen=True)
class HadamardConversionReport:
    converted: tuple[str, ...]
    skipped_tied: tuple[str, ...]
    skipped_non_power_of_two: tuple[str, ...]
    weight_storage: str = "float32_reference"

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return the versioned module-structure metadata required on reload."""
        return {
            "format": PAPER_HADAMARD_FORMAT,
            "converted_linear_names": list(self.converted),
            "skipped_tied_linear_names": list(self.skipped_tied),
            "skipped_non_power_of_two_linear_names": list(self.skipped_non_power_of_two),
            "weight_storage": self.weight_storage,
        }


class HadamardRotatedLinear(nn.Module):
    """Store ``W H`` and apply ``X H`` online before a Linear operation.

    For the normalized symmetric Hadamard matrix ``H``, ``H H.T = I``. Thus
    ``(X H) (W H).T = X W.T`` before quantization. This module is a correctness
    reference for the local weight/activation-pair rotation in paper Section
    4.3. It is not a fused kernel and currently supports only power-of-two input
    dimensions, which cover the Linear inputs used by Llama-3.2-1B.
    """

    def __init__(
        self,
        rotated_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(rotated_weight, torch.Tensor) or not rotated_weight.is_floating_point():
            raise TypeError("rotated_weight must be a floating-point tensor")
        if rotated_weight.ndim != 2 or not rotated_weight.numel():
            raise ValueError("rotated_weight must be nonempty with shape [out_features, in_features]")
        in_features = rotated_weight.shape[1]
        if in_features & (in_features - 1):
            raise ValueError("in_features must be a power of two")
        if bias is not None:
            if bias.shape != (rotated_weight.shape[0],):
                raise ValueError("bias must match out_features")
            if bias.device != rotated_weight.device or bias.dtype != rotated_weight.dtype:
                raise ValueError("bias must match rotated_weight device and dtype")
        self.in_features = in_features
        self.out_features = rotated_weight.shape[0]
        self.output_dtype = rotated_weight.dtype if output_dtype is None else output_dtype
        self.weight = nn.Parameter(rotated_weight.detach().clone())
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone())
        self._rotated_input_hooks: OrderedDict[int, Callable] = OrderedDict()

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, *, preserve_source_dtype: bool = False,
    ) -> "HadamardRotatedLinear":
        """Create the exact rotated-coordinate representation of ``linear``."""
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be torch.nn.Linear")
        if linear.in_features & (linear.in_features - 1):
            raise ValueError("linear.in_features must be a power of two")
        output_dtype = linear.weight.dtype
        compute_dtype = (
            torch.float32
            if output_dtype in (torch.float16, torch.bfloat16)
            else output_dtype
        )
        rotated_weight = hadamard(linear.weight.detach().to(compute_dtype))
        bias = None if linear.bias is None else linear.bias.detach().to(compute_dtype)
        if preserve_source_dtype:
            rotated_weight = rotated_weight.to(output_dtype)
            bias = None if bias is None else bias.to(output_dtype)
        result = cls(rotated_weight, bias, output_dtype=output_dtype)
        result.train(linear.training)
        return result

    def reconstructed_weight(self) -> torch.Tensor:
        """Return the current rotated weight mapped back to original coordinates."""
        return hadamard(self.weight)

    def register_rotated_input_hook(
        self,
        hook: Callable[["HadamardRotatedLinear", torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> RemovableHandle:
        """Register a transform applied after ``X H`` and before the Linear.

        The hook receives ``(module, original_input, rotated_input)`` and must
        return a tensor with the same shape as ``rotated_input``.  It allows A4
        quality simulation to quantize directly in the paper's local Hadamard
        coordinates without an inverse/forward Hadamard round trip.
        """
        if not callable(hook):
            raise TypeError("rotated input hook must be callable")
        handle = RemovableHandle(self._rotated_input_hooks)
        self._rotated_input_hooks[handle.id] = hook
        return handle

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"input feature dimension {inputs.shape[-1]} does not match {self.in_features}"
            )
        rotated = hadamard(inputs.to(self.weight.dtype))
        for hook in tuple(self._rotated_input_hooks.values()):
            transformed = hook(self, inputs, rotated)
            if not isinstance(transformed, torch.Tensor):
                raise TypeError("rotated input hook must return a tensor")
            if transformed.shape != rotated.shape:
                raise ValueError("rotated input hook must preserve input shape")
            rotated = transformed
        outputs = F.linear(rotated, self.weight, self.bias)
        return outputs.to(self.output_dtype)


def convert_linears_to_hadamard_coordinates_(
    model: nn.Module,
    *,
    include_names: Collection[str] | None = None,
    strict_power_of_two: bool = True,
    preserve_source_dtype: bool = False,
) -> HadamardConversionReport:
    """Replace selected Linear modules by exact local Hadamard coordinates.

    A Linear sharing storage with the input embedding is skipped so conversion
    does not silently break tied embedding/``lm_head`` semantics. The function
    mutates ``model`` only after every requested module name and feature shape
    has been validated.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    requested = None if include_names is None else frozenset(include_names)
    modules = dict(model.named_modules())
    linear_names = {
        name for name, module in modules.items() if name and isinstance(module, nn.Linear)
    }
    if requested is not None:
        unknown = sorted(requested - linear_names)
        if unknown:
            raise ValueError(f"unknown Linear module names: {unknown}")

    tied_weight = None
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embedding = get_embeddings()
        tied_weight = None if embedding is None else getattr(embedding, "weight", None)

    selected = sorted(linear_names if requested is None else requested)
    tied, unsupported, convertible = [], [], []
    for name in selected:
        module = modules[name]
        if tied_weight is not None and module.weight.data_ptr() == tied_weight.data_ptr():
            tied.append(name)
        elif module.in_features & (module.in_features - 1):
            unsupported.append(name)
        else:
            convertible.append(name)
    if strict_power_of_two and unsupported:
        raise ValueError(f"non-power-of-two Linear input dimensions: {unsupported}")

    for name in convertible:
        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else modules[parent_name]
        setattr(parent, child_name, HadamardRotatedLinear.from_linear(
            modules[name], preserve_source_dtype=preserve_source_dtype
        ))
    return HadamardConversionReport(
        converted=tuple(convertible),
        skipped_tied=tuple(tied),
        skipped_non_power_of_two=tuple(unsupported),
        weight_storage=("source_model_dtype" if preserve_source_dtype else "float32_reference"),
    )


def restore_hadamard_structure_from_metadata_(
    model: nn.Module, metadata: object,
) -> HadamardConversionReport:
    """Rebuild the module structure described by versioned NQMC metadata."""
    if not isinstance(metadata, dict) or metadata.get("format") != PAPER_HADAMARD_FORMAT:
        raise ValueError("unsupported or malformed Hadamard checkpoint metadata")
    names = metadata.get("converted_linear_names")
    skipped_tied = metadata.get("skipped_tied_linear_names")
    skipped_non_power = metadata.get("skipped_non_power_of_two_linear_names")
    weight_storage = metadata.get("weight_storage")
    if not all(isinstance(value, list) and all(isinstance(name, str) and name for name in value)
               for value in (names, skipped_tied, skipped_non_power)):
        raise ValueError("Hadamard checkpoint module lists must contain nonempty strings")
    if weight_storage not in {"float32_reference", "source_model_dtype"}:
        raise ValueError("unsupported Hadamard checkpoint weight storage mode")
    if len(names) != len(set(names)):
        raise ValueError("Hadamard checkpoint converted module names must be unique")
    requested = [*names, *skipped_tied, *skipped_non_power]
    if len(requested) != len(set(requested)):
        raise ValueError("Hadamard checkpoint module categories must be disjoint")
    report = convert_linears_to_hadamard_coordinates_(
        model,
        include_names=requested,
        strict_power_of_two=False,
        preserve_source_dtype=weight_storage == "source_model_dtype",
    )
    if (report.converted != tuple(sorted(names))
            or report.skipped_tied != tuple(sorted(skipped_tied))
            or report.skipped_non_power_of_two != tuple(sorted(skipped_non_power))):
        raise ValueError("Hadamard checkpoint skip metadata disagrees with the model")
    return report
