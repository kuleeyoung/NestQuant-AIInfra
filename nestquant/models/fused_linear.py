"""Inference-only Linear modules backed by packed NQRT fused CUDA kernels."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch
from torch import nn

from ..core.model_checkpoint_file import NQMCFileReader
from ..core.rotation import hadamard
from ..core.tensor_checkpoint import PackedNestQuantTensor
from ..runtime.triton_fused_linear import (
    PreparedNQRTLinear,
    linear_fused_triton,
    prepare_fused_linear,
)
from .rotated_linear import PAPER_HADAMARD_FORMAT, HadamardRotatedLinear


class FusedNQRTLinear(nn.Module):
    """Apply one packed NQRT v4 weight without materializing a BF16 matrix.

    The NQMC record is read and its compact streams are copied to the input
    CUDA device on the first call.  Subsequent calls reuse that prepared state.
    Moving the module invalidates the device-local cache.
    """

    def __init__(
        self,
        reader: NQMCFileReader,
        weight_name: str,
        bias: torch.Tensor | None = None,
        *,
        input_hadamard: bool = False,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(reader, NQMCFileReader):
            raise TypeError("reader must be an NQMCFileReader")
        entry = reader.get_entry(weight_name)
        if entry.kind != "nestquant":
            raise ValueError("FusedNQRTLinear requires a packed NestQuant record")
        if len(entry.shape) != 2 or not all(entry.shape):
            raise ValueError("weight record must have nonempty [out, in] shape")
        self.out_features, self.in_features = entry.shape
        if input_hadamard and self.in_features & (self.in_features - 1):
            raise ValueError("Hadamard input transform requires power-of-two in_features")
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
                raise TypeError("bias must be a floating-point tensor")
            if tuple(bias.shape) != (self.out_features,):
                raise ValueError("bias must match out_features")
        if output_dtype is not None and output_dtype not in {
            torch.float16, torch.bfloat16, torch.float32,
        }:
            raise ValueError("output_dtype must be float16, bfloat16, float32, or None")

        self.reader = reader
        self.weight_name = weight_name
        self.input_hadamard = input_hadamard
        self.output_dtype = output_dtype
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self._prepared: PreparedNQRTLinear | None = None

    @property
    def prepared(self) -> bool:
        return self._prepared is not None

    @property
    def packed_device_bytes(self) -> int:
        prepared = self._prepared
        if prepared is None:
            return 0
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                prepared.code_payload,
                prepared.beta_payload,
                prepared.row_norms,
                prepared.beta_table,
            )
        )

    def clear_prepared(self) -> None:
        """Release the cached device-local packed streams and metadata."""
        self._prepared = None

    def _apply(self, fn, recurse: bool = True):
        # PreparedNQRTLinear is not registered module state; never retain a
        # stale device copy after ``to()``, ``cuda()``, or dtype conversion.
        self.clear_prepared()
        return super()._apply(fn, recurse=recurse)

    def _prepared_for(self, device: torch.device) -> PreparedNQRTLinear:
        prepared = self._prepared
        if prepared is not None and prepared.code_payload.device == device:
            return prepared
        _, record = self.reader.read_named_record(self.weight_name)
        if not isinstance(record, PackedNestQuantTensor):
            raise TypeError("fused Linear record must remain packed")
        prepared = prepare_fused_linear(record, device=device)
        self._prepared = prepared
        return prepared

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("inputs must be a floating-point tensor")
        if inputs.ndim == 0 or inputs.shape[-1] != self.in_features:
            actual = None if inputs.ndim == 0 else inputs.shape[-1]
            raise ValueError(
                f"input feature dimension {actual} does not match {self.in_features}"
            )
        if inputs.device.type != "cuda":
            raise ValueError("FusedNQRTLinear requires CUDA inputs")
        original_dtype = inputs.dtype
        compute_inputs = hadamard(inputs) if self.input_hadamard else inputs
        bias = None if self.bias is None else self.bias.to(inputs.device)
        outputs = linear_fused_triton(
            compute_inputs, self._prepared_for(inputs.device), bias
        )
        target_dtype = original_dtype if self.output_dtype is None else self.output_dtype
        return outputs.to(target_dtype)


@dataclass(frozen=True)
class FusedNQRTConversionReport:
    replaced_linear_names: tuple[str, ...]
    input_hadamard_names: tuple[str, ...]


def convert_linears_to_fused_nqrt_(
    model: nn.Module,
    reader: NQMCFileReader,
    include_names: Collection[str],
    *,
    output_dtype: torch.dtype | None = None,
) -> FusedNQRTConversionReport:
    """Transactionally replace selected model Linears with fused NQRT modules.

    Dense state outside the selected modules is intentionally left untouched;
    callers must start from the same base-model revision as the checkpoint.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(reader, NQMCFileReader):
        raise TypeError("reader must be an NQMCFileReader")
    if isinstance(include_names, (str, bytes)):
        raise TypeError("include_names must be a collection of module names")
    requested = tuple(include_names)
    if not requested or any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("include_names must contain nonempty module names")
    if len(requested) != len(set(requested)):
        raise ValueError("include_names must be unique")

    modules = dict(model.named_modules())
    hadamard_names = _hadamard_names(reader.model_metadata)
    staged = []
    for name in requested:
        module = modules.get(name)
        if not isinstance(module, (nn.Linear, HadamardRotatedLinear)):
            raise ValueError(f"module is not a Linear: {name}")
        entry = reader.get_entry(f"{name}.weight")
        if entry.kind != "nestquant":
            raise ValueError(f"module weight is not packed: {name}")
        if tuple(module.weight.shape) != entry.shape:
            raise ValueError(f"quantized weight shape mismatch: {name}.weight")
        staged.append((name, module))

    for name, module in staged:
        fused = FusedNQRTLinear(
            reader,
            f"{name}.weight",
            None if module.bias is None else module.bias.detach(),
            input_hadamard=name in hadamard_names,
            output_dtype=(
                getattr(module, "output_dtype", module.weight.dtype)
                if output_dtype is None else output_dtype
            ),
        )
        fused.train(module.training)
        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else modules[parent_name]
        setattr(parent, child_name, fused)

    return FusedNQRTConversionReport(
        replaced_linear_names=tuple(sorted(requested)),
        input_hadamard_names=tuple(sorted(set(requested) & hadamard_names)),
    )


def _hadamard_names(metadata: dict[str, object]) -> set[str]:
    value = metadata.get("hadamard_rotation")
    if value is None:
        return set()
    if not isinstance(value, dict) or value.get("format") != PAPER_HADAMARD_FORMAT:
        raise ValueError("unsupported or malformed Hadamard checkpoint metadata")
    converted = value.get("converted_linear_names")
    if not isinstance(converted, list) or not all(
        isinstance(name, str) and name for name in converted
    ):
        raise ValueError("Hadamard converted module names must be nonempty strings")
    return set(converted)
