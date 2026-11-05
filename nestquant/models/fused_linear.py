"""Inference-only Linear modules backed by packed NQRT fused CUDA kernels."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import math

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
from ..runtime.triton_dot_linear import (
    linear_fused_dot_triton,
    select_nqrt_dot_config,
)
from ..runtime.triton_grouped_linear import linear_fused_qkv_triton
from ..runtime.cuda_mma_linear import can_use_mma_j2, linear_fused_mma_j2
from ..runtime.cuda_w4a4_mma import (
    can_use_w4a4_mma_small_m,
    can_use_w4a4_mma_large_m_staged,
    can_use_w4a4_mma_chunked_staged,
    can_use_w4a4_gemv_m1,
    can_use_w4a4_mma_m16,
    linear_packed_w4a4_gemv_m1,
    linear_packed_w4a4_gemv_m1_splitk_probe,
    linear_packed_w4a4_mma_small_m,
    linear_packed_w4a4_mma_large_m_staged,
    linear_packed_w4a4_mma_chunked_staged,
    linear_packed_w4a4_mma_m16,
)
from ..runtime.triton_w4a4_linear import (
    PreparedNQA4Activation,
    linear_packed_w4a4_tiled_triton,
    linear_packed_w4a4_triton,
    quantize_activation_cuda,
    select_w4a4_tiled_config,
)
from .rotated_linear import PAPER_HADAMARD_FORMAT, HadamardRotatedLinear


class SharedHadamardCache:
    """Reuse one inference-only Hadamard result for modules sharing an input."""

    def __init__(self, uses_per_input: int) -> None:
        if isinstance(uses_per_input, bool) or not isinstance(uses_per_input, int):
            raise TypeError("uses_per_input must be an integer")
        if uses_per_input < 2:
            raise ValueError("uses_per_input must be at least two")
        self.uses_per_input = uses_per_input
        self._input: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._remaining_uses = 0
        self.hits = 0
        self.misses = 0

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("shared Hadamard caching requires inference or no-grad mode")
        if self._input is inputs and self._output is not None:
            self.hits += 1
            output = self._output
            self._remaining_uses -= 1
            if self._remaining_uses == 0:
                self.clear()
            return output
        output = hadamard(inputs)
        self._input = inputs
        self._output = output
        self._remaining_uses = self.uses_per_input - 1
        self.misses += 1
        return output

    def clear(self) -> None:
        self._input = None
        self._output = None
        self._remaining_uses = 0


class SharedPackedActivationCache:
    """Reuse one packed A4 encoding for projections sharing an input tensor."""

    def __init__(self, uses_per_input: int) -> None:
        if isinstance(uses_per_input, bool) or not isinstance(uses_per_input, int):
            raise TypeError("uses_per_input must be an integer")
        if uses_per_input < 2:
            raise ValueError("uses_per_input must be at least two")
        self.uses_per_input = uses_per_input
        self._input: torch.Tensor | None = None
        self._packed: PreparedNQA4Activation | None = None
        self._remaining_uses = 0
        self.hits = 0
        self.misses = 0

    def encode(
        self, inputs: torch.Tensor, *, q: int, betas: tuple[float, ...],
        beta_table: torch.Tensor | None = None,
    ) -> tuple[PreparedNQA4Activation, bool]:
        if torch.is_grad_enabled():
            raise RuntimeError("shared A4 caching requires inference or no-grad mode")
        if self._input is inputs and self._packed is not None:
            self.hits += 1
            packed = self._packed
            self._remaining_uses -= 1
            if self._remaining_uses == 0:
                self.clear()
            return packed, False
        packed = quantize_activation_cuda(
            inputs, q=q, betas=betas, beta_table=beta_table
        )
        self._input = inputs
        self._packed = packed
        self._remaining_uses = self.uses_per_input - 1
        self.misses += 1
        return packed, True

    def clear(self) -> None:
        self._input = None
        self._packed = None
        self._remaining_uses = 0


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
        input_transform_cache: SharedHadamardCache | None = None,
        output_dtype: torch.dtype | None = None,
        compute_backend: str = "tensor_core_dot",
        activation_q: int = 14,
        activation_betas: tuple[float, ...] | None = None,
        packed_activation_cache: SharedPackedActivationCache | None = None,
        force_w4a4_tiled_fallback: bool = False,
        enable_m1_down_splitk: bool = False,
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
        if input_transform_cache is not None and not isinstance(
            input_transform_cache, SharedHadamardCache
        ):
            raise TypeError("input_transform_cache must be a SharedHadamardCache or None")
        if input_transform_cache is not None and not input_hadamard:
            raise ValueError("input_transform_cache requires input_hadamard=True")
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
                raise TypeError("bias must be a floating-point tensor")
            if tuple(bias.shape) != (self.out_features,):
                raise ValueError("bias must match out_features")
        if output_dtype is not None and output_dtype not in {
            torch.float16, torch.bfloat16, torch.float32,
        }:
            raise ValueError("output_dtype must be float16, bfloat16, float32, or None")
        if compute_backend not in {
            "tensor_core_dot", "scalar", "cuda_mma_j2",
            "packed_w4a4", "packed_w4a4_tiled", "packed_w4a4_cuda_mma",
            "packed_w4a4_cuda_gemv_m1",
        }:
            raise ValueError(
                "unsupported fused NQRT compute_backend"
            )
        if compute_backend.startswith("packed_w4a4") and not activation_betas:
            raise ValueError("packed_w4a4 requires activation_betas")
        if packed_activation_cache is not None and not compute_backend.startswith("packed_w4a4"):
            raise ValueError("packed_activation_cache requires packed_w4a4")
        if not isinstance(enable_m1_down_splitk, bool):
            raise TypeError("enable_m1_down_splitk must be a bool")

        self.reader = reader
        self.weight_name = weight_name
        self.input_hadamard = input_hadamard
        self.input_transform_cache = input_transform_cache
        self.output_dtype = output_dtype
        self.compute_backend = compute_backend
        self.activation_q = activation_q
        self.activation_betas = (
            None if activation_betas is None
            else tuple(float(value) for value in activation_betas)
        )
        self.packed_activation_cache = packed_activation_cache
        self.force_w4a4_tiled_fallback = bool(force_w4a4_tiled_fallback)
        self.enable_m1_down_splitk = enable_m1_down_splitk
        self.a4_call_count = 0
        self.a4_encode_count = 0
        self.a4_packed_input_bytes = 0
        self.a4_encoded_input_bytes = 0
        self.a4_max_resident_bytes = 0
        self.w4a4_cuda_mma_dispatch_count = 0
        self.w4a4_cuda_mma_m1_dispatch_count = 0
        self.w4a4_cuda_mma_m1_splitk_dispatch_count = 0
        self.w4a4_cuda_mma_m16_dispatch_count = 0
        self.w4a4_cuda_mma_small_m_dispatch_count = 0
        self.w4a4_cuda_mma_m128_staged_dispatch_count = 0
        self.w4a4_cuda_mma_large_m_staged_dispatch_count = 0
        self.w4a4_cuda_mma_chunked_staged_dispatch_count = 0
        self.w4a4_tiled_fallback_count = 0
        self.register_buffer(
            "activation_beta_table",
            None if self.activation_betas is None else torch.tensor(
                self.activation_betas, dtype=torch.float32
            ),
        )
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
        if self.input_transform_cache is not None:
            self.input_transform_cache.clear()
        if self.packed_activation_cache is not None:
            self.packed_activation_cache.clear()
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
        if self.input_transform_cache is not None:
            compute_inputs = self.input_transform_cache.transform(inputs)
        else:
            compute_inputs = hadamard(inputs) if self.input_hadamard else inputs
        bias = None if self.bias is None else self.bias.to(inputs.device)
        prepared = self._prepared_for(inputs.device)
        if self.compute_backend.startswith("packed_w4a4"):
            if self.packed_activation_cache is None:
                packed_activation = quantize_activation_cuda(
                    compute_inputs, q=self.activation_q,
                    betas=self.activation_betas,
                    beta_table=self.activation_beta_table,
                )
                encoded = True
            else:
                packed_activation, encoded = self.packed_activation_cache.encode(
                    compute_inputs, q=self.activation_q,
                    betas=self.activation_betas,
                    beta_table=self.activation_beta_table,
                )
            self.a4_call_count += 1
            self.a4_encode_count += int(encoded)
            self.a4_packed_input_bytes += packed_activation.resident_bytes
            self.a4_encoded_input_bytes += (
                packed_activation.resident_bytes if encoded else 0
            )
            self.a4_max_resident_bytes = max(
                self.a4_max_resident_bytes, packed_activation.resident_bytes
            )
            if (
                self.compute_backend == "packed_w4a4_cuda_mma"
                and not self.force_w4a4_tiled_fallback
                and bias is None
                and can_use_w4a4_mma_large_m_staged(packed_activation, prepared)
            ):
                self.w4a4_cuda_mma_dispatch_count += 1
                self.w4a4_cuda_mma_large_m_staged_dispatch_count += 1
                self.w4a4_cuda_mma_m128_staged_dispatch_count += int(
                    packed_activation.token_count == 128
                )
                outputs = linear_packed_w4a4_mma_large_m_staged(
                    packed_activation, prepared
                )
            elif (
                self.compute_backend == "packed_w4a4_cuda_mma"
                and not self.force_w4a4_tiled_fallback
                and bias is None
                and can_use_w4a4_mma_m16(packed_activation, prepared)
            ):
                self.w4a4_cuda_mma_dispatch_count += 1
                self.w4a4_cuda_mma_m16_dispatch_count += 1
                outputs = linear_packed_w4a4_mma_m16(
                    packed_activation, prepared
                )
            elif (
                self.compute_backend == "packed_w4a4_cuda_mma"
                and not self.force_w4a4_tiled_fallback
                and bias is None
                and can_use_w4a4_mma_small_m(packed_activation, prepared)
            ):
                self.w4a4_cuda_mma_dispatch_count += 1
                self.w4a4_cuda_mma_small_m_dispatch_count += 1
                outputs = linear_packed_w4a4_mma_small_m(
                    packed_activation, prepared
                )
            elif (
                self.compute_backend in {
                    "packed_w4a4_cuda_mma", "packed_w4a4_cuda_gemv_m1",
                }
                and not self.force_w4a4_tiled_fallback
                and bias is None
                and can_use_w4a4_gemv_m1(packed_activation, prepared)
            ):
                self.w4a4_cuda_mma_dispatch_count += 1
                self.w4a4_cuda_mma_m1_dispatch_count += 1
                if (
                    self.enable_m1_down_splitk
                    and packed_activation.feature_count == 8192
                    and self.out_features == 2048
                ):
                    self.w4a4_cuda_mma_m1_splitk_dispatch_count += 1
                    outputs = linear_packed_w4a4_gemv_m1_splitk_probe(
                        packed_activation, prepared, split_k=8
                    )
                else:
                    outputs = linear_packed_w4a4_gemv_m1(
                        packed_activation, prepared
                    )
            elif (
                self.compute_backend == "packed_w4a4_cuda_mma"
                and not self.force_w4a4_tiled_fallback
                and bias is None
                and can_use_w4a4_mma_chunked_staged(
                    packed_activation, prepared
                )
            ):
                self.w4a4_cuda_mma_dispatch_count += 1
                self.w4a4_cuda_mma_chunked_staged_dispatch_count += 1
                outputs = linear_packed_w4a4_mma_chunked_staged(
                    packed_activation, prepared
                )
            elif self.compute_backend in {
                "packed_w4a4_tiled", "packed_w4a4_cuda_mma",
                "packed_w4a4_cuda_gemv_m1",
            }:
                if self.compute_backend in {
                    "packed_w4a4_cuda_mma", "packed_w4a4_cuda_gemv_m1",
                }:
                    self.w4a4_tiled_fallback_count += 1
                outputs = linear_packed_w4a4_tiled_triton(
                    packed_activation, prepared, bias,
                    **select_w4a4_tiled_config(
                        packed_activation.token_count,
                        self.out_features,
                        self.in_features,
                    ),
                )
            else:
                outputs = linear_packed_w4a4_triton(
                    packed_activation, prepared, bias
                )
        elif self.compute_backend == "cuda_mma_j2" and can_use_mma_j2(compute_inputs, prepared):
            outputs = linear_fused_mma_j2(compute_inputs, prepared, bias)
        elif self.compute_backend in {"tensor_core_dot", "cuda_mma_j2"}:
            token_count = math.prod(compute_inputs.shape[:-1])
            config = select_nqrt_dot_config(
                self.out_features, self.in_features, token_count
            )
            outputs = linear_fused_dot_triton(
                compute_inputs, prepared, bias, **config
            )
        else:
            outputs = linear_fused_triton(compute_inputs, prepared, bias)
        target_dtype = original_dtype if self.output_dtype is None else self.output_dtype
        return outputs.to(target_dtype)


class GroupedQKVCoordinator:
    """Compute and distribute Q/K/V outputs from one packed kernel launch."""

    def __init__(self, reader: NQMCFileReader, weight_names: tuple[str, str, str]) -> None:
        if not isinstance(reader, NQMCFileReader):
            raise TypeError("reader must be an NQMCFileReader")
        if not isinstance(weight_names, tuple) or len(weight_names) != 3 or not all(
            isinstance(name, str) and name for name in weight_names
        ):
            raise ValueError("weight_names must contain three nonempty names")
        self.reader = reader
        self.weight_names = weight_names
        self._prepared: tuple[PreparedNQRTLinear, PreparedNQRTLinear, PreparedNQRTLinear] | None = None
        self._input: torch.Tensor | None = None
        self._outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._consumed: set[int] = set()
        self.launches = 0

    def _prepared_for(self, device: torch.device):
        prepared = self._prepared
        if prepared is not None and prepared[0].code_payload.device == device:
            return prepared
        values = []
        for name in self.weight_names:
            _, record = self.reader.read_named_record(name)
            if not isinstance(record, PackedNestQuantTensor):
                raise TypeError("grouped QKV record must remain packed")
            values.append(prepare_fused_linear(record, device=device))
        prepared = tuple(values)
        self._prepared = prepared
        return prepared

    def project(self, index: int, inputs: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("grouped QKV execution requires inference or no-grad mode")
        if index not in {0, 1, 2}:
            raise ValueError("QKV projection index must be 0, 1, or 2")
        if self._outputs is None:
            transformed = hadamard(inputs)
            self._outputs = linear_fused_qkv_triton(
                transformed, self._prepared_for(inputs.device)
            )
            self._input = inputs
            self._consumed.clear()
            self.launches += 1
        elif self._input is not inputs:
            raise RuntimeError("grouped QKV outputs were not fully consumed before a new input")
        if index in self._consumed:
            raise RuntimeError("grouped QKV projection was requested more than once")
        assert self._outputs is not None
        output = self._outputs[index]
        self._consumed.add(index)
        if len(self._consumed) == 3:
            self._input = None
            self._outputs = None
            self._consumed.clear()
        return output

    def clear(self, *, prepared: bool = False) -> None:
        self._input = None
        self._outputs = None
        self._consumed.clear()
        if prepared:
            self._prepared = None

    @property
    def packed_device_bytes(self) -> int:
        if self._prepared is None:
            return 0
        return sum(
            tensor.numel() * tensor.element_size()
            for item in self._prepared
            for tensor in (
                item.code_payload, item.beta_payload, item.row_norms, item.beta_table
            )
        )


class GroupedQKVProjection(nn.Module):
    """Linear-compatible Q, K, or V view backed by a shared coordinator."""

    def __init__(
        self,
        coordinator: GroupedQKVCoordinator,
        index: int,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "coordinator", coordinator)
        self.index = index
        self.in_features = in_features
        self.out_features = out_features
        self.output_dtype = output_dtype
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    def _apply(self, fn, recurse: bool = True):
        self.coordinator.clear(prepared=True)
        return super()._apply(fn, recurse=recurse)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.coordinator.project(self.index, inputs)
        if self.bias is not None:
            output = output + self.bias.to(output.device)
        return output.to(self.output_dtype)


@dataclass(frozen=True)
class FusedNQRTConversionReport:
    replaced_linear_names: tuple[str, ...]
    input_hadamard_names: tuple[str, ...]
    shared_hadamard_groups: tuple[tuple[str, ...], ...]
    grouped_qkv_groups: tuple[tuple[str, str, str], ...]
    w4a4_tiled_fallback_names: tuple[str, ...] = ()
    m1_down_splitk_fallback_names: tuple[str, ...] = ()


def convert_linears_to_fused_nqrt_(
    model: nn.Module,
    reader: NQMCFileReader,
    include_names: Collection[str],
    *,
    output_dtype: torch.dtype | None = None,
    group_qkv: bool = False,
    compute_backend: str = "tensor_core_dot",
    activation_q: int = 14,
    activation_betas: tuple[float, ...] | None = None,
    w4a4_tiled_fallback_names: Collection[str] = (),
    enable_m1_down_splitk: bool = False,
    m1_down_splitk_fallback_names: Collection[str] = (),
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
    if not isinstance(group_qkv, bool):
        raise TypeError("group_qkv must be a bool")
    if not isinstance(enable_m1_down_splitk, bool):
        raise TypeError("enable_m1_down_splitk must be a bool")
    if isinstance(m1_down_splitk_fallback_names, (str, bytes)):
        raise TypeError("m1_down_splitk_fallback_names must be a collection of names")
    splitk_fallback_names = frozenset(m1_down_splitk_fallback_names)
    if any(not isinstance(name, str) or not name for name in splitk_fallback_names):
        raise ValueError("m1_down_splitk_fallback_names must contain nonempty names")
    if isinstance(w4a4_tiled_fallback_names, (str, bytes)):
        raise TypeError("w4a4_tiled_fallback_names must be a collection of names")
    fallback_names = frozenset(w4a4_tiled_fallback_names)
    if any(not isinstance(name, str) or not name for name in fallback_names):
        raise ValueError("w4a4_tiled_fallback_names must contain nonempty names")
    if not fallback_names <= set(requested):
        raise ValueError("w4a4_tiled_fallback_names must be included in include_names")
    if not splitk_fallback_names <= set(requested):
        raise ValueError("m1_down_splitk_fallback_names must be included in include_names")
    if splitk_fallback_names and not enable_m1_down_splitk:
        raise ValueError(
            "m1_down_splitk_fallback_names requires enable_m1_down_splitk=True"
        )
    if compute_backend not in {
        "tensor_core_dot", "scalar", "cuda_mma_j2",
        "packed_w4a4", "packed_w4a4_tiled", "packed_w4a4_cuda_mma",
        "packed_w4a4_cuda_gemv_m1",
    }:
        raise ValueError(
            "unsupported fused NQRT compute_backend"
        )
    if compute_backend.startswith("packed_w4a4") and not activation_betas:
        raise ValueError("packed_w4a4 requires activation_betas")

    modules = dict(model.named_modules())
    hadamard_names = _hadamard_names(reader.model_metadata)
    requested_set = set(requested)
    shared_groups = _shared_hadamard_groups(requested_set & hadamard_names)
    grouped_qkv_groups = _complete_qkv_groups(requested_set & hadamard_names) if group_qkv else ()
    grouped_names = {name for group in grouped_qkv_groups for name in group}
    if fallback_names & grouped_names:
        raise ValueError(
            "w4a4_tiled_fallback_names cannot use grouped QKV replacement"
        )
    caches = {
        name: cache
        for group in shared_groups
        for cache in (SharedHadamardCache(len(group)),)
        for name in group
    }
    packed_caches = {
        name: cache
        for group in shared_groups
        for cache in (SharedPackedActivationCache(len(group)),)
        for name in group
    } if compute_backend.startswith("packed_w4a4") else {}
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

    for group in grouped_qkv_groups:
        coordinator = GroupedQKVCoordinator(
            reader, tuple(f"{name}.weight" for name in group)
        )
        for index, name in enumerate(group):
            module = modules[name]
            projection = GroupedQKVProjection(
                coordinator,
                index,
                module.weight.shape[1],
                module.weight.shape[0],
                None if module.bias is None else module.bias.detach(),
                getattr(module, "output_dtype", module.weight.dtype)
                if output_dtype is None else output_dtype,
            )
            projection.train(module.training)
            parent_name, _, child_name = name.rpartition(".")
            parent = model if not parent_name else modules[parent_name]
            setattr(parent, child_name, projection)

    for name, module in staged:
        if name in grouped_names:
            continue
        fused = FusedNQRTLinear(
            reader,
            f"{name}.weight",
            None if module.bias is None else module.bias.detach(),
            input_hadamard=name in hadamard_names,
            input_transform_cache=caches.get(name),
            output_dtype=(
                getattr(module, "output_dtype", module.weight.dtype)
                if output_dtype is None else output_dtype
            ),
            compute_backend=compute_backend,
            activation_q=activation_q,
            activation_betas=activation_betas,
            packed_activation_cache=packed_caches.get(name),
            force_w4a4_tiled_fallback=name in fallback_names,
            enable_m1_down_splitk=(
                enable_m1_down_splitk and name not in splitk_fallback_names
            ),
        )
        fused.to(device=module.weight.device)
        fused.train(module.training)
        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else modules[parent_name]
        setattr(parent, child_name, fused)

    return FusedNQRTConversionReport(
        replaced_linear_names=tuple(sorted(requested)),
        input_hadamard_names=tuple(sorted(requested_set & hadamard_names)),
        shared_hadamard_groups=shared_groups,
        grouped_qkv_groups=grouped_qkv_groups,
        w4a4_tiled_fallback_names=tuple(sorted(fallback_names)),
        m1_down_splitk_fallback_names=tuple(sorted(splitk_fallback_names)),
    )


def _shared_hadamard_groups(names: set[str]) -> tuple[tuple[str, ...], ...]:
    candidates: dict[str, list[str]] = {}
    for name in sorted(names):
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            key = name.rsplit(".", 1)[0] + ":qkv"
        elif name.endswith((".gate_proj", ".up_proj")):
            key = name.rsplit(".", 1)[0] + ":gate_up"
        else:
            continue
        candidates.setdefault(key, []).append(name)
    return tuple(tuple(group) for group in candidates.values() if len(group) > 1)


def _complete_qkv_groups(names: set[str]) -> tuple[tuple[str, str, str], ...]:
    prefixes = sorted({name.rsplit(".", 1)[0] for name in names})
    groups = []
    for prefix in prefixes:
        group = tuple(f"{prefix}.{suffix}_proj" for suffix in ("q", "k", "v"))
        if all(name in names for name in group):
            groups.append(group)
    return tuple(groups)


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
