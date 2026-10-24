"""Triton fixed-width unpack for NQPK code and beta-index streams."""

from __future__ import annotations

import importlib.util
import math

import torch

from ..core.packing import PackedNestQuantIndices


_TRITON_INSTALLED = importlib.util.find_spec("triton") is not None
if _TRITON_INSTALLED:
    import triton
    import triton.language as tl

    @triton.jit
    def _unpack_fixed_width(payload, output, value_count: tl.constexpr,
                            width: tl.constexpr, block_size: tl.constexpr):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < value_count
        values_per_byte: tl.constexpr = 8 // width
        packed = tl.load(payload + offsets // values_per_byte, mask=mask, other=0)
        shifts = (offsets % values_per_byte) * width
        values = (packed >> shifts) & ((1 << width) - 1)
        tl.store(output + offsets, values, mask=mask)
else:
    triton = None
    _unpack_fixed_width = None


def triton_available() -> bool:
    return _TRITON_INSTALLED and torch.cuda.is_available()


def _payload_tensor(payload: bytes, device: torch.device) -> torch.Tensor:
    # Clone detaches the tensor from Python's immutable bytes buffer before H2D.
    host = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    return host.to(device=device, non_blocking=False)


def unpack_indices_triton(
    packed: PackedNestQuantIndices,
    *,
    device: torch.device | str = "cuda",
    output_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack complete NQPK streams on CUDA without CPU integer expansion.

    This is an isolated compiled-unpack prototype. It still uploads packed
    payload bytes and materializes expanded code/index tensors; it does not
    perform E8 reconstruction, dither replay, row scaling, or GEMM.
    """
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Triton unpack requires an available CUDA device")
    if output_dtype not in {torch.int16, torch.int32, torch.int64}:
        raise ValueError("output_dtype must be int16, int32, or int64")
    if not _TRITON_INSTALLED:
        raise RuntimeError("Triton is not installed")
    assert triton is not None and _unpack_fixed_width is not None
    code_count = math.prod(packed.shape)
    beta_shape = packed.shape[:-1]
    beta_count = math.prod(beta_shape)
    code_payload = _payload_tensor(packed.code_payload, target)
    beta_payload = _payload_tensor(packed.beta_payload, target)
    codes = torch.empty(code_count, device=target, dtype=output_dtype)
    beta_indices = torch.empty(beta_count, device=target, dtype=output_dtype)
    block_size = 256
    _unpack_fixed_width[(triton.cdiv(code_count, block_size),)](
        code_payload, codes, value_count=code_count,
        width=packed.code_bit_width, block_size=block_size,
    )
    _unpack_fixed_width[(triton.cdiv(beta_count, block_size),)](
        beta_payload, beta_indices, value_count=beta_count,
        width=packed.beta_index_bit_width, block_size=block_size,
    )
    return codes.reshape(packed.shape), beta_indices.reshape(beta_shape)
