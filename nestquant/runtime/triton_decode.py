"""Hybrid decoder using Triton unpack with the reference E8 reconstruction."""

from __future__ import annotations

from collections.abc import Callable
import time

import torch

from ..core.quantizer import QuantizedRows
from ..core.codec import generate_e8_dither_stateless
from ..core.tensor_checkpoint import (
    PackedNestQuantTensor,
    _DTYPE_TO_NAME,
    _generate_dither_from_state,
)
from .triton_unpack import unpack_indices_triton


def dequantize_with_triton_unpack(
    record: PackedNestQuantTensor,
    *,
    device: torch.device | str = "cuda",
    output_dtype: torch.dtype | None = None,
    rows_per_chunk: int = 128,
    profile_callback: Callable[[dict[str, object]], None] | None = None,
) -> torch.Tensor:
    """Decode with compiled packed-stream expansion and reference E8 math.

    This isolates the benefit of replacing NumPy unpack and per-chunk integer
    transfers. Dither replay and E8 reconstruction intentionally retain their
    existing semantics and PyTorch implementation.
    """
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("record must be a PackedNestQuantTensor")
    if isinstance(rows_per_chunk, bool) or not isinstance(rows_per_chunk, int) or rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be a positive integer")
    target_dtype = record.output_dtype if output_dtype is None else output_dtype
    if target_dtype not in _DTYPE_TO_NAME:
        raise ValueError("output_dtype must be float16, bfloat16, float32, or float64")
    target_device = torch.device(device)
    def measured(phase: str, operation, *, synchronize: bool = False):
        if profile_callback is None:
            return operation()
        if synchronize and target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        value = operation()
        if synchronize and target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        profile_callback({"phase": phase, "seconds": time.perf_counter() - started})
        return value

    codes, beta_indices = measured("triton_unpack", lambda: unpack_indices_triton(
        record.indices, device=target_device, output_dtype=torch.int16
    ), synchronize=True)
    # PyTorch advanced indexing requires int32/int64; beta IDs are only one
    # value per E8 block, so widening this smaller stream is inexpensive.
    beta_indices = beta_indices.to(torch.int64)
    generated_dither = None
    if record.dither_generator_state is not None:
        generated_dither = measured("cpu_dither_replay", lambda: _generate_dither_from_state(
            record.dither_generator_state,
            record.indices.shape,
            device=record.dither_generator_device,
            dtype=record.output_dtype,
            output_device="cpu",
            codec_semantics=record.codec_semantics,
        ))
    rows = record.indices.shape[0]
    output = torch.empty(
        (rows, record.feature_count), device=target_device, dtype=target_dtype
    )
    def reconstruct() -> None:
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            dither = None
            if record.dither is not None:
                dither = record.dither[start:end].to(target_device)
            elif generated_dither is not None:
                dither = generated_dither[start:end].to(target_device)
            elif record.dither_seed is not None:
                blocks_per_row = record.indices.shape[1]
                dither = generate_e8_dither_stateless(
                    (end - start) * blocks_per_row,
                    record.dither_seed,
                    block_offset=start * blocks_per_row,
                    device=target_device,
                    dtype=record.output_dtype,
                    codec_semantics=record.codec_semantics,
                    algorithm=record.dither_algorithm,
                ).reshape(end - start, blocks_per_row, 8)
            chunk = QuantizedRows(
                codes=codes[start:end],
                beta_indices=beta_indices[start:end],
                dither=dither,
                row_norms=record.row_norms[start:end].to(target_device),
                q=record.indices.q,
                betas=record.betas,
                feature_count=record.feature_count,
                output_dtype=record.output_dtype,
                codec_semantics=record.codec_semantics,
            ).dequantize()
            output[start:end].copy_(chunk.to(target_dtype))

    measured("e8_reconstruct_scale", reconstruct, synchronize=True)
    return output
