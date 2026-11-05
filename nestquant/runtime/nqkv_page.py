"""GPU-resident NQKV v1 pages and a compiled reference decoder."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..core.kv_format import PackedNQKVPage
from ..core.codec import _GENERATOR, _GENERATOR_INVERSE
from .triton_unpack import _TRITON_INSTALLED, _payload_tensor


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from .triton_tiled_decode import _nearest_e8_upstream

    @triton.jit
    def _nqkv_decode_kernel(
        code_payload,
        beta_payload,
        row_norms,
        beta_table,
        output,
        block_count: tl.constexpr,
        blocks_per_row: tl.constexpr,
        feature_count: tl.constexpr,
        sqrt_feature: tl.constexpr,
        q: tl.constexpr,
        code_width: tl.constexpr,
        beta_width: tl.constexpr,
        tile_blocks: tl.constexpr,
    ):
        local_blocks = tl.arange(0, tile_blocks)[:, None]
        coordinates = tl.arange(0, 8)[None, :]
        blocks = tl.program_id(0) * tile_blocks + local_blocks
        block_mask = blocks < block_count

        code_offsets = blocks * 8 + coordinates
        codes_per_byte: tl.constexpr = 8 // code_width
        packed_codes = tl.load(
            code_payload + code_offsets // codes_per_byte,
            mask=block_mask,
            other=0,
        )
        code_shifts = (code_offsets % codes_per_byte) * code_width
        codes = ((packed_codes >> code_shifts) & ((1 << code_width) - 1)).to(tl.float32)

        betas_per_byte: tl.constexpr = 8 // beta_width
        packed_betas = tl.load(
            beta_payload + blocks // betas_per_byte,
            mask=block_mask,
            other=0,
        )
        beta_shifts = (blocks % betas_per_byte) * beta_width
        beta_ids = (packed_betas >> beta_shifts) & ((1 << beta_width) - 1)
        beta = tl.load(beta_table + beta_ids, mask=block_mask, other=0.0)

        # Apply the same E8 generator-basis transform used by the established
        # NQRT tiled decoder, but without weight-only subtractive dither.
        c7 = tl.sum(tl.where(coordinates == 7, codes, 0.0), axis=1)[:, None]
        next_offsets = blocks * 8 + tl.minimum(coordinates + 1, 7)
        packed_next = tl.load(
            code_payload + next_offsets // codes_per_byte,
            mask=block_mask,
            other=0,
        )
        next_shifts = (next_offsets % codes_per_byte) * code_width
        next_codes = (
            (packed_next >> next_shifts) & ((1 << code_width) - 1)
        ).to(tl.float32)
        coset = tl.where(
            coordinates == 0,
            2.0 * codes - next_codes + 0.5 * c7,
            tl.where(
                coordinates < 6,
                codes - next_codes + 0.5 * c7,
                tl.where(coordinates == 6, codes + 0.5 * c7, 0.5 * c7),
            ),
        )
        coarse = _nearest_e8_upstream(coset / q, coordinates)
        reconstructed = coset - q * coarse

        rows = blocks // blocks_per_row
        row_norm = tl.load(row_norms + rows, mask=block_mask, other=0.0)
        scale = row_norm / sqrt_feature
        tl.store(
            output + code_offsets,
            reconstructed * beta * scale,
            mask=block_mask,
        )

    @triton.jit
    def _nqkv_encode_kernel(
        matrix,
        row_norms,
        beta_table,
        output_codes,
        output_betas,
        block_count: tl.constexpr,
        blocks_per_row: tl.constexpr,
        feature_count: tl.constexpr,
        sqrt_feature: tl.constexpr,
        q: tl.constexpr,
        beta_count: tl.constexpr,
        tile_blocks: tl.constexpr,
    ):
        flat_blocks = tl.program_id(0) * tile_blocks + tl.arange(0, tile_blocks)
        local_blocks = tl.arange(0, tile_blocks)[:, None]
        coordinates = tl.arange(0, 8)[None, :]
        blocks = tl.program_id(0) * tile_blocks + local_blocks
        block_mask = blocks < block_count
        rows = blocks // blocks_per_row
        norms = tl.load(row_norms + rows, mask=block_mask, other=0.0)
        values = tl.load(
            matrix + blocks * 8 + coordinates, mask=block_mask, other=0.0
        ).to(tl.float32)
        normalized = tl.where(
            norms > 0.0,
            values * (sqrt_feature / norms),
            0.0,
        )
        best_error = tl.full((tile_blocks,), float("inf"), tl.float32)
        best_beta = tl.zeros((tile_blocks,), tl.int32)
        best_codes = tl.zeros((tile_blocks, 8), tl.int32)
        for beta_id in tl.static_range(0, beta_count):
            beta = tl.load(beta_table + beta_id)
            lattice = _nearest_e8_upstream(normalized / beta, coordinates)
            x0 = tl.sum(tl.where(coordinates == 0, lattice, 0.0), axis=1)
            x1 = tl.sum(tl.where(coordinates == 1, lattice, 0.0), axis=1)
            x2 = tl.sum(tl.where(coordinates == 2, lattice, 0.0), axis=1)
            x3 = tl.sum(tl.where(coordinates == 3, lattice, 0.0), axis=1)
            x4 = tl.sum(tl.where(coordinates == 4, lattice, 0.0), axis=1)
            x5 = tl.sum(tl.where(coordinates == 5, lattice, 0.0), axis=1)
            x6 = tl.sum(tl.where(coordinates == 6, lattice, 0.0), axis=1)
            x7 = tl.sum(tl.where(coordinates == 7, lattice, 0.0), axis=1)
            c7 = 2.0 * x7
            c6 = x6 - x7
            c5 = x5 + c6 - x7
            c4 = x4 + c5 - x7
            c3 = x3 + c4 - x7
            c2 = x2 + c3 - x7
            c1 = x1 + c2 - x7
            c0 = 0.5 * (x0 + c1 - x7)
            basis = tl.where(
                coordinates == 0, c0[:, None],
                tl.where(coordinates == 1, c1[:, None],
                tl.where(coordinates == 2, c2[:, None],
                tl.where(coordinates == 3, c3[:, None],
                tl.where(coordinates == 4, c4[:, None],
                tl.where(coordinates == 5, c5[:, None],
                tl.where(coordinates == 6, c6[:, None], c7[:, None])))))))
            codes = libdevice.nearbyint(basis).to(tl.int32) % q
            codes = tl.where(codes < 0, codes + q, codes)
            code7 = tl.sum(tl.where(coordinates == 7, codes, 0), axis=1)
            # Build the shifted coordinate vector with scalar lane reductions
            # so this remains compatible with the project's Triton release.
            c_next = tl.where(
                coordinates == 0, tl.sum(tl.where(coordinates == 1, codes, 0), axis=1)[:, None],
                tl.where(coordinates == 1, tl.sum(tl.where(coordinates == 2, codes, 0), axis=1)[:, None],
                tl.where(coordinates == 2, tl.sum(tl.where(coordinates == 3, codes, 0), axis=1)[:, None],
                tl.where(coordinates == 3, tl.sum(tl.where(coordinates == 4, codes, 0), axis=1)[:, None],
                tl.where(coordinates == 4, tl.sum(tl.where(coordinates == 5, codes, 0), axis=1)[:, None],
                tl.where(coordinates == 5, tl.sum(tl.where(coordinates == 6, codes, 0), axis=1)[:, None],
                tl.where(coordinates == 6, code7[:, None], code7[:, None])))))))
            coset = tl.where(
                coordinates == 0, 2.0 * codes - c_next + 0.5 * code7[:, None],
                tl.where(coordinates < 6, codes - c_next + 0.5 * code7[:, None],
                tl.where(coordinates == 6, codes + 0.5 * code7[:, None],
                         0.5 * code7[:, None])),
            ).to(tl.float32)
            coarse = _nearest_e8_upstream(coset / q, coordinates)
            reconstructed = (coset - q * coarse) * beta
            delta = reconstructed - normalized
            error = tl.sum(delta * delta, axis=1)
            choose = error < best_error
            best_error = tl.where(choose, error, best_error)
            best_beta = tl.where(choose, beta_id, best_beta)
            best_codes = tl.where(choose[:, None], codes, best_codes)
        tl.store(output_codes + blocks * 8 + coordinates, best_codes, mask=block_mask)
        tl.store(output_betas + flat_blocks, best_beta, mask=flat_blocks < block_count)
else:
    triton = None
    _nqkv_decode_kernel = None
    _nqkv_encode_kernel = None


@dataclass(frozen=True)
class PreparedNQKVPage:
    """Fixed-address CUDA tensors for one NQKV v1 physical page."""

    key_code_payload: torch.Tensor
    key_beta_payload: torch.Tensor
    key_norms: torch.Tensor
    value_code_payload: torch.Tensor
    value_beta_payload: torch.Tensor
    value_norms: torch.Tensor
    key_beta_table: torch.Tensor
    value_beta_table: torch.Tensor
    page_size: int
    valid_tokens: int
    num_kv_heads: int
    head_dim: int
    q: int
    code_width: int
    beta_width: int
    output_dtype: torch.dtype

    @property
    def device(self) -> torch.device:
        return self.key_code_payload.device

    @property
    def beta_table(self) -> torch.Tensor:
        """Legacy alias for v1 callers whose Key/Value tables are shared."""
        return self.key_beta_table

    @property
    def resident_bytes(self) -> int:
        tensors = (
                self.key_code_payload,
                self.key_beta_payload,
                self.key_norms,
                self.value_code_payload,
                self.value_beta_payload,
                self.value_norms,
                self.key_beta_table,
                self.value_beta_table,
            )
        return sum(
            tensor.numel() * tensor.element_size()
            for index, tensor in enumerate(tensors)
            if tensor.data_ptr() not in {item.data_ptr() for item in tensors[:index]}
        )


def quantize_nqkv_page_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    page_size: int,
    q: int,
    betas: tuple[float, ...] | list[float] | None = None,
    key_betas: tuple[float, ...] | list[float] | None = None,
    value_betas: tuple[float, ...] | list[float] | None = None,
    key_beta_table: torch.Tensor | None = None,
    value_beta_table: torch.Tensor | None = None,
) -> PreparedNQKVPage:
    """Quantize and bit-pack one NHD page without a CPU payload round trip.

    This is a correctness-first GPU writer. E8 encoding uses the established
    PyTorch CUDA oracle and fixed-width packing stays on CUDA. A later fused
    writer may replace its launch structure while retaining the byte order.
    """
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("key and value must be torch tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key and value must share NHD shape [tokens, kv_heads, head_dim]")
    if key.device != value.device or key.dtype != value.dtype or key.device.type != "cuda":
        raise ValueError("key and value must share CUDA device and dtype")
    if not key.is_floating_point():
        raise TypeError("key and value must be floating point")
    if not 1 <= key.shape[0] <= page_size:
        raise ValueError("token count must be in [1, page_size]")
    if key.shape[2] % 8:
        raise ValueError("head_dim must be divisible by 8")
    if betas is not None:
        if key_betas is not None or value_betas is not None:
            raise ValueError("pass legacy betas or key_betas/value_betas, not both")
        key_beta_values = tuple(float(beta) for beta in betas)
        value_beta_values = key_beta_values
    else:
        if key_betas is None or value_betas is None:
            raise ValueError("v2 requires both key_betas and value_betas")
        key_beta_values = tuple(float(beta) for beta in key_betas)
        value_beta_values = tuple(float(beta) for beta in value_betas)
    if not key_beta_values or not value_beta_values:
        raise ValueError("beta tables must be nonempty")

    valid_tokens, num_heads, head_dim = key.shape
    rows = page_size * num_heads
    valid_rows = valid_tokens * num_heads
    key_rows = torch.zeros((rows, head_dim), device=key.device, dtype=torch.float32)
    value_rows = torch.zeros_like(key_rows)
    key_rows[:valid_rows].copy_(key.reshape(valid_rows, head_dim).float())
    value_rows[:valid_rows].copy_(value.reshape(valid_rows, head_dim).float())
    key_codes, key_betas, key_norms = _quantize_rows_cuda_fused(
        key_rows, q=q, betas=key_beta_values, beta_table=key_beta_table
    )
    value_codes, value_betas, value_norms = _quantize_rows_cuda_fused(
        value_rows, q=q, betas=value_beta_values, beta_table=value_beta_table
    )
    code_width = max(1, math.ceil(math.log2(q)))
    key_beta_width = max(1, math.ceil(math.log2(len(key_beta_values))))
    value_beta_width = max(1, math.ceil(math.log2(len(value_beta_values))))
    if (
        code_width not in {1, 2, 4, 8}
        or key_beta_width not in {1, 2, 4, 8}
        or value_beta_width != key_beta_width
    ):
        raise ValueError("NQKV v1 supports only 1, 2, 4, or 8 bit fields")
    return PreparedNQKVPage(
        key_code_payload=_pack_unsigned_cuda(key_codes, code_width),
        key_beta_payload=_pack_unsigned_cuda(key_betas, key_beta_width),
        key_norms=key_norms,
        value_code_payload=_pack_unsigned_cuda(value_codes, code_width),
        value_beta_payload=_pack_unsigned_cuda(value_betas, value_beta_width),
        value_norms=value_norms,
        key_beta_table=(
            key_beta_table if key_beta_table is not None else
            torch.tensor(key_beta_values, device=key.device, dtype=torch.float32)
        ),
        value_beta_table=(
            value_beta_table if value_beta_table is not None else
            torch.tensor(value_beta_values, device=key.device, dtype=torch.float32)
        ),
        page_size=page_size,
        valid_tokens=valid_tokens,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        q=q,
        code_width=code_width,
        beta_width=key_beta_width,
        output_dtype=key.dtype,
    )


def _quantize_rows_cuda_fused(
    matrix: torch.Tensor, *, q: int, betas: tuple[float, ...],
    beta_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse E8 multi-beta selection into one Triton launch per tensor."""
    if not _TRITON_INSTALLED or _nqkv_encode_kernel is None:
        return _quantize_rows_cuda_fast(matrix, q=q, betas=betas)
    assert triton is not None
    rows, feature_count = matrix.shape
    blocks_per_row = feature_count // 8
    block_count = rows * blocks_per_row
    row_norms = torch.linalg.vector_norm(matrix, ord=2, dim=-1).to(torch.float32)
    if beta_table is None:
        beta_table = torch.tensor(
            betas, device=matrix.device, dtype=torch.float32
        )
    elif (
        beta_table.device != matrix.device
        or beta_table.dtype != torch.float32
        or beta_table.ndim != 1
        or beta_table.numel() != len(betas)
        or not beta_table.is_contiguous()
    ):
        raise ValueError(
            "beta_table must be contiguous float32 on the matrix device"
        )
    codes = torch.empty((rows, blocks_per_row, 8), device=matrix.device, dtype=torch.uint8)
    beta_indices = torch.empty((rows, blocks_per_row), device=matrix.device, dtype=torch.uint8)
    tile_blocks = 16
    _nqkv_encode_kernel[(triton.cdiv(block_count, tile_blocks),)](
        matrix, row_norms, beta_table, codes, beta_indices,
        block_count=block_count,
        blocks_per_row=blocks_per_row,
        feature_count=feature_count,
        sqrt_feature=math.sqrt(feature_count),
        q=q,
        beta_count=beta_table.numel(),
        tile_blocks=tile_blocks,
        num_warps=4,
    )
    return codes, beta_indices, row_norms


def _nearest_e8_cuda_unchecked(points: torch.Tensor) -> torch.Tensor:
    """Upstream-tie E8 nearest point without host-synchronizing validation."""
    rounded = torch.round(points)
    residual = points - rounded
    coordinate = residual.abs().argmax(dim=-1, keepdim=True)
    direction = torch.where(
        residual.gather(-1, coordinate) >= 0,
        torch.ones_like(coordinate, dtype=points.dtype),
        -torch.ones_like(coordinate, dtype=points.dtype),
    )
    odd = torch.remainder(rounded.sum(dim=-1, keepdim=True), 2) != 0
    integer = rounded + torch.zeros_like(rounded).scatter(
        -1, coordinate, direction
    ) * odd.to(points.dtype)
    half = torch.full((), 0.5, device=points.device, dtype=points.dtype)
    shifted = points - half
    shifted_rounded = torch.round(shifted)
    shifted_residual = shifted - shifted_rounded
    shifted_coordinate = shifted_residual.abs().argmax(dim=-1, keepdim=True)
    shifted_direction = torch.where(
        shifted_residual.gather(-1, shifted_coordinate) >= 0,
        torch.ones_like(shifted_coordinate, dtype=points.dtype),
        -torch.ones_like(shifted_coordinate, dtype=points.dtype),
    )
    shifted_odd = torch.remainder(
        shifted_rounded.sum(dim=-1, keepdim=True), 2
    ) != 0
    half_integer = shifted_rounded + torch.zeros_like(shifted_rounded).scatter(
        -1, shifted_coordinate, shifted_direction
    ) * shifted_odd.to(points.dtype) + half
    integer_error = (points - integer).square().sum(dim=-1)
    half_error = (points - half_integer).square().sum(dim=-1)
    return torch.where((integer_error < half_error).unsqueeze(-1), integer, half_integer)


def _quantize_rows_cuda_fast(
    matrix: torch.Tensor, *, q: int, betas: tuple[float, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runtime-only multi-beta encoder with no scalar CUDA-to-host checks.

    Public input validation happens once in :func:`quantize_nqkv_page_cuda`.
    Removing reference-code ``.item()`` checks avoids a device synchronize for
    every E8 encode/decode operation while preserving the canonical byte order.
    """
    row_norms = torch.linalg.vector_norm(matrix, ord=2, dim=-1)
    nonzero = row_norms > 0
    safe_norms = torch.where(nonzero, row_norms, torch.ones_like(row_norms))
    normalized = matrix * (math.sqrt(matrix.shape[1]) / safe_norms).unsqueeze(-1)
    normalized = torch.where(nonzero.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    blocks = normalized.reshape(matrix.shape[0], -1, 8)
    generator = _GENERATOR.to(device=matrix.device, dtype=matrix.dtype)
    inverse = _GENERATOR_INVERSE.to(device=matrix.device, dtype=matrix.dtype)
    candidate_codes = []
    candidate_errors = []
    for beta in betas:
        lattice = _nearest_e8_cuda_unchecked(blocks / beta)
        coordinates = torch.round(lattice @ inverse.T)
        codes = torch.remainder(coordinates.to(torch.int64), q)
        coset = codes.to(matrix.dtype) @ generator.T
        coarse = _nearest_e8_cuda_unchecked(coset / q)
        reconstructed = (coset - q * coarse) * beta
        candidate_codes.append(codes)
        candidate_errors.append((reconstructed - blocks).square().sum(dim=-1))
    code_stack = torch.stack(candidate_codes, dim=0)
    errors = torch.stack(candidate_errors, dim=0)
    best_beta = errors.argmin(dim=0)
    gather = best_beta.unsqueeze(0).unsqueeze(-1).expand(1, *best_beta.shape, 8)
    best_codes = torch.gather(code_stack, 0, gather).squeeze(0)
    return best_codes, best_beta, row_norms.to(torch.float32)


def _pack_unsigned_cuda(values: torch.Tensor, width: int) -> torch.Tensor:
    per_byte = 8 // width
    flat = values.reshape(-1).to(torch.uint8)
    padded_count = ((flat.numel() + per_byte - 1) // per_byte) * per_byte
    if padded_count != flat.numel():
        padded = torch.zeros(padded_count, device=flat.device, dtype=torch.uint8)
        padded[:flat.numel()].copy_(flat)
        flat = padded
    groups = flat.reshape(-1, per_byte).to(torch.int32)
    shifts = torch.arange(per_byte, device=flat.device, dtype=torch.int32) * width
    return torch.sum(groups << shifts, dim=1).to(torch.uint8)


def prepare_nqkv_page(
    page: PackedNQKVPage, *, device: torch.device | str = "cuda"
) -> PreparedNQKVPage:
    """Upload only packed NQKV streams and compact metadata to CUDA."""
    if not isinstance(page, PackedNQKVPage):
        raise TypeError("page must be a PackedNQKVPage")
    page._validate()
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("prepared NQKV pages require an available CUDA device")
    key_beta_table = torch.tensor(page.betas, device=target, dtype=torch.float32)
    value_beta_table = (
        key_beta_table if page.value_betas is None else
        torch.tensor(page.value_betas, device=target, dtype=torch.float32)
    )
    return PreparedNQKVPage(
        key_code_payload=_payload_tensor(page.key_indices.code_payload, target),
        key_beta_payload=_payload_tensor(page.key_indices.beta_payload, target),
        key_norms=page.key_norms.to(device=target),
        value_code_payload=_payload_tensor(page.value_indices.code_payload, target),
        value_beta_payload=_payload_tensor(page.value_indices.beta_payload, target),
        value_norms=page.value_norms.to(device=target),
        key_beta_table=key_beta_table,
        value_beta_table=value_beta_table,
        page_size=page.page_size,
        valid_tokens=page.valid_tokens,
        num_kv_heads=page.num_kv_heads,
        head_dim=page.head_dim,
        q=page.q,
        code_width=page.key_indices.code_bit_width,
        beta_width=page.key_indices.beta_index_bit_width,
        output_dtype=page.output_dtype,
    )


def dequantize_nqkv_page_triton(
    page: PreparedNQKVPage,
    *,
    output_dtype: torch.dtype | None = None,
    tile_blocks: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a resident NQKV page without expanded code or beta tensors."""
    if not isinstance(page, PreparedNQKVPage):
        raise TypeError("page must be a PreparedNQKVPage")
    if not _TRITON_INSTALLED or not torch.cuda.is_available():
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if tile_blocks not in {16, 32, 64, 128}:
        raise ValueError("tile_blocks must be one of 16, 32, 64, or 128")
    dtype = page.output_dtype if output_dtype is None else output_dtype
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("output_dtype must be float16, bfloat16, or float32")
    assert triton is not None and _nqkv_decode_kernel is not None
    rows = page.page_size * page.num_kv_heads
    blocks_per_row = page.head_dim // 8
    block_count = rows * blocks_per_row
    key = torch.empty((rows, page.head_dim), device=page.device, dtype=dtype)
    value = torch.empty_like(key)
    grid = (triton.cdiv(block_count, tile_blocks),)
    common = {
        "block_count": block_count,
        "blocks_per_row": blocks_per_row,
        "feature_count": page.head_dim,
        "sqrt_feature": math.sqrt(page.head_dim),
        "q": page.q,
        "code_width": page.code_width,
        "beta_width": page.beta_width,
        "tile_blocks": tile_blocks,
        "num_warps": 4,
    }
    _nqkv_decode_kernel[grid](
        page.key_code_payload, page.key_beta_payload, page.key_norms,
        page.key_beta_table, key, **common,
    )
    _nqkv_decode_kernel[grid](
        page.value_code_payload, page.value_beta_payload, page.value_norms,
        page.value_beta_table, value, **common,
    )
    shape = (page.page_size, page.num_kv_heads, page.head_dim)
    return key.reshape(shape)[:page.valid_tokens], value.reshape(shape)[:page.valid_tokens]
