"""Nested E8 Voronoi-code encoder and decoder (reference implementation).

For integer nesting ratio q, the encoder transmits the E8 lattice coordinates
modulo q. The decoder maps that coset to its minimum-energy representative in
the scaled Voronoi region q V_E8.
"""

from __future__ import annotations

import torch

from .lattice import E8_DIMENSION, nearest_e8


UPSTREAM_CODEC_SEMANTICS = "e8_upstream_half_coset_tie_v1"
LEGACY_CODEC_SEMANTICS = "e8_legacy_integer_coset_tie_v1"
CURRENT_CODEC_SEMANTICS = UPSTREAM_CODEC_SEMANTICS
STATELESS_DITHER_ALGORITHM = "lcg31_mix_v1"
_STATELESS_MODULUS = 1 << 31


def _tie_break(codec_semantics: str) -> str:
    if codec_semantics == UPSTREAM_CODEC_SEMANTICS:
        return "upstream_half"
    if codec_semantics == LEGACY_CODEC_SEMANTICS:
        return "legacy_integer"
    raise ValueError(f"unsupported codec semantics: {codec_semantics}")


_GENERATOR = torch.tensor(
    [
        [2, -1, 0, 0, 0, 0, 0, 0.5],
        [0, 1, -1, 0, 0, 0, 0, 0.5],
        [0, 0, 1, -1, 0, 0, 0, 0.5],
        [0, 0, 0, 1, -1, 0, 0, 0.5],
        [0, 0, 0, 0, 1, -1, 0, 0.5],
        [0, 0, 0, 0, 0, 1, -1, 0.5],
        [0, 0, 0, 0, 0, 0, 1, 0.5],
        [0, 0, 0, 0, 0, 0, 0, 0.5],
    ],
    dtype=torch.float64,
)
_GENERATOR_INVERSE = torch.linalg.inv(_GENERATOR)


def _validate_q(q: int) -> None:
    if isinstance(q, bool) or not isinstance(q, int):
        raise TypeError("q must be an integer")
    if q < 2:
        raise ValueError("q must be at least 2")


def encode_nested(
    x: torch.Tensor, q: int, *, dither: torch.Tensor | None = None,
    codec_semantics: str = CURRENT_CODEC_SEMANTICS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode vectors as E8 lattice coordinates modulo ``q``.

    Args:
        x: Floating-point tensor with final dimension 8. Leading dimensions
            represent independent vectors.
        q: Integer nesting ratio, at least 2.
        dither: Optional subtractive E8 dither with the same shape, device,
            and dtype as ``x``. The same dither must be passed to the decoder.

    Returns:
        ``(code, overloaded)``. ``code`` is an int64 tensor with the input
        shape and entries in ``[0, q)``. ``overloaded`` is a bool tensor with
        shape ``x.shape[:-1]``; true means the nearest lattice point lies
        outside ``q V_E8``, so decoding returns the coset representative rather
        than that original nearest lattice point.
    """
    _validate_q(q)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError("x must have a floating-point dtype")
    if x.ndim == 0 or x.shape[-1] != E8_DIMENSION:
        raise ValueError(f"the final dimension of x must be {E8_DIMENSION}")
    if not torch.isfinite(x).all().item():
        raise ValueError("x must contain only finite values")

    _validate_dither(x, dither)
    shifted = x if dither is None else x + dither
    tie_break = _tie_break(codec_semantics)
    lattice_point, _ = nearest_e8(shifted, tie_break=tie_break)
    generator = _GENERATOR.to(device=x.device, dtype=x.dtype)
    inverse = _GENERATOR_INVERSE.to(device=x.device, dtype=x.dtype)
    coordinates_float = lattice_point @ inverse.T
    coordinates = torch.round(coordinates_float)
    # A large residual means the closest-point routine and basis disagree.
    tolerance = 1e-5 if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-10
    if ((coordinates_float - coordinates).abs() > tolerance).any().item():
        raise RuntimeError("nearest E8 point did not map to integer generator coordinates")

    code = torch.remainder(coordinates.to(torch.int64), q)
    unshifted_lattice_point = lattice_point if dither is None else lattice_point - dither
    scaled_nearest, _ = nearest_e8(
        unshifted_lattice_point / q, tie_break=tie_break
    )
    overloaded = (scaled_nearest != 0).any(dim=-1)
    return code, overloaded


def decode_nested(
    code: torch.Tensor,
    q: int,
    *,
    dtype: torch.dtype = torch.float32,
    dither: torch.Tensor | None = None,
    codec_semantics: str = CURRENT_CODEC_SEMANTICS,
) -> torch.Tensor:
    """Decode E8 coordinates modulo ``q`` to the minimum-energy representative.

    ``code`` must contain valid integer residues in ``[0, q)``. The returned
    tensor is floating point on the same device as ``code``. Pass ``dtype`` to
    select the reconstruction precision; the default is float32.
    """
    _validate_q(q)
    if not isinstance(code, torch.Tensor):
        raise TypeError("code must be a torch.Tensor")
    if code.dtype == torch.bool or code.is_floating_point() or code.is_complex():
        raise TypeError("code must have an integer dtype")
    if code.ndim == 0 or code.shape[-1] != E8_DIMENSION:
        raise ValueError(f"the final dimension of code must be {E8_DIMENSION}")
    if ((code < 0) | (code >= q)).any().item():
        raise ValueError("all code coordinates must be in [0, q)")
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point torch.dtype")

    generator = _GENERATOR.to(device=code.device, dtype=dtype)
    coset_point = code.to(dtype) @ generator.T
    if dither is not None:
        if dither.shape != code.shape or dither.device != code.device or dither.dtype != dtype:
            raise ValueError("dither must match code shape/device and requested decode dtype")
        if not torch.isfinite(dither).all().item():
            raise ValueError("dither must contain only finite values")
        coset_point = coset_point - dither
    nearest_to_scaled, _ = nearest_e8(
        coset_point / q, tie_break=_tie_break(codec_semantics)
    )
    return coset_point - q * nearest_to_scaled


def generate_e8_dither(
    count: int,
    generator: torch.Generator,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    codec_semantics: str = CURRENT_CODEC_SEMANTICS,
) -> torch.Tensor:
    """Generate subtractive dither as in the author reference implementation.

    Draw ``U`` uniformly from ``[0, 2)^8`` and return ``U - Q_E8(U)``, which
    lies in a fundamental Voronoi region of E8.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    target_device = torch.device(device)
    generator_device = torch.device(generator.device)
    if generator_device != target_device and generator_device.type != "cpu":
        raise ValueError("a non-CPU generator must match the requested dither device")
    # Keep the random stream portable by drawing on CPU even when the E8
    # nearest-point operation runs on an accelerator.
    uniform = torch.rand((count, 8), generator=generator,
                         device=generator_device, dtype=dtype) * 2.0
    if uniform.device != target_device:
        uniform = uniform.to(target_device)
    nearest, _ = nearest_e8(uniform, tie_break=_tie_break(codec_semantics))
    return uniform - nearest


def generate_e8_dither_stateless(
    count: int,
    seed: int,
    *,
    block_offset: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    codec_semantics: str = CURRENT_CODEC_SEMANTICS,
    algorithm: str = STATELESS_DITHER_ALGORITHM,
) -> torch.Tensor:
    """Generate random-access E8 dither from global block coordinates.

    The integer hash uses only signed int64 operations whose intermediate
    values stay below ``2**63``. CPU and CUDA therefore execute the same
    specified arithmetic. Uniform samples are constructed in float32 before
    the requested output cast, making a block independent of call shape,
    chunk size, execution order, and PyTorch RNG state.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if isinstance(block_offset, bool) or not isinstance(block_offset, int) or block_offset < 0:
        raise ValueError("block_offset must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _STATELESS_MODULUS:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if algorithm != STATELESS_DITHER_ALGORITHM:
        raise ValueError(f"unsupported stateless dither algorithm: {algorithm}")
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point torch.dtype")
    if block_offset + count > (_STATELESS_MODULUS // E8_DIMENSION):
        raise ValueError("block range exceeds the stateless dither counter space")

    target_device = torch.device(device)
    first = block_offset * E8_DIMENSION
    counters = torch.arange(
        first, first + count * E8_DIMENSION,
        device=target_device, dtype=torch.int64,
    )
    values = torch.remainder(counters + seed, _STATELESS_MODULUS)
    # A coordinate hash rather than a sequential RNG: every output is a pure
    # function of its counter. Constants and modulus are part of the v1 spec.
    values = torch.bitwise_xor(values, values >> 16)
    values = torch.remainder(values * 1_103_515_245 + 12_345, _STATELESS_MODULUS)
    values = torch.bitwise_xor(values, values >> 13)
    values = torch.remainder(values * 134_775_813 + 1, _STATELESS_MODULUS)
    values = torch.bitwise_xor(values, values >> 16)
    uniform = values.to(torch.float32).reshape(count, E8_DIMENSION)
    uniform = uniform * (2.0 / float(_STATELESS_MODULUS))
    nearest, _ = nearest_e8(uniform, tie_break=_tie_break(codec_semantics))
    return (uniform - nearest).to(dtype)


def _validate_dither(x: torch.Tensor, dither: torch.Tensor | None) -> None:
    if dither is None:
        return
    if not isinstance(dither, torch.Tensor) or dither.shape != x.shape:
        raise ValueError("dither must be a tensor with the same shape as x")
    if dither.device != x.device or dither.dtype != x.dtype:
        raise ValueError("dither must match x device and dtype")
    if not torch.isfinite(dither).all().item():
        raise ValueError("dither must contain only finite values")
