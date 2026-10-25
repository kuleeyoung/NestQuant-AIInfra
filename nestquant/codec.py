"""Compatibility import; use :mod:`nestquant.core.codec` in new code."""

from .core.codec import (
    decode_nested,
    encode_nested,
    generate_e8_dither,
    generate_e8_dither_stateless,
)

__all__ = [
    "decode_nested", "encode_nested", "generate_e8_dither",
    "generate_e8_dither_stateless",
]
