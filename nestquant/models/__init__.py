"""Adapters applying algorithm primitives to neural-network modules."""

from .quantization import quantize_nestquant_w4_, quantize_rtn_w4_
from .rotated_linear import (
    HadamardConversionReport,
    HadamardRotatedLinear,
    convert_linears_to_hadamard_coordinates_,
)

__all__ = [
    "HadamardConversionReport",
    "HadamardRotatedLinear",
    "convert_linears_to_hadamard_coordinates_",
    "quantize_nestquant_w4_",
    "quantize_rtn_w4_",
]
