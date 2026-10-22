"""NestQuant reproduction toolkit with stable top-level core exports.

The implementation is separated into ``core``, ``models``, ``evaluation``
and ``experiments``. Top-level names remain available for small demos and
backwards compatibility; new code should import from the domain packages.
"""

from .core import (
    E8_DIMENSION,
    BetaSearchResult,
    QuantizedRows,
    build_candidate_tables,
    decode_nested,
    encode_nested,
    nearest_e8,
    pack_nestquant_indices,
    PackedNestQuantIndices,
    PackedNestQuantTensor,
    PackedNestQuantModel,
    quantize_rows,
    select_beta_subset,
)

__all__ = [
    "E8_DIMENSION",
    "BetaSearchResult",
    "QuantizedRows",
    "build_candidate_tables",
    "decode_nested",
    "encode_nested",
    "nearest_e8",
    "pack_nestquant_indices",
    "PackedNestQuantIndices",
    "PackedNestQuantTensor",
    "PackedNestQuantModel",
    "quantize_rows",
    "select_beta_subset",
]
