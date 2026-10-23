"""NestQuant algorithm primitives: lattice, codec, beta search, quantizer, packing."""

from .lattice import E8_DIMENSION, nearest_e8
from .codec import decode_nested, encode_nested, generate_e8_dither
from .beta_search import BetaSearchResult, build_candidate_tables, select_beta_subset
from .quantizer import (
    QuantizedBlocks, QuantizedRows, quantize_e8_blocks, quantize_rows,
    quantize_rows_ldlq,
)
from .packing import PackedNestQuantIndices, pack_nestquant_indices
from .tensor_checkpoint import PackedNestQuantTensor
from .model_checkpoint import PackedNestQuantModel
from .model_checkpoint_file import NQMCEntry, NQMCFileReader
from .ldlq import (
    ActivationCovarianceAccumulator,
    block_unit_lower_ldl,
    hessian_proxy_loss,
    ldlq_quantize_blocks,
    ldlq_quantize_columns,
    qa_ldlq_adjustment,
    regularize_hessian,
    unit_lower_ldl,
)

__all__ = [
    "E8_DIMENSION", "nearest_e8", "decode_nested", "encode_nested",
    "generate_e8_dither", "BetaSearchResult", "build_candidate_tables",
    "select_beta_subset", "QuantizedBlocks", "QuantizedRows", "quantize_e8_blocks",
    "quantize_rows", "quantize_rows_ldlq",
    "PackedNestQuantIndices", "pack_nestquant_indices", "PackedNestQuantTensor",
    "PackedNestQuantModel", "NQMCEntry", "NQMCFileReader",
    "ActivationCovarianceAccumulator", "regularize_hessian", "unit_lower_ldl",
    "block_unit_lower_ldl", "ldlq_quantize_columns", "ldlq_quantize_blocks",
    "hessian_proxy_loss", "qa_ldlq_adjustment",
]
