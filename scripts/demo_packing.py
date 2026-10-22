"""Demonstrate q-ary residue and beta-index packing on a tiny matrix."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.packing import PackedNestQuantIndices, pack_nestquant_indices
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor


def main() -> None:
    generator = torch.Generator().manual_seed(58)
    weight = torch.randn((4, 32), generator=generator, dtype=torch.float32)
    betas = (0.5, 0.75, 1.0, 1.25)
    quantized = quantize_rows(weight, q=14, betas=betas)
    packed = pack_nestquant_indices(
        quantized.codes, quantized.beta_indices,
        q=quantized.q, beta_count=len(quantized.betas))
    restored = PackedNestQuantIndices.from_bytes(packed.to_bytes())
    restored_codes, restored_beta_indices = restored.unpack()
    assert torch.equal(restored_codes, quantized.codes.cpu())
    assert torch.equal(restored_beta_indices, quantized.beta_indices.cpu())
    tensor_file = PackedNestQuantTensor.from_quantized_rows("toy.proj.weight", quantized)
    tensor_blob = tensor_file.to_bytes()
    restored_tensor = PackedNestQuantTensor.from_bytes(tensor_blob)
    assert torch.equal(restored_tensor.dequantize(), quantized.dequantize())

    print(f"weight shape: {tuple(weight.shape)}")
    print(f"residue stream: {len(packed.code_payload)} bytes ({packed.code_bit_width} bits/code)")
    print(f"beta-index stream: {len(packed.beta_payload)} bytes ({packed.beta_index_bit_width} bits/E8 block)")
    print(f"container header: {packed.header_bytes} bytes")
    print(f"serialized integer streams: {packed.serialized_bytes} bytes")
    print("round-trip: exact for codes and beta indices")
    print(f"self-contained single-tensor file: {len(tensor_blob)} bytes")
    print("round-trip: exact dequantized tensor reconstruction")
    print("scope: toy single-tensor artifact; not a full model checkpoint or inference format")


if __name__ == "__main__":
    main()
