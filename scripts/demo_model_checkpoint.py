"""End-to-end toy model checkpoint round trip with tied weights."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.model_checkpoint import PackedNestQuantModel
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 16)
        self.lm_head = nn.Linear(16, 16, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.norm = nn.LayerNorm(16)


def main() -> None:
    model = TinyLanguageModel()
    state = model.state_dict()
    quantized_rows = quantize_rows(
        state["lm_head.weight"].float(), q=14, betas=(0.5, 0.75, 1.0, 1.25)
    )
    packed_weight = PackedNestQuantTensor.from_quantized_rows("lm_head.weight", quantized_rows)
    checkpoint = PackedNestQuantModel.from_state_dict(
        state, {"lm_head.weight": packed_weight},
        model_metadata={"model_id": "tiny-language-model", "format_demo": True},
    )
    blob = checkpoint.to_bytes()
    loaded = PackedNestQuantModel.from_bytes(blob)
    restored = loaded.state_dict()
    assert set(restored) == set(state)
    assert restored["embed_tokens.weight"] is restored["lm_head.weight"]
    assert torch.equal(restored["lm_head.weight"], quantized_rows.dequantize())
    assert torch.equal(restored["norm.weight"], state["norm.weight"])

    print(f"state-dict names restored: {len(restored)}")
    print(f"unique dense tensors: {len(loaded.dense_tensors)}")
    print(f"unique NestQuant tensors: {len(loaded.quantized_tensors)}")
    print(f"tied aliases: {loaded.aliases}")
    print(f"serialized model container: {len(blob)} bytes")
    print("round-trip: exact dense tensors, exact alias sharing, exact quantized reconstruction")
    print("scope: toy model only; quantized records dequantize on load and do not enable low-bit inference")


if __name__ == "__main__":
    main()
