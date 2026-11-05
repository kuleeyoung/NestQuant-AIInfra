from __future__ import annotations

import pytest

from scripts.diagnose_m1_splitk_layers import layer_index


def test_layer_index_extracts_llama_layer() -> None:
    assert layer_index("model.layers.15.mlp.down_proj") == 15


def test_layer_index_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        layer_index("model.mlp.down_proj")
