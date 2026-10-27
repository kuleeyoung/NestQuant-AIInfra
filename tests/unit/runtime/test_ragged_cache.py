from types import SimpleNamespace

import pytest
import torch

from nestquant.runtime.ragged_cache import RaggedCacheController


def make_cache(batch: int = 3, heads: int = 2, length: int = 6, dim: int = 4):
    layer = SimpleNamespace(
        keys=torch.zeros(batch, heads, length, dim),
        values=torch.zeros(batch, heads, length, dim),
        is_initialized=True,
    )
    return SimpleNamespace(layers=[layer]), layer


def test_per_row_scatter_writes_independent_positions() -> None:
    cache, layer = make_cache()
    controller = RaggedCacheController(cache, batch_size=3, max_cache_len=6)
    controller.install()
    positions = torch.tensor([[1], [4], [2]])
    controller.set_write_state(positions, torch.ones_like(positions, dtype=torch.bool))
    keys = torch.stack([torch.full((2, 1, 4), value) for value in (1.0, 2.0, 3.0)])
    values = keys + 10
    output_keys, output_values = layer.update(keys, values)
    assert torch.equal(output_keys[0, :, 1], keys[0, :, 0])
    assert torch.equal(output_keys[1, :, 4], keys[1, :, 0])
    assert torch.equal(output_keys[2, :, 2], keys[2, :, 0])
    assert torch.equal(output_values[1, :, 4], values[1, :, 0])
    assert torch.count_nonzero(output_keys).item() == keys.numel()


def test_inactive_slot_preserves_previous_cache_value() -> None:
    cache, layer = make_cache()
    layer.keys[1, :, 3] = 7
    layer.values[1, :, 3] = 8
    controller = RaggedCacheController(cache, batch_size=3, max_cache_len=6)
    controller.install()
    positions = torch.tensor([[0], [3], [5]])
    controller.set_write_state(positions, torch.tensor([[True], [False], [True]]))
    incoming = torch.ones(3, 2, 1, 4)
    layer.update(incoming, incoming)
    assert torch.all(layer.keys[1, :, 3] == 7)
    assert torch.all(layer.values[1, :, 3] == 8)


def test_attention_mask_uses_each_rows_position_and_length() -> None:
    cache, _ = make_cache()
    controller = RaggedCacheController(cache, batch_size=3, max_cache_len=6)
    positions = torch.tensor([[1], [4], [2]])
    lengths = torch.tensor([2, 5, 3])
    mask = controller.attention_mask(positions, valid_lengths=lengths, dtype=torch.float32)
    assert mask.shape == (3, 1, 1, 6)
    assert torch.equal(mask[:, 0, 0].eq(0).sum(dim=1), lengths)
    assert torch.all(mask[0, 0, 0, 2:] < -1e30)


def test_clear_slots_removes_stale_values_only_from_selected_rows() -> None:
    cache, layer = make_cache()
    layer.keys.fill_(2)
    layer.values.fill_(3)
    controller = RaggedCacheController(cache, batch_size=3, max_cache_len=6)
    controller.clear_slots(torch.tensor([False, True, False]))
    assert torch.count_nonzero(layer.keys[1]).item() == 0
    assert torch.count_nonzero(layer.values[1]).item() == 0
    assert torch.all(layer.keys[[0, 2]] == 2)


def test_controller_rejects_out_of_range_position() -> None:
    cache, _ = make_cache()
    controller = RaggedCacheController(cache, batch_size=3, max_cache_len=6)
    with pytest.raises(ValueError, match="outside"):
        controller.set_write_state(
            torch.tensor([[0], [6], [1]]), torch.ones(3, 1, dtype=torch.bool)
        )


def test_tiny_llama_ragged_decode_matches_independent_full_forward() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(20261026)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    model = transformers.LlamaForCausalLM(config).eval()
    cache = transformers.StaticCache(config, max_cache_len=8)
    prompt = torch.tensor([[1, 2], [11, 12]])
    with torch.inference_mode():
        model(
            input_ids=prompt,
            past_key_values=cache,
            position_ids=torch.tensor([[0, 1], [0, 1]]),
            use_cache=True,
        )

    controller = RaggedCacheController(cache, batch_size=2, max_cache_len=8)
    controller.install()

    # Request 0 advances to position 2 while request 1 is paused.
    positions = torch.tensor([[2], [2]])
    write_mask = torch.tensor([[True], [False]])
    controller.set_write_state(positions, write_mask)
    attention_mask = controller.attention_mask(
        positions, valid_lengths=torch.tensor([3, 2]), dtype=torch.float32
    )
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([[3], [0]]),
            past_key_values=cache,
            position_ids=positions,
            attention_mask=attention_mask,
            use_cache=True,
        )

    # Both run now, at independent positions 3 and 2.
    positions = torch.tensor([[3], [2]])
    controller.set_write_state(positions, torch.ones(2, 1, dtype=torch.bool))
    attention_mask = controller.attention_mask(
        positions, valid_lengths=torch.tensor([4, 3]), dtype=torch.float32
    )
    with torch.inference_mode():
        ragged_logits = model(
            input_ids=torch.tensor([[4], [13]]),
            past_key_values=cache,
            position_ids=positions,
            attention_mask=attention_mask,
            use_cache=True,
        ).logits[:, -1]
        reference_0 = model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits[0, -1]
        reference_1 = model(input_ids=torch.tensor([[11, 12, 13]])).logits[0, -1]

    assert torch.allclose(ragged_logits[0], reference_0, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ragged_logits[1], reference_1, atol=1e-6, rtol=1e-5)
