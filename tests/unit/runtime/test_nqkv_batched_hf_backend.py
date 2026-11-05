"""Integration tests for fixed-capacity batched NQKV Llama attention."""

from pathlib import Path
import copy
import tempfile
import types
import unittest

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
)

from nestquant.core.codec import generate_e8_dither_stateless
from nestquant.core.model_checkpoint import PackedNestQuantModel
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.quantizer import quantize_rows
from nestquant.core.rotation import hadamard
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.models.fused_linear import (
    FusedNQRTLinear,
    convert_linears_to_fused_nqrt_,
)
from nestquant.models.rotated_linear import PAPER_HADAMARD_FORMAT
from nestquant.runtime.nqkv_batched_hf_backend import (
    BatchedNQKVLlamaCache,
    install_batched_nqkv_llama_backend,
)
from nestquant.runtime.nqkv_batched_runner import BatchedNQKVLlamaRunner
from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.inference_engine import ContinuousBatchInferenceEngine
from nestquant.runtime.triton_unpack import triton_available


class _OneAttentionModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = LlamaAttention(config, layer_idx=0)


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestBatchedNQKVHFBackend(unittest.TestCase):
    def setUp(self):
        self.config = LlamaConfig(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            attention_dropout=0.0,
        )

    def _cache(self, capacity=4):
        return BatchedNQKVLlamaCache(
            capacity=capacity,
            num_layers=1,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=64,
            max_cache_len=32,
            page_size=4,
            q=14,
            key_betas=(0.25, 0.3271428571, 0.4621428571, 1.2185714286),
            value_betas=(0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286),
            dtype=torch.bfloat16,
            device="cuda",
        )

    def _position_embeddings(self, batch, tokens):
        cos = torch.ones(batch, tokens, 64, device="cuda", dtype=torch.bfloat16)
        return cos, torch.zeros_like(cos)

    def _packed_checkpoint(self, model, directory):
        layer_prefix = "model.layers.0" if "model.layers.0.self_attn.q_proj.weight" in model.state_dict() else "layers.0"
        projection_names = (
            f"{layer_prefix}.self_attn.q_proj",
            f"{layer_prefix}.self_attn.k_proj",
            f"{layer_prefix}.self_attn.v_proj",
            f"{layer_prefix}.self_attn.o_proj",
            f"{layer_prefix}.mlp.gate_proj",
            f"{layer_prefix}.mlp.up_proj",
            f"{layer_prefix}.mlp.down_proj",
        )
        state = dict(model.state_dict())
        records = {}
        for offset, name in enumerate(projection_names):
            tensor_name = f"{name}.weight"
            stored = hadamard(state[tensor_name])
            seed = 20261101 + offset
            dither = generate_e8_dither_stateless(
                stored.numel() // 8, seed
            ).to(device=stored.device, dtype=stored.dtype).reshape(
                stored.shape[0], stored.shape[1] // 8, 8
            )
            rows = quantize_rows(
                stored,
                q=14,
                betas=(0.5, 0.75, 1.0, 1.25),
                dither=dither,
            )
            state[tensor_name] = stored
            records[tensor_name] = PackedNestQuantTensor.from_quantized_rows(
                tensor_name, rows, dither_seed=seed
            )
        path = Path(directory) / "tiny_llama.nqmc"
        PackedNestQuantModel.from_state_dict(
            state,
            records,
            model_metadata={
                "hadamard_rotation": {
                    "format": PAPER_HADAMARD_FORMAT,
                    "converted_linear_names": list(projection_names),
                    "skipped_tied_linear_names": [],
                    "skipped_non_power_of_two_linear_names": [],
                    "weight_storage": "source_model_dtype",
                }
            },
        ).write_to_file(path)
        return NQMCFileReader(path), projection_names

    def test_attention_decode_writes_only_active_slots(self):
        model = _OneAttentionModel(self.config).eval().bfloat16().cuda()
        cache = self._cache()
        cache.register_batch((("left", 0), ("right", 2)))
        install_batched_nqkv_llama_backend(model, cache)
        active = torch.tensor([True, False, True, False], device="cuda")
        cache.set_batch_state(
            active_mask=active, write_mask=active[:, None]
        )
        hidden = torch.randn(4, 1, 256, device="cuda", dtype=torch.bfloat16)
        output = model.attention(
            hidden,
            position_embeddings=self._position_embeddings(4, 1),
            past_key_values=cache,
        )[0]
        self.assertEqual(output.shape, (4, 1, 256))
        self.assertEqual(cache.manager.layer_caches[0].total_tokens("left"), 1)
        self.assertEqual(cache.manager.layer_caches[0].total_tokens("right"), 1)
        self.assertEqual(torch.count_nonzero(output[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(output[3]).item(), 0)

    def test_runner_rejects_unknown_execution_policy(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        with self.assertRaisesRegex(ValueError, "execution_policy"):
            BatchedNQKVLlamaRunner(
                model,
                self._cache(),
                execution_policy="unknown",
            )

    def test_compact_decode_rows_map_to_noncontiguous_logical_slots(self):
        model = _OneAttentionModel(self.config).eval().bfloat16().cuda()
        cache = self._cache(capacity=4)
        cache.register_batch((("left", 0), ("right", 2)))
        install_batched_nqkv_llama_backend(model, cache)
        execution_slots = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
        execution_active = torch.ones(2, device="cuda", dtype=torch.bool)
        cache.set_decode_execution_state(
            slot_ids=execution_slots,
            active_mask=execution_active,
        )
        cache.enable_graph_decode()
        hidden = torch.randn(2, 1, 256, device="cuda", dtype=torch.bfloat16)
        output = model.attention(
            hidden,
            position_embeddings=self._position_embeddings(2, 1),
            past_key_values=cache,
        )[0]

        self.assertEqual(output.shape, (2, 1, 256))
        layer = cache.manager.layer_caches[0]
        self.assertEqual(layer.hot_lengths.tolist(), [1, 0, 1, 0])

    def test_compact_decode_rejects_duplicate_active_slot_mapping(self):
        cache = self._cache(capacity=4)
        with self.assertRaisesRegex(ValueError, "unique logical slots"):
            cache.set_decode_execution_state(
                slot_ids=torch.tensor([1, 1], device="cuda", dtype=torch.int32),
                active_mask=torch.ones(2, device="cuda", dtype=torch.bool),
            )

    def test_ragged_prefill_uses_mask_and_stores_only_valid_tokens(self):
        model = _OneAttentionModel(self.config).eval().bfloat16().cuda()
        cache = self._cache()
        cache.register_batch((("short", 0), ("long", 2)))
        install_batched_nqkv_llama_backend(model, cache)
        write_mask = torch.tensor(
            [[True, True, True, False, False],
             [False, False, False, False, False],
             [True, True, True, True, True],
             [False, False, False, False, False]],
            device="cuda",
        )
        active = write_mask.any(dim=1)
        cache.set_batch_state(active_mask=active, write_mask=write_mask)
        hidden = torch.randn(4, 5, 256, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(5, device="cuda")
        allowed = positions[None, :] <= positions[:, None]
        valid_keys = write_mask[:, None, None, :]
        allowed = allowed[None, None] & valid_keys
        attention_mask = torch.zeros(
            (4, 1, 5, 5), device="cuda", dtype=torch.bfloat16
        ).masked_fill(~allowed, torch.finfo(torch.bfloat16).min)
        output = model.attention(
            hidden,
            attention_mask=attention_mask,
            position_embeddings=self._position_embeddings(4, 5),
            past_key_values=cache,
        )[0]
        self.assertEqual(output.shape, (4, 5, 256))
        self.assertTrue(torch.isfinite(output).all())
        layer = cache.manager.layer_caches[0]
        self.assertEqual(layer.total_tokens("short"), 3)
        self.assertEqual(layer.total_tokens("long"), 5)

    def test_complete_llama_model_runs_fixed_capacity_decode(self):
        model = LlamaModel(self.config).eval().bfloat16().cuda()
        cache = self._cache(capacity=2)
        cache.register_batch((("left", 0), ("right", 1)))
        install_batched_nqkv_llama_backend(model, cache)
        active = torch.ones(2, device="cuda", dtype=torch.bool)
        cache.set_batch_state(active_mask=active, write_mask=active[:, None])
        with torch.no_grad():
            first = model(
                input_ids=torch.tensor([[1], [2]], device="cuda"),
                position_ids=torch.zeros((2, 1), device="cuda", dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state
            cache.set_batch_state(active_mask=active, write_mask=active[:, None])
            second = model(
                input_ids=torch.tensor([[3], [4]], device="cuda"),
                position_ids=torch.ones((2, 1), device="cuda", dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state
        self.assertEqual(first.shape, (2, 1, 256))
        self.assertEqual(second.shape, (2, 1, 256))
        layer = cache.manager.layer_caches[0]
        self.assertEqual(layer.total_tokens("left"), 2)
        self.assertEqual(layer.total_tokens("right"), 2)

    def test_full_model_prefill_fuses_ragged_page_fill_and_pack(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        cache = self._cache(capacity=2)
        install_batched_nqkv_llama_backend(model, cache)
        runner = BatchedNQKVLlamaRunner(model, cache)

        prompt = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda"
        )
        logits = runner.prefill(prompt)

        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(cache.ragged_subpage_layer_launch_count, 1)
        self.assertEqual(cache.ragged_full_page_layer_pack_count, 1)
        layer = cache.manager.layer_caches[0]
        self.assertEqual(layer.hot_lengths.tolist(), [0, 0])
        self.assertEqual(layer.cold_cache.page_counts.tolist(), [1, 1])
        for sequence_id in cache.manager.request_ids:
            self.assertEqual(layer.total_tokens(sequence_id), 4)

        next_token = torch.tensor([[9], [10]], device="cuda")
        next_positions = torch.full(
            (2, 1), 4, device="cuda", dtype=torch.long
        )
        next_mask = torch.ones((2, 1), device="cuda", dtype=torch.bool)
        next_logits = runner.prefill_slots(
            next_token,
            positions=next_positions,
            write_mask=next_mask,
            valid_lengths=torch.full(
                (2,), 5, device="cuda", dtype=torch.long
            ),
        )
        self.assertTrue(torch.isfinite(next_logits).all())
        self.assertEqual(layer.hot_lengths.tolist(), [1, 1])
        self.assertEqual(layer.cold_cache.page_counts.tolist(), [1, 1])
        for sequence_id in cache.manager.request_ids:
            self.assertEqual(layer.total_tokens(sequence_id), 5)

    def test_runner_segments_one_boundary_crossing_prefill(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        cache = self._cache(capacity=2)
        install_batched_nqkv_llama_backend(model, cache)
        runner = BatchedNQKVLlamaRunner(model, cache)

        first = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        runner.prefill(first)
        second = torch.tensor([[7, 8, 9], [10, 11, 12]], device="cuda")
        positions = torch.arange(3, 6, device="cuda").expand(2, -1)
        mask = torch.ones((2, 3), device="cuda", dtype=torch.bool)
        logits = runner.prefill_slots(
            second,
            positions=positions,
            write_mask=mask,
            valid_lengths=torch.full((2,), 6, device="cuda", dtype=torch.long),
        )

        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(cache.ragged_cross_page_layer_launch_count, 1)
        layer = cache.manager.layer_caches[0]
        self.assertEqual(layer.cold_cache.page_counts.tolist(), [1, 1])
        self.assertEqual(layer.hot_lengths.tolist(), [2, 2])
        for sequence_id in cache.manager.request_ids:
            self.assertEqual(layer.total_tokens(sequence_id), 6)
            self.assertEqual(len(layer.cold_cache.handles(sequence_id)), 1)

    def test_multi_page_planner_handles_mixed_ragged_rows(self):
        cache = self._cache(capacity=2)
        cache.register_batch((("left", 0), ("right", 1)))
        for layer in cache.manager.layer_caches:
            layer.synchronize_host_length(0, 3)
            layer.hot_lengths[0] = 3
            layer.synchronize_host_length(1, 1)
            layer.hot_lengths[1] = 1

        plan = cache.manager.ragged_multi_page_plan((10, 4))

        self.assertIsNotNone(plan)
        starts, counts, boundaries, final_lengths, flush_counts = plan
        self.assertEqual(starts, ((0, 0), (1, 3), (5, 4), (9, 4)))
        self.assertEqual(counts, ((1, 3), (4, 1), (4, 0), (1, 0)))
        self.assertEqual(boundaries, ((0, 1), (0,), (0,), ()))
        self.assertEqual(final_lengths, (1, 1))
        self.assertEqual(flush_counts, (3, 1))

    def test_multi_page_planner_leaves_short_appends_on_existing_paths(self):
        cache = self._cache(capacity=2)
        cache.register_batch((("left", 0), ("right", 1)))

        self.assertIsNone(cache.manager.ragged_multi_page_plan((4, 3)))
        self.assertIsNone(cache.manager.ragged_multi_page_plan((0, 0)))

    def test_multi_page_planner_rejects_layer_length_drift(self):
        cache = BatchedNQKVLlamaCache(
            capacity=2,
            num_layers=2,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=64,
            max_cache_len=32,
            page_size=4,
            q=14,
            key_betas=(0.25, 0.3271428571, 0.4621428571, 1.2185714286),
            value_betas=(0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286),
            dtype=torch.bfloat16,
            device="cuda",
        )
        cache.register_batch((("left", 0), ("right", 1)))
        cache.manager.layer_caches[0].synchronize_host_length(0, 1)
        cache.manager.layer_caches[0].hot_lengths[0] = 1

        self.assertIsNone(cache.manager.ragged_multi_page_plan((8, 0)))

    def test_runner_segments_mixed_crossing_and_subpage_rows(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        cache = self._cache(capacity=2)
        install_batched_nqkv_llama_backend(model, cache)
        runner = BatchedNQKVLlamaRunner(model, cache)
        runner.prefill(torch.tensor([[1], [2]], device="cuda"))

        middle = torch.tensor([[3, 4], [0, 0]], device="cuda")
        middle_mask = torch.tensor(
            [[True, True], [False, False]], device="cuda"
        )
        runner.prefill_slots(
            middle,
            positions=torch.tensor([[1, 2], [0, 0]], device="cuda"),
            write_mask=middle_mask,
            valid_lengths=torch.tensor([3, 1], device="cuda"),
        )

        final = torch.tensor([[5, 6, 7], [8, 9, 0]], device="cuda")
        final_mask = torch.tensor(
            [[True, True, True], [True, True, False]], device="cuda"
        )
        logits = runner.prefill_slots(
            final,
            positions=torch.tensor([[3, 4, 5], [1, 2, 0]], device="cuda"),
            write_mask=final_mask,
            valid_lengths=torch.tensor([6, 3], device="cuda"),
        )

        self.assertTrue(torch.isfinite(logits).all())
        layer = cache.manager.layer_caches[0]
        self.assertEqual(layer.cold_cache.page_counts.tolist(), [1, 0])
        self.assertEqual(layer.hot_lengths.tolist(), [2, 3])
        left, right = cache.manager.request_ids
        self.assertEqual(layer.total_tokens(left), 6)
        self.assertEqual(layer.total_tokens(right), 3)

    def test_cross_page_forward_failure_releases_affected_requests(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        cache = self._cache(capacity=2)
        install_batched_nqkv_llama_backend(model, cache)
        runner = BatchedNQKVLlamaRunner(model, cache)
        runner.prefill(torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda"))
        original_forward = model.forward

        def failing_forward(this, *args, **kwargs):
            original_forward(*args, **kwargs)
            raise RuntimeError("injected post-KV failure")

        model.forward = types.MethodType(failing_forward, model)
        mask = torch.ones((2, 3), device="cuda", dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "injected post-KV failure"):
            runner.prefill_slots(
                torch.tensor([[7, 8, 9], [10, 11, 12]], device="cuda"),
                positions=torch.arange(3, 6, device="cuda").expand(2, -1),
                write_mask=mask,
                valid_lengths=torch.full(
                    (2,), 6, device="cuda", dtype=torch.long
                ),
            )

        self.assertEqual(cache.manager.active_requests, 0)
        for layer in cache.manager.layer_caches:
            self.assertEqual(layer.cold_cache.pool.allocated_pages, 0)
            self.assertEqual(layer.cold_cache.page_counts.tolist(), [0, 0])

    def test_runner_packs_multiple_prefill_pages_then_continues(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        cache = self._cache(capacity=2)
        install_batched_nqkv_llama_backend(model, cache)
        runner = BatchedNQKVLlamaRunner(model, cache)
        cache.ragged_multi_page_enabled = True
        prompt = torch.arange(1, 23, device="cuda").reshape(2, 11)

        logits = runner.prefill(prompt)

        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(cache.ragged_multi_page_layer_round_count, 3)
        layer = cache.manager.layer_caches[0]
        self.assertEqual(layer.cold_cache.page_counts.tolist(), [2, 2])
        self.assertEqual(layer.hot_lengths.tolist(), [3, 3])
        for sequence_id in cache.manager.request_ids:
            self.assertEqual(layer.total_tokens(sequence_id), 11)
            self.assertEqual(len(layer.cold_cache.handles(sequence_id)), 2)

        tail = torch.tensor([[23, 24], [25, 26]], device="cuda")
        tail_mask = torch.ones((2, 2), device="cuda", dtype=torch.bool)
        continued = runner.prefill_slots(
            tail,
            positions=torch.arange(11, 13, device="cuda").expand(2, -1),
            write_mask=tail_mask,
            valid_lengths=torch.full((2,), 13, device="cuda", dtype=torch.long),
        )

        self.assertTrue(torch.isfinite(continued).all())
        self.assertEqual(layer.cold_cache.page_counts.tolist(), [3, 3])
        self.assertEqual(layer.hot_lengths.tolist(), [1, 1])
        for sequence_id in cache.manager.request_ids:
            self.assertEqual(layer.total_tokens(sequence_id), 13)
            self.assertEqual(len(layer.cold_cache.handles(sequence_id)), 3)

    def test_multi_page_forward_failure_releases_all_pages_and_slots(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        cache = self._cache(capacity=2)
        install_batched_nqkv_llama_backend(model, cache)
        runner = BatchedNQKVLlamaRunner(model, cache)
        cache.ragged_multi_page_enabled = True
        original_forward = model.forward

        def failing_forward(this, *args, **kwargs):
            original_forward(*args, **kwargs)
            raise RuntimeError("injected multi-page failure")

        model.forward = types.MethodType(failing_forward, model)
        with self.assertRaisesRegex(RuntimeError, "injected multi-page failure"):
            runner.prefill(
                torch.arange(1, 23, device="cuda").reshape(2, 11)
            )

        self.assertEqual(cache.manager.active_requests, 0)
        for layer in cache.manager.layer_caches:
            self.assertEqual(layer.cold_cache.pool.allocated_pages, 0)
            self.assertEqual(layer.cold_cache.page_counts.tolist(), [0, 0])
            self.assertTrue(bool((layer.cold_cache.page_indices == -1).all().item()))
            self.assertEqual(layer.hot_lengths.tolist(), [0, 0])

    def test_complete_llama_combines_packed_w4a4_with_batched_kv4(self):
        model = LlamaModel(self.config).eval().bfloat16().cuda()
        with tempfile.TemporaryDirectory() as directory:
            reader, names = self._packed_checkpoint(model, directory)
            report = convert_linears_to_fused_nqrt_(
                model,
                reader,
                names,
                output_dtype=torch.bfloat16,
                compute_backend="packed_w4a4_tiled",
                activation_q=14,
                activation_betas=tuple(
                    value / 14 for value in (3.47, 4.74, 6.90, 18.11)
                ),
            )
            cache = self._cache(capacity=2)
            cache.register_batch((("left", 0), ("right", 1)))
            install_batched_nqkv_llama_backend(model, cache)
            active = torch.ones(2, device="cuda", dtype=torch.bool)
            with torch.inference_mode():
                for position, tokens in enumerate(((1, 2), (3, 4))):
                    cache.set_batch_state(
                        active_mask=active, write_mask=active[:, None]
                    )
                    output = model(
                        input_ids=torch.tensor(tokens, device="cuda")[:, None],
                        position_ids=torch.full(
                            (2, 1), position, device="cuda", dtype=torch.long
                        ),
                        past_key_values=cache,
                        use_cache=True,
                    ).last_hidden_state

            fused = [
                module for module in model.modules()
                if isinstance(module, FusedNQRTLinear)
            ]
            self.assertEqual(report.replaced_linear_names, tuple(sorted(names)))
            self.assertEqual(len(fused), 7)
            self.assertTrue(all(module.a4_call_count > 0 for module in fused))
            self.assertTrue(torch.isfinite(output).all())
            layer = cache.manager.layer_caches[0]
            self.assertEqual(layer.total_tokens("left"), 2)
            self.assertEqual(layer.total_tokens("right"), 2)

    def test_continuous_batch_engine_reuses_packed_w4a4kv4_slots(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        with tempfile.TemporaryDirectory() as directory:
            reader, names = self._packed_checkpoint(model, directory)
            convert_linears_to_fused_nqrt_(
                model,
                reader,
                names,
                output_dtype=torch.bfloat16,
                compute_backend="packed_w4a4_tiled",
                activation_q=14,
                activation_betas=tuple(
                    value / 14 for value in (3.47, 4.74, 6.90, 18.11)
                ),
            )
            cache = self._cache(capacity=2)
            install_batched_nqkv_llama_backend(model, cache)
            runner = BatchedNQKVLlamaRunner(model, cache)
            engine = ContinuousBatchInferenceEngine(runner)
            first = DecodeRequest("first", (1, 2, 3), 1)
            second = DecodeRequest("second", (4,), 1)
            engine.submit(first)
            engine.submit(second)
            initial = engine.step()
            self.assertEqual(initial.admitted_request_ids, ("first", "second"))
            self.assertEqual(initial.finished_request_ids, ("first", "second"))
            self.assertEqual(cache.manager.active_requests, 0)

            third = DecodeRequest("third", (5, 6), 2)
            engine.submit(third)
            reused = engine.step()
            self.assertEqual(reused.admitted_request_ids, ("third",))
            self.assertEqual(cache.manager.active_requests, 1)
            self.assertEqual(cache.manager.slot(cache.manager.request_ids[0]), 0)
            self.assertTrue(runner.graph_captured)

    def test_full_model_graph_decode_matches_eager_with_dynamic_active_rows(self):
        torch.manual_seed(20261101)
        eager_model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        graph_model = copy.deepcopy(eager_model)
        eager_cache = self._cache(capacity=2)
        graph_cache = self._cache(capacity=2)
        install_batched_nqkv_llama_backend(eager_model, eager_cache)
        install_batched_nqkv_llama_backend(graph_model, graph_cache)
        eager = BatchedNQKVLlamaRunner(eager_model, eager_cache)
        graph = BatchedNQKVLlamaRunner(graph_model, graph_cache)

        dummy = torch.zeros((2, 1), device="cuda", dtype=torch.long)
        positions = torch.zeros_like(dummy)
        active = torch.ones_like(dummy, dtype=torch.bool)
        lengths = torch.ones(2, device="cuda", dtype=torch.long)
        eager.prefill(dummy)
        graph.prefill(dummy)
        graph.capture(
            dummy, positions=positions, active_mask=active,
            valid_lengths=lengths,
        )
        clear = torch.ones(2, device="cuda", dtype=torch.bool)
        eager.clear_slots(clear)
        graph.clear_slots(clear)

        prompts = torch.tensor(
            [[1, 2], [4, 0]], device="cuda", dtype=torch.long
        )
        prompt_positions = torch.arange(2, device="cuda").expand(2, -1)
        prompt_mask = torch.tensor(
            [[True, True], [True, False]], device="cuda"
        )
        prompt_lengths = torch.tensor([2, 1], device="cuda")
        eager_prefill = eager.prefill_slots(
            prompts, positions=prompt_positions, write_mask=prompt_mask,
            valid_lengths=prompt_lengths,
        )
        graph_prefill = graph.prefill_slots(
            prompts, positions=prompt_positions, write_mask=prompt_mask,
            valid_lengths=prompt_lengths,
        )
        torch.testing.assert_close(graph_prefill, eager_prefill, rtol=0, atol=0)

        token = torch.tensor([[6], [7]], device="cuda")
        decode_positions = torch.tensor([[2], [1]], device="cuda")
        decode_active = torch.ones((2, 1), device="cuda", dtype=torch.bool)
        decode_lengths = torch.tensor([3, 2], device="cuda")
        graph_logits = graph.decode(
            token, positions=decode_positions, active_mask=decode_active,
            valid_lengths=decode_lengths,
        )
        eager_logits = eager._forward(token, decode_positions, decode_active)
        for eager_layer, graph_layer in zip(
            eager_cache.manager.layer_caches,
            graph_cache.manager.layer_caches,
            strict=True,
        ):
            self.assertEqual(
                eager_layer.hot_lengths.tolist(), graph_layer.hot_lengths.tolist()
            )
            torch.testing.assert_close(
                graph_layer.hot_keys, eager_layer.hot_keys, rtol=0, atol=0
            )
            torch.testing.assert_close(
                graph_layer.hot_values, eager_layer.hot_values, rtol=0, atol=0
            )
        torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)

        token.copy_(torch.tensor([[8], [9]], device="cuda"))
        decode_positions.copy_(torch.tensor([[2], [2]], device="cuda"))
        decode_active.copy_(torch.tensor([[False], [True]], device="cuda"))
        decode_lengths.copy_(torch.tensor([3, 3], device="cuda"))
        graph_logits = graph.decode(
            token, positions=decode_positions, active_mask=decode_active,
            valid_lengths=decode_lengths,
        )
        eager_logits = eager._forward(token, decode_positions, decode_active)
        torch.testing.assert_close(
            graph_logits[decode_active[:, 0]],
            eager_logits[decode_active[:, 0]],
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            torch.count_nonzero(graph_logits[~decode_active[:, 0]]).item(), 0
        )
        self.assertEqual(graph.graph_replay_counts, {2: 1, 1: 1})
        for eager_layer, graph_layer in zip(
            eager_cache.manager.layer_caches,
            graph_cache.manager.layer_caches,
            strict=True,
        ):
            self.assertEqual(
                eager_layer.hot_lengths.tolist(), graph_layer.hot_lengths.tolist()
            )

        # Both rows now have three hot tokens. The next token fills a page and
        # must take the boundary packing Graph without an eager fallback.
        token.copy_(torch.tensor([[10], [11]], device="cuda"))
        decode_positions.copy_(torch.tensor([[3], [3]], device="cuda"))
        decode_active.fill_(True)
        decode_lengths.fill_(4)
        graph_logits = graph.decode(
            token, positions=decode_positions, active_mask=decode_active,
            valid_lengths=decode_lengths,
        )
        eager_logits = eager._forward(token, decode_positions, decode_active)
        torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)
        self.assertEqual(graph.graph_boundary_fallback_count, 0)
        self.assertEqual(graph.graph_boundary_replay_count, 1)
        for eager_layer, graph_layer in zip(
            eager_cache.manager.layer_caches,
            graph_cache.manager.layer_caches,
            strict=True,
        ):
            self.assertEqual(
                eager_layer.hot_lengths.tolist(), graph_layer.hot_lengths.tolist()
            )
            self.assertEqual(graph_layer.hot_lengths.tolist(), [0, 0])
            self.assertEqual(
                eager_layer.cold_cache.page_counts.tolist(),
                graph_layer.cold_cache.page_counts.tolist(),
            )

        # Exercise a second complete page. This catches stale host ownership,
        # physical-page reuse, and pending-reservation bugs that a single
        # boundary transition cannot expose.
        for position, values in enumerate(((12, 13), (14, 15), (16, 17), (18, 19)), start=4):
            token.copy_(torch.tensor(values, device="cuda")[:, None])
            decode_positions.fill_(position)
            decode_active.fill_(True)
            decode_lengths.fill_(position + 1)
            graph_logits = graph.decode(
                token,
                positions=decode_positions,
                active_mask=decode_active,
                valid_lengths=decode_lengths,
            )
            eager_logits = eager._forward(token, decode_positions, decode_active)
            torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)

        self.assertEqual(graph.graph_boundary_fallback_count, 0)
        self.assertEqual(graph.graph_boundary_replay_count, 2)
        for eager_layer, graph_layer in zip(
            eager_cache.manager.layer_caches,
            graph_cache.manager.layer_caches,
            strict=True,
        ):
            self.assertEqual(graph_layer.hot_lengths.tolist(), [0, 0])
            self.assertEqual(graph_layer.cold_cache.page_counts.tolist(), [2, 2])
            self.assertEqual(
                eager_layer.cold_cache.page_counts.tolist(),
                graph_layer.cold_cache.page_counts.tolist(),
            )
            for slot in range(2):
                eager_sequence = eager_cache.manager.sequence_id(slot)
                graph_sequence = graph_cache.manager.sequence_id(slot)
                eager_key, eager_value = eager_layer.materialize(eager_sequence)
                graph_key, graph_value = graph_layer.materialize(graph_sequence)
                torch.testing.assert_close(graph_key, eager_key, rtol=0, atol=0)
                torch.testing.assert_close(graph_value, eager_value, rtol=0, atol=0)

        # Releasing graph-committed pages must return every physical page to
        # the pool and leave no sequence ownership behind.
        eager.clear_slots(clear)
        graph.clear_slots(clear)
        self.assertEqual(eager_cache.manager.active_requests, 0)
        self.assertEqual(graph_cache.manager.active_requests, 0)
        for graph_layer in graph_cache.manager.layer_caches:
            self.assertEqual(graph_layer.cold_cache.pool.allocated_pages, 0)


if __name__ == "__main__":
    unittest.main()
