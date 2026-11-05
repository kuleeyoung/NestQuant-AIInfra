"""Tests for fused ragged-batch attention over packed NQKV pages."""

import unittest

import torch

from nestquant.runtime.nqkv_attention_k1 import (
    BatchedNQKVAttentionGraphRunner,
    BatchedNQKVAttentionWorkspace,
    nqkv_decode_attention_batched_k1,
    nqkv_decode_attention_k1,
)
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestBatchedNQKVAttentionK1(unittest.TestCase):
    def _cache(self, *, max_sequences: int = 4, max_pages: int = 4):
        pool = NQKVPagePool(
            capacity=max_sequences * max_pages,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        return NQKVHotColdCache(
            PagedNQKVCache(
                pool,
                max_sequences=max_sequences,
                max_pages_per_sequence=max_pages,
            )
        )

    def setUp(self):
        self.cache = self._cache()

    def _tokens(self, seed: int, count: int):
        generator = torch.Generator().manual_seed(seed)
        shape = (count, 2, 64)
        key = torch.randn(shape, generator=generator).to(torch.bfloat16).cuda()
        value = torch.randn(shape, generator=generator).to(torch.bfloat16).cuda()
        return key, value

    def test_mixed_hot_and_cold_slots_match_single_sequence_k1(self):
        lengths = {"hot": 3, "mixed": 6, "cold": 8}
        slots = {"hot": 2, "mixed": 0, "cold": 3}
        for index, (sequence_id, length) in enumerate(lengths.items()):
            self.cache.register(sequence_id, slot=slots[sequence_id])
            key, value = self._tokens(10 + index, length)
            self.cache.append(sequence_id, key, value)
        query = torch.randn(3, 4, 64, device="cuda", dtype=torch.bfloat16)
        slot_ids = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
        active = torch.ones(3, device="cuda", dtype=torch.bool)
        actual = nqkv_decode_attention_batched_k1(
            query, self.cache, slot_ids=slot_ids, active_mask=active
        )
        for row, sequence_id in enumerate(("hot", "mixed", "cold")):
            expected = nqkv_decode_attention_k1(query[row], self.cache, sequence_id)
            self.assertTrue(torch.equal(actual[row], expected))

    def test_inactive_rows_are_zero_and_workspace_addresses_stay_fixed(self):
        self.cache.register("left", slot=1)
        self.cache.register("right", slot=3)
        for seed, sequence_id in ((20, "left"), (21, "right")):
            key, value = self._tokens(seed, 5)
            self.cache.append(sequence_id, key, value)
        query = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        slot_ids = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
        active = torch.tensor([True, False], device="cuda")
        workspace = BatchedNQKVAttentionWorkspace.create(
            self.cache, batch_size=2, query_heads=4
        )
        pointers = workspace.pointers()
        first = nqkv_decode_attention_batched_k1(
            query,
            self.cache,
            slot_ids=slot_ids,
            active_mask=active,
            workspace=workspace,
        ).clone()
        second = nqkv_decode_attention_batched_k1(
            query,
            self.cache,
            slot_ids=slot_ids,
            active_mask=active,
            workspace=workspace,
        )
        expected = nqkv_decode_attention_k1(query[0], self.cache, "left")
        self.assertTrue(torch.equal(first[0], expected))
        self.assertEqual(torch.count_nonzero(first[1]).item(), 0)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(workspace.pointers(), pointers)

    def test_rejects_duplicate_active_or_empty_slots(self):
        self.cache.register("empty", slot=0)
        query = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "unique"):
            nqkv_decode_attention_batched_k1(
                query,
                self.cache,
                slot_ids=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
                active_mask=torch.ones(2, device="cuda", dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "empty"):
            nqkv_decode_attention_batched_k1(
                query[:1],
                self.cache,
                slot_ids=torch.tensor([0], device="cuda", dtype=torch.int32),
                active_mask=torch.ones(1, device="cuda", dtype=torch.bool),
            )

    def test_batch_1_2_4_8_mixed_lengths_match_single_k1(self):
        lengths = (1, 3, 4, 5, 7, 9, 15, 31)
        slot_order = (7, 0, 5, 2, 6, 1, 4, 3)
        for batch_size in (1, 2, 4, 8):
            with self.subTest(batch_size=batch_size):
                cache = self._cache(max_sequences=8, max_pages=8)
                sequence_ids = []
                for row in range(batch_size):
                    sequence_id = f"request-{row}"
                    sequence_ids.append(sequence_id)
                    cache.register(sequence_id, slot=slot_order[row])
                    key, value = self._tokens(100 + row, lengths[row])
                    cache.append(sequence_id, key, value)
                query = torch.randn(
                    batch_size, 4, 64, device="cuda", dtype=torch.bfloat16
                )
                slot_ids = torch.tensor(
                    slot_order[:batch_size], device="cuda", dtype=torch.int32
                )
                active = torch.ones(batch_size, device="cuda", dtype=torch.bool)
                actual = nqkv_decode_attention_batched_k1(
                    query, cache, slot_ids=slot_ids, active_mask=active
                )
                for row, sequence_id in enumerate(sequence_ids):
                    expected = nqkv_decode_attention_k1(
                        query[row], cache, sequence_id
                    )
                    self.assertTrue(torch.equal(actual[row], expected))

    def test_cuda_graph_tracks_append_release_and_slot_reuse(self):
        cache = self._cache(max_sequences=4, max_pages=4)
        cache.register("left", slot=0)
        cache.register("right", slot=2)
        for seed, sequence_id in ((200, "left"), (201, "right")):
            key, value = self._tokens(seed, 5)
            cache.append(sequence_id, key, value)
        query = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
        active = torch.ones(2, device="cuda", dtype=torch.bool)
        runner = BatchedNQKVAttentionGraphRunner(
            cache, batch_size=2, query_heads=4
        )
        runner.capture(query, slot_ids=slots, active_mask=active)
        pointers = runner.input_pointers() + runner.workspace.pointers()

        new_query = torch.randn_like(query)
        expected_left = nqkv_decode_attention_k1(new_query[0], cache, "left")
        expected_right = nqkv_decode_attention_k1(new_query[1], cache, "right")
        first = runner.replay(
            new_query, slot_ids=slots, active_mask=active
        ).clone()
        self.assertTrue(torch.equal(first[0], expected_left))
        self.assertTrue(torch.equal(first[1], expected_right))

        cache.release("right")
        released = runner.replay(
            new_query, slot_ids=slots, active_mask=active
        ).clone()
        self.assertTrue(torch.equal(released[0], expected_left))
        self.assertEqual(torch.count_nonzero(released[1]).item(), 0)

        invalid_slots = torch.tensor([-1, 99], device="cuda", dtype=torch.int32)
        guarded = runner.replay(
            new_query, slot_ids=invalid_slots, active_mask=active
        ).clone()
        self.assertEqual(torch.count_nonzero(guarded).item(), 0)

        cache.register("replacement", slot=2)
        key, value = self._tokens(202, 3)
        cache.append("replacement", key, value)
        expected_replacement = nqkv_decode_attention_k1(
            new_query[1], cache, "replacement"
        )
        reused = runner.replay(
            new_query, slot_ids=slots, active_mask=active
        )
        self.assertTrue(torch.equal(reused[0], expected_left))
        self.assertTrue(torch.equal(reused[1], expected_replacement))
        self.assertEqual(runner.input_pointers() + runner.workspace.pointers(), pointers)


if __name__ == "__main__":
    unittest.main()
