"""Tests for pure ragged multi-page NQKV append planning."""

import unittest

from nestquant.runtime.nqkv_segment_plan import plan_ragged_multi_page_append


class TestNQKVSegmentPlan(unittest.TestCase):
    def _plan(self, hot, counts, pages=None, *, maximum=8):
        if pages is None:
            pages = [0] * len(hot)
        return plan_ragged_multi_page_append(
            hot,
            counts,
            pages,
            page_size=4,
            max_pages_per_sequence=maximum,
        )

    def test_zero_and_subpage_rows(self):
        plan = self._plan([0, 1, 3], [0, 2, 1])
        self.assertEqual(plan.required_pages_per_row, (0, 0, 1))
        self.assertEqual(plan.final_hot_lengths, (0, 3, 0))
        self.assertEqual(plan.max_rounds, 1)
        self.assertEqual(plan.rows[0].segments, ())
        self.assertFalse(plan.rows[1].segments[0].flush_after)
        self.assertTrue(plan.rows[2].segments[0].flush_after)

    def test_multiple_full_pages_and_tail(self):
        plan = self._plan([2], [11], [1])
        row = plan.rows[0]
        self.assertEqual(row.required_pages, 3)
        self.assertEqual(row.final_hot_length, 1)
        self.assertEqual(
            [
                (item.source_start, item.token_count, item.destination_start,
                 item.flush_after, item.page_ordinal)
                for item in row.segments
            ],
            [
                (0, 2, 2, True, 0),
                (2, 4, 0, True, 1),
                (6, 4, 0, True, 2),
                (10, 1, 0, False, 3),
            ],
        )
        self.assertEqual(plan.total_required_pages, 3)

    def test_ragged_rows_expose_rounds(self):
        plan = self._plan([3, 0, 2], [6, 2, 9])
        self.assertEqual(plan.required_pages_per_row, (2, 0, 2))
        self.assertEqual(plan.final_hot_lengths, (1, 2, 3))
        self.assertEqual(plan.max_rounds, 3)
        round_one = plan.round(1)
        self.assertEqual(round_one[0].source_start, 1)
        self.assertIsNone(round_one[1])
        self.assertEqual(round_one[2].source_start, 2)
        self.assertTrue(round_one[2].flush_after)

    def test_exhaustive_conservation_and_boundaries(self):
        for hot in range(4):
            for count in range(33):
                plan = self._plan([hot], [count], maximum=16)
                row = plan.rows[0]
                self.assertEqual(sum(item.token_count for item in row.segments), count)
                self.assertEqual(
                    row.required_pages, (hot + count) // 4
                )
                self.assertEqual(row.final_hot_length, (hot + count) % 4)
                source = 0
                for item in row.segments:
                    self.assertEqual(item.source_start, source)
                    self.assertGreater(item.token_count, 0)
                    self.assertLessEqual(
                        item.destination_start + item.token_count, 4
                    )
                    self.assertEqual(
                        item.flush_after,
                        item.destination_start + item.token_count == 4,
                    )
                    source += item.token_count

    def test_rejects_sequence_page_overflow_before_execution(self):
        with self.assertRaisesRegex(RuntimeError, "max_pages"):
            self._plan([3], [6], [1], maximum=2)

    def test_rejects_invalid_shapes_and_values(self):
        with self.assertRaisesRegex(ValueError, "align"):
            plan_ragged_multi_page_append(
                [0], [1, 2], [0], page_size=4, max_pages_per_sequence=2
            )
        with self.assertRaisesRegex(ValueError, "hot length"):
            self._plan([4], [1])
        with self.assertRaisesRegex(ValueError, "negative"):
            self._plan([0], [-1])
        with self.assertRaises(TypeError):
            self._plan([False], [1])


if __name__ == "__main__":
    unittest.main()
