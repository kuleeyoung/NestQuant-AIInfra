"""Tests for strict downstream comparison and report rendering."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from nestquant.evaluation.downstream_summary import render_markdown, summarize_downstream_results


def result(method, piqa=(1, 0), hellaswag=(0, 1)):
    def samples(values):
        return [
            {"doc_id": index, "doc_hash": f"hash-{index}", "acc": value, "acc_norm": value}
            for index, value in enumerate(values)
        ]

    return {
        "status": "completed",
        "method": method,
        "tasks": ["piqa", "hellaswag"],
        "limit_per_task": None,
        "zero_shot": True,
        "model": {"id": "model", "revision": "revision"},
        "environment": {"lm_eval": "0.4.13"},
        "task_provenance": {"piqa": {"dataset_fingerprint": "p"}, "hellaswag": {"dataset_fingerprint": "h"}},
        "elapsed_seconds": 120.0,
        "evidence_boundary": "quality only",
        "lm_eval_results": {
            "results": {
                "piqa": {"sample_len": 2, "acc,none": sum(piqa) / 2, "acc_norm,none": sum(piqa) / 2},
                "hellaswag": {"sample_len": 2, "acc,none": sum(hellaswag) / 2, "acc_norm,none": sum(hellaswag) / 2},
            },
            "samples": {"piqa": samples(piqa), "hellaswag": samples(hellaswag)},
        },
    }


class TestDownstreamSummary(unittest.TestCase):
    def write_results(self, root, payloads):
        paths = []
        for payload in payloads:
            path = root / f"{payload['method']}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
        return paths

    def test_reports_metric_deltas_and_correctness_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_results(root, [
                result("bf16"), result("w4", piqa=(0, 0)), result("w4a4kv4", hellaswag=(1, 1)),
            ])
            summary = summarize_downstream_results(paths)
        self.assertEqual(
            summary["methods"]["w4"]["tasks"]["piqa"]["delta_acc,none_pp_vs_bf16"],
            -50.0,
        )
        transition = summary["correctness_transitions_vs_bf16"]["w4"]["piqa"]["acc,none"]
        self.assertEqual(transition["bf16_correct_quantized_wrong"], 1)
        self.assertEqual(transition["net_correct_change"], -1)
        self.assertIn("-50.00 pp", render_markdown(summary))

    def test_rejects_protocol_or_sample_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mismatched = result("w4")
            mismatched["task_provenance"] = {"changed": True}
            paths = self.write_results(root, [result("bf16"), mismatched, result("w4a4kv4")])
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                summarize_downstream_results(paths)

            changed_sample = result("w4")
            changed_sample["lm_eval_results"]["samples"]["piqa"][0]["doc_hash"] = "changed"
            paths = self.write_results(root, [result("bf16"), changed_sample, result("w4a4kv4")])
            with self.assertRaisesRegex(ValueError, "sample identity mismatch"):
                summarize_downstream_results(paths)


if __name__ == "__main__":
    unittest.main()
