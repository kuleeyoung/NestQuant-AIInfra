import json
import tempfile
import unittest
from pathlib import Path

from nestquant.evaluation.result_summary import collect_rows


class TestPplSummary(unittest.TestCase):
    def test_delta_uses_bf16_from_same_run_and_retains_run_status(self):
        payload = {
            "run_id": "run-1",
            "status": "smoke_completed",
            "config": {"model_id": "toy", "model_revision": "rev", "context_length": 8},
            "config_sha256": "abc",
            "results": {
                "bf16": {"quality_stage": "bf16_reference", "evaluation": {"perplexity": 10.0, "predicted_token_count": 14}},
                "rtn_w4": {"quality_stage": "fake_quant_quality_simulation", "evaluation": {"perplexity": 11.0, "predicted_token_count": 14}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "run.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            rows = collect_rows([manifest])

        self.assertEqual([row["method"] for row in rows], ["bf16", "rtn_w4"])
        self.assertEqual(rows[0]["status"], "smoke_completed")
        self.assertEqual(rows[0]["ppl_delta_vs_bf16"], "")
        self.assertAlmostEqual(rows[1]["ppl_delta_vs_bf16"], 10.0)
        self.assertEqual(rows[1]["predicted_tokens"], 14)

    def test_failed_manifest_without_results_is_not_dropped(self):
        payload = {"run_id": "failed-1", "status": "failed", "failed_method": "bf16",
                   "error": {"message": "gated"}}
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "failed.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            rows = collect_rows([manifest])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["method"], "bf16")
        self.assertEqual(rows[0]["error"], "gated")
        self.assertEqual(rows[0]["perplexity"], "")


if __name__ == "__main__":
    unittest.main()
