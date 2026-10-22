"""Tests for the guarded reproduction orchestrator."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from nestquant.experiments.reproduction import (
    DEFAULT_CONFIG,
    load_reproduction_config,
    main,
    plan_stage,
    validate_reviewed_result,
)


class TestReproduction(unittest.TestCase):
    def test_versioned_config_has_safe_and_full_stages(self):
        config = load_reproduction_config(DEFAULT_CONFIG)
        self.assertGreaterEqual(len(config["stages"]["smoke"]), 2)
        self.assertGreaterEqual(len(config["stages"]["full"]), 6)

    def test_plan_renders_python_and_reports_existing_outputs(self):
        config = load_reproduction_config(DEFAULT_CONFIG)
        plan = plan_stage(config, "full", DEFAULT_CONFIG.parents[1])
        self.assertTrue(plan[0]["command"][0])
        self.assertIn("collect_llama_hessian_groups.py", plan[0]["command"][1])
        self.assertIsInstance(plan[0]["outputs_complete"], bool)

    def test_full_execution_requires_exact_confirmation(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--stage", "full", "--config", str(DEFAULT_CONFIG)])

    def test_full_dry_run_does_not_require_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            with redirect_stdout(io.StringIO()):
                code = main([
                    "--stage", "full", "--dry-run", "--config", str(DEFAULT_CONFIG),
                    "--report", str(report),
                ])
            self.assertEqual(code, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(all(step["status"] in ("planned", "skipped_existing")
                                for step in payload["full"]))

    def test_all_dry_run_validates_reviewed_result_instead_of_missing_future_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            with redirect_stdout(io.StringIO()):
                code = main([
                    "--stage", "all", "--dry-run", "--config", str(DEFAULT_CONFIG),
                    "--report", str(report),
                ])
            self.assertEqual(code, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["validation"]["status"], "passed")

    def test_reviewed_result_is_valid_without_ignored_raw_evidence(self):
        config = load_reproduction_config(DEFAULT_CONFIG)
        result = validate_reviewed_result(
            config, DEFAULT_CONFIG.parents[1], require_raw=False
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["raw_evidence"]), 4)


if __name__ == "__main__":
    unittest.main()
