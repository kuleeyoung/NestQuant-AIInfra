"""Tests for the unified CPU/CUDA runtime verification orchestrator."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from nestquant.cli.verify_runtime import (
    build_stages,
    run_verification,
    select_stages,
)


class TestRuntimeVerification(unittest.TestCase):
    def test_graph_stage_requires_checkpoint(self):
        stages = build_stages(
            python="python", checkpoint=None, graph_output=Path("graph.json"))

        with self.assertRaisesRegex(ValueError, "--checkpoint"):
            select_stages(stages, ["graph"])

    def test_dry_run_records_commands_without_starting_processes(self):
        stages = build_stages(
            python="python", checkpoint=Path("model.nqmc"),
            graph_output=Path("graph.json"))
        chosen = select_stages(stages, ["cpu", "graph"])
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            with mock.patch("nestquant.cli.verify_runtime.execute_stage") as run:
                report = run_verification(
                    chosen, dry_run=True, report_path=report_path)

            run.assert_not_called()
            self.assertEqual(report["status"], "planned")
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual([item["name"] for item in saved["stages"]],
                             ["cpu", "graph"])

    def test_failure_stops_later_stages_and_writes_report(self):
        stages = build_stages(
            python="python", checkpoint=None, graph_output=Path("graph.json"))[:2]
        outcomes = [
            subprocess.CompletedProcess(stages[0].command, 0),
            subprocess.CompletedProcess(stages[1].command, 7),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            with mock.patch("nestquant.cli.verify_runtime.execute_stage",
                            side_effect=outcomes) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_verification(stages, dry_run=False, report_path=report_path)

            self.assertEqual(run.call_count, 2)
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["stages"][0]["status"], "passed")
            self.assertEqual(saved["stages"][1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
