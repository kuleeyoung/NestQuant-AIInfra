"""Protocol checks preventing beta-profile provenance from drifting."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class TestBetaProfiles(unittest.TestCase):
    def test_project_weight_coefficients_match_named_profile(self):
        project = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
        experiment = project["llm_ppl_experiment"]
        registry = json.loads(
            (ROOT / experiment["weight_beta_profile_registry"]).read_text(encoding="utf-8")
        )
        profile = registry["profiles"][experiment["weight_beta_profile_id"]]
        self.assertEqual(profile["scope"], "llm_weight")
        self.assertFalse(profile["paper_equivalent"])
        self.assertEqual(profile["values"], experiment["beta_coefficients_project_scale"])
        self.assertEqual(
            profile["selected_count"],
            project["initial_algorithm_config"]["selected_beta_count"],
        )

    def test_paper_grid_is_separate_from_upstream_cli_defaults(self):
        registry = json.loads(
            (ROOT / "configs/beta_profiles.v1.json").read_text(encoding="utf-8")
        )["profiles"]
        self.assertTrue(registry["paper_synthetic_grid"]["paper_equivalent"])
        self.assertEqual(
            registry["upstream_weight_cli_default_q14"]["paper_equivalent"],
            "not_confirmed",
        )
        self.assertNotEqual(
            registry["upstream_weight_cli_default_q14"]["values"],
            registry["project_extended_llama32_1b_w4_v1"]["values"],
        )


if __name__ == "__main__":
    unittest.main()
