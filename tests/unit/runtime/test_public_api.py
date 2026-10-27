"""Regression tests for the compiled-runtime public API."""

import unittest

import nestquant.runtime as runtime


class TestRuntimePublicAPI(unittest.TestCase):
    def test_all_retains_existing_exports_and_graph_runner(self):
        expected = {
            "PreparedNQRTLinear",
            "linear_fused_mma_j2",
            "NQRTGraphDecodeRunner",
            "NQRTGraphDecodePool",
            "RaggedCacheController",
            "NQRTRaggedGraphDecodeRunner",
            "ContinuousBatchScheduler",
            "DecodeRequest",
            "RequestStatus",
            "ContinuousBatchInferenceEngine",
            "EngineStep",
            "GreedyHostSampler",
            "AdmissionPolicy",
            "ResidencyGroup",
            "ResidencyPlan",
            "atomic_projection_groups",
            "make_residency_groups",
            "residency_groups_from_payload",
            "select_residency_plan",
            "nondominated_indices",
        }

        self.assertTrue(expected.issubset(set(runtime.__all__)))
        for name in expected:
            self.assertIs(getattr(runtime, name), runtime.__dict__[name])


if __name__ == "__main__":
    unittest.main()
