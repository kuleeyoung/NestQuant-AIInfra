import tempfile
from pathlib import Path
import unittest

import torch

from nestquant.core.checkpoint_inspection import inspect_nqmc
from nestquant.core.model_checkpoint import PackedNestQuantModel
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor


class TestCheckpointInspection(unittest.TestCase):
    def test_manifest_accounting_and_record_semantics(self):
        dense = torch.arange(4, dtype=torch.float32)
        source = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        quantized = PackedNestQuantTensor.from_quantized_rows(
            "weight", quantize_rows(source, q=14, betas=(0.5, 0.75, 1.0, 1.25))
        )
        checkpoint = PackedNestQuantModel.from_state_dict(
            {"weight": source, "alias": source, "bias": dense},
            {"weight": quantized}, model_metadata={"model_id": "inspect-test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.nqmc"
            checkpoint.write_to_file(path)
            report = inspect_nqmc(
                path, inspect_records=True, compute_file_sha256=True, chunk_bytes=7
            )

        self.assertEqual(report["format"], "NQMC")
        self.assertEqual(report["version"], 1)
        self.assertEqual(report["tensor_count_including_aliases"], 3)
        self.assertEqual(report["unique_tensor_record_count"], 2)
        self.assertEqual(report["quantized_tensor_count"], 1)
        self.assertEqual(report["quantized_parameter_count"], 16)
        self.assertEqual(report["dense_tensor_count"], 1)
        self.assertEqual(report["dense_unique_value_count"], 4)
        self.assertEqual(report["tied_alias_count"], 1)
        self.assertEqual(report["model_metadata"]["model_id"], "inspect-test")
        self.assertEqual(len(report["file_sha256"]), 64)
        records = report["record_inspection"]
        self.assertEqual(records["quantized_records_parsed"], 1)
        self.assertEqual(records["nqrt_version_counts"], {"4": 1})
        self.assertEqual(records["q_counts"], {"14": 1})
        self.assertEqual(records["beta_count_counts"], {"4": 1})
        self.assertEqual(records["dither_storage_counts"], {"none": 1})

    def test_manifest_only_does_not_parse_records(self):
        checkpoint = PackedNestQuantModel.from_state_dict(
            {"value": torch.ones(3)}, {}, model_metadata={}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.nqmc"
            checkpoint.write_to_file(path)
            report = inspect_nqmc(path, verify_checksum=False)
        self.assertFalse(report["payload_checksum_verified"])
        self.assertIsNone(report["record_inspection"])
        self.assertNotIn("file_sha256", report)


if __name__ == "__main__":
    unittest.main()
