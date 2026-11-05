from collections import Counter
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "summarize_service_m_distribution.py"
SPEC = importlib.util.spec_from_file_location("summarize_service_m_distribution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_padded_rows = MODULE._padded_rows
_summary = MODULE._summary


def test_chunk_padding_plan_uses_m128_chunks_and_supported_tail() -> None:
    assert _padded_rows(40) == (64, 1)
    assert _padded_rows(120) == (128, 1)
    assert _padded_rows(240) == (256, 2)
    assert _padded_rows(384) == (384, 3)


def test_distribution_summary_separates_exact_coverage_and_padding() -> None:
    summary = _summary(Counter({8: 2, 40: 1, 240: 1}))
    assert summary["forward_count"] == 4
    assert summary["exact_supported_forward_count"] == 2
    assert summary["fallback_forward_count"] == 2
    assert summary["chunk_padding_kernel_launch_count"] == 5
    assert summary["useful_rows"] == 296
    assert summary["physical_rows_after_chunk_padding"] == 336
