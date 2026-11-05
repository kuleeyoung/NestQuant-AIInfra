from __future__ import annotations

import argparse

from scripts.serve_w4a4kv4 import add_m1_down_splitk_arguments


def test_service_enables_quality_gated_m1_splitk_by_default() -> None:
    parser = argparse.ArgumentParser()
    add_m1_down_splitk_arguments(parser)

    assert parser.parse_args([]).enable_m1_down_splitk is True


def test_service_exposes_explicit_unsplit_rollback() -> None:
    parser = argparse.ArgumentParser()
    add_m1_down_splitk_arguments(parser)

    assert parser.parse_args(["--disable-m1-down-splitk"]).enable_m1_down_splitk is False
