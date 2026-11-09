"""Tests for `cli/market_cap_gap_cli.py` — the pure bucketing and summary."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.cli.market_cap_gap_cli import GapRecord, float_age_bucket, summarize


def test_float_age_bucket_edges():
    assert float_age_bucket(None) == "none"
    assert float_age_bucket(365) == "<=365d"
    assert float_age_bucket(366) == "<=548d"
    assert float_age_bucket(731) == ">730d"


def test_summarize_groups_by_reason_largest_first():
    records = [
        GapRecord(1, "no_price", 100),
        GapRecord(0, "no_price", None),
        GapRecord(0, "no_price", 600),
        GapRecord(1, "no_cash", 400),
    ]
    rows = summarize(records)
    assert [r["reason"] for r in rows] == ["no_price", "no_cash"]
    no_price = rows[0]
    assert no_price["rows"] == 3 and no_price["positives"] == 1 and no_price["positive_rate"] == 0.3333
    assert no_price["float_none"] == 1 and no_price["float_<=365d"] == 1 and no_price["float_<=730d"] == 1
