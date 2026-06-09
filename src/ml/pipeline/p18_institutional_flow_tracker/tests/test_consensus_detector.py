"""Unit tests for ConsensusDetector."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.ml.pipeline.p18_institutional_flow_tracker.processors.consensus_detector import ConsensusDetector


@pytest.fixture()
def detector() -> ConsensusDetector:
    return ConsensusDetector(min_institutions=3)


def _make_exits(tickers_ciks: list) -> pd.DataFrame:
    rows = [
        {"cik": cik, "ticker": ticker, "value_usd_prev": 50_000_000, "delta_pct": -0.5}
        for ticker, cik in tickers_ciks
    ]
    return pd.DataFrame(rows)


def test_ticker_with_three_institutions_flagged(detector: ConsensusDetector) -> None:
    exits = _make_exits([("AAPL", "1"), ("AAPL", "2"), ("AAPL", "3")])
    result = detector.detect(exits)
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "AAPL"
    assert result.iloc[0]["institution_count"] == 3


def test_ticker_with_two_institutions_not_flagged(detector: ConsensusDetector) -> None:
    exits = _make_exits([("MSFT", "1"), ("MSFT", "2")])
    result = detector.detect(exits)
    assert result.empty


def test_sorted_by_institution_count(detector: ConsensusDetector) -> None:
    exits = _make_exits([
        ("GOOG", "1"), ("GOOG", "2"), ("GOOG", "3"), ("GOOG", "4"),
        ("META", "1"), ("META", "2"), ("META", "3"),
    ])
    result = detector.detect(exits)
    assert result.iloc[0]["ticker"] == "GOOG"
    assert result.iloc[0]["institution_count"] == 4
    assert result.iloc[1]["ticker"] == "META"


def test_empty_exits_returns_empty(detector: ConsensusDetector) -> None:
    result = detector.detect(pd.DataFrame())
    assert result.empty


def test_total_value_aggregated(detector: ConsensusDetector) -> None:
    exits = _make_exits([("AMZN", "1"), ("AMZN", "2"), ("AMZN", "3")])
    result = detector.detect(exits)
    assert result.iloc[0]["total_value_sold_usd"] == 3 * 50_000_000
