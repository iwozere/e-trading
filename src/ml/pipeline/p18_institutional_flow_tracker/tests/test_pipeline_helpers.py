"""Unit tests for module-level helpers in pipeline.py."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import _build_top_tickers


def _scored_df(n: int) -> pd.DataFrame:
    """Build a CompositeScorer-shaped frame, already sorted by total_score desc."""
    rows = []
    for i in range(n):
        detail = {"consensus_exit_3plus": True, "volume_spike_confirmed": i % 2 == 0}
        rows.append(
            {
                "ticker": f"TICK{i}",
                "total_score": 100 - i,
                "signals_active": sum(1 for v in detail.values() if v),
                "signal_detail": json.dumps(detail),
            }
        )
    return pd.DataFrame(rows)


def test_build_top_tickers_empty_returns_empty_list() -> None:
    assert _build_top_tickers(pd.DataFrame(), 10) == []


def test_build_top_tickers_caps_at_top_n() -> None:
    top = _build_top_tickers(_scored_df(50), 10)
    assert len(top) == 10
    # Preserves the existing descending order from the scorer.
    assert [t["ticker"] for t in top] == [f"TICK{i}" for i in range(10)]


def test_build_top_tickers_fewer_than_n() -> None:
    top = _build_top_tickers(_scored_df(3), 10)
    assert len(top) == 3


def test_build_top_tickers_structure_and_signals() -> None:
    top = _build_top_tickers(_scored_df(1), 10)
    entry = top[0]
    assert entry["ticker"] == "TICK0"
    assert entry["score"] == 100
    assert entry["signals_active"] == 2
    # Only active (truthy) signals are listed, sorted alphabetically.
    assert entry["signals"] == ["consensus_exit_3plus", "volume_spike_confirmed"]


def test_build_top_tickers_handles_malformed_signal_detail() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "BAD",
                "total_score": 70,
                "signals_active": 1,
                "signal_detail": None,
            }
        ]
    )
    top = _build_top_tickers(df, 10)
    assert top[0]["ticker"] == "BAD"
    assert top[0]["signals"] == []
