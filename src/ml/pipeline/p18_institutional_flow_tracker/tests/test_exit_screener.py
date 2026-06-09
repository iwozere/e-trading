"""Unit tests for ExitScreener."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.ml.pipeline.p18_institutional_flow_tracker.processors.exit_screener import ExitScreener


@pytest.fixture()
def screener() -> ExitScreener:
    return ExitScreener(
        exit_threshold_pct=0.30,
        min_position_pct_of_portfolio=0.005,
        min_position_value_usd=25_000_000,
    )


def _make_delta(rows: list) -> pd.DataFrame:
    cols = ["cik", "ticker", "exit_type", "delta_pct", "pct_of_portfolio_prev", "value_usd_prev"]
    return pd.DataFrame(rows, columns=cols)


def test_significant_full_exit_passes(screener: ExitScreener) -> None:
    df = _make_delta([{
        "cik": "1", "ticker": "AAPL", "exit_type": "full_exit",
        "delta_pct": -1.0, "pct_of_portfolio_prev": 0.02, "value_usd_prev": 50_000_000,
    }])
    result = screener.screen(df)
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "AAPL"


def test_tiny_position_filtered_out(screener: ExitScreener) -> None:
    df = _make_delta([{
        "cik": "1", "ticker": "XYZ", "exit_type": "full_exit",
        "delta_pct": -1.0, "pct_of_portfolio_prev": 0.001, "value_usd_prev": 1_000_000,
    }])
    result = screener.screen(df)
    assert result.empty


def test_partial_exit_below_threshold_filtered(screener: ExitScreener) -> None:
    df = _make_delta([{
        "cik": "1", "ticker": "TSLA", "exit_type": "partial_exit",
        "delta_pct": -0.10, "pct_of_portfolio_prev": 0.05, "value_usd_prev": 100_000_000,
    }])
    result = screener.screen(df)
    assert result.empty


def test_partial_exit_above_threshold_passes(screener: ExitScreener) -> None:
    df = _make_delta([{
        "cik": "1", "ticker": "TSLA", "exit_type": "partial_exit",
        "delta_pct": -0.55, "pct_of_portfolio_prev": 0.05, "value_usd_prev": 100_000_000,
    }])
    result = screener.screen(df)
    assert len(result) == 1


def test_empty_input_returns_empty(screener: ExitScreener) -> None:
    result = screener.screen(pd.DataFrame())
    assert result.empty
