"""Unit tests for PositionDeltaCalculator."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.ml.pipeline.p18_institutional_flow_tracker.processors.position_delta_calculator import (
    PositionDeltaCalculator,
)


@pytest.fixture()
def calc(tmp_path: Path) -> PositionDeltaCalculator:
    return PositionDeltaCalculator(results_dir=tmp_path)


def _make_holdings(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cik", "institution_name", "ticker", "shares", "value_usd", "pct_of_portfolio"])


def test_full_exit_detected(calc: PositionDeltaCalculator) -> None:
    prev = _make_holdings([{"cik": "100", "institution_name": "Fund A", "ticker": "AAPL", "shares": 10000, "value_usd": 1_500_000, "pct_of_portfolio": 0.02}])
    curr = _make_holdings([])  # AAPL fully exited

    result = calc.calculate(curr, prev)

    assert not result.empty
    row = result[result["ticker"] == "AAPL"].iloc[0]
    assert row["exit_type"] == "full_exit"
    assert row["shares_curr"] == 0


def test_partial_exit_detected(calc: PositionDeltaCalculator) -> None:
    prev = _make_holdings([{"cik": "100", "institution_name": "Fund A", "ticker": "MSFT", "shares": 10000, "value_usd": 3_000_000, "pct_of_portfolio": 0.03}])
    curr = _make_holdings([{"cik": "100", "institution_name": "Fund A", "ticker": "MSFT", "shares": 5000, "value_usd": 1_500_000, "pct_of_portfolio": 0.015}])

    result = calc.calculate(curr, prev)

    row = result[result["ticker"] == "MSFT"].iloc[0]
    assert row["exit_type"] == "partial_exit"
    assert abs(row["delta_pct"] - (-0.5)) < 0.001


def test_new_position_detected(calc: PositionDeltaCalculator) -> None:
    prev = _make_holdings([])
    curr = _make_holdings([{"cik": "100", "institution_name": "Fund A", "ticker": "NVDA", "shares": 5000, "value_usd": 2_000_000, "pct_of_portfolio": 0.02}])

    result = calc.calculate(curr, prev)

    row = result[result["ticker"] == "NVDA"].iloc[0]
    assert row["exit_type"] == "new_position"


def test_empty_inputs_return_empty(calc: PositionDeltaCalculator) -> None:
    result = calc.calculate(pd.DataFrame(), pd.DataFrame())
    assert result.empty


def test_unchanged_position(calc: PositionDeltaCalculator) -> None:
    holdings = _make_holdings([{"cik": "100", "institution_name": "Fund A", "ticker": "GOOG", "shares": 1000, "value_usd": 1_000_000, "pct_of_portfolio": 0.01}])
    result = calc.calculate(holdings, holdings.copy())

    row = result[result["ticker"] == "GOOG"].iloc[0]
    assert row["exit_type"] == "unchanged"
    assert row["delta_pct"] == 0.0
