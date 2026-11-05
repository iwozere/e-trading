"""Tests for the P24 portfolio layer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml.pipeline.p24_falling_knife.config import PortfolioParams
from src.ml.pipeline.p24_falling_knife.portfolio import simulate_portfolio


def _trades(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["security_id", "entry_date", "exit_date", "net_return", "score"])


def test_slot_limit_ranks_by_score_then_id() -> None:
    trades = _trades([("B", "2022-01-03", "2022-02-01", 0.10, 1.0),
                      ("A", "2022-01-03", "2022-02-01", 0.20, 1.0),
                      ("C", "2022-01-03", "2022-02-01", 0.30, 2.0)])
    result = simulate_portfolio(trades, PortfolioParams(max_positions=2, init_cash=1000.0))
    assert list(result.taken["security_id"]) == ["C", "A"]
    assert result.skipped == 1
    assert result.max_concurrent == 2
    assert result.equity.iloc[-1] == pytest.approx(1000.0 + 500.0 * 0.30 + 500.0 * 0.20)


def test_exit_frees_slot_same_day_and_compounds() -> None:
    trades = _trades([("A", "2022-01-03", "2022-01-10", 0.50, 1.0),
                      ("B", "2022-01-10", "2022-01-20", 0.10, 1.0)])
    result = simulate_portfolio(trades, PortfolioParams(max_positions=1, init_cash=1000.0))
    assert result.skipped == 0
    assert list(result.taken["allocation"]) == pytest.approx([1000.0, 1500.0])
    assert result.summary()["total_return"] == pytest.approx(1.5 * 1.1 - 1.0)


def test_liquidity_cap_and_same_day_round_trip() -> None:
    trades = _trades([("A", "2022-01-03", "2022-01-03", -1.0, 1.0)])
    trades["dollar_volume_20"] = [2000.0]
    result = simulate_portfolio(trades, PortfolioParams(max_positions=1, liquidity_cap_frac=0.05, init_cash=1000.0))
    assert result.taken["allocation"].iloc[0] == pytest.approx(100.0)
    assert result.equity.iloc[-1] == pytest.approx(900.0)
    assert result.summary()["max_drawdown"] == pytest.approx(0.0)


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError):
        simulate_portfolio(pd.DataFrame({"security_id": []}), PortfolioParams())
