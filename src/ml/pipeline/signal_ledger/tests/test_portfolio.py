"""Tests for the paper-book simulation."""

from __future__ import annotations

import pytest

from src.ml.pipeline.signal_ledger.portfolio import simulate_book, summarize_book
from src.ml.pipeline.signal_ledger.tests.helpers import flat, panel, ramp, sessions

N = 15
IDX = sessions(N)


def _signal(ticker: str, day: int, rank: int = 1, hold: int = 5, side: str = "long") -> dict:
    return {"signal_date": IDX[day].date(), "ticker": ticker, "rank": rank, "horizon_days": hold, "side": side}


def _book(signals, **kwargs):
    opens = panel(IDX, {"VT": flat(N), "AAA": flat(N, 10.0), "BBB": flat(N, 10.0), "CCC": flat(N, 10.0)})
    closes = panel(IDX, {"VT": flat(N), "AAA": ramp(N, 10.0, 1.0), "BBB": flat(N, 10.0), "CCC": flat(N, 10.0)})
    return simulate_book("p05_top_picks", signals, opens, closes, start_capital=10_000.0, position_usd=1_000.0, **kwargs)


def test_position_is_bought_next_open_and_sold_after_hold() -> None:
    rows = _book([_signal("AAA", 0, hold=5)])

    assert rows[0]["date"] == IDX[1].date()
    assert rows[0]["opened"] == 1
    closed_row = next(r for r in rows if r["closed"])
    assert closed_row["date"] == IDX[5].date()  # entry session counts as day 1
    # bought 1000 * (1 - 5bp) / 10 shares, sold at close 15 (ramp index 5) minus 5bp
    expected = 10_000.0 - 1_000.0 + 1_000.0 * (1 - 0.0005) / 10.0 * 15.0 * (1 - 0.0005)
    assert closed_row["nav"] == pytest.approx(expected)
    assert rows[-1]["positions"] == 0


def test_position_cap_keeps_best_ranks_and_skips_duplicates() -> None:
    signals = [_signal("BBB", 0, rank=2), _signal("AAA", 0, rank=1), _signal("CCC", 0, rank=3), _signal("AAA", 1, rank=1)]
    rows = _book(signals, max_positions=2)
    assert rows[0]["opened"] == 2
    assert rows[1]["opened"] == 0  # AAA already held, cap reached anyway


def test_short_and_flagged_signals_are_skipped() -> None:
    assert _book([_signal("AAA", 0, side="short")]) == []
    assert _book([{**_signal("AAA", 0), "data_flag": "price_jump"}]) == []


def test_summary_against_benchmark() -> None:
    summary = summarize_book(_book([_signal("AAA", 0, hold=5)]))
    assert summary["benchmark_return"] == pytest.approx(0.0)
    assert summary["excess_return"] > 0
    assert summary["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    assert 0 < summary["avg_invested_pct"] < 1
