"""Tests for the paper-book simulation and its exit policies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.ml.pipeline.signal_ledger.exits import POLICIES, ExitPolicy, policies_for, policy_by_name
from src.ml.pipeline.signal_ledger.portfolio import BookResult, simulate_book, simulate_uncapped, summarize_book, summarize_trades
from src.ml.pipeline.signal_ledger.tests.helpers import flat, panel, ramp, sessions

N = 15
IDX = sessions(N)


def _signal(ticker: str, day: int, rank: int = 1, hold: int = 5, side: str = "long", **extra: Any) -> dict:
    return {"id": day * 10 + rank, "signal_date": IDX[day].date(), "ticker": ticker, "rank": rank, "horizon_days": hold,
            "side": side, **extra}


def _book(signals, **kwargs) -> BookResult:
    opens = panel(IDX, {"VT": flat(N), "AAA": flat(N, 10.0), "BBB": flat(N, 10.0), "CCC": flat(N, 10.0)})
    closes = panel(IDX, {"VT": flat(N), "AAA": ramp(N, 10.0, 1.0), "BBB": flat(N, 10.0), "CCC": flat(N, 10.0)})
    return simulate_book("p05_top_picks", signals, opens, closes, start_capital=10_000.0, position_usd=1_000.0, **kwargs)


def _single(closes_aaa: List[float], policy: ExitPolicy, opens_aaa: Optional[List[float]] = None,
            highs_aaa: Optional[List[float]] = None, lows_aaa: Optional[List[float]] = None,
            source: str = "p10_final_passed", signals: Optional[List[Dict[str, Any]]] = None) -> BookResult:
    n = len(closes_aaa)
    idx = sessions(n)
    opens = panel(idx, {"VT": flat(n), "AAA": opens_aaa or [closes_aaa[0]] + closes_aaa[:-1]})
    closes = panel(idx, {"VT": flat(n), "AAA": closes_aaa})
    highs = panel(idx, {"AAA": highs_aaa}) if highs_aaa else None
    lows = panel(idx, {"AAA": lows_aaa}) if lows_aaa else None
    signals = signals or [{"id": 1, "signal_date": idx[0].date(), "ticker": "AAA", "rank": 1, "horizon_days": 5, "side": "long"}]
    return simulate_book(source, signals, opens, closes, highs, lows, policy=policy, start_capital=10_000.0, position_usd=1_000.0)


def test_position_is_bought_next_open_and_sold_after_hold() -> None:
    rows = _book([_signal("AAA", 0, hold=5)]).nav

    assert rows[0]["date"] == IDX[1].date()
    assert rows[0]["opened"] == 1
    closed_row = next(r for r in rows if r["closed"])
    assert closed_row["date"] == IDX[5].date()  # entry session counts as day 1
    # bought 1000 * (1 - 5bp) / 10 shares, sold at close 15 (ramp index 5) minus 5bp
    expected = 10_000.0 - 1_000.0 + 1_000.0 * (1 - 0.0005) / 10.0 * 15.0 * (1 - 0.0005)
    assert closed_row["nav"] == pytest.approx(expected)
    assert rows[-1]["positions"] == 0


def test_closed_trade_row_has_reason_return_and_excursions() -> None:
    trade = _book([_signal("AAA", 0, hold=5)]).trades[0]
    assert trade["exit_reason"] == "time"
    assert trade["entry_date"] == IDX[1].date() and trade["exit_date"] == IDX[5].date()
    assert trade["sessions_held"] == 5
    assert trade["net_return"] == pytest.approx((1 - 0.0005) ** 2 * 1.5 - 1)
    assert trade["mfe"] == pytest.approx(0.5)  # no high/low panels: the close range is used
    assert trade["signal_id"] == 1


def test_position_cap_keeps_best_ranks_and_skips_duplicates() -> None:
    signals = [_signal("BBB", 0, rank=2), _signal("AAA", 0, rank=1), _signal("CCC", 0, rank=3), _signal("AAA", 1, rank=1)]
    rows = _book(signals, max_positions=2).nav
    assert rows[0]["opened"] == 2
    assert rows[1]["opened"] == 0  # AAA already held, cap reached anyway


def test_uncapped_run_trades_every_signal() -> None:
    signals = [_signal("BBB", 0, rank=2), _signal("AAA", 0, rank=1), _signal("CCC", 0, rank=3)]
    opens = panel(IDX, {"VT": flat(N), "AAA": flat(N, 10.0), "BBB": flat(N, 10.0), "CCC": flat(N, 10.0)})
    closes = panel(IDX, {"VT": flat(N), "AAA": flat(N, 10.0), "BBB": flat(N, 10.0), "CCC": flat(N, 10.0)})
    capped = simulate_book("p05_top_picks", signals, opens, closes, max_positions=1)
    uncapped = simulate_uncapped("p05_top_picks", signals, opens, closes)
    assert [t["ticker"] for t in capped.trades] == ["AAA"]
    assert sorted(t["ticker"] for t in uncapped.trades) == ["AAA", "BBB", "CCC"]


def test_short_and_flagged_signals_are_skipped() -> None:
    assert _book([_signal("AAA", 0, side="short")]).nav == []
    assert _book([{**_signal("AAA", 0), "data_flag": "price_jump"}]).nav == []


def test_still_held_positions_are_reported_as_open_trades() -> None:
    trades = _book([_signal("AAA", 0, hold=50)]).trades
    assert [(t["exit_reason"], t["exit_date"]) for t in trades] == [("open", None)]
    assert trades[0]["last_price"] == pytest.approx(24.0)


def test_summary_against_benchmark() -> None:
    summary = summarize_book(_book([_signal("AAA", 0, hold=5)]).nav)
    assert summary["benchmark_return"] == pytest.approx(0.0)
    assert summary["excess_return"] > 0
    assert summary["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    assert 0 < summary["avg_invested_pct"] < 1


def test_bracket_stop_fills_at_the_stop_level_intraday() -> None:
    closes = [100.0] * 4 + [95.0] * 8
    lows = [99.0] * 4 + [80.0] + [94.0] * 7
    trade = _single(closes, policy_by_name("bracket"), lows_aaa=lows).trades[0]  # type: ignore[arg-type]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == pytest.approx(85.0)
    assert trade["mae"] == pytest.approx(-0.20)


def test_stop_gapped_through_fills_at_the_open() -> None:
    closes = [100.0] * 4 + [70.0] * 8
    opens = [100.0] * 4 + [70.0] * 8
    trade = _single(closes, policy_by_name("bracket"), opens_aaa=opens).trades[0]  # type: ignore[arg-type]
    assert (trade["exit_reason"], trade["exit_price"]) == ("stop", 70.0)


def test_bracket_target_is_taken_at_the_target_price() -> None:
    closes = [100.0, 100.0, 110.0, 120.0, 128.0, 128.0]
    highs = [100.0, 101.0, 112.0, 125.0, 140.0, 129.0]
    trade = _single(closes, policy_by_name("bracket"), highs_aaa=highs).trades[0]  # type: ignore[arg-type]
    assert (trade["exit_reason"], trade["exit_price"]) == ("target", pytest.approx(130.0))


def test_trailing_stop_follows_the_highest_close() -> None:
    closes = [100.0] * 16 + [110.0, 120.0, 130.0, 118.0, 100.0, 100.0]
    policy = ExitPolicy("trail_test", max_hold=100, trail_atr=3.0)
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]  # ATR = 2 before the rally, so the trail is 6 below the high-water close
    signals = [{"id": 1, "signal_date": sessions(len(closes))[14].date(), "ticker": "AAA", "rank": 1, "horizon_days": 5, "side": "long"}]
    trade = _single(closes, policy, highs_aaa=highs, lows_aaa=lows, signals=signals).trades[0]
    assert trade["exit_reason"] == "trail"
    assert trade["exit_price"] == pytest.approx(124.0)


def test_native_exit_uses_the_meta_exit_signal_date() -> None:
    idx = sessions(12)
    signals = [{"id": 1, "signal_date": idx[0].date(), "ticker": "AAA", "rank": 1, "horizon_days": 250, "side": "long",
                "meta": {"exit_signal_date": idx[3].isoformat()}}]
    trade = _single(ramp(12, 10.0, 1.0), policy_by_name("native"), source="p22_tier_entry", signals=signals).trades[0]  # type: ignore[arg-type]
    assert trade["exit_reason"] == "native"
    assert trade["exit_date"] == idx[4].date()  # next open after the exit signal


def test_decay_exit_waits_for_the_ticker_to_stop_being_signalled() -> None:
    idx = sessions(20)
    signals = [{"id": d, "signal_date": idx[d].date(), "ticker": "AAA", "rank": 1, "horizon_days": 20, "side": "long"} for d in range(0, 4)]
    result = _single(flat(20, 10.0), policy_by_name("native"), source="p05_top_picks", signals=signals)  # type: ignore[arg-type]
    trade = result.trades[0]
    assert len(result.trades) == 1
    assert trade["exit_reason"] == "decay"
    assert trade["exit_date"] == idx[9].date()  # last signal enters at session 4; 5 sessions without one -> open of 9


def test_price_jump_closes_at_the_last_good_close() -> None:
    closes = [1.0] * 5 + [30.0] * 5
    trade = _single(closes, policy_by_name("fixed_20")).trades[0]  # type: ignore[arg-type]
    assert (trade["exit_reason"], trade["exit_price"]) == ("data", 1.0)


def test_policies_for_adds_native_only_where_it_exists() -> None:
    assert "native" in {p.name for p in policies_for("p22_tier_entry")}
    assert "native" in {p.name for p in policies_for("p05_top_picks")}
    assert "native" not in {p.name for p in policies_for("p19_momentum_alerts")}
    assert len({p.name for p in POLICIES}) == len(POLICIES)


def test_summarize_trades_counts_reasons_and_open_positions() -> None:
    trades = [
        {"exit_reason": "time", "net_return": 0.10, "sessions_held": 5, "mae": -0.02, "mfe": 0.12},
        {"exit_reason": "stop", "net_return": -0.15, "sessions_held": 3, "mae": -0.15, "mfe": 0.01},
        {"exit_reason": "open", "net_return": 0.05, "sessions_held": 2, "mae": 0.0, "mfe": 0.05},
    ]
    stats = summarize_trades(trades)
    assert (stats["trades"], stats["closed"], stats["open"]) == (3, 2, 1)
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["exit_reasons"] == {"time": 1, "stop": 1}
    assert summarize_trades([])["mean_return"] is None
