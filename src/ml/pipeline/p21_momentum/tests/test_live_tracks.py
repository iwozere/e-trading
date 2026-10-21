"""Tests for the live Track B-E replay, NAV rewrite and the NaN-safe regime inputs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.pipeline.p21_momentum.config import NAV_TOTAL_USD, PROXY_TICKER
from src.ml.pipeline.p21_momentum.results.run_io import append_nav_row, rewrite_track_navs
from src.ml.pipeline.p21_momentum.strategy.live_tracks import (
    TrackExecution,
    load_executions,
    replay_tracks,
    summarize_navs,
    wide_frames,
)
from src.ml.pipeline.p21_momentum.strategy.regime import PriorRegimeState, compute_regime

IDX = pd.bdate_range("2026-09-01", periods=10)


def _frames(spy_close=None, mtum_close=None, aaa_close=None):
    n = len(IDX)
    opens = pd.DataFrame({"SPY": [100.0] * n, PROXY_TICKER: [100.0] * n, "AAA": [10.0] * n}, index=IDX)
    closes = pd.DataFrame({
        "SPY": spy_close or [100.0] * n,
        PROXY_TICKER: mtum_close or [100.0] * n,
        "AAA": aaa_close or [10.0] * n,
    }, index=IDX)
    return opens, closes


def test_tracks_start_at_nav_and_hold_their_sleeves() -> None:
    opens, closes = _frames(spy_close=[110.0] * len(IDX), mtum_close=[120.0] * len(IDX))
    executions = [TrackExecution(IDX[0].date(), {"AAA": 2_500.0}, scalar=0.5)]

    navs = replay_tracks(executions, opens, closes)

    first = navs.iloc[0]
    assert navs.index[0] == IDX[0]
    assert first["nav_b"] == pytest.approx(NAV_TOTAL_USD, rel=1e-3)  # 2,500 of AAA at flat prices, minus costs
    # C: 20% of NAV in MTUM bought at 100, marked at 120 -> +4%; D: half that sleeve -> +2%
    assert first["nav_c"] == pytest.approx(NAV_TOTAL_USD * 1.04, rel=1e-3)
    assert first["nav_d"] == pytest.approx(NAV_TOTAL_USD * 1.02, rel=1e-3)
    assert first["nav_e"] == pytest.approx(NAV_TOTAL_USD * 1.10)


def test_ter_drag_shrinks_mtum_tracks_daily() -> None:
    opens, closes = _frames()
    navs = replay_tracks([TrackExecution(IDX[0].date(), {}, 1.0)], opens, closes)
    assert navs["nav_c"].iloc[-1] < navs["nav_c"].iloc[0]


def test_track_b_catastrophic_stop_sells_next_open() -> None:
    aaa = [10.0, 10.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    opens, closes = _frames(aaa_close=aaa)
    opens.loc[IDX[3], "AAA"] = 6.0
    navs = replay_tracks([TrackExecution(IDX[0].date(), {"AAA": 2_500.0}, 1.0)], opens, closes)
    # sold at the 6.0 open on day 4: NAV stays flat afterwards
    assert navs["nav_b"].iloc[4] == pytest.approx(navs["nav_b"].iloc[-1])
    assert navs["nav_b"].iloc[-1] == pytest.approx(NAV_TOTAL_USD - 1_000.0, rel=2e-3)


def test_no_executions_gives_empty_frame() -> None:
    opens, closes = _frames()
    assert replay_tracks([], opens, closes).empty


def test_load_executions_uses_targets_b_or_unscales_track_a(tmp_path: Path) -> None:
    for as_of, payload in (
        ("2026-08-31", {"as_of": "2026-08-31", "targets": [{"ticker": "AAA", "target_usd": 1_250.0}]}),
        ("2026-09-30", {"as_of": "2026-09-30", "targets": [{"ticker": "BBB", "target_usd": 1.0}], "targets_b": {"BBB": 2_500.0}}),
    ):
        (tmp_path / as_of).mkdir()
        (tmp_path / as_of / "targets.json").write_text(json.dumps(payload), encoding="utf-8")
    history = [{"date": "2026-08-31", "scalar_applied": 0.5}, {"date": "2026-09-30", "scalar_applied": 0.25}]

    executions = load_executions(tmp_path, history)

    assert [e.execution_date for e in executions] == [date(2026, 9, 1), date(2026, 10, 1)]
    assert executions[0].targets_b == {"AAA": 2_500.0}
    assert executions[1].targets_b == {"BBB": 2_500.0}
    assert executions[1].scalar == 0.25


def test_wide_frames_keeps_only_spy_sessions() -> None:
    days = pd.date_range("2026-09-04", periods=4)  # Fri, Sat, Sun, Mon
    spy = pd.DataFrame({"timestamp": days[[0, 3]], "open": [1.0, 2.0], "close": [1.0, 2.0]})
    btc = pd.DataFrame({"timestamp": days, "open": [5.0] * 4, "close": [5.0] * 4})
    _opens, closes = wide_frames({"SPY": spy, "BTC-USD": btc})
    assert list(closes.index) == list(days[[0, 3]])


def test_rewrite_track_navs_leaves_track_a(tmp_path: Path) -> None:
    path = tmp_path / "nav_daily.csv"
    for day in ("2026-08-31", "2026-09-01"):
        append_nav_row({"date": day, "nav_a": 249_000.0, "nav_b": 250_000, "nav_c": 250_000, "nav_d": 250_000, "nav_e": 250_000}, path)
    navs = pd.DataFrame({"nav_b": [1.0], "nav_c": [2.0], "nav_d": [3.0], "nav_e": [4.0]}, index=[pd.Timestamp("2026-09-01")])

    assert rewrite_track_navs(path, navs) == 1
    frame = pd.read_csv(path)
    assert frame["nav_a"].tolist() == [249_000.0, 249_000.0]
    assert frame["nav_e"].tolist() == [250_000.0, 4.0]


def test_summarize_navs_month_ytd_and_drawdown() -> None:
    index = pd.to_datetime(["2026-09-01", "2026-09-30", "2026-10-15", "2026-10-30"])
    navs = pd.DataFrame({f"nav_{t}": [100.0, 110.0, 99.0, 121.0] for t in "abcde"}, index=index)
    cum, dd = summarize_navs(navs, date(2026, 10, 30))
    assert cum["A"]["month"] == pytest.approx(0.10)
    assert cum["A"]["since_inception"] == pytest.approx(0.21)
    assert dd["A"] == pytest.approx(-0.10)


def test_regime_ignores_gaps_in_the_spx_series() -> None:
    days = pd.bdate_range("2025-01-01", periods=300)
    spx = pd.Series(np.linspace(100.0, 80.0, 300), index=days)  # falling market: below its 200-day mean
    spx.iloc[150] = np.nan  # a union-of-calendars gap, as yfinance batch downloads produce
    vix = pd.Series([15.0] * 300, index=days)

    result = compute_regime(spx, vix, PriorRegimeState(scalar_applied=1.0, months_at_state=1), as_of=days[-1].date())

    assert not np.isnan(result.spx_vs_200dma)
    assert result.bear is True
