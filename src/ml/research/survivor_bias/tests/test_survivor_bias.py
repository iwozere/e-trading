"""Unit tests for the survivor-bias study (synthetic data only)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ml.research.survivor_bias import config
from src.ml.research.survivor_bias.delisting import apply_to_trades, classify_end, terminal_factor
from src.ml.research.survivor_bias.jobs.run_survivor_bias import bias_view, clean_trades
from src.ml.research.survivor_bias.panel import bad_print_days, daily_features, month_rows
from src.ml.research.survivor_bias.portfolios import eligible_rows, ew_universe, momentum_decile, momentum_p21, period_stats

STORE_DATE = date(2026, 11, 20)


def _series(values, start="2020-01-01"):
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, dtype=float)


def _bars(closes, start="2020-01-01", volume=1_000_000.0):
    close = _series(closes, start)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=close.index)


def test_classify_end_live_when_active_or_recent():
    prices = _series([10.0] * 100)
    assert classify_end("ABC", True, date(2020, 1, 1), STORE_DATE, prices, prices) == config.LIVE
    assert classify_end("ABC", False, date(2026, 11, 15), STORE_DATE, prices, prices) == config.LIVE


def test_classify_end_distressed_rules():
    flat = _series([10.0] * 100)
    assert classify_end("ABCDQ", False, date(2010, 1, 1), STORE_DATE, flat, flat) == config.DISTRESSED
    cheap = _series([0.5] * 100)
    assert classify_end("ABC", False, date(2010, 1, 1), STORE_DATE, cheap, cheap) == config.DISTRESSED
    falling = _series(list(np.linspace(20.0, 5.0, 100)))
    assert classify_end("ABC", False, date(2010, 1, 1), STORE_DATE, falling, falling) == config.DISTRESSED
    assert classify_end("ABC", False, date(2010, 1, 1), STORE_DATE, flat, flat) == config.ORDINARY


def test_terminal_factor_versions():
    assert terminal_factor(config.DISTRESSED, config.LAST_CLOSE) == 1.0
    assert terminal_factor(config.DISTRESSED, config.WITH_HAIRCUT) == pytest.approx(0.7)
    assert terminal_factor(config.ORDINARY, config.WITH_HAIRCUT) == 1.0
    with pytest.raises(ValueError):
        terminal_factor(config.DISTRESSED, "other")


def test_apply_to_trades_only_hits_delisting_trades():
    trades = pd.DataFrame({"security_id": [1, 1, 2], "delisted_in_trade": [True, False, True], "net_return": [0.1, 0.1, 0.1]})
    out = apply_to_trades(trades, {1: config.DISTRESSED, 2: config.ORDINARY}, config.WITH_HAIRCUT)
    assert out["net_return"].tolist() == pytest.approx([1.1 * 0.7 - 1.0, 0.1, 0.1])


def test_daily_features_momentum_and_dollar_volume():
    closes = list(np.linspace(10.0, 30.0, 300))
    features = daily_features(_bars(closes), None, None)
    assert features["dollar_volume"].iloc[-1] == pytest.approx(closes[-32] * 1_000_000.0, rel=0.05)
    expected = closes[-1 - config.SKIP_BARS] / closes[-1 - config.LOOKBACK_BARS] - 1.0
    assert features["mom_ret"].iloc[-1] == pytest.approx(expected)
    assert np.isnan(features["mom_ret"].iloc[100])
    assert features["mom_vol"].iloc[-1] > 0


def test_month_rows_terminal_flags_for_delisted():
    daily = daily_features(_bars([10.0] * 21 + [11.0] * 20 + [12.0] * 5, start="2021-01-01"), None, None)
    rows = month_rows(daily, config.DISTRESSED)
    assert rows["ret_next"].iloc[0] == pytest.approx(0.1)
    assert rows["next_terminal"].tolist()[-2:] == [True, True]
    assert rows["ret_next"].iloc[-1] == 0.0
    live = month_rows(daily, config.LIVE)
    assert not live["next_terminal"].any()
    assert np.isnan(live["ret_next"].iloc[-1])


def _panel():
    months = pd.date_range("2015-01-01", periods=24, freq="MS")
    rows = []
    for sid in range(1, 31):
        for i, month in enumerate(months):
            rows.append({"security_id": sid, "month": month, "last_date": (month + pd.offsets.MonthEnd(0)).date(),
                         "close_tr": 10.0, "close_raw": 10.0, "dollar_volume": 1e8 + sid, "bars": 500 + i,
                         "mom_ret": sid / 100.0, "mom_vol": 0.2, "ret_next": 0.01 * (sid % 3),
                         "next_terminal": sid == 30 and i == 23, "end_kind": config.DISTRESSED if sid == 30 else config.LIVE})
    return pd.DataFrame(rows)


def test_eligible_rows_applies_haircut_and_filters():
    panel = _panel()
    panel.loc[panel["security_id"] == 1, "close_raw"] = 2.0
    rows = eligible_rows(panel, set(range(1, 31)), config.WITH_HAIRCUT)
    assert 1 not in set(rows["security_id"])
    last = rows[(rows["security_id"] == 30) & rows["next_terminal"]]
    assert last["ret"].iloc[0] == pytest.approx(0.7 - 1.0)
    clipped = eligible_rows(panel.assign(ret_next=10.0), {2}, config.LAST_CLOSE, clip=True)
    assert clipped["ret"].max() == config.CLIP_MONTHLY_RETURN


def test_eligible_rows_drops_source_disagreement():
    panel = _panel()
    panel["ret_vendor"] = panel["ret_next"]
    panel.loc[(panel["security_id"] == 2) & (panel["month"] == panel["month"].min()), "ret_vendor"] = 5.0
    rows = eligible_rows(panel, set(range(1, 31)), config.LAST_CLOSE)
    assert len(rows) == len(panel) - 1


def test_daily_features_drops_momentum_when_sources_disagree():
    bars = _bars(list(np.linspace(10.0, 30.0, 300)))
    bars["closeadj_vendor"] = bars["close"]
    assert np.isfinite(daily_features(bars, None, None)["mom_ret"].iloc[-1])
    bars["closeadj_vendor"] = bars["close"] * np.where(np.arange(300) < 100, 0.1, 1.0)
    assert np.isnan(daily_features(bars, None, None)["mom_ret"].iloc[-1])


def test_bad_print_days_spike_and_impossible_move():
    spike = _series([10.0, 10.0, 40.0, 10.5, 10.0, 10.0])
    assert bad_print_days(spike, None).tolist() == [False, False, True, False, False, False]
    crash = _series([10.0, 10.0, 2.0, 2.0, 2.0, 2.0])
    assert not bad_print_days(crash, None).any()
    currency = _series([985.0, 985.0, 0.01, 0.01])
    assert bad_print_days(currency, None).iloc[2]
    split_day = pd.DataFrame({"date": [currency.index[2].date()], "numerator": [1.0], "denominator": [100000.0]})
    assert not bad_print_days(currency, split_day).iloc[2]


def test_month_rows_marks_suspect_months():
    closes = [10.0] * 21 + [10.0, 40.0, 10.0] + [10.0] * 22
    daily = daily_features(_bars(closes, start="2021-01-01"), None, None)
    rows = month_rows(daily, config.LIVE)
    assert rows["suspect"].tolist()[:2] == [True, True]
    assert not rows["suspect"].iloc[-1]


def test_strategies_pick_expected_names():
    rows = eligible_rows(_panel(), set(range(1, 31)), config.LAST_CLOSE)
    bench = ew_universe(rows)
    assert bench["names"].iloc[0] == 30
    decile = momentum_decile(rows)
    assert decile["names"].iloc[0] == 3
    p21 = momentum_p21(rows)
    assert p21["names"].iloc[0] == 20
    assert p21["cost"].iloc[0] == pytest.approx(config.P21_COST_PER_SIDE)
    assert p21["cost"].iloc[1] == 0.0
    stats = period_stats(p21, bench)
    assert stats[0]["period"] == "2012-2021" and stats[0]["months"] == 24


def test_clean_trades_flags_suspect_window():
    panel = pd.DataFrame({"security_id": [1, 2], "month": pd.to_datetime(["2015-03-01", "2010-01-01"]), "suspect": [True, True]})
    trades = pd.DataFrame({"security_id": [1, 1, 2], "signal_date": ["2015-06-10", "2017-01-05", "2015-06-10"],
                           "exit_date": ["2015-08-01", "2017-03-01", "2015-08-01"]})
    assert clean_trades(trades, panel).tolist() == [False, True, True]


def test_bias_view_differences():
    table = pd.DataFrame({"k": ["a"] * 3, "universe": [config.FULL, config.VINTAGE_LIKE, config.SURVIVORS], "v": [0.05, 0.08, 0.10]})
    wide = bias_view(table, ["k"], "v")
    assert wide["bias_vintage"].iloc[0] == pytest.approx(0.03)
    assert wide["bias_survivors"].iloc[0] == pytest.approx(0.05)
