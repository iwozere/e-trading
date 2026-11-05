"""Tests for the P24 backtest runner on synthetic securities (no vintage needed)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ml.pipeline.p24_falling_knife.backtest import (
    FULL,
    HOLDOUT,
    TRAIN,
    UPPER_BOUND,
    RunSpec,
    add_benchmark_excess,
    period_of,
    price_bucket,
    run_security,
    trades_frame,
)
from src.ml.pipeline.p24_falling_knife.config import DAILY, P24Config
from src.ml.pipeline.p24_falling_knife.tests.helpers import KNIFE_LEGS, bars_from_closes, path_from_legs
from src.ml.pipeline.p24_falling_knife.universe import Security

CONFIG = P24Config(random_control_draws=10)


def _security() -> Security:
    return Security(security_id=7, symbol="KNF", name="Knife Corp", exchange="NYSE", instrument_type="stock",
                    first_price_date=date(2021, 1, 4), last_price_date=None, delisted_date=None,
                    is_actively_trading=True, valid_to=None, identity_uncertain=False)


def _daily(start: str) -> pd.DataFrame:
    bars = bars_from_closes(path_from_legs(KNIFE_LEGS), start=start, volume=2_000_000.0)
    bars["open_raw"], bars["close_raw"], bars["volume_raw"] = bars["open"], bars["close"], bars["volume"]
    bars["jump_no_split"] = False
    return bars


def test_period_labels() -> None:
    assert period_of(date(2010, 5, 1), CONFIG) == UPPER_BOUND
    assert period_of(date(2022, 5, 1), CONFIG) == TRAIN
    assert period_of(date(2025, 5, 1), CONFIG) == HOLDOUT
    assert period_of(date(2001, 5, 1), CONFIG) is None


def test_price_buckets() -> None:
    assert [price_bucket(p) for p in (25.0, 7.0, 2.0, 0.4)] == [">=10", "5-10", "1-5", "<1"]


def test_run_security_produces_full_and_control_rows() -> None:
    # The synthetic knife's signal lands in 2022 when the series starts in mid-2020.
    spec = RunSpec(config=CONFIG, timeframes=(DAILY,), variants=(FULL, "drop_only"), include_holdout=False,
                   as_of=date(2026, 10, 16))
    result = run_security(_security(), _daily("2020-08-03"), spec, rng_seed=1)
    full = [r for r in result.trades if r["variant"] == FULL]
    assert len(full) == 1
    row = full[0]
    assert row["period"] == TRAIN and row["timeframe"] == DAILY
    assert row["entry_date"] > row["signal_date"]
    assert row["price_bucket"] == ">=10"
    assert any(r["variant"] == "drop_only" for r in result.trades)
    assert len(result.random_control) == 1
    assert len(result.random_control[0]["returns"]) == 10


def test_holdout_rows_need_explicit_request() -> None:
    daily = _daily("2023-03-01")  # signal in 2024 -> holdout
    closed = RunSpec(config=CONFIG, timeframes=(DAILY,), variants=(FULL,), include_holdout=False, as_of=None)
    assert run_security(_security(), daily, closed, rng_seed=1).trades == []
    opened = replace(closed, include_holdout=True)
    assert [r["period"] for r in run_security(_security(), daily, opened, rng_seed=1).trades] == [HOLDOUT]


def test_run_security_is_deterministic() -> None:
    spec = RunSpec(config=CONFIG, timeframes=(DAILY,), variants=(FULL,), include_holdout=False, as_of=None)
    a = run_security(_security(), _daily("2020-08-03"), spec, rng_seed=3)
    b = run_security(_security(), _daily("2020-08-03"), spec, rng_seed=3)
    pd.testing.assert_frame_equal(trades_frame(a.trades), trades_frame(b.trades))
    assert a.random_control[0]["returns"] == b.random_control[0]["returns"]


def test_benchmark_excess() -> None:
    closes = pd.Series([100.0, 101.0, 102.0, 110.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                                                                           "2024-01-05"]))
    trades = pd.DataFrame({"entry_date": [date(2024, 1, 3)], "exit_date": [date(2024, 1, 5)], "net_return": [0.2]})
    out = add_benchmark_excess(trades, closes)
    assert out["spy_return"].iloc[0] == pytest.approx(0.10)
    assert out["excess_spy"].iloc[0] == pytest.approx(0.10)
    assert np.isfinite(out["excess_spy"]).all()
