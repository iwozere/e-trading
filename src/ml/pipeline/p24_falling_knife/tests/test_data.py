"""Tests for the P24 data and universe layers."""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from src.data.fmp_pipeline.builders.flags import JUMP_NO_SPLIT, OHLC_VIOLATION
from src.ml.pipeline.p24_falling_knife.config import UniverseParams, WEEKLY
from src.ml.pipeline.p24_falling_knife.data import prepare_daily, series_for_timeframe, to_weekly
from src.ml.pipeline.p24_falling_knife.universe import (
    Security,
    eligibility_mask,
    is_common_stock,
    survivorship_table,
)


def _raw_frame(dates: pd.DatetimeIndex, close: Any, volume: float = 1e6, flags: Optional[Any] = None) -> pd.DataFrame:
    close_arr = np.asarray(close, dtype=float) * np.ones(len(dates))
    return pd.DataFrame({"open": close_arr, "high": close_arr * 1.01, "low": close_arr * 0.99, "close": close_arr,
                         "volume": np.full(len(dates), volume),
                         "flags": np.zeros(len(dates), dtype=np.int64) if flags is None else flags}, index=dates)


def _security(**kw: Any) -> Security:
    base: dict[str, Any] = dict(security_id=1, symbol="ABC", name="Abc Inc", exchange="NASDAQ", instrument_type="stock",
                                first_price_date=date(2010, 1, 4), last_price_date=date(2026, 10, 16), delisted_date=None,
                                is_actively_trading=True, valid_to=None, identity_uncertain=False)
    base.update(kw)
    return Security(**base)


def test_weekly_resample_aggregates_and_labels_last_session() -> None:
    # 2024-07-01 .. 2024-07-12: the 4th of July holiday week has four sessions.
    dates = pd.DatetimeIndex(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05",
                              "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-11", "2024-07-12"])
    daily = pd.DataFrame({"open": np.arange(9) + 10.0, "high": np.arange(9) + 11.0, "low": np.arange(9) + 9.0,
                          "close": np.arange(9) + 10.5, "volume": np.ones(9)}, index=dates)
    weekly = to_weekly(daily)
    assert list(weekly.index) == [pd.Timestamp("2024-07-05"), pd.Timestamp("2024-07-12")]
    first = weekly.iloc[0]
    assert first["open"] == 10.0 and first["close"] == 13.5 and first["high"] == 14.0 and first["low"] == 9.0
    assert first["volume"] == 4.0


def test_incomplete_current_week_is_dropped() -> None:
    dates = pd.bdate_range("2024-07-08", "2024-07-17")  # Mon .. Wed of the next week
    daily = _raw_frame(dates, 10.0).drop(columns=["flags"])
    assert to_weekly(daily, as_of=date(2024, 7, 17)).index[-1] == pd.Timestamp("2024-07-12")
    # A delisted security's last short week stays: the data is complete up to a later as_of.
    assert to_weekly(daily, as_of=date(2026, 1, 2)).index[-1] == pd.Timestamp("2024-07-17")


def test_prepare_daily_drops_bad_prints_marks_jumps_and_keeps_raw() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    flags = np.array([0, OHLC_VIOLATION, 0, JUMP_NO_SPLIT, 0, 0], dtype=np.int64)
    frame = _raw_frame(dates, [10, 10, 10, 20, 20, 20], flags=flags)
    splits = pd.DataFrame({"date": [pd.Timestamp("2024-01-08")], "numerator": [2.0], "denominator": [1.0]})
    out = prepare_daily(frame, splits, None)
    assert len(out) == 5
    assert out["jump_no_split"].tolist() == [False, False, True, False, False]
    # 2:1 split on 2024-01-08: earlier closes halved, raw close kept.
    assert out["close"].iloc[0] == pytest.approx(5.0)
    assert out["close_raw"].iloc[0] == pytest.approx(10.0)
    assert out["volume"].iloc[0] == pytest.approx(2e6)


def test_eligibility_uses_raw_price_and_past_volume_only() -> None:
    dates = pd.bdate_range("2024-01-02", periods=40)
    daily = pd.DataFrame({"close": np.full(40, 5.0), "close_raw": np.full(40, 0.5), "volume": np.full(40, 1e6),
                          "volume_raw": np.full(40, 1e7)}, index=dates)
    params = UniverseParams()
    assert not eligibility_mask(daily, params).any()  # traded at $0.50 although split-adjusted shows $5
    daily["close_raw"] = 5.0
    mask = eligibility_mask(daily, params)
    assert mask.iloc[-1] and not mask.iloc[0]  # needs half a dollar-volume window first
    shocked = daily.copy()
    shocked.loc[shocked.index[-1], "volume_raw"] = 1e12
    pd.testing.assert_series_equal(eligibility_mask(shocked, params).iloc[:-1], mask.iloc[:-1])


def test_weekly_series_carries_eligibility() -> None:
    dates = pd.bdate_range("2024-01-02", periods=60)
    daily = _raw_frame(dates, 20.0).drop(columns=["flags"])
    weekly = series_for_timeframe(daily, WEEKLY, UniverseParams(), as_of=date(2026, 1, 2))
    assert weekly["eligible"].dtype == bool
    assert weekly["eligible"].iloc[-1]
    assert "dollar_volume_20" in weekly.columns


def test_common_stock_filter() -> None:
    assert is_common_stock(_security())
    assert not is_common_stock(_security(instrument_type="etf"))
    assert not is_common_stock(_security(instrument_type="adr"))
    assert is_common_stock(_security(instrument_type="adr"), include_adr=True)
    assert not is_common_stock(_security(symbol="ABCDW"))
    assert not is_common_stock(_security(symbol="BRK-B"))
    assert not is_common_stock(_security(identity_uncertain=True))
    assert not is_common_stock(_security(symbol="ZXZZT"))
    assert not is_common_stock(_security(symbol="CBX", name="IEX Test Common Stock ZXIET"))
    assert is_common_stock(_security(symbol="INTT", name="inTEST Corporation"))
    assert is_common_stock(_security(symbol="AEHR", name="Aehr Test Systems"))


def test_survivorship_table_flags_thin_years() -> None:
    securities = [_security(security_id=i, first_price_date=date(2015, 1, 2)) for i in range(100)]
    securities += [_security(security_id=1000 + i, first_price_date=date(2015, 1, 2), last_price_date=date(2022, 6, 1),
                             delisted_date=date(2022, 6, 1)) for i in range(10)]
    securities.append(_security(security_id=5000, first_price_date=date(2015, 1, 2), last_price_date=date(2018, 3, 1),
                                delisted_date=date(2018, 3, 1)))
    table = survivorship_table(securities, 2017, 2023, (2022, 2022), 0.5).set_index("year")
    assert table.loc[2022, "est_coverage"] == pytest.approx(1.0)
    assert table.loc[2018, "upper_bound_only"] and not table.loc[2022, "upper_bound_only"]
    assert table.loc[2018, "delisted"] == 1
