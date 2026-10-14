"""
Tests for ``src.data.fmp_pipeline.adjust.yfinance_to_raw`` (spec §10.2.3, including the §10.2.5 gap case).

All series are synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.fmp_pipeline.adjust.adjust import adjust
from src.data.fmp_pipeline.adjust.yfinance_to_raw import (
    extract_yfinance_dividends,
    extract_yfinance_splits,
    yfinance_to_raw,
)

DAYS = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]))


def _yf(closes: list[float], volumes: list[float], splits: list[float], dividends: list[float]) -> pd.DataFrame:
    """Build a yfinance-shaped history (auto_adjust=False, actions=True)."""
    close = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close * 0.9,
            "Volume": volumes,
            "Dividends": dividends,
            "Stock Splits": splits,
        },
        index=DAYS[: len(closes)],
    )


def test_no_split_is_identity():
    df = _yf([10, 11, 12, 13, 14], [5, 5, 5, 5, 5], [0] * 5, [0] * 5)
    out = yfinance_to_raw(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    np.testing.assert_allclose(out["close"], [10, 11, 12, 13, 14])
    np.testing.assert_allclose(out["volume"], [5] * 5)


def test_one_split():
    # yfinance shows the pre-split bars already divided by 4 and volume multiplied by 4.
    df = _yf([100, 101, 101, 102, 103], [400, 400, 100, 100, 100], [0, 0, 4, 0, 0], [0] * 5)
    out = yfinance_to_raw(df)
    np.testing.assert_allclose(out["close"], [400, 404, 101, 102, 103])
    np.testing.assert_allclose(out["high"], [404, 408, 102, 103, 104])
    np.testing.assert_allclose(out["volume"], [100] * 5)


def test_two_splits():
    df = _yf([100] * 5, [6000, 3000, 3000, 1000, 1000], [0, 2, 0, 3, 0], [0] * 5)
    out = yfinance_to_raw(df)
    np.testing.assert_allclose(out["close"], [600, 300, 300, 100, 100])
    np.testing.assert_allclose(out["volume"], [1000] * 5)


def test_reverse_split():
    df = _yf([10, 11, 10.5, 11, 12], [1000] * 5, [0, 0, 0.1, 0, 0], [0] * 5)
    out = yfinance_to_raw(df)
    np.testing.assert_allclose(out["close"], [1.0, 1.1, 10.5, 11, 12])
    np.testing.assert_allclose(out["volume"], [10_000, 10_000, 1000, 1000, 1000])


def test_gap_with_split_inside_window_round_trips():
    # Spec §10.2.5: ingestion paused for a long time, a 3:1 split happened during the gap. The download covers
    # the whole gap, so the reported split un-adjusts every bar before it.
    index = pd.bdate_range("2023-01-02", "2024-06-28")
    raw_close = np.where(index < pd.Timestamp("2023-09-01"), 300.0, 100.0) + np.arange(len(index)) * 0.01
    raw = pd.DataFrame({"open": raw_close, "high": raw_close, "low": raw_close, "close": raw_close,
                        "volume": 1000.0}, index=index)
    splits = pd.DataFrame({"date": ["2023-09-01"], "numerator": [3.0], "denominator": [1.0]})
    yf_view = adjust(raw, splits, mode="split").rename(columns=str.capitalize)
    yf_view["Stock Splits"] = np.where(index == pd.Timestamp("2023-09-01"), 3.0, 0.0)
    out = yfinance_to_raw(yf_view)
    pd.testing.assert_frame_equal(out, raw.astype(float))


def test_split_after_window_must_be_passed_explicitly():
    # The window ends before a 2:1 split that yfinance has already applied; it is not in "Stock Splits".
    df = _yf([50, 51, 52], [200, 200, 200], [0, 0, 0], [0, 0, 0])
    later_split = pd.DataFrame({"date": ["2024-02-01"], "numerator": [2.0], "denominator": [1.0]})
    out = yfinance_to_raw(df, splits=later_split)
    np.testing.assert_allclose(out["close"], [100, 102, 104])
    np.testing.assert_allclose(out["volume"], [100, 100, 100])


def test_tz_aware_index_and_lowercase_columns():
    df = _yf([100, 101, 101, 102, 103], [400] * 5, [0, 0, 4, 0, 0], [0] * 5)
    df.index = pd.DatetimeIndex(df.index).tz_localize("America/New_York")
    lower = df.rename(columns={"Close": "close", "Volume": "volume"})
    out = yfinance_to_raw(lower)
    np.testing.assert_allclose(out["close"], [400, 404, 101, 102, 103])
    assert pd.DatetimeIndex(out.index).tz is not None


def test_no_price_columns_raises():
    with pytest.raises(ValueError, match="no price columns"):
        yfinance_to_raw(pd.DataFrame({"Volume": [1.0]}, index=DAYS[:1]))


def test_empty_frame():
    assert yfinance_to_raw(_yf([], [], [], [])).empty


def test_extract_splits():
    df = _yf([100] * 5, [1] * 5, [0, 2, 0, 0.5, 0], [0] * 5)
    splits = extract_yfinance_splits(df)
    assert list(splits["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-05"]
    np.testing.assert_allclose(splits["numerator"], [2, 0.5])
    np.testing.assert_allclose(splits["denominator"], [1, 1])
    assert extract_yfinance_splits(df.drop(columns="Stock Splits")).empty


def test_extract_dividends_restated_on_raw_basis():
    # A 0.205 split-adjusted dividend before a 4:1 split was 0.82 per pre-split share (the AAPL 2020 case).
    df = _yf([100] * 5, [1] * 5, [0, 0, 4, 0, 0], [0.205, 0, 0, 0.21, 0])
    dividends = extract_yfinance_dividends(df)
    np.testing.assert_allclose(dividends["amount"], [0.82, 0.21])
    assert extract_yfinance_dividends(df.drop(columns="Dividends")).empty


def test_extract_dividends_with_explicit_splits():
    df = _yf([100] * 3, [1] * 3, [0] * 3, [0.5, 0, 0])
    later_split = pd.DataFrame({"date": ["2024-03-01"], "numerator": [2.0], "denominator": [1.0]})
    np.testing.assert_allclose(extract_yfinance_dividends(df, splits=later_split)["amount"], [1.0])


def test_round_trip_raw_to_yfinance_to_raw():
    raw = pd.DataFrame(
        {"open": [399.0, 403, 100, 101, 102], "high": [402.0, 406, 103, 104, 105], "low": [398.0, 402, 99, 100, 101],
         "close": [400.0, 404, 101, 102, 103], "volume": [100.0, 100, 400, 400, 400]},
        index=DAYS,
    )
    splits = pd.DataFrame({"date": ["2024-01-04"], "numerator": [4.0], "denominator": [1.0]})
    yf_view = adjust(raw, splits, mode="split").rename(columns=str.capitalize)
    pd.testing.assert_frame_equal(yfinance_to_raw(yf_view, splits=splits), raw)
