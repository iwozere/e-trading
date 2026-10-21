"""
Tests for ``src.data.fmp_pipeline.adjust`` (spec §8.2–§8.3).

Every expected value is hand-computed. The real-world vectors use the actual split dates and ratios from
spec §8.3; their prices are synthetic round numbers near the real ones, never downloaded data (D5).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.data.fmp_pipeline.adjust.adjust import adjust
from src.data.fmp_pipeline.adjust.factors import adjustment_factors, bar_dates, normalize_dividends, normalize_splits


def _bars(dates: list[str], closes: list[float], volumes: Optional[list[float]] = None) -> pd.DataFrame:
    """Build a raw OHLCV frame where open/high/low are fixed offsets of close."""
    close = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": volumes if volumes is not None else [1000.0] * len(closes),
        },
        index=pd.DatetimeIndex(pd.to_datetime(dates)),
    )


def _splits(*rows: tuple[str, float, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "numerator", "denominator"])


def _dividends(*rows: tuple[str, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "amount"])


DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


def test_no_events_is_identity_for_every_mode():
    raw = _bars(DAYS, [10, 11, 12, 13, 14])
    for mode in ("raw", "split", "total"):
        pd.testing.assert_frame_equal(adjust(raw, mode=mode), raw.astype(float) if mode != "raw" else raw)


def test_raw_mode_returns_a_copy():
    raw = _bars(DAYS, [10, 11, 12, 13, 14])
    out = adjust(raw, _splits(("2024-01-04", 2, 1)), mode="raw")
    out.loc[out.index[0], "close"] = -1
    assert raw["close"].iloc[0] == 10


def test_single_forward_split():
    raw = _bars(DAYS, [400, 404, 101, 102, 103], [100, 100, 400, 400, 400])
    out = adjust(raw, _splits(("2024-01-04", 4, 1)), mode="split")
    np.testing.assert_allclose(out["close"], [100, 101, 101, 102, 103])
    np.testing.assert_allclose(out["open"], [399 / 4, 403 / 4, 100, 101, 102])
    np.testing.assert_allclose(out["volume"], [400, 400, 400, 400, 400])


def test_multiple_splits_compound():
    raw = _bars(DAYS, [600, 300, 300, 100, 100])
    out = adjust(raw, _splits(("2024-01-03", 2, 1), ("2024-01-05", 3, 1)), mode="split")
    np.testing.assert_allclose(out["close"], [100, 100, 100, 100, 100])
    np.testing.assert_allclose(out["volume"], [6000, 3000, 3000, 1000, 1000])


def test_reverse_split():
    raw = _bars(DAYS, [1.0, 1.1, 10.5, 11, 12], [10_000, 10_000, 1000, 1000, 1000])
    out = adjust(raw, _splits(("2024-01-04", 1, 10)), mode="split")
    np.testing.assert_allclose(out["close"], [10, 11, 10.5, 11, 12])
    np.testing.assert_allclose(out["volume"], [1000, 1000, 1000, 1000, 1000])


def test_single_dividend_total_mode():
    raw = _bars(DAYS, [100, 100, 99, 99, 99])
    out = adjust(raw, dividends=_dividends(("2024-01-04", 1.0)), mode="total")
    np.testing.assert_allclose(out["close"], [99, 99, 99, 99, 99])
    # split mode ignores dividends; volume is never dividend-adjusted
    split_only = adjust(raw, dividends=_dividends(("2024-01-04", 1.0)), mode="split")
    np.testing.assert_allclose(split_only["close"], raw["close"])
    np.testing.assert_allclose(out["volume"], raw["volume"])


def test_dividend_on_split_date_uses_restated_previous_close():
    # 4:1 split and a 0.25 post-split dividend share the ex-date 2024-01-04. C_prev = 400 pre-split = 100 on the
    # dividend's basis, so DF = 1 - 0.25 / 100 = 0.9975 (not 1 - 0.25 / 400).
    raw = _bars(DAYS, [400, 400, 99.75, 99.75, 99.75])
    out = adjust(raw, _splits(("2024-01-04", 4, 1)), _dividends(("2024-01-04", 0.25)), mode="total")
    np.testing.assert_allclose(out["close"], [99.75, 99.75, 99.75, 99.75, 99.75])


def test_dividend_with_no_prior_bar_has_no_effect():
    raw = _bars(DAYS, [10, 11, 12, 13, 14])
    for ex_date in ("2024-01-02", "2023-12-01"):
        out = adjust(raw, dividends=_dividends((ex_date, 0.5)), mode="total")
        np.testing.assert_allclose(out["close"], raw["close"])


def test_duplicate_events_are_counted_once():
    raw = _bars(DAYS, [400, 404, 101, 102, 103])
    once = adjust(raw, _splits(("2024-01-04", 4, 1)), _dividends(("2024-01-05", 1.02)), mode="total")
    twice = adjust(
        raw,
        _splits(("2024-01-04", 4, 1), ("2024-01-04", 4, 1)),
        _dividends(("2024-01-05", 1.02), ("2024-01-05", 1.02)),
        mode="total",
    )
    pd.testing.assert_frame_equal(once, twice)


def test_two_distinct_dividends_on_one_date_both_apply():
    raw = _bars(DAYS, [100, 100, 100, 100, 100])
    out = adjust(raw, dividends=_dividends(("2024-01-04", 1.0), ("2024-01-04", 2.0)), mode="total")
    np.testing.assert_allclose(out["close"].iloc[:2], [100 * 0.99 * 0.98] * 2)


def test_conflicting_split_ratios_raise():
    raw = _bars(DAYS, [10, 11, 12, 13, 14])
    with pytest.raises(ValueError, match="conflicting split ratios"):
        adjust(raw, _splits(("2024-01-04", 4, 1), ("2024-01-04", 2, 1)), mode="split")


@pytest.mark.parametrize("numerator,denominator", [(0, 1), (-2, 1), (1, 0)])
def test_invalid_split_ratio_raises(numerator: float, denominator: float):
    raw = _bars(DAYS, [10, 11, 12, 13, 14])
    with pytest.raises(ValueError, match="invalid split ratio"):
        adjust(raw, _splits(("2024-01-04", numerator, denominator)), mode="split")


@pytest.mark.parametrize("amount", [5.5, 11.0, 30.0])
def test_dividend_of_half_the_previous_close_or_more_is_left_out(amount: float):
    raw = _bars(DAYS, [10, 11, 12, 13, 14])
    out = adjust(raw, dividends=_dividends(("2024-01-04", amount)), mode="total")
    assert out["close"].tolist() == raw["close"].tolist()


def test_dividend_just_below_half_the_previous_close_is_applied():
    raw = _bars(DAYS, [10, 11, 12, 13, 14])
    out = adjust(raw, dividends=_dividends(("2024-01-04", 5.4)), mode="total")
    assert out["close"].iloc[0] == pytest.approx(10 * (1 - 5.4 / 11))


def test_negative_dividend_raises():
    with pytest.raises(ValueError, match="invalid dividend amount"):
        normalize_dividends(_dividends(("2024-01-04", -1.0)))


def test_events_after_last_bar_are_ignored():
    raw = _bars(DAYS, [10, 11, 12, 13, 14])
    out = adjust(raw, _splits(("2024-01-09", 2, 1)), _dividends(("2024-02-01", 0.5)), mode="total")
    np.testing.assert_allclose(out["close"], raw["close"])


def test_event_on_last_bar_date_applies_to_earlier_bars():
    raw = _bars(DAYS, [20, 20, 20, 20, 10])
    out = adjust(raw, _splits(("2024-01-08", 2, 1)), mode="split")
    np.testing.assert_allclose(out["close"], [10, 10, 10, 10, 10])


def test_missing_previous_close_skips_to_last_finite_close():
    raw = _bars(DAYS, [100, np.nan, 50, 50, 50])
    out = adjust(raw, dividends=_dividends(("2024-01-04", 1.0)), mode="total")
    assert out["close"].iloc[0] == pytest.approx(100 * 0.99)


def test_other_columns_pass_through_and_missing_columns_are_tolerated():
    raw = _bars(DAYS, [400, 404, 101, 102, 103])[["close"]].assign(symbol="X")
    out = adjust(raw, _splits(("2024-01-04", 4, 1)), mode="split")
    assert list(out.columns) == ["close", "symbol"]
    assert (out["symbol"] == "X").all()


def test_unsorted_index_raises():
    raw = _bars(DAYS, [10, 11, 12, 13, 14]).iloc[::-1]
    with pytest.raises(ValueError, match="sorted ascending"):
        adjust(raw, mode="split")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        adjust(_bars(DAYS, [10, 11, 12, 13, 14]), mode="legacy")  # type: ignore[arg-type]


def test_empty_prices():
    empty = _bars([], [])
    assert adjust(empty, _splits(("2024-01-04", 2, 1)), mode="total").empty
    assert adjustment_factors(empty).empty


def test_missing_event_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        normalize_splits(pd.DataFrame({"date": ["2024-01-04"], "ratio": [2.0]}))


def test_bar_dates_requires_datetime_index():
    with pytest.raises(TypeError):
        bar_dates(pd.RangeIndex(3))


def test_intraday_bars_use_calendar_dates_and_reference_closes():
    index = pd.DatetimeIndex(
        ["2024-01-03 09:30", "2024-01-03 15:30", "2024-01-03 17:00", "2024-01-04 09:30", "2024-01-04 10:30"]
    )
    raw = pd.DataFrame({"close": [400.0, 402.0, 410.0, 100.0, 100.5], "volume": [10.0] * 5}, index=index)
    daily = pd.Series([402.0], index=pd.DatetimeIndex(["2024-01-03"]))
    splits = _splits(("2024-01-04", 4, 1))
    dividends = _dividends(("2024-01-04", 0.5025))  # 0.5025 / (402 / 4) = 0.005

    out = adjust(raw, splits, dividends, mode="total", reference_closes=daily)
    np.testing.assert_allclose(out["close"].iloc[:3], np.array([400, 402, 410]) / 4 * 0.995)
    np.testing.assert_allclose(out["close"].iloc[3:], [100, 100.5])
    np.testing.assert_allclose(out["volume"], [40, 40, 40, 10, 10])
    # Without reference closes the extended-hours 17:00 print (410) would be used as C_prev.
    fallback = adjustment_factors(raw, splits, dividends)
    assert fallback["div_factor"].iloc[0] == pytest.approx(1 - 0.5025 / (410 / 4))


def test_tz_aware_index_uses_local_date():
    index = pd.DatetimeIndex(["2024-01-03 15:30", "2024-01-04 09:30"]).tz_localize("America/New_York")
    raw = pd.DataFrame({"close": [400.0, 100.0]}, index=index)
    out = adjust(raw, _splits(("2024-01-04", 4, 1)), mode="split")
    np.testing.assert_allclose(out["close"], [100, 100])


def test_total_equals_split_times_dividend_factor():
    raw = _bars(DAYS, [400, 404, 101, 102, 103])
    splits, dividends = _splits(("2024-01-04", 4, 1)), _dividends(("2024-01-03", 1.0), ("2024-01-08", 0.5))
    factors = adjustment_factors(raw, splits, dividends)
    total = adjust(raw, splits, dividends, mode="total")
    split = adjust(raw, splits, dividends, mode="split")
    np.testing.assert_allclose(total["close"], split["close"] * factors["div_factor"])
    # DF(2024-01-03) = 1 - 1/400; DF(2024-01-08) = 1 - 0.5/102
    expected = [(1 - 1 / 400) * (1 - 0.5 / 102)] + [1 - 0.5 / 102] * 3 + [1.0]
    np.testing.assert_allclose(factors["div_factor"], expected)


# --- Spec §8.3 real-world vectors (real event dates and ratios, synthetic prices) ---------------------------


@pytest.mark.parametrize(
    "split_rows,last_pre_split_day,first_post_split_day,raw_before,expected_before",
    [
        ([("2014-06-09", 7, 1)], "2014-06-06", "2014-06-09", 645.57, 645.57 / 7),  # AAPL 7:1
        ([("2014-06-09", 7, 1), ("2020-08-31", 4, 1)], "2014-06-06", "2020-08-31", 645.57, 645.57 / 28),  # AAPL both
        ([("2020-08-31", 4, 1)], "2020-08-28", "2020-08-31", 499.23, 499.23 / 4),  # AAPL 4:1
        ([("2020-08-31", 5, 1), ("2022-08-25", 3, 1)], "2020-08-28", "2022-08-25", 2213.40, 2213.40 / 15),  # TSLA
        ([("2022-06-06", 20, 1)], "2022-06-03", "2022-06-06", 2447.00, 2447.00 / 20),  # AMZN
        ([("2022-07-18", 20, 1)], "2022-07-15", "2022-07-18", 2235.55, 2235.55 / 20),  # GOOGL
        ([("2024-06-10", 10, 1)], "2024-06-07", "2024-06-10", 1208.88, 1208.88 / 10),  # NVDA
    ],
)
def test_spec_split_vectors(
    split_rows: list[tuple[str, float, float]],
    last_pre_split_day: str,
    first_post_split_day: str,
    raw_before: float,
    expected_before: float,
):
    raw = _bars([last_pre_split_day, first_post_split_day], [raw_before, 50.0], [1_000_000, 1_000_000])
    out = adjust(raw, _splits(*split_rows), mode="split")
    assert out["close"].iloc[0] == pytest.approx(expected_before)
    assert out["close"].iloc[1] == 50.0
    assert out["volume"].iloc[0] == pytest.approx(1_000_000 * raw_before / expected_before)


def test_spec_aapl_dividend_2020_08_07():
    # $0.82 pre-split dividend, ex 2020-08-07; C_prev = 2020-08-06 close (≈455.61 raw). Then the 4:1 split.
    raw = _bars(["2020-08-06", "2020-08-07", "2020-08-28", "2020-08-31"], [455.61, 444.45, 499.23, 129.04])
    out = adjust(raw, _splits(("2020-08-31", 4, 1)), _dividends(("2020-08-07", 0.82)), mode="total")
    df = 1 - 0.82 / 455.61
    assert out["close"].iloc[0] == pytest.approx(455.61 / 4 * df)
    assert out["close"].iloc[1] == pytest.approx(444.45 / 4)
    assert out["close"].iloc[3] == pytest.approx(129.04)


def test_event_dates_of_any_dtype_are_floored_to_midnight():
    expected = np.array(["1965-03-01", "2024-01-04"], dtype="datetime64[ns]")
    for column in (["1965-03-01 15:30", "2024-01-04 00:00"],
                   pd.to_datetime(["1965-03-01 15:30", "2024-01-04 00:00"]),
                   pd.to_datetime(["1965-03-01 15:30", "2024-01-04 00:00"]).tz_localize("America/New_York")):
        events = normalize_splits(pd.DataFrame({"date": column, "numerator": [2, 3], "denominator": [1, 1]}))
        assert (events.dates == expected).all() and list(events.ratios) == [2.0, 3.0]
    index = pd.DatetimeIndex(["1965-03-01 09:30", "1965-03-01 23:59", "2024-01-04 00:00"])
    assert (bar_dates(index) == np.array(["1965-03-01", "1965-03-01", "2024-01-04"], dtype="datetime64[ns]")).all()


def test_normalized_events_are_sorted_and_deduplicated():
    splits = normalize_splits(_splits(("2024-03-01", 2, 1), ("2020-01-02", 4, 1), ("2024-03-01", 2, 1)))
    assert list(pd.to_datetime(splits.dates).date.astype(str)) == ["2020-01-02", "2024-03-01"]
    assert list(splits.ratios) == [4.0, 2.0]
    dividends = normalize_dividends(_dividends(("2024-03-01", 0.5), ("2020-01-02", 0.1), ("2024-03-01", 0.2),
                                               ("2024-03-01", 0.5)))
    assert list(dividends.amounts) == [0.1, 0.2, 0.5] and len(dividends.dates) == 3
