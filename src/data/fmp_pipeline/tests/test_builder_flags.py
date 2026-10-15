"""Synthetic unit tests for the suspect-data flags (spec §7.5): each flag fires on its case and not on near misses."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.fmp_pipeline.builders.flags import (
    JUMP_NO_SPLIT,
    OHLC_VIOLATION,
    SPIKE_REVERTED,
    STALE,
    ZERO_VOLUME,
    Jump,
    compute_flags,
    near_events,
)
from src.data.fmp_pipeline.planner.config import FlagConfig

CFG = FlagConfig()


def _dates(n: int) -> np.ndarray:
    start = np.datetime64("2020-01-01", "D")
    return np.arange(start, start + np.timedelta64(n, "D"))


def _series(closes: list[float], volume: float = 1000.0) -> tuple[np.ndarray, ...]:
    c = np.array(closes, dtype=float)
    return _dates(len(c)), c, c * 1.01, c * 0.99, c, np.full(len(c), volume)


def _noisy(n: int = 80, level: float = 100.0) -> list[float]:
    rng = np.random.default_rng(7)
    return list(level * np.cumprod(1 + rng.normal(0, 0.01, n)))


def _flags(closes: list[float], **kwargs: Any) -> tuple[np.ndarray, list[Jump]]:
    d, o, h, low, c, v = _series(closes)
    return compute_flags(d, o, h, low, c, v, CFG, **kwargs)


def test_clean_series_has_no_flags():
    flags, jumps = _flags(_noisy())
    assert not flags.any() and jumps == []


def test_ohlc_violation_and_bad_prices():
    d, o, h, low, c, v = _series([10.0, 10.0, 10.0, 10.0])
    low[1] = 10.5  # low above open/close
    h[2] = 9.0  # high below
    c[3] = 0.0  # non-positive
    o[0] = np.nan
    flags, _ = compute_flags(d, o, h, low, c, v, CFG, daily=False)
    assert all(flags & OHLC_VIOLATION)


def test_spike_reverted_one_and_two_bars():
    closes = _noisy()
    closes[70] *= 1.6  # one-bar spike, back the next day
    flags, _ = _flags(closes)
    assert flags[70] & SPIKE_REVERTED and not flags[71] & SPIKE_REVERTED

    closes = _noisy()
    closes[70] *= 1.6
    closes[71] = closes[70] * 1.01  # stays up one more bar, then back
    flags, _ = _flags(closes)
    assert flags[70] & SPIKE_REVERTED and flags[71] & SPIKE_REVERTED


def test_real_crash_is_not_a_spike():
    closes = _noisy()
    closes[70:] = [c * 0.5 for c in closes[70:]]  # drops and stays down
    flags, jumps = _flags(closes)
    assert not (flags & SPIKE_REVERTED).any()
    assert flags[70] & JUMP_NO_SPLIT and jumps[0].matched_ratio == 2.0  # a halving looks like a 2:1 split


def test_spike_near_a_dividend_or_without_history_is_not_flagged():
    closes = _noisy()
    closes[70] *= 1.6
    dates = _dates(len(closes))
    flags, _ = _flags(closes, dividend_dates=dates[[72]])
    assert not flags[70] & SPIKE_REVERTED
    short = _noisy(15)
    short[10] *= 1.6
    flags, _ = _flags(short)
    assert not (flags & SPIKE_REVERTED).any()  # fewer than spike_min_history prior returns


def test_spike_against_a_flat_history_has_infinite_z():
    closes = [10.0] * 40 + [16.0, 10.0, 10.0]
    flags, _ = _flags(closes)
    assert flags[40] & SPIKE_REVERTED


def test_jump_without_split_and_with_split():
    closes = _noisy()
    closes[60:] = [c / 4 for c in closes[60:]]
    flags, jumps = _flags(closes)
    assert flags[60] & JUMP_NO_SPLIT and jumps[0].index == 60 and jumps[0].matched_ratio == 4.0
    assert abs(jumps[0].implied_ratio - 4.0) < 0.2
    flags, jumps = _flags(closes, split_dates=_dates(len(closes))[[61]])
    assert not flags[60] & JUMP_NO_SPLIT and jumps == []
    reverse = _noisy()
    reverse[60:] = [c * 10 for c in reverse[60:]]
    _, jumps = _flags(reverse)
    assert jumps[0].matched_ratio == 0.1


def test_stale_needs_a_full_run_of_identical_zero_volume_bars():
    d, o, h, low, c, v = _series(_noisy(20))
    for i in range(5, 10):
        o[i], h[i], low[i], c[i], v[i] = 5.0, 5.0, 5.0, 5.0, 0.0
    flags, _ = compute_flags(d, o, h, low, c, v, CFG)  # daily: N = 5
    assert all(flags[5:10] & STALE) and not (flags[:5] & STALE).any() and not (flags[10:] & STALE).any()
    flags, _ = compute_flags(d, o, h, low, c, v, CFG, daily=False)  # intraday: N = 35
    assert not (flags & STALE).any()
    v[7] = 1.0  # breaks the run into 2 + 2
    flags, _ = compute_flags(d, o, h, low, c, v, CFG)
    assert not (flags & STALE).any()


def test_zero_volume_only_with_a_price_range():
    d, o, h, low, c, v = _series([10.0, 10.0, 10.0])
    v[:] = [0.0, 0.0, np.nan]
    h[1] = low[1] = o[1] = c[1] = 10.0  # degenerate bar
    flags, _ = compute_flags(d, o, h, low, c, v, CFG, daily=False)
    assert flags[0] & ZERO_VOLUME and not flags[1] & ZERO_VOLUME and not flags[2] & ZERO_VOLUME


def test_near_events_ignores_events_outside_the_series():
    dates = _dates(10)
    assert not near_events(dates, np.array(["2019-01-01", "2021-01-01"], dtype="datetime64[D]"), 2).any()
    mask = near_events(dates, np.array(["2020-01-05"], dtype="datetime64[D]"), 2)
    assert list(np.flatnonzero(mask)) == [2, 3, 4, 5, 6]
    assert not near_events(dates[:0], dates, 2).any() and not near_events(dates, None, 2).any()


def test_intraday_skips_spike_and_jump():
    closes = _noisy()
    closes[60:] = [c / 4 for c in closes[60:]]
    flags, jumps = _flags(closes, daily=False)
    assert not (flags & (JUMP_NO_SPLIT | SPIKE_REVERTED)).any() and jumps == []
