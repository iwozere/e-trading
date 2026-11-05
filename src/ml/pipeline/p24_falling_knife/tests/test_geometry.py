"""Tests for P24 geometry: causal ZigZag, reference peak, channel width."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.pipeline.p24_falling_knife.geometry import (
    HIGH,
    LOW,
    channel_width,
    nan_argmax,
    nan_argmin,
    reference_peak_bars,
    swing_lows_between,
    zigzag,
)
from src.ml.pipeline.p24_falling_knife.tests.helpers import knife_bars


def test_zigzag_known_path() -> None:
    close = [10, 11, 12, 9, 8, 9, 10, 11, 7]
    swings = zigzag(close, 0.25)
    # 9 <= 12 * 0.75 confirms the 12 top on bar 3; 10 >= 8 * 1.25 confirms the 8 low on bar 6.
    assert [(s.kind, s.bar, s.confirmed_bar) for s in swings] == [(HIGH, 2, 3), (LOW, 4, 6), (HIGH, 7, 8)]


def test_zigzag_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        zigzag([1.0, 2.0], 0.0)


@pytest.mark.parametrize("pct", [0.08, 0.25])
def test_zigzag_is_causal(pct: float) -> None:
    close = knife_bars()["close"].to_numpy()
    full = zigzag(close, pct)
    for cut in range(50, len(close), 7):
        prefix = zigzag(close[:cut + 1], pct)
        assert prefix == [s for s in full if s.confirmed_bar <= cut]


def test_reference_peak_uses_confirmed_swing_high() -> None:
    close = np.array([10, 12, 15, 14, 11, 10, 9, 8], dtype=float)
    peaks = reference_peak_bars(close, zigzag(close, 0.25), lookback=252)
    # The 15 top is confirmed at bar 4 (11 <= 15 * 0.75); before that the rolling max stands in.
    assert peaks[4] == 2 and peaks[7] == 2
    assert peaks[1] == 1


def test_reference_peak_falls_back_to_window_max() -> None:
    close = np.linspace(10.0, 20.0, 30)
    peaks = reference_peak_bars(close, zigzag(close, 0.25), lookback=10)
    assert peaks[-1] == 29


def test_swing_lows_between_respects_confirmation() -> None:
    close = [10, 8, 9, 7, 9]
    swings = zigzag(close, 0.12)
    assert [s.bar for s in swing_lows_between(swings, 0, 4, known_at=2)] == [1]
    assert [s.bar for s in swing_lows_between(swings, 0, 4, known_at=4)] == [1, 3]
    assert swing_lows_between(swings, 1, 4, known_at=4)[0].bar == 3


def test_channel_width_extremes() -> None:
    adx = pd.Series([10.0, 20.0, 40.0, 30.0])
    ppo = pd.Series([1.0, -2.0, -4.0, -3.0])
    width = channel_width(adx, ppo, range_bars=10)
    assert np.isnan(width["w"].iloc[0])  # no negative PPO yet: no valley
    assert width["w"].iloc[2] == pytest.approx(0.0)  # ADX at its max, PPO at its floor
    assert width["w"].iloc[3] == pytest.approx((1 - 30 / 40) + (1 - 3 / 4))


def test_nan_arg_helpers_prefer_latest_and_skip_nan() -> None:
    values = np.array([np.nan, 3.0, 1.0, 3.0, 1.0])
    assert nan_argmax(values, 0, 5) == 3
    assert nan_argmin(values, 0, 5) == 4
    assert nan_argmax(values, 0, 1) is None
    assert nan_argmin(values, 3, 3) is None
