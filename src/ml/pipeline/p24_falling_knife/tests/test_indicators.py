"""Tests for P24 indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from src.ml.pipeline.p24_falling_knife.config import DAILY_PARAMS
from src.ml.pipeline.p24_falling_knife.indicators import INDICATOR_COLUMNS, compute_indicators, ppo
from src.ml.pipeline.p24_falling_knife.tests.helpers import knife_bars


def _manual_ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def test_ppo_matches_hand_computed_ema() -> None:
    close = knife_bars()["close"]
    frame = ppo(close, 12, 26, 9)
    values = close.to_numpy()
    line = 100.0 * (_manual_ema(values, 12) - _manual_ema(values, 26)) / _manual_ema(values, 26)
    np.testing.assert_allclose(frame["ppo"].to_numpy()[25:], line[25:], rtol=0, atol=1e-9)
    signal = _manual_ema(line[25:], 9)
    np.testing.assert_allclose(frame["ppo_signal"].to_numpy()[33:], signal[8:], rtol=0, atol=1e-9)
    np.testing.assert_allclose(frame["ppo_hist"].dropna(), (frame["ppo"] - frame["ppo_signal"]).dropna(), atol=1e-12)


def test_ppo_warm_up_is_nan() -> None:
    frame = ppo(knife_bars()["close"], 12, 26, 9)
    assert frame["ppo"].iloc[:25].isna().all()
    assert frame["ppo"].iloc[25:].notna().all()
    assert frame["ppo_signal"].iloc[:33].isna().all()
    assert frame["ppo_signal"].iloc[33:].notna().all()


def test_ppo_is_in_percent_and_scale_free() -> None:
    close = knife_bars()["close"]
    scaled = ppo(close * 1000.0, 12, 26, 9)["ppo"]
    pd.testing.assert_series_equal(ppo(close, 12, 26, 9)["ppo"], scaled, check_exact=False, atol=1e-9)


def test_compute_indicators_matches_talib() -> None:
    bars = knife_bars()
    out = compute_indicators(bars, DAILY_PARAMS)
    assert set(INDICATOR_COLUMNS) <= set(out.columns)
    h, low, c = (bars[k].to_numpy(dtype=np.float64) for k in ("high", "low", "close"))
    np.testing.assert_allclose(out["adx"], talib.ADX(h, low, c, timeperiod=14), equal_nan=True)
    np.testing.assert_allclose(out["cci"], talib.CCI(h, low, c, timeperiod=20), equal_nan=True)
    np.testing.assert_allclose(out["minus_di"], talib.MINUS_DI(h, low, c, timeperiod=14), equal_nan=True)
    assert len(out) == len(bars)
