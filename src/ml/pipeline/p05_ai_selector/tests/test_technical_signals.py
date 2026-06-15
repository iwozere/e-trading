"""Tests for technical signal computation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.ml.pipeline.p05_ai_selector.signals.technical import (
    compute_rsi,
    compute_sma,
    compute_volume_surge_ratio,
    score_technicals,
)


def _make_ohlcv(n: int = 60, trend: str = "up") -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame."""
    base = 100.0
    closes = [base + (i if trend == "up" else -i) for i in range(n)]
    closes = [max(0.01, c) for c in closes]
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * n,
    })


class TestComputeRsi:
    def test_rsi_oversold_scores(self):
        """Declining price series yields RSI < 30."""
        ohlcv = _make_ohlcv(60, trend="down")
        rsi = compute_rsi(ohlcv["close"])
        assert rsi < 30

    def test_rsi_overbought(self):
        """Steadily rising prices yield RSI > 70."""
        ohlcv = _make_ohlcv(60, trend="up")
        rsi = compute_rsi(ohlcv["close"])
        assert rsi > 70

    def test_rsi_insufficient_data(self):
        """With fewer than 15 rows, RSI returns 50.0 (neutral)."""
        prices = pd.Series([100.0, 101.0, 99.0])
        assert compute_rsi(prices) == 50.0


class TestVolumeSurge:
    def test_volume_surge_scores(self):
        """When last volume is 2× the average, ratio > 1.5."""
        volumes = pd.Series([1_000_000.0] * 20 + [2_000_000.0])
        ratio = compute_volume_surge_ratio(volumes)
        assert ratio > 1.5

    def test_no_surge_ratio_near_one(self):
        """Constant volume returns a ratio near 1."""
        volumes = pd.Series([1_000_000.0] * 21)
        ratio = compute_volume_surge_ratio(volumes)
        assert 0.9 <= ratio <= 1.1


class TestSmaCrossover:
    def test_sma_crossover_bullish(self):
        """Rising trend: price > SMA20 > SMA50 → bullish signal fires."""
        ohlcv = _make_ohlcv(70, trend="up")
        score, breakdown = score_technicals(ohlcv)
        assert breakdown["sma_crossover_bullish"] is True
        assert score > 0

    def test_sma_crossover_bearish(self):
        """Declining trend: price < SMA20 < SMA50 → bearish signal fires."""
        ohlcv = _make_ohlcv(70, trend="down")
        score, breakdown = score_technicals(ohlcv)
        assert breakdown["sma_crossover_bearish"] is True

    def test_missing_data_returns_zero(self):
        """Empty DataFrame returns (0.0, {})."""
        score, breakdown = score_technicals(pd.DataFrame())
        assert score == 0.0
        assert breakdown == {}
