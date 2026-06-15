"""Technical signal computation — pure functions, no side effects."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Compute simple moving average.

    Args:
        prices: Close price series.
        window: Rolling window size.

    Returns:
        SMA series (same index as prices).
    """
    return prices.rolling(window=window, min_periods=window).mean()


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Compute RSI using Wilder's exponential smoothing.

    Args:
        prices: Close price series (at least period+1 rows).
        period: RSI look-back period.

    Returns:
        Latest RSI value (0–100), or 50.0 if insufficient data.
    """
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_loss = float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0
    rs = float(gain.iloc[-1]) / last_loss
    return round(100 - 100 / (1 + rs), 2)


def compute_volume_surge_ratio(volumes: pd.Series, window: int = 20) -> float:
    """
    Return today's volume divided by the rolling mean over the window.

    Args:
        volumes: Volume series.
        window: Rolling average window.

    Returns:
        Surge ratio, or 1.0 if the rolling mean is zero/unavailable.
    """
    if len(volumes) < 2:
        return 1.0
    avg = volumes.iloc[:-1].rolling(window=window, min_periods=5).mean().iloc[-1]
    if avg is None or avg == 0 or np.isnan(float(avg)):
        return 1.0
    return round(float(volumes.iloc[-1]) / float(avg), 4)


def compute_momentum_pct(prices: pd.Series, days: int = 5) -> float:
    """
    Return percentage price change over the last `days` periods.

    Args:
        prices: Close price series.
        days: Number of periods to look back.

    Returns:
        Momentum % (e.g. 3.4 for +3.4%), or 0.0 if insufficient data.
    """
    if len(prices) < days + 1:
        return 0.0
    old_price = float(prices.iloc[-(days + 1)])
    new_price = float(prices.iloc[-1])
    if old_price == 0:
        return 0.0
    return round((new_price - old_price) / old_price * 100, 4)


def compute_atr_compression(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short: int = 5,
    long: int = 20,
) -> bool:
    """
    Return True when short-window ATR is below 70 % of long-window ATR (consolidation setup).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        short: Short ATR window.
        long: Long ATR window.

    Returns:
        True when short ATR < 0.7 × long ATR.
    """
    if len(close) < long + 1:
        return False
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift()).abs(),
        "lc": (low - close.shift()).abs(),
    }).max(axis=1)
    short_atr = float(tr.rolling(short, min_periods=short).mean().iloc[-1])
    long_atr = float(tr.rolling(long, min_periods=long).mean().iloc[-1])
    if long_atr == 0 or np.isnan(long_atr) or np.isnan(short_atr):
        return False
    return short_atr < 0.7 * long_atr


def compute_52w_proximity(prices: pd.Series) -> Tuple[float, float]:
    """
    Compute how far the latest price is from 52-week high and low.

    Args:
        prices: Close price series (up to 252 days).

    Returns:
        Tuple (pct_from_high, pct_from_low) as negative/positive percentages.
        E.g. (-8.2, 34.1) means 8.2 % below 52w-high, 34.1 % above 52w-low.
    """
    window = prices.iloc[-252:] if len(prices) >= 252 else prices
    if window.empty:
        return (0.0, 0.0)
    high_52w = float(window.max())
    low_52w = float(window.min())
    last = float(prices.iloc[-1])
    pct_from_high = round((last - high_52w) / high_52w * 100, 2) if high_52w else 0.0
    pct_from_low = round((last - low_52w) / low_52w * 100, 2) if low_52w else 0.0
    return (pct_from_high, pct_from_low)


def score_technicals(
    ohlcv: pd.DataFrame,
    weights: Optional[Dict[str, int]] = None,
) -> Tuple[float, Dict[str, object]]:
    """
    Compute all technical signals and return (total_score, breakdown_dict).

    Args:
        ohlcv: DataFrame with columns: open, high, low, close, volume.
        weights: Signal weight overrides. Defaults to spec §6.1 values.

    Returns:
        Tuple of (total_score, signal_breakdown).
    """
    from src.ml.pipeline.p05_ai_selector.config import TECHNICAL_WEIGHTS
    w = weights or TECHNICAL_WEIGHTS

    if ohlcv.empty or len(ohlcv) < 10:
        return (0.0, {})

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    last_close = float(close.iloc[-1])
    last_sma20 = float(sma20.iloc[-1]) if not sma20.dropna().empty else None
    last_sma50 = float(sma50.iloc[-1]) if not sma50.dropna().empty else None

    rsi = compute_rsi(close)
    vol_surge_ratio = compute_volume_surge_ratio(volume)
    momentum_5d = compute_momentum_pct(close)
    atr_comp = compute_atr_compression(high, low, close)
    pct_from_high, pct_from_low = compute_52w_proximity(close)

    sma_bullish = (
        last_sma20 is not None and last_sma50 is not None
        and last_close > last_sma20 > last_sma50
    )
    sma_bearish = (
        last_sma20 is not None and last_sma50 is not None
        and last_close < last_sma20 < last_sma50
    )
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    volume_surge = vol_surge_ratio > 1.5
    momentum_top = momentum_5d > 0
    near_52w_high = pct_from_high >= -5.0
    near_52w_low = pct_from_low <= 5.0

    breakdown: Dict[str, object] = {
        "sma_crossover_bullish": sma_bullish,
        "sma_crossover_bearish": sma_bearish,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "volume_surge": volume_surge,
        "momentum_5d_positive": momentum_top,
        "atr_compression": atr_comp,
        "near_52w_high": near_52w_high,
        "near_52w_low": near_52w_low,
        "rsi_14": rsi,
        "sma20_above_sma50": sma_bullish,
        "volume_surge_ratio": vol_surge_ratio,
        "momentum_5d_pct": momentum_5d,
        "pct_from_52w_high": pct_from_high,
        "pct_from_52w_low": pct_from_low,
    }

    score = 0.0
    if sma_bullish:
        score += w.get("sma_crossover_bullish", 15)
    elif sma_bearish:
        score += w.get("sma_crossover_bearish", 10)
    if rsi_oversold:
        score += w.get("rsi_oversold", 12)
    elif rsi_overbought:
        score += w.get("rsi_overbought", 8)
    if volume_surge:
        score += w.get("volume_surge", 15)
    if momentum_top:
        score += w.get("momentum_5d", 10)
    if atr_comp:
        score += w.get("atr_compression", 5)
    if near_52w_high:
        score += w.get("near_52w_high", 8)
    if near_52w_low:
        score += w.get("near_52w_low", 8)

    return (score, breakdown)
