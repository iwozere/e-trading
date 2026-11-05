"""
P24 Falling Knife — chart geometry (spec §4.2, §4.6, §4.7).

Swing points come from a causal ZigZag on closes: a swing high exists only from the bar on which price has closed
``pct`` below it (``confirmed_bar``), so a scan at bar ``t`` sees exactly the swings a chart reader would have seen.
The channel width ``W`` puts ADX and PPO on one scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

HIGH = "high"
LOW = "low"


@dataclass(frozen=True)
class Swing:
    """One ZigZag pivot."""

    kind: str
    bar: int
    price: float
    confirmed_bar: int


def zigzag(close: npt.ArrayLike, pct: float) -> List[Swing]:
    """
    Causal ZigZag on closes.

    A high is confirmed when a later close is at least ``pct`` below it, a low when a later close is at least ``pct``
    above it. Pivots alternate high/low. NaN closes are skipped.

    Args:
        close: Close prices in bar order.
        pct: Reversal threshold as a fraction (0.25 = 25%).

    Returns:
        Confirmed swings in bar order.

    Raises:
        ValueError: ``pct`` not in (0, 1).
    """
    if not 0.0 < pct < 1.0:
        raise ValueError(f"zigzag pct must be in (0, 1), got {pct}")
    values = np.asarray(close, dtype=np.float64)
    swings: List[Swing] = []
    trend = 0
    hi_idx = lo_idx = -1
    hi = lo = np.nan
    for i, c in enumerate(values):
        if np.isnan(c) or c <= 0.0:
            continue
        if hi_idx < 0:
            hi_idx = lo_idx = i
            hi = lo = c
            continue
        if trend >= 0:
            if c > hi:
                hi, hi_idx = c, i
            if trend == 0 and c < lo:
                lo, lo_idx = c, i
            if c <= hi * (1.0 - pct) and (trend > 0 or hi_idx < i):
                swings.append(Swing(HIGH, hi_idx, hi, i))
                trend, lo, lo_idx = -1, c, i
                continue
            if trend == 0 and c >= lo * (1.0 + pct):
                swings.append(Swing(LOW, lo_idx, lo, i))
                trend, hi, hi_idx = 1, c, i
        else:
            if c < lo:
                lo, lo_idx = c, i
            if c >= lo * (1.0 + pct):
                swings.append(Swing(LOW, lo_idx, lo, i))
                trend, hi, hi_idx = 1, c, i
    return swings


def last_confirmed_high_bars(swings: Sequence[Swing], n_bars: int) -> np.ndarray:
    """
    For every bar ``t``, the bar of the most recent swing high confirmed at or before ``t`` (−1 if none).

    Args:
        swings: Output of ``zigzag``.
        n_bars: Length of the series.

    Returns:
        Integer array of length ``n_bars``.
    """
    out = np.full(n_bars, -1, dtype=np.int64)
    for swing in swings:
        if swing.kind == HIGH and swing.confirmed_bar < n_bars:
            out[swing.confirmed_bar:] = swing.bar
    return out


def reference_peak_bars(close: np.ndarray, swings: Sequence[Swing], lookback: int) -> np.ndarray:
    """
    Reference peak P1 per bar (spec §4.2).

    The most recent confirmed swing high within the last ``lookback`` bars; if there is none, the highest close in
    that window (ties: the latest bar).

    Args:
        close: Close prices.
        swings: ZigZag swings with the peak threshold.
        lookback: Window in bars.

    Returns:
        Integer array of peak bars (−1 while no close is available yet).
    """
    n = len(close)
    confirmed = last_confirmed_high_bars(swings, n)
    bars = np.arange(n)
    use_swing = (confirmed >= 0) & (bars - confirmed <= lookback)
    out = np.where(use_swing, confirmed, -1).astype(np.int64)
    filled = np.where(np.isnan(close), -np.inf, close)
    for t in np.flatnonzero(~use_swing):
        start = max(0, int(t) - lookback)
        window = filled[start:t + 1]
        if np.isfinite(window).any():
            out[t] = start + int(len(window) - 1 - np.argmax(window[::-1]))
    return out


def low_since_peak(close: np.ndarray, peak_bars: np.ndarray) -> np.ndarray:
    """
    For every bar ``t``, the bar of the lowest close in ``(peak_bars[t], t]`` (latest on ties; −1 if undefined).

    Vectorized per run of equal ``peak_bars`` values.

    Args:
        close: Closes (NaN never wins).
        peak_bars: Reference peak per bar (``reference_peak_bars``).

    Returns:
        Integer array of low bars.
    """
    n = len(close)
    out = np.full(n, -1, dtype=np.int64)
    values = np.where(np.isnan(close), np.inf, close)
    change = np.flatnonzero(np.diff(peak_bars) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    for a, b in zip(starts, ends):
        p = int(peak_bars[a])
        if p < 0 or p + 1 >= b:
            continue
        seg = values[p + 1:b]
        running = np.minimum.accumulate(seg)
        positions = np.where(seg <= running, np.arange(seg.size), -1)
        latest = np.maximum.accumulate(positions)
        first = max(a, p + 1)
        offset = first - (p + 1)
        lows = latest[offset:] + p + 1
        valid = np.isfinite(running[offset:])
        out[first:b] = np.where(valid, lows, -1)
    return out


def swing_lows_between(swings: Sequence[Swing], after_bar: int, before_bar: int, known_at: int) -> List[Swing]:
    """
    Swing lows with ``after_bar < bar < before_bar`` that were confirmed by ``known_at``.

    Args:
        swings: ZigZag swings (support threshold).
        after_bar: Exclusive lower bound.
        before_bar: Exclusive upper bound.
        known_at: Bar of evaluation; later confirmations are invisible.

    Returns:
        Matching lows in bar order.
    """
    return [s for s in swings if s.kind == LOW and after_bar < s.bar < before_bar and s.confirmed_bar <= known_at]


def channel_width(adx: pd.Series, ppo: pd.Series, range_bars: int) -> pd.DataFrame:
    """
    Normalized ADX/PPO channel width (spec §4.6).

    ``a = ADX / rolling max ADX``, ``p = PPO / |rolling min PPO|``, ``W = (1 − a) + (1 + p)``. ``W`` is NaN while the
    rolling PPO floor is not negative (no valley exists). Rolling windows use ``min_periods=1``, so early bars use the
    history available so far.

    Args:
        adx: ADX series.
        ppo: PPO series.
        range_bars: Context window in bars.

    Returns:
        Frame with ``adx_norm``, ``ppo_floor``, ``ppo_norm``, ``w``.
    """
    adx_max = adx.rolling(range_bars, min_periods=1).max()
    ppo_floor = ppo.rolling(range_bars, min_periods=1).min()
    floor_abs = ppo_floor.abs().where(ppo_floor < 0)
    adx_norm = adx / adx_max.where(adx_max > 0)
    ppo_norm = ppo / floor_abs
    width = (1.0 - adx_norm) + (1.0 + ppo_norm)
    return pd.DataFrame({"adx_norm": adx_norm, "ppo_floor": ppo_floor, "ppo_norm": ppo_norm, "w": width},
                        index=adx.index)


def nan_argmax(values: np.ndarray, start: int, stop: int) -> Optional[int]:
    """
    Index (absolute) of the maximum of ``values[start:stop]``, latest on ties; ``None`` if all NaN or empty.

    Args:
        values: Array.
        start: Inclusive start.
        stop: Exclusive stop.

    Returns:
        Absolute index or ``None``.
    """
    return _nan_arg(values, start, stop, maximum=True)


def nan_argmin(values: np.ndarray, start: int, stop: int) -> Optional[int]:
    """
    Index (absolute) of the minimum of ``values[start:stop]``, latest on ties; ``None`` if all NaN or empty.

    Args:
        values: Array.
        start: Inclusive start.
        stop: Exclusive stop.

    Returns:
        Absolute index or ``None``.
    """
    return _nan_arg(values, start, stop, maximum=False)


def _nan_arg(values: np.ndarray, start: int, stop: int, maximum: bool) -> Optional[int]:
    start = max(start, 0)
    window = values[start:stop]
    if window.size == 0:
        return None
    reversed_window = window[::-1]
    filled = np.where(np.isnan(reversed_window), -np.inf if maximum else np.inf, reversed_window)
    pos = int(np.argmax(filled) if maximum else np.argmin(filled))
    if np.isnan(reversed_window[pos]):
        return None
    return start + window.size - 1 - pos


def window_min(values: np.ndarray, start: int, stop: int) -> float:
    """
    NaN-aware minimum of ``values[start:stop]`` (NaN if empty or all NaN).

    Args:
        values: Array.
        start: Inclusive start (clipped at 0).
        stop: Exclusive stop.

    Returns:
        Minimum or NaN.
    """
    idx = nan_argmin(values, start, stop)
    return float("nan") if idx is None else float(values[idx])


def window_max(values: np.ndarray, start: int, stop: int) -> float:
    """
    NaN-aware maximum of ``values[start:stop]`` (NaN if empty or all NaN).

    Args:
        values: Array.
        start: Inclusive start (clipped at 0).
        stop: Exclusive stop.

    Returns:
        Maximum or NaN.
    """
    idx = nan_argmax(values, start, stop)
    return float("nan") if idx is None else float(values[idx])
