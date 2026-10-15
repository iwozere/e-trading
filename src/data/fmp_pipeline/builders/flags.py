"""
Suspect-data flags per price bar (spec §7.5, D11: flag, never alter). Pure numpy, deterministic, no I/O.

==== ===================== ==========================================================================================
Bit  Name                  Rule
==== ===================== ==========================================================================================
0    ``OHLC_VIOLATION``    ``low > min(open, close)``, ``high < max(open, close)``, or a price missing or ``<= 0``
1    ``SPIKE_REVERTED``    ``|r| >= 35%``, robust z-score against the prior 60 returns ``>= 8``, reverted within one
                           or two bars (``|r_t + r_t+1 (+ r_t+2)| < 0.5 |r_t|``), and no split or dividend within
                           ±2 sessions. The outlier bar is flagged, plus the next bar when the reversion takes two.
2    ``JUMP_NO_SPLIT``     ``close_prev / close`` within 3% of 2, 3, 4, 5, 10, 3/2 or their inverses, with no split
                           within ±2 sessions (a probable missing split)
3    ``STALE``             a run of at least N bars with identical OHLC and zero volume (every bar of the run)
4    ``ZERO_VOLUME``       volume 0 while ``high > low``
==== ===================== ==========================================================================================

Daily bars get all five flags. Hourly bars get 0, 3 and 4: the spike and jump rules are calibrated on daily
returns, and a split shows up in the daily series anyway (D45).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data.fmp_pipeline.planner.config import FlagConfig

OHLC_VIOLATION = 1 << 0
SPIKE_REVERTED = 1 << 1
JUMP_NO_SPLIT = 1 << 2
STALE = 1 << 3
ZERO_VOLUME = 1 << 4
FLAG_NAMES = {OHLC_VIOLATION: "OHLC_VIOLATION", SPIKE_REVERTED: "SPIKE_REVERTED", JUMP_NO_SPLIT: "JUMP_NO_SPLIT",
              STALE: "STALE", ZERO_VOLUME: "ZERO_VOLUME"}
MAD_SCALE = 1.4826


@dataclass(frozen=True)
class Jump:
    """A ``JUMP_NO_SPLIT`` hit, for ``missing_splits_candidates.csv``."""

    index: int
    implied_ratio: float
    matched_ratio: float


def near_events(dates: np.ndarray, events: Optional[np.ndarray], window: int) -> np.ndarray:
    """
    Mark the bars within ``window`` sessions of any event.

    An event is placed on the first bar on or after its date; events before the first bar or after the last bar
    mark nothing.

    Args:
        dates: Ascending bar dates (``datetime64[D]``).
        events: Event dates (``datetime64[D]``), any order; ``None`` for none.
        window: Sessions on each side.

    Returns:
        Boolean mask, one per bar.
    """
    near = np.zeros(len(dates), dtype=bool)
    if events is None or len(events) == 0 or len(dates) == 0:
        return near
    events = events[(events >= dates[0]) & (events <= dates[-1])]
    for pos in np.searchsorted(dates, events, side="left"):
        near[max(0, pos - window): pos + window + 1] = True
    return near


def ohlc_violation(o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Bit 0: inconsistent OHLC, or any price missing or not positive."""
    prices = np.vstack([o, h, low, c])
    with np.errstate(invalid="ignore"):
        bad_price = np.isnan(prices).any(axis=0) | (prices <= 0).any(axis=0)
        inconsistent = (low > np.fmin(o, c)) | (h < np.fmax(o, c))
    return bad_price | inconsistent


def _returns(c: np.ndarray) -> np.ndarray:
    r = np.full(len(c), np.nan)
    if len(c) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            prev = c[:-1]
            r[1:] = np.where((prev > 0) & np.isfinite(prev) & np.isfinite(c[1:]), c[1:] / prev - 1.0, np.nan)
    return r


def spike_reverted(c: np.ndarray, near_event: np.ndarray, cfg: FlagConfig) -> np.ndarray:
    """Bit 1: single-bar outliers that revert within one or two bars, away from corporate events."""
    r = _returns(c)
    flags = np.zeros(len(c), dtype=bool)
    with np.errstate(invalid="ignore"):
        candidates = np.flatnonzero(np.abs(r) >= cfg.spike_min_return)
    for i in candidates:
        if near_event[i] or i + 1 >= len(c):
            continue
        prior = r[max(1, i - cfg.spike_lookback): i]
        prior = prior[np.isfinite(prior)]
        if len(prior) < cfg.spike_min_history:
            continue
        median = np.median(prior)
        mad = np.median(np.abs(prior - median)) * MAD_SCALE
        deviation = abs(r[i] - median)
        z = np.inf if mad == 0 and deviation > 0 else (deviation / mad if mad > 0 else 0.0)
        if z < cfg.spike_z:
            continue
        limit = cfg.spike_revert_ratio * abs(r[i])
        one = r[i] + r[i + 1]
        if np.isfinite(one) and abs(one) < limit:
            flags[i] = True
        elif i + 2 < len(c) and np.isfinite(one + r[i + 2]) and abs(one + r[i + 2]) < limit:
            flags[i] = flags[i + 1] = True
    return flags


def jumps_without_split(c: np.ndarray, near_split: np.ndarray, cfg: FlagConfig) -> list[Jump]:
    """Bit 2: close-to-close ratios near a split ratio with no split nearby."""
    found: list[Jump] = []
    ratios = np.asarray(cfg.jump_ratios, dtype=float)
    for i in range(1, len(c)):
        prev, cur = c[i - 1], c[i]
        if near_split[i] or not (np.isfinite(prev) and np.isfinite(cur)) or prev <= 0 or cur <= 0:
            continue
        implied = prev / cur
        close = np.abs(implied / ratios - 1.0) <= cfg.jump_tolerance
        if close.any():
            found.append(Jump(i, float(implied), float(ratios[np.argmax(close)])))
    return found


def stale(o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray, v: np.ndarray, min_bars: int) -> np.ndarray:
    """Bit 3: runs of at least ``min_bars`` bars with identical OHLC and zero volume."""
    n = len(c)
    if n == 0:
        return np.zeros(0, dtype=bool)
    zero = v == 0
    same = np.zeros(n, dtype=bool)
    same[1:] = (o[1:] == o[:-1]) & (h[1:] == h[:-1]) & (low[1:] == low[:-1]) & (c[1:] == c[:-1])
    cont = zero & np.concatenate([[False], zero[:-1]]) & same
    run_id = np.cumsum(~cont)
    lengths = np.bincount(run_id[zero], minlength=run_id.max() + 1)
    return zero & (lengths[run_id] >= min_bars)


def zero_volume(h: np.ndarray, low: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bit 4: no volume on a bar with a real price range."""
    with np.errstate(invalid="ignore"):
        return (v == 0) & (h > low)


def compute_flags(
    dates: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    cfg: FlagConfig,
    daily: bool = True,
    split_dates: Optional[np.ndarray] = None,
    dividend_dates: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[Jump]]:
    """
    Compute the flag bitmask of one security's bars.

    Args:
        dates: Ascending session dates (``datetime64[D]``); for hourly bars the exchange-local date of each bar.
        o, h, low, c: Raw prices (float, NaN for missing).
        v: Volume (float, NaN for missing).
        cfg: Thresholds.
        daily: Daily bars (all flags) or intraday (bits 0, 3, 4).
        split_dates: Split effective dates.
        dividend_dates: Dividend ex-dates.

    Returns:
        ``(int32 flags per bar, JUMP_NO_SPLIT hits)``.
    """
    flags = np.zeros(len(c), dtype=np.int32)
    flags[ohlc_violation(o, h, low, c)] |= OHLC_VIOLATION
    flags[stale(o, h, low, c, v, cfg.stale_bars_daily if daily else cfg.stale_bars_intraday)] |= STALE
    flags[zero_volume(h, low, v)] |= ZERO_VOLUME
    jumps: list[Jump] = []
    if daily:
        near_split = near_events(dates, split_dates, cfg.event_window)
        events = [e for e in (split_dates, dividend_dates) if e is not None and len(e)]
        near_any = near_events(dates, np.concatenate(events) if events else None, cfg.event_window)
        flags[spike_reverted(c, near_any, cfg)] |= SPIKE_REVERTED
        jumps = jumps_without_split(c, near_split, cfg)
        for jump in jumps:
            flags[jump.index] |= JUMP_NO_SPLIT
    return flags, jumps
