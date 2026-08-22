"""
P21 Momentum — Regime overlay (docs/pipeline-specification.md §8).

Computed at signal date T, applied at execution T+1. **The 20-day VIX
smoothing is mandatory** — spot VIX triggers on single-day spikes and
produces expensive exposure chatter. Regime hysteresis is asymmetric:
downward changes to scalar apply immediately; upward changes require two
consecutive months in the new (raw) state. Protection engages fast,
disengages slowly, by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Tuple

import pandas as pd

from src.ml.pipeline.p21_momentum.config import (
    BEAR_LOOKBACK_DAYS,
    MA_DAYS,
    SCALAR_BEAR_HIGHVOL,
    SCALAR_BEAR_LOWVOL,
    SCALAR_NORMAL,
    UPGRADE_CONFIRMATION_MONTHS,
    VIX_SMOOTHING_DAYS,
    VIX_THRESHOLD,
)


@dataclass(slots=True)
class PriorRegimeState:
    """The minimal state compute_regime() needs from prior months, passed in explicitly.

    Kept out of the function's own I/O so compute_regime() stays pure and
    testable — the caller (jobs/run_monthly_rebalance.py) reads
    _state/regime_history.json and builds this from it, then appends the
    RegimeResult after the call.
    """

    scalar_applied: float
    months_at_state: int
    # (bear, high_vol) raw state for the most recent prior months, most
    # recent first, up to (upgrade_confirmation_months - 1) entries. Derived
    # directly from regime_history.json's own bear/high_vol fields — no
    # separate counter to keep in sync, which is what makes this safe under
    # the bit-identical-rerun requirement (spec §14.9 B10).
    recent_raw_states: List[Tuple[bool, bool]] = field(default_factory=list)


@dataclass(slots=True)
class RegimeResult:
    """Output of compute_regime() — backs one _state/regime_history.json entry."""

    date: str
    spx_12m_return: float
    spx_vs_200dma: float
    vix_20d_avg: float
    bear: bool
    high_vol: bool
    scalar_raw: float
    scalar_applied: float
    months_at_state: int


def _raw_scalar(bear: bool, high_vol: bool) -> float:
    if not bear:
        return SCALAR_NORMAL
    if bear and not high_vol:
        return SCALAR_BEAR_LOWVOL
    return SCALAR_BEAR_HIGHVOL


def compute_regime(
    spx: pd.Series,
    vix: pd.Series,
    prior_state: PriorRegimeState,
    as_of: date,
    bear_lookback_days: int = BEAR_LOOKBACK_DAYS,
    ma_days: int = MA_DAYS,
    vix_smoothing_days: int = VIX_SMOOTHING_DAYS,
    vix_threshold: float = VIX_THRESHOLD,
    upgrade_confirmation_months: int = UPGRADE_CONFIRMATION_MONTHS,
) -> RegimeResult:
    """
    Compute this month's regime scalar with asymmetric hysteresis.

    Args:
        spx: ^GSPC adjusted close series, ascending by date, most recent last.
            Must have at least bear_lookback_days + 1 bars.
        vix: ^VIX close series, ascending by date, most recent last. Must
            have at least vix_smoothing_days bars.
        prior_state: Prior months' state (read from _state/regime_history.json
            by the caller).
        as_of: Signal date T.
        bear_lookback_days, ma_days, vix_smoothing_days, vix_threshold,
            upgrade_confirmation_months: Spec §16 parameters.

    Returns:
        RegimeResult with both scalar_raw (this month's mechanical scalar)
        and scalar_applied (after hysteresis — the one actually used by
        strategy/sizing.py).
    """
    spx_12m_return = float(spx.iloc[-1] / spx.iloc[-bear_lookback_days] - 1.0)
    ma = spx.rolling(ma_days).mean().iloc[-1]
    spx_vs_200dma = float(spx.iloc[-1] / ma)
    bear = bool(spx_12m_return < 0 or spx.iloc[-1] < ma)

    vix_20d_avg = float(vix.iloc[-vix_smoothing_days:].mean())
    high_vol = bool(vix_20d_avg > vix_threshold)

    scalar_raw = _raw_scalar(bear, high_vol)
    raw_state = (bear, high_vol)

    # Asymmetric hysteresis: downward move (lower scalar) applies immediately;
    # upward move (higher scalar) requires N consecutive months in the new
    # raw state (including this one) before it takes effect.
    if scalar_raw <= prior_state.scalar_applied:
        scalar_applied = scalar_raw
        months_at_state = prior_state.months_at_state + 1 if scalar_raw == prior_state.scalar_applied else 1
    else:
        streak = 1  # this month counts as the first
        for past_raw_state in prior_state.recent_raw_states:
            if past_raw_state == raw_state:
                streak += 1
            else:
                break
        if streak >= upgrade_confirmation_months:
            scalar_applied = scalar_raw
            months_at_state = 1
        else:
            scalar_applied = prior_state.scalar_applied
            months_at_state = prior_state.months_at_state + 1

    return RegimeResult(
        date=as_of.isoformat(),
        spx_12m_return=spx_12m_return,
        spx_vs_200dma=spx_vs_200dma,
        vix_20d_avg=vix_20d_avg,
        bear=bear,
        high_vol=high_vol,
        scalar_raw=scalar_raw,
        scalar_applied=scalar_applied,
        months_at_state=months_at_state,
    )
