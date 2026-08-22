"""
P21 Momentum — Four/five-track attribution (docs/pipeline-specification.md §9).

All tracks computed simultaneously from the same data, starting from an
identical notional NAV. Costs apply to A and B (commissions + slippage,
already embedded in their NAV series by execution/); TER applies to C and D
(0.20%/252 daily on position value, applied here since neither is a real
simulated-fill track).

Decomposition:
    B - C  = stock selection effect      (20 names vs. ~500)
    A - B  = overlay effect on stocks
    D - C  = overlay effect on the ETF
    A - D  = total DIY benefit over "QDVA + overlay"  <- the only number
             that answers the real question (spec §9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from src.ml.pipeline.p21_momentum.config import TER_ADJUSTMENT_ANNUAL

CostModel = Literal["commissions_slippage", "ter"]

TRADING_DAYS_PER_YEAR = 252


@dataclass(slots=True)
class Track:
    """
    One of the five attribution tracks (A/B/C/D/E).

    cost_model distinguishes the two ways a track's daily return is
    adjusted: "commissions_slippage" tracks (A, B) already have costs baked
    into their NAV series by execution/fills.py's simulated fills — this
    dataclass applies no further adjustment for them. "ter" tracks (C, D)
    are a raw index/ETF return series with the daily TER drag applied here.
    E (SPY) uses neither adjustment — it is a pure anchor.
    """

    name: str  # "A", "B", "C", "D", "E"
    cost_model: CostModel
    nav: Dict[str, float]  # {date_iso: nav}


def apply_ter_drag(daily_returns: Dict[str, float], annual_ter: float = TER_ADJUSTMENT_ANNUAL) -> Dict[str, float]:
    """
    Subtract the daily TER drag from a series of raw daily returns.

    Args:
        daily_returns: {date_iso: raw daily return}, ascending date order.
        annual_ter: Annual TER adjustment (spec: 0.0005, MTUM 0.15% vs QDVA
            0.20%).

    Returns:
        {date_iso: adjusted daily return}, each reduced by annual_ter / 252.
    """
    daily_drag = annual_ter / TRADING_DAYS_PER_YEAR
    return {d: r - daily_drag for d, r in daily_returns.items()}


def build_nav_series(daily_returns: Dict[str, float], initial_nav: float) -> Dict[str, float]:
    """
    Compound a daily-return series into a NAV series starting at initial_nav.

    Args:
        daily_returns: {date_iso: daily return}, ascending date order
            (Python 3.7+ dict preserves insertion order — callers must pass
            dates already sorted for a correct compounding path).
        initial_nav: Starting NAV (spec: 250,000 for every track).

    Returns:
        {date_iso: nav}, same keys as daily_returns, compounded.
    """
    nav: Dict[str, float] = {}
    running = initial_nav
    for d, r in daily_returns.items():
        running = running * (1 + r)
        nav[d] = running
    return nav


@dataclass(slots=True)
class AttributionResult:
    """The four spec §9 decomposition figures, plus the raw track NAVs, for one date."""

    date: str
    nav_a: float
    nav_b: float
    nav_c: float
    nav_d: float
    nav_e: float
    stock_selection_effect: float  # B - C
    overlay_effect_on_stocks: float  # A - B
    overlay_effect_on_etf: float  # D - C
    total_diy_benefit: float  # A - D  <- the answer to the real question


def compute_attribution(
    nav_a: float, nav_b: float, nav_c: float, nav_d: float, nav_e: float, as_of: str, initial_nav: float
) -> AttributionResult:
    """
    Compute the §9 decomposition for one date, expressed as % return since inception.

    Args:
        nav_a..nav_e: Each track's NAV on `as_of`.
        as_of: ISO date.
        initial_nav: The common starting NAV all five tracks share (spec: 250,000).

    Returns:
        AttributionResult with cumulative-return-based differences (not raw
        dollar NAV differences), since that is what §9's decomposition and
        §12.5's attribution table both report.
    """

    def cum_return(nav: float) -> float:
        return nav / initial_nav - 1.0

    ra, rb, rc, rd = cum_return(nav_a), cum_return(nav_b), cum_return(nav_c), cum_return(nav_d)
    return AttributionResult(
        date=as_of,
        nav_a=nav_a,
        nav_b=nav_b,
        nav_c=nav_c,
        nav_d=nav_d,
        nav_e=nav_e,
        stock_selection_effect=rb - rc,
        overlay_effect_on_stocks=ra - rb,
        overlay_effect_on_etf=rd - rc,
        total_diy_benefit=ra - rd,
    )
