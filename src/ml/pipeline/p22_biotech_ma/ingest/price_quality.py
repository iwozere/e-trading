"""
P22 — price/share series sanity check against the 10-K public float (2026-10-27).

`dei:EntityPublicFloat` is the market value of non-affiliate shares on the last
day of the filer's second fiscal quarter, so a correct market cap on that same
day is at least the float (and rarely more than a few times it). Recomputing
`market_cap.compute_market_cap` AS OF the float's own measurement date and
dividing by the float therefore tests the price and share series themselves,
independent of how the stock moved afterwards.

The cross-check that motivated this (`docs/Design.md` "Size floor") found the
ratio far outside [0.5, 10] for three kinds of data problem, none of which a
market-cap number alone can reveal:

- corrupt vintage "raw" prices for some serial reverse-splitters (ACOR traded
  ~$30 but the FMP vintage holds ~$0.004 for 2006-2020; TNXP partly adjusted);
- ADRs, whose SEC share count is in ordinary shares (BeiGene: 13 per ADS);
- share counts from before a reverse merger.

A filer occasionally tags the float itself 1000x too large (Sarepta, Fate), which
looks exactly like a bad price. `size_floor.py` copes with that; for the M6
backtest a persistently inconsistent company is simply excluded, so a false
positive costs one company, not a wrong label.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import (
    FLOAT_CONSISTENCY_MAX_RATIO,
    FLOAT_CONSISTENCY_MIN_RATIO,
    PUBLIC_FLOAT_MAX_AGE_DAYS,
)
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import compute_market_cap
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class PublicFloat:
    """One 10-K public-float value."""

    value: float
    period_end: date


@dataclass(frozen=True)
class FloatConsistency:
    """Derived market cap vs the public float, both on the float's measurement date."""

    public_float: PublicFloat
    market_cap: float
    ratio: float
    consistent: bool


def public_floats_as_of(repo: Any, company_id: int, as_of: date) -> List[PublicFloat]:
    """
    Every positive public-float value known on `as_of`, newest measurement date first.

    Args:
        repo: A `P22Repo`-shaped object.
        company_id: The company.
        as_of: Lookahead bound (`known_from <= as_of`, via `get_financial_facts_as_of`).
    """
    floats = [
        PublicFloat(value=float(fact["value"]), period_end=fact["period_end"])
        for fact in repo.get_financial_facts_as_of(company_id, "public_float", as_of)
        if fact.get("value") is not None and float(fact["value"]) > 0 and fact.get("period_end") is not None
    ]
    return sorted(floats, key=lambda f: f.period_end, reverse=True)


def latest_public_float(repo: Any, company_id: int, as_of: date) -> Optional[PublicFloat]:
    """The newest public float known on `as_of`, or None when absent or older than `PUBLIC_FLOAT_MAX_AGE_DAYS`."""
    floats = public_floats_as_of(repo, company_id, as_of)
    if not floats or (as_of - floats[0].period_end).days > PUBLIC_FLOAT_MAX_AGE_DAYS:
        return None
    return floats[0]


def check_float_consistency(repo: Any, company_id: int, public_float: PublicFloat) -> Optional[FloatConsistency]:
    """
    Compare `compute_market_cap` on the float's measurement date with the float.

    Point-in-time safe: the market cap uses only data known on `public_float.period_end`,
    which is earlier than the date the float itself became known.

    Returns:
        None when no market cap can be derived for that date (nothing to compare).
    """
    result, _ = compute_market_cap(repo, company_id, public_float.period_end)
    if result is None:
        return None
    ratio = result.market_cap / public_float.value
    return FloatConsistency(
        public_float=public_float,
        market_cap=result.market_cap,
        ratio=ratio,
        consistent=FLOAT_CONSISTENCY_MIN_RATIO <= ratio <= FLOAT_CONSISTENCY_MAX_RATIO,
    )


def float_consistency_history(repo: Any, company_id: int, as_of: date) -> List[FloatConsistency]:
    """`check_float_consistency` for every public float known on `as_of`, newest first; uncomparable dates are left out."""
    checks = (check_float_consistency(repo, company_id, f) for f in public_floats_as_of(repo, company_id, as_of))
    return [c for c in checks if c is not None]


def summarize_consistency(checks: List[FloatConsistency]) -> Dict[str, Any]:
    """
    Per-company summary for reports: how many float dates were checked and how many failed, low vs high.

    Returns:
        `{"checked", "inconsistent", "too_low", "too_high", "median_ratio"}`; `median_ratio` is None when nothing was checked.
    """
    ratios = sorted(c.ratio for c in checks)
    median: Optional[float] = None
    if ratios:
        mid = len(ratios) // 2
        median = ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2
    return {
        "checked": len(checks),
        "inconsistent": sum(1 for c in checks if not c.consistent),
        "too_low": sum(1 for c in checks if c.ratio < FLOAT_CONSISTENCY_MIN_RATIO),
        "too_high": sum(1 for c in checks if c.ratio > FLOAT_CONSISTENCY_MAX_RATIO),
        "median_ratio": median,
    }
