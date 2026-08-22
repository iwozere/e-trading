"""
P21 Momentum — Filters F1-F6 (docs/pipeline-specification.md §5).

Order matters: cheap filters first, expensive (network) filters only on
survivors. Each filter is a pure function returning a FilterResult; callers
compose them via run_all(). F1/F2/F3/F5 exclude on failure; F4 always
passes (flags only, spec §5 "on missing data: Pass + flag" and the general
loose-quality-filter design); F6 only excludes NEW entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.ml.pipeline.p21_momentum.config import (
    ADV_WINDOW_DAYS,
    EARNINGS_BLACKOUT_DAYS,
    GAP_FILTER_TOP3_SHARE,
    MIN_ADV_USD,
    MIN_HISTORY,
)


@dataclass(slots=True)
class FilterResult:
    """Result of one filter check for one ticker."""

    passed: bool
    flag: Optional[str] = None


def f1_history(adj_close: pd.Series) -> FilterResult:
    """F1: exclude if fewer than MIN_HISTORY (260) bars of price history."""
    if len(adj_close) < MIN_HISTORY:
        return FilterResult(passed=False, flag="F1_INSUFFICIENT_HISTORY")
    return FilterResult(passed=True)


def f2_liquidity(adj_close: pd.Series, volume: pd.Series, window_days: int = ADV_WINDOW_DAYS) -> FilterResult:
    """F2: exclude if median (close * volume) over window_days < MIN_ADV_USD."""
    if len(adj_close) < window_days or len(volume) < window_days:
        return FilterResult(passed=False, flag="F2_INSUFFICIENT_DATA")
    dollar_volume = (adj_close.iloc[-window_days:] * volume.iloc[-window_days:]).median()
    if dollar_volume < MIN_ADV_USD:
        return FilterResult(passed=False, flag="F2_ILLIQUID")
    return FilterResult(passed=True)


def f3_gap(window: pd.Series) -> FilterResult:
    """
    F3: exclude if the top-3 daily log returns dominate the total log return.

    Args:
        window: The same lookback window used for signal computation (§4).

    Purpose (spec §5): exclude names whose "trend" consists of a single gap
    (acquisition announcement, one-off surprise) that does not persist.
    """
    log_rets = (window / window.shift(1)).apply(np.log).dropna()
    if log_rets.empty:
        return FilterResult(passed=True)  # nothing to evaluate; do not spuriously exclude
    total = log_rets.sum()
    if total <= 0:
        return FilterResult(passed=True)  # negative momentum will be filtered by ranking
    top3 = log_rets.nlargest(3).sum()
    passes = (top3 / total) <= GAP_FILTER_TOP3_SHARE
    return FilterResult(passed=passes, flag=None if passes else "F3_GAP_DOMINATED")


def f4_quality(fcf_ttm: Optional[float], net_income_ttm: Optional[float]) -> FilterResult:
    """
    F4: exclude only if simultaneously loss-making on TTM FCF AND TTM net income.

    On missing data, pass rather than exclude (spec §5 F4: "yfinance
    fundamentals are unreliable ... specified loosely"). Callers must tally
    the missing-data flag for the §12.6 D5 decision criterion.
    """
    if fcf_ttm is None or net_income_ttm is None:
        return FilterResult(passed=True, flag="F4_DATA_MISSING")
    if fcf_ttm < 0 and net_income_ttm < 0:
        return FilterResult(passed=False, flag="F4_LOSS_MAKING")
    return FilterResult(passed=True)


def f5_exclusions(ticker: str, excluded: Set[str]) -> FilterResult:
    """F5: exclude if ticker is in the operator-maintained exclusion list."""
    if ticker in excluded:
        return FilterResult(passed=False, flag="F5_MANUAL_EXCLUSION")
    return FilterResult(passed=True)


def f6_earnings(
    next_earnings: Optional[date],
    execution_date: date,
    is_new_entry: bool,
    blackout_days: int = EARNINGS_BLACKOUT_DAYS,
) -> FilterResult:
    """
    F6: exclude NEW entries only, if earnings fall within blackout_days after execution.

    Existing holdings are unaffected (spec §5 F6: "Exclude from NEW entries
    only; holdings unaffected").
    """
    if not is_new_entry or next_earnings is None:
        return FilterResult(passed=True)
    days_to_earnings = (next_earnings - execution_date).days
    if 0 <= days_to_earnings <= blackout_days:
        return FilterResult(passed=False, flag="F6_EARNINGS_BLACKOUT")
    return FilterResult(passed=True)


@dataclass(slots=True)
class CandidateFilterOutcome:
    """Aggregate F1-F6 outcome for one candidate ticker."""

    ticker: str
    passed: bool
    filters: Dict[str, FilterResult]


def run_all(
    ticker: str,
    adj_close: pd.Series,
    volume: pd.Series,
    signal_window: pd.Series,
    fcf_ttm: Optional[float],
    net_income_ttm: Optional[float],
    excluded: Set[str],
    next_earnings: Optional[date],
    execution_date: date,
    is_new_entry: bool,
) -> CandidateFilterOutcome:
    """
    Run F1-F6 for one candidate, cheap filters first, short-circuiting on the first exclude.

    F4 never excludes (loose by design) so it always runs to completion and
    is recorded regardless of earlier results, for the f4_data_missing tally
    (spec §5, §12.6 D5).

    Returns:
        CandidateFilterOutcome with passed=True only if every exclusion-
        capable filter (F1, F2, F3, F5, F6) passed. F4's result is always
        included in .filters for reporting, even when it "passes" with a
        flag.
    """
    results: Dict[str, FilterResult] = {}
    overall_passed = True

    results["F1"] = f1_history(adj_close)
    if not results["F1"].passed:
        overall_passed = False

    if overall_passed:
        results["F2"] = f2_liquidity(adj_close, volume)
        if not results["F2"].passed:
            overall_passed = False

    if overall_passed:
        results["F3"] = f3_gap(signal_window)
        if not results["F3"].passed:
            overall_passed = False

    # F4 always runs (loose filter, never gates on its own — but still tallied)
    results["F4"] = f4_quality(fcf_ttm, net_income_ttm)
    if not results["F4"].passed:
        overall_passed = False

    if overall_passed:
        results["F5"] = f5_exclusions(ticker, excluded)
        if not results["F5"].passed:
            overall_passed = False

    if overall_passed:
        results["F6"] = f6_earnings(next_earnings, execution_date, is_new_entry)
        if not results["F6"].passed:
            overall_passed = False

    return CandidateFilterOutcome(ticker=ticker, passed=overall_passed, filters=results)


def tally_f4_missing_pct(outcomes: List[CandidateFilterOutcome]) -> float:
    """Fraction of candidates with F4 flagged F4_DATA_MISSING (spec §12.6 D5)."""
    if not outcomes:
        return 0.0
    missing = sum(1 for o in outcomes if o.filters.get("F4") is not None and o.filters["F4"].flag == "F4_DATA_MISSING")
    return missing / len(outcomes)
