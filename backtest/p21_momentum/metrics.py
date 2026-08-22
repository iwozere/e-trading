"""
P21 Momentum backtest — required metrics (docs/pipeline-specification.md §14.8).

Three tiers, in the spec's own bias-resistance order:

- **Mechanical** (:func:`compute_mechanical_metrics`) — turnover, position
  count, sector concentration, filter attrition, regime histogram, holding
  period, trade size/commission, event counts. Largely immune to the Option A
  survivorship bias (docs/implementation-plan.md §8.1) — this is the primary
  output ``phase0_report.py`` leads with.
- **Risk** (:func:`compute_risk_metrics`) — volatility, drawdown, tracking
  error, beta/correlation. Moderately bias-resistant.
- **Return** (:func:`compute_return_metrics`) — CAGR, Sharpe, Sortino,
  information ratio. Must always be reported alongside the §14.3 banner, or
  suppressed entirely — never presented bare. Callers, not this module,
  own that suppression decision (``phase0_report.py`` gates it).

All functions here are pure: they consume a :class:`~backtest.p21_momentum.runner.BacktestResult`
(or a bare NAV series) and return dataclasses/DataFrames — no file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.pipeline.p21_momentum.schemas import LedgerEntry

if TYPE_CHECKING:
    from backtest.p21_momentum.runner import BacktestResult

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RiskMetrics:
    """Spec §14.8 "Risk" tier, for one track's NAV series."""

    annualized_vol: float
    max_drawdown: float
    max_drawdown_duration_days: int
    time_to_recovery_days: Optional[int]  # None if the series ends still underwater
    worst_month: float
    worst_quarter: float
    downside_deviation: float


def _drawdown_series(nav: pd.Series) -> pd.Series:
    running_max = nav.cummax()
    return nav / running_max - 1.0


def _max_drawdown_and_duration(nav: pd.Series) -> tuple[float, int, Optional[int]]:
    dd = _drawdown_series(nav)
    trough_idx = pd.Timestamp(dd.idxmin())
    max_dd = float(dd.loc[trough_idx])
    peak_idx = pd.Timestamp(nav.loc[:trough_idx].idxmax())
    duration_days = int((trough_idx - peak_idx).days)

    peak_value = nav.loc[peak_idx]
    after_trough = nav.loc[trough_idx:]
    recovered = after_trough[after_trough >= peak_value]
    time_to_recovery = int((pd.Timestamp(recovered.index[0]) - trough_idx).days) if len(recovered) > 0 else None
    return max_dd, duration_days, time_to_recovery


def compute_risk_metrics(nav: pd.Series) -> RiskMetrics:
    """
    Compute spec §14.8's risk tier for one track's daily NAV series.

    Args:
        nav: Daily NAV, ascending DatetimeIndex, no gaps (as produced by
            ``runner.run_backtest()``'s ``nav_daily`` columns).

    Returns:
        RiskMetrics. Requires at least 2 observations; degenerate inputs
        (all-NaN or single-row) return zeros/None rather than raising, since
        a stress window can be shorter than a full track history.
    """
    nav = nav.dropna()
    if len(nav) < 2:
        return RiskMetrics(0.0, 0.0, 0, None, 0.0, 0.0, 0.0)

    daily_returns = nav.pct_change().dropna()
    annualized_vol = (
        float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_returns) > 1 else 0.0
    )

    max_dd, dd_duration, time_to_recovery = _max_drawdown_and_duration(nav)

    monthly = nav.resample("ME").last().pct_change().dropna()
    worst_month = float(monthly.min()) if len(monthly) else 0.0
    quarterly = nav.resample("QE").last().pct_change().dropna()
    worst_quarter = float(quarterly.min()) if len(quarterly) else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_deviation = (
        float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) else 0.0
    )

    return RiskMetrics(
        annualized_vol=annualized_vol,
        max_drawdown=max_dd,
        max_drawdown_duration_days=dd_duration,
        time_to_recovery_days=time_to_recovery,
        worst_month=worst_month,
        worst_quarter=worst_quarter,
        downside_deviation=downside_deviation,
    )


def compute_rolling_tracking_error(
    nav: pd.Series, nav_c: pd.Series, window_days: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Rolling N-day (default 12-month) tracking error of ``nav`` versus track C's ``nav_c`` (spec §14.8)."""
    active = nav.pct_change() - nav_c.pct_change()
    return active.rolling(window_days).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_rolling_beta_corr(
    nav: pd.Series, nav_market: pd.Series, window_days: int = 2 * TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """Rolling N-day (default 24-month) beta/correlation of ``nav`` versus the market NAV (spec §14.8: vs SPY)."""
    r = nav.pct_change()
    m = nav_market.pct_change()
    cov = r.rolling(window_days).cov(m)
    var_m = m.rolling(window_days).var()
    beta = cov / var_m
    corr = r.rolling(window_days).corr(m)
    return pd.DataFrame({"beta": beta, "corr": corr})


# ---------------------------------------------------------------------------
# Return metrics — always subject to the §14.3 banner / suppression by caller
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReturnMetrics:
    """Spec §14.8 "Return" tier. Contaminated by Option A survivorship bias (§14.2) — never present bare."""

    cagr: float
    sharpe: float
    sortino: float
    information_ratio_vs_c: float
    hit_rate_vs_c_monthly: float


def compute_return_metrics(nav: pd.Series, nav_c: pd.Series, risk_free_annual: float = 0.0) -> ReturnMetrics:
    """
    Compute spec §14.8's return tier for one track's NAV versus track C (the TER-only MTUM proxy).

    Args:
        nav: Daily NAV for the track being evaluated.
        nav_c: Daily NAV for track C (the benchmark this tier is measured against, spec §9).
        risk_free_annual: Annualized risk-free rate for the Sharpe excess-return term; 0.0 by
            default since spec §16 does not specify one and the backtest is mechanical-first.

    Returns:
        ReturnMetrics. Degenerate (< 2 observations) inputs return zeros.
    """
    nav = nav.dropna()
    if len(nav) < 2:
        return ReturnMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    n_days = (nav.index[-1] - nav.index[0]).days
    years = n_days / 365.25
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0

    daily_returns = nav.pct_change().dropna()
    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    excess_std = excess.std(ddof=1)
    sharpe = float(excess.mean() / excess_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if excess_std > 0 else 0.0

    downside = excess[excess < 0]
    downside_std = float(np.sqrt((downside**2).mean())) if len(downside) else 0.0
    sortino = float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std > 0 else 0.0

    nav_c = nav_c.reindex(nav.index).dropna()
    common_idx = nav.index.intersection(nav_c.index)
    active_daily = nav.loc[common_idx].pct_change() - nav_c.loc[common_idx].pct_change()
    active_daily = active_daily.dropna()
    information_ratio = (
        float(active_daily.mean() / active_daily.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(active_daily) > 1 and active_daily.std(ddof=1) > 0
        else 0.0
    )

    monthly = nav.resample("ME").last().pct_change().dropna()
    monthly_c = nav_c.resample("ME").last().pct_change().dropna()
    common_months = monthly.index.intersection(monthly_c.index)
    hit_rate = float((monthly.loc[common_months] > monthly_c.loc[common_months]).mean()) if len(common_months) else 0.0

    return ReturnMetrics(
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        information_ratio_vs_c=information_ratio,
        hit_rate_vs_c_monthly=hit_rate,
    )


# ---------------------------------------------------------------------------
# Mechanical metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MechanicalMetrics:
    """Spec §14.8 "Mechanical" tier — the backtest's primary, bias-resistant output."""

    turnover_annualized_median_pct: float
    turnover_monthly_pct: List[float]
    position_count_mean: float
    position_count_min: int
    pct_months_below_20: float
    pct_months_below_12: float
    max_sector_count_ever: int
    f1_removed_total: int
    f2_removed_total: int
    f3_removed_total: int
    f1_removed_pct_of_universe: Optional[float]
    f2_removed_pct_of_universe: Optional[float]
    f3_removed_pct_of_universe: Optional[float]
    regime_scalar_histogram: Dict[float, int]
    regime_state_transitions: int
    holding_period_days_median: Optional[float]
    holding_period_days_mean: Optional[float]
    trade_size_usd_median: Optional[float]
    commission_pct_of_trade_value_median: Optional[float]
    warn_underfilled_count: int
    manual_review_count_total: int
    exit_delisted_count_total: int


def compute_holding_periods_days(trades: List[LedgerEntry], track: str = "A") -> List[int]:
    """
    Reconstruct §14.8's "holding period" distribution from a track's trade ledger.

    Simplification (documented, not silent): a position's lifespan runs from
    an ``ENTRY_RANK_*``-reasoned BUY to the next ``EXIT_*``-reasoned trade for
    the same ticker; ``REBAL_ADD``/``REBAL_TRIM`` partial rebalances do not
    close or reopen a position. A ticker still held at the series' end
    contributes no observation (its holding period is right-censored, not
    zero) — consistent with how survival-style holding-period stats are
    usually reported.
    """
    opens: Dict[str, date] = {}
    periods: List[int] = []
    for t in sorted((t for t in trades if t.track == track), key=lambda t: (t.ts, t.ticker)):
        d = date.fromisoformat(t.ts)
        if t.reason.startswith("ENTRY_RANK"):
            opens[t.ticker] = d
        elif t.reason.startswith("EXIT_"):
            start = opens.pop(t.ticker, None)
            if start is not None:
                periods.append((d - start).days)
    return periods


def compute_mechanical_metrics(
    result: "BacktestResult", track: str = "A", universe_size: Optional[int] = None
) -> MechanicalMetrics:
    """
    Compute spec §14.8's mechanical tier from a full backtest result.

    Args:
        result: Output of ``runner.run_backtest()``.
        track: Which track's trade ledger to draw turnover/holding-period/
            trade-size stats from — spec's own note (runner.py module
            docstring) is that only track A's turnover is a required metric,
            so this defaults to "A".
        universe_size: If given, filter-attrition counts are also expressed
            as a percentage of this (spec §14.8: "absolute and percentage").
            Left ``None`` (percentages omitted) when the universe size isn't
            known to the caller, e.g. in a stress-window slice.

    Returns:
        MechanicalMetrics.
    """
    months = result.monthly_metrics
    turnover_pct: List[float] = []
    nav_a_by_month = {}
    if len(result.nav_daily):
        nav_index = pd.DatetimeIndex(result.nav_daily.index)
        for m in months:
            year, mon = (int(x) for x in m.month.split("-"))
            month_rows = result.nav_daily[(nav_index.year == year) & (nav_index.month == mon)]
            nav_a_by_month[m.month] = float(month_rows["nav_a"].iloc[-1]) if len(month_rows) else None

    for m in months:
        nav = nav_a_by_month.get(m.month)
        if nav and nav > 0:
            turnover_pct.append(m.turnover_two_way_usd / nav * 100.0)

    turnover_annualized_median = float(np.median(turnover_pct) * 12) if turnover_pct else 0.0

    position_counts = [m.position_count for m in months]
    position_count_mean = float(np.mean(position_counts)) if position_counts else 0.0
    position_count_min = min(position_counts) if position_counts else 0
    n_months = len(months) or 1
    pct_below_20 = sum(1 for c in position_counts if c < 20) / n_months
    pct_below_12 = sum(1 for c in position_counts if c < 12) / n_months
    max_sector_ever = max((m.max_sector_count for m in months), default=0)

    f1_total = sum(m.f1_removed for m in months)
    f2_total = sum(m.f2_removed for m in months)
    f3_total = sum(m.f3_removed for m in months)
    denom = universe_size * n_months if universe_size else None
    f1_pct = f1_total / denom if denom else None
    f2_pct = f2_total / denom if denom else None
    f3_pct = f3_total / denom if denom else None

    histogram: Dict[float, int] = {}
    transitions = 0
    prev_state: Optional[tuple] = None
    for entry in result.regime_history:
        scalar = round(float(entry["scalar_applied"]), 4)
        histogram[scalar] = histogram.get(scalar, 0) + 1
        state = (entry["bear"], entry["high_vol"])
        if prev_state is not None and state != prev_state:
            transitions += 1
        prev_state = state

    holding_periods = compute_holding_periods_days(result.trades, track=track)
    holding_median = float(np.median(holding_periods)) if holding_periods else None
    holding_mean = float(np.mean(holding_periods)) if holding_periods else None

    track_trades = [t for t in result.trades if t.track == track]
    trade_sizes = [t.gross_usd for t in track_trades if t.gross_usd > 0]
    commission_pcts = [t.commission_usd / t.gross_usd for t in track_trades if t.gross_usd > 0]
    trade_size_median = float(np.median(trade_sizes)) if trade_sizes else None
    commission_pct_median = float(np.median(commission_pcts)) if commission_pcts else None

    return MechanicalMetrics(
        turnover_annualized_median_pct=turnover_annualized_median,
        turnover_monthly_pct=turnover_pct,
        position_count_mean=position_count_mean,
        position_count_min=position_count_min,
        pct_months_below_20=pct_below_20,
        pct_months_below_12=pct_below_12,
        max_sector_count_ever=max_sector_ever,
        f1_removed_total=f1_total,
        f2_removed_total=f2_total,
        f3_removed_total=f3_total,
        f1_removed_pct_of_universe=f1_pct,
        f2_removed_pct_of_universe=f2_pct,
        f3_removed_pct_of_universe=f3_pct,
        regime_scalar_histogram=histogram,
        regime_state_transitions=transitions,
        holding_period_days_median=holding_median,
        holding_period_days_mean=holding_mean,
        trade_size_usd_median=trade_size_median,
        commission_pct_of_trade_value_median=commission_pct_median,
        warn_underfilled_count=sum(1 for m in months if m.warn_underfilled),
        manual_review_count_total=sum(m.manual_review_count for m in months),
        exit_delisted_count_total=sum(m.exit_delisted_count for m in months),
    )
