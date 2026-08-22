"""
P21 Momentum backtest — transaction cost sensitivity (docs/pipeline-specification.md §14.7).

Two sweeps, both reusing ``runner.run_backtest()``'s single-config path in a loop — no
separate simulation logic (docs/implementation-plan.md §8.4):

- :func:`run_slippage_sweep` — the full backtest at four slippage levels
  (0/3/10/25 bps). Spec's own acceptance test (§14.9 B8): if the edge over
  track C disappears between 3 and 10 bps, the strategy is not viable.
- :func:`run_turnover_curve` — ``hold_rank`` in ``[20, 40, 60, 100, 150]`` at
  10 bps, net return plotted against realized turnover. The curve's maximum
  is the economically correct hysteresis setting (spec: "if it lands far
  from 60, update the specification").
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Dict, List, Optional, Sequence

import pandas as pd

from backtest.p21_momentum.metrics import compute_mechanical_metrics, compute_return_metrics
from backtest.p21_momentum.runner import BacktestParams, run_backtest
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

SLIPPAGE_LEVELS_BPS: Sequence[float] = (0.0, 3.0, 10.0, 25.0)
TURNOVER_CURVE_HOLD_RANKS: Sequence[int] = (20, 40, 60, 100, 150)
TURNOVER_CURVE_SLIPPAGE_BPS = 10.0


@dataclass(slots=True)
class SlippageResult:
    """One slippage level's outcome — one row of ``slippage_0_3_10_25.csv``."""

    slippage_bps: float
    cagr_a: float
    sharpe_a: float
    cagr_c: float
    sharpe_c: float
    edge_cagr_a_minus_c: float

    def to_dict(self) -> dict:
        return {
            "slippage_bps": self.slippage_bps,
            "cagr_a": self.cagr_a,
            "sharpe_a": self.sharpe_a,
            "cagr_c": self.cagr_c,
            "sharpe_c": self.sharpe_c,
            "edge_cagr_a_minus_c": self.edge_cagr_a_minus_c,
        }


@dataclass(slots=True)
class TurnoverCurvePoint:
    """One hold_rank's outcome — one row of the turnover/net-return curve."""

    hold_rank: int
    turnover_annualized_median_pct: float
    net_cagr_a: float

    def to_dict(self) -> dict:
        return {
            "hold_rank": self.hold_rank,
            "turnover_annualized_median_pct": self.turnover_annualized_median_pct,
            "net_cagr_a": self.net_cagr_a,
        }


def run_slippage_sweep(
    panel: Dict[str, pd.DataFrame],
    sector_by_ticker: Dict[str, str],
    start: date,
    end: date,
    slippage_levels_bps: Sequence[float] = SLIPPAGE_LEVELS_BPS,
    base_params: Optional[BacktestParams] = None,
) -> List[SlippageResult]:
    """
    Run the full backtest at each of ``slippage_levels_bps``, holding every other parameter fixed.

    Returns:
        One SlippageResult per level, same order as ``slippage_levels_bps``.
    """
    base = base_params or BacktestParams()
    results = []
    for bps in slippage_levels_bps:
        params = replace(base, slippage_bps=bps)
        result = run_backtest(panel, sector_by_ticker, start, end, params=params)
        if result.nav_daily.empty:
            results.append(SlippageResult(bps, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        return_a = compute_return_metrics(result.nav_daily["nav_a"], result.nav_daily["nav_c"])
        return_c = compute_return_metrics(result.nav_daily["nav_c"], result.nav_daily["nav_c"])
        results.append(
            SlippageResult(
                slippage_bps=bps,
                cagr_a=return_a.cagr,
                sharpe_a=return_a.sharpe,
                cagr_c=return_c.cagr,
                sharpe_c=return_c.sharpe,
                edge_cagr_a_minus_c=return_a.cagr - return_c.cagr,
            )
        )
    _logger.info("Slippage sweep: %d levels evaluated", len(results))
    return results


def edge_survives_10bps(results: List[SlippageResult]) -> bool:
    """
    Spec §14.9 B8: does the edge over track C survive between 3 and 10 bps?

    Returns:
        True if the edge at 10 bps is still positive whenever the edge at 3 bps was positive
        (an edge that was never positive at 3 bps has nothing to "survive," and is a
        base-case viability question, not this specific acceptance test).
    """
    by_bps = {r.slippage_bps: r for r in results}
    if 3.0 not in by_bps or 10.0 not in by_bps:
        raise ValueError("edge_survives_10bps requires both 3bps and 10bps in the results")
    edge_3 = by_bps[3.0].edge_cagr_a_minus_c
    edge_10 = by_bps[10.0].edge_cagr_a_minus_c
    if edge_3 <= 0:
        return False
    return edge_10 > 0


def run_turnover_curve(
    panel: Dict[str, pd.DataFrame],
    sector_by_ticker: Dict[str, str],
    start: date,
    end: date,
    hold_ranks: Sequence[int] = TURNOVER_CURVE_HOLD_RANKS,
    slippage_bps: float = TURNOVER_CURVE_SLIPPAGE_BPS,
    base_params: Optional[BacktestParams] = None,
) -> List[TurnoverCurvePoint]:
    """
    Re-run the backtest across ``hold_ranks`` at a fixed (pessimistic) slippage level,
    recording realized turnover against net return (spec §14.7's cost-neutral turnover curve).

    Returns:
        One TurnoverCurvePoint per hold_rank, same order as ``hold_ranks``.
    """
    points = []
    base = base_params or BacktestParams()
    for hr in hold_ranks:
        params = replace(base, hold_rank=hr, slippage_bps=slippage_bps)
        result = run_backtest(panel, sector_by_ticker, start, end, params=params)
        if result.nav_daily.empty:
            points.append(TurnoverCurvePoint(hr, 0.0, 0.0))
            continue
        mechanical = compute_mechanical_metrics(result)
        return_a = compute_return_metrics(result.nav_daily["nav_a"], result.nav_daily["nav_c"])
        points.append(
            TurnoverCurvePoint(
                hold_rank=hr,
                turnover_annualized_median_pct=mechanical.turnover_annualized_median_pct,
                net_cagr_a=return_a.cagr,
            )
        )
    _logger.info("Turnover curve: %d hold_rank values evaluated", len(points))
    return points


def best_hold_rank(points: List[TurnoverCurvePoint]) -> Optional[int]:
    """The hold_rank maximizing net return on the turnover curve — spec: 'the economically correct setting'."""
    if not points:
        return None
    return max(points, key=lambda p: p.net_cagr_a).hold_rank


def render_slippage_csv(results: List[SlippageResult]) -> pd.DataFrame:
    """Return the slippage sweep as a DataFrame, ready for ``slippage_0_3_10_25.csv`` (spec §14.10)."""
    return pd.DataFrame([r.to_dict() for r in results])


def render_turnover_curve_png(points: List[TurnoverCurvePoint], out_path) -> None:
    """Render spec §14.10's ``turnover_net_return_curve.png`` deliverable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    turnover = [p.turnover_annualized_median_pct for p in points]
    net_return = [p.net_cagr_a * 100 for p in points]
    labels = [str(p.hold_rank) for p in points]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(turnover, net_return, marker="o")
    for x, y, label in zip(turnover, net_return, labels):
        ax.annotate(f"hold_rank={label}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Annualized turnover (median, %)")
    ax.set_ylabel("Net CAGR, track A (%)")
    ax.set_title(f"Turnover vs. net return curve (slippage={TURNOVER_CURVE_SLIPPAGE_BPS:.0f}bps)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
