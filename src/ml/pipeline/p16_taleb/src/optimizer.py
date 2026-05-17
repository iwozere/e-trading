"""
Strike grid optimization for the P16 Taleb barbell pipeline.

Sweeps the full (moneyness × T_days × rebalance_days) parameter grid,
computing summary statistics for each combination. Uses ProcessPoolExecutor
for parallelism.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from .simulator import simulate_barbell

_logger = logging.getLogger(__name__)

# Crisis threshold: drawdown worse than -15% at expiry = "crisis" period
_CRISIS_DRAWDOWN = -0.15


# ---------------------------------------------------------------------------
# Summary statistics helper
# ---------------------------------------------------------------------------

def _summarize(
    sim: pd.DataFrame,
    moneyness: float,
    T_days: int,
    rebalance_days: int,
    initial_capital: float,
) -> dict:
    """
    Compute summary statistics from a single simulation run.

    Args:
        sim:             Output of simulate_barbell().
        moneyness:       K/S ratio used in the simulation.
        T_days:          Tenor in calendar days.
        rebalance_days:  Trading-day interval between purchases.
        initial_capital: Starting capital (for ROI denominator).

    Returns:
        Dict with one row of optimizer output metrics.
    """
    if sim.empty:
        return {}

    n_periods = len(sim)
    total_cost = float(sim["budget_spent"].sum())
    total_payoff = float(sim["payoff"].sum())
    total_pnl = float(sim["pnl"].sum())

    net_roi_pct = (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0
    win_rate_pct = float((sim["payoff"] > 0).mean() * 100.0)
    avg_premium_pct = float(sim["put_price_pct_of_S"].mean())
    max_single_payoff = float(sim["payoff"].max())
    payoff_to_cost_ratio = (total_payoff / total_cost) if total_cost > 0 else 0.0

    # Crisis capture: % of crisis periods (drawdown < -15% at expiry) with payoff > 0
    crisis_mask = sim["drawdown_at_expiry"] < _CRISIS_DRAWDOWN
    n_crisis = int(crisis_mask.sum())
    if n_crisis > 0:
        crisis_capture_rate = float((sim.loc[crisis_mask, "payoff"] > 0).mean() * 100.0)
    else:
        crisis_capture_rate = float("nan")

    pnl_pct = sim["pnl_pct"]
    sharpe_analog = float(pnl_pct.mean() / pnl_pct.std()) if pnl_pct.std() > 0 else 0.0

    return {
        "moneyness":           moneyness,
        "strike_otm_pct":      round((1.0 - moneyness) * 100.0, 1),
        "T_days":              T_days,
        "rebalance_days":      rebalance_days,
        "n_periods":           n_periods,
        "total_cost":          total_cost,
        "total_payoff":        total_payoff,
        "total_pnl":           total_pnl,
        "net_roi_pct":         net_roi_pct,
        "win_rate_pct":        win_rate_pct,
        "avg_premium_pct":     avg_premium_pct,
        "max_single_payoff":   max_single_payoff,
        "payoff_to_cost_ratio": payoff_to_cost_ratio,
        "crisis_capture_rate": crisis_capture_rate,
        "sharpe_analog":       sharpe_analog,
        "n_crisis_periods":    n_crisis,
    }


# ---------------------------------------------------------------------------
# Worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> dict:
    """Run one simulation and return summary statistics."""
    df_json, moneyness, T_days, rebalance_days, budget_pct, initial_capital, skew_slope = args
    df = pd.read_json(df_json, orient="split")
    df.index = pd.to_datetime(df.index)
    sim = simulate_barbell(
        df,
        moneyness=moneyness,
        T_days=T_days,
        rebalance_days=rebalance_days,
        put_budget_pct=budget_pct,
        initial_capital=initial_capital,
        skew_slope=skew_slope,
    )
    return _summarize(sim, moneyness, T_days, rebalance_days, initial_capital)


# ---------------------------------------------------------------------------
# Public optimizer
# ---------------------------------------------------------------------------

def optimize_strikes(
    df: pd.DataFrame,
    moneyness_grid: Optional[list] = None,
    T_days_grid: Optional[list] = None,
    rebalance_days_grid: Optional[list] = None,
    budget_pct: float = 0.02,
    initial_capital: float = 100_000.0,
    skew_slope: float = 0.015,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Run simulate_barbell across a parameter grid and rank by all metrics.

    Args:
        df:                   Master daily DataFrame (output of build_features).
        moneyness_grid:       List of K/S ratios to test. Default: spec §5 grid.
        T_days_grid:          List of tenors in calendar days. Default: [60, 90, 120].
        rebalance_days_grid:  List of rebalance intervals (trading days). Default: [21].
        budget_pct:           Put budget as fraction of initial_capital per period.
        initial_capital:      Starting capital.
        skew_slope:           Skew model slope (passed through to simulator).
        max_workers:          ProcessPoolExecutor worker count.

    Returns:
        DataFrame sorted by net_roi_pct descending. One row per parameter combo.
    """
    if moneyness_grid is None:
        moneyness_grid = [
            0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82,
            0.84, 0.85, 0.86, 0.87, 0.88, 0.90, 0.92, 0.94, 0.95,
        ]
    if T_days_grid is None:
        T_days_grid = [60, 90, 120]
    if rebalance_days_grid is None:
        rebalance_days_grid = [21]

    # Serialize df once (shared across workers via JSON — avoids shared-memory complexity)
    df_json = df.to_json(orient="split", date_format="iso")

    combos = [
        (df_json, m, t, r, budget_pct, initial_capital, skew_slope)
        for m in moneyness_grid
        for t in T_days_grid
        for r in rebalance_days_grid
    ]

    total = len(combos)
    _logger.info("Optimizer: %d combinations, max_workers=%d", total, max_workers)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, c): c for c in combos}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
                if row:
                    results.append(row)
            except Exception:
                combo = futures[fut]
                _logger.warning("Worker failed for moneyness=%.2f T=%d", combo[1], combo[2])
            if i % 10 == 0 or i == total:
                _logger.info("Optimizer progress: %d/%d", i, total)

    if not results:
        _logger.error("Optimizer produced no results")
        return pd.DataFrame()

    opt_df = (
        pd.DataFrame(results)
        .sort_values("net_roi_pct", ascending=False)
        .reset_index(drop=True)
    )

    _logger.info(
        "Optimizer complete: %d results. Best: moneyness=%.2f T=%d → ROI=%.1f%%",
        len(opt_df),
        opt_df["moneyness"].iloc[0],
        opt_df["T_days"].iloc[0],
        opt_df["net_roi_pct"].iloc[0],
    )
    return opt_df
