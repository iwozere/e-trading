"""
P21 Momentum backtest — parameter robustness (docs/pipeline-specification.md §14.5).

This is, per the spec, "where backtests are most often ruined" — so two of
its rules are enforced mechanically here, not left as operator discipline:

- **Rule 3** (deflated Sharpe): :func:`deflated_sharpe_band` computes the
  expected-by-chance maximum Sharpe over N trials so a caller can tell
  whether the grid's winner is actually informative.
- **Rule 4** (out-of-sample, touched once): :func:`run_grid` refuses to
  evaluate the out-of-sample window more than once per process unless
  ``acknowledge_oos_reaccess=True`` is passed, and every evaluation —
  in-sample or out — appends a timestamped line to ``oos_access_log.md``
  regardless (:func:`log_oos_access`).

**Rule 1** (do not optimize the core signal) is a human discipline this
module cannot enforce in code — ``lookback_start``/``skip_recent`` are left
in the grid only "to confirm the implementation behaves sensibly," per spec;
nothing here stops a caller from misusing the grid's winner as a tuned
signal. That responsibility is documented, not code-enforced.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest.p21_momentum.metrics import compute_return_metrics
from backtest.p21_momentum.runner import BacktestParams, run_backtest
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Spec §14.5 Rule 2's exact grid.
ROBUSTNESS_GRID: Dict[str, Sequence] = {
    "lookback_start": (189, 252, 315),
    "skip_recent": (10, 21, 42),
    "entry_rank": (10, 20, 30),
    "hold_rank": (40, 60, 100),
    "max_per_sector": (3, 4, 6),
    "vix_threshold": (24.0, 28.0, 32.0),
}

# Spec §14.5 Rule 4.
IN_SAMPLE_START = date(2005, 1, 1)
IN_SAMPLE_END = date(2016, 12, 31)
OUT_OF_SAMPLE_START = date(2017, 1, 1)
OUT_OF_SAMPLE_END = date(2026, 6, 30)

DEFAULT_OOS_LOG_PATH = Path(__file__).resolve().parent / "oos_access_log.md"


class OutOfSampleReaccessError(RuntimeError):
    """Raised when a grid run would touch the out-of-sample window more than once (spec §14.5 Rule 4)."""


@dataclass(slots=True)
class GridRow:
    """One row of ``grid_729.csv`` — one parameter combination's outcome."""

    lookback_start: int
    skip_recent: int
    entry_rank: int
    hold_rank: int
    max_per_sector: int
    vix_threshold: float
    sharpe_a: float
    sharpe_c: float
    turnover_annualized_median_pct: float

    def to_dict(self) -> dict:
        return {
            "lookback_start": self.lookback_start,
            "skip_recent": self.skip_recent,
            "entry_rank": self.entry_rank,
            "hold_rank": self.hold_rank,
            "max_per_sector": self.max_per_sector,
            "vix_threshold": self.vix_threshold,
            "sharpe_a": self.sharpe_a,
            "sharpe_c": self.sharpe_c,
            "turnover_annualized_median_pct": self.turnover_annualized_median_pct,
        }


def log_oos_access(
    window_label: str,
    reason: str,
    log_path: Path = DEFAULT_OOS_LOG_PATH,
) -> None:
    """Append a timestamped line to ``oos_access_log.md`` (spec §14.5 Rule 4 — mandatory, every evaluation)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"- {timestamp} — window={window_label} — {reason}\n"
    if not log_path.exists():
        log_path.write_text(
            "# Out-of-Sample Access Log (spec §14.5 Rule 4)\n\n"
            "Append-only. Every evaluation of the 2017-01 -> 2026-06 out-of-sample window is logged here,\n"
            "so accidental repeated peeking cannot happen unnoticed.\n\n",
            encoding="utf-8",
        )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def deflated_sharpe_band(n_trials: int, n_observations: int) -> Tuple[float, float]:
    """
    Estimate the expected-by-chance maximum Sharpe over ``n_trials`` independent trials
    (Bailey & Lopez de Prado's approximation), spec §14.5 Rule 3.

    Uses the classical extreme-value approximation for the expected maximum of N standard
    normal variates, converted to an annualized Sharpe band via the observation count:

        E[max Sharpe] ~= ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))) / sqrt(T)

    where gamma is the Euler-Mascheroni constant and T is the number of monthly
    observations, annualized by sqrt(12).

    Args:
        n_trials: Number of independent parameter combinations tested (729 for the full grid).
        n_observations: Number of monthly return observations in the sample (~245 for
            2005-2026).

    Returns:
        (low, high) — spec quotes "roughly 0.9-1.1" for 729 trials over ~20 years; this
        computes the same quantity for arbitrary trial/observation counts so the band scales
        correctly if the study period or grid size changes.
    """
    from scipy.stats import norm

    euler_mascheroni = 0.5772156649
    if n_trials <= 1 or n_observations <= 1:
        return (0.0, 0.0)
    z_a = norm.ppf(1 - 1 / n_trials)
    z_b = norm.ppf(1 - 1 / (n_trials * np.e))
    expected_max_z = (1 - euler_mascheroni) * z_a + euler_mascheroni * z_b
    monthly_sharpe = expected_max_z / np.sqrt(n_observations)
    annualized = float(monthly_sharpe * np.sqrt(12))
    # Spec's quoted band has width comparable to a +/-10% relative spread around the
    # point estimate; reported as a band, not a point, per Rule 3's own framing.
    return (annualized * 0.9, annualized * 1.1)


def is_top_quartile_separated(sharpes: Sequence[float]) -> bool:
    """
    Spec §14.5 Rule 3's practical decision rule: is the best config clearly separated
    from the median of the top quartile?

    Returns:
        True if the maximum is more than one top-quartile standard deviation above the
        top-quartile median — a simple, auditable separation test. False means "treat the
        whole surface as flat and use the literature defaults," per spec.
    """
    if len(sharpes) < 4:
        return False
    arr = np.sort(np.asarray(sharpes))
    top_quartile = arr[int(len(arr) * 0.75):]
    if len(top_quartile) < 2:
        return False
    median = float(np.median(top_quartile))
    std = float(np.std(top_quartile, ddof=1))
    return bool(arr[-1] > median + std) if std > 0 else False


# Process-lifetime guard for Rule 4 — intentionally module-level state, not persisted:
# a fresh process (fresh Phase 0 study session) gets a fresh single out-of-sample touch.
_OOS_TOUCHED = {"value": False}


def run_grid(
    panel: Dict[str, pd.DataFrame],
    sector_by_ticker: Dict[str, str],
    start: date,
    end: date,
    grid: Optional[Dict[str, Sequence]] = None,
    acknowledge_oos_reaccess: bool = False,
    oos_log_path: Path = DEFAULT_OOS_LOG_PATH,
) -> List[GridRow]:
    """
    Run the full parameter grid (spec §14.5 Rule 2) over [start, end].

    Args:
        panel, sector_by_ticker: Same frozen-panel inputs as ``runner.run_backtest()``.
        start, end: Date range for this grid run — pass the in-sample window
            (:data:`IN_SAMPLE_START`/:data:`IN_SAMPLE_END`) for parameter inspection, and the
            out-of-sample window only once (spec §14.5 Rule 4).
        grid: Override for :data:`ROBUSTNESS_GRID`; mainly for tests, which use a far
            smaller grid to stay fast.
        acknowledge_oos_reaccess: Must be True to run this function again with a date range
            overlapping the out-of-sample window within the same process — see
            :class:`OutOfSampleReaccessError`.
        oos_log_path: Where to append the mandatory access-log line.

    Returns:
        One GridRow per parameter combination, in grid iteration order.
    """
    grid = grid or ROBUSTNESS_GRID
    touches_oos = start <= OUT_OF_SAMPLE_END and end >= OUT_OF_SAMPLE_START
    if touches_oos:
        if _OOS_TOUCHED["value"] and not acknowledge_oos_reaccess:
            raise OutOfSampleReaccessError(
                "This process has already evaluated the out-of-sample window once (spec §14.5 "
                "Rule 4). Pass acknowledge_oos_reaccess=True if this repeated access is deliberate."
            )
        _OOS_TOUCHED["value"] = True

    window_label = f"{start.isoformat()}..{end.isoformat()}"
    reason = "out-of-sample grid run" if touches_oos else "in-sample grid run"
    log_oos_access(window_label, reason, log_path=oos_log_path)

    keys = list(grid.keys())
    rows: List[GridRow] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params_kwargs = dict(zip(keys, combo))
        params = BacktestParams(**params_kwargs)
        result = run_backtest(panel, sector_by_ticker, start, end, params=params)
        if result.nav_daily.empty:
            continue
        return_a = compute_return_metrics(result.nav_daily["nav_a"], result.nav_daily["nav_c"])
        return_c = compute_return_metrics(result.nav_daily["nav_c"], result.nav_daily["nav_c"])
        turnover_pct: List[float] = []
        nav_index = pd.DatetimeIndex(result.nav_daily.index)
        for m in result.monthly_metrics:
            year, mon = (int(x) for x in m.month.split("-"))
            month_rows = result.nav_daily[(nav_index.year == year) & (nav_index.month == mon)]
            if len(month_rows):
                nav = float(month_rows["nav_a"].iloc[-1])
                if nav > 0:
                    turnover_pct.append(m.turnover_two_way_usd / nav * 100.0)
        turnover_annualized = float(np.median(turnover_pct) * 12) if turnover_pct else 0.0

        rows.append(
            GridRow(
                lookback_start=params.lookback_start,
                skip_recent=params.skip_recent,
                entry_rank=params.entry_rank,
                hold_rank=params.hold_rank,
                max_per_sector=params.max_per_sector,
                vix_threshold=params.vix_threshold,
                sharpe_a=return_a.sharpe,
                sharpe_c=return_c.sharpe,
                turnover_annualized_median_pct=turnover_annualized,
            )
        )
    _logger.info("Robustness grid: %d combinations evaluated over %s", len(rows), window_label)
    return rows


def render_grid_csv(rows: List[GridRow]) -> pd.DataFrame:
    """Return the grid as a DataFrame, ready for ``grid_729.csv`` (spec §14.10)."""
    return pd.DataFrame([r.to_dict() for r in rows])


def render_deflated_sharpe_md(rows: List[GridRow], n_observations: int) -> str:
    """Render spec §14.10's ``deflated_sharpe.md`` deliverable."""
    sharpes = [r.sharpe_a for r in rows]
    low, high = deflated_sharpe_band(len(rows), n_observations)
    best = max(sharpes) if sharpes else 0.0
    separated = is_top_quartile_separated(sharpes)
    lines = [
        "# Deflated Sharpe Analysis (spec §14.5 Rule 3)",
        "",
        f"Trials: {len(rows)}. Observations: {n_observations}.",
        f"Expected-by-chance maximum Sharpe band: {low:.2f} - {high:.2f}.",
        f"Best observed Sharpe (track A): {best:.2f}.",
        "",
    ]
    if best < low:
        lines.append(
            "**The best configuration's Sharpe falls below the by-chance band.** No configuration "
            "in this grid carries evidence of skill; use the literature defaults."
        )
    elif not separated:
        lines.append(
            "**The best configuration is not clearly separated from the top quartile's median** "
            "(spec §14.5 Rule 3's practical decision rule). Treat the whole surface as flat and "
            "use the literature defaults — this is the expected, correct outcome."
        )
    else:
        lines.append(
            "The best configuration is separated from the top-quartile median and above the "
            "by-chance band. Inspect the marginal surface before trusting this — a narrow peak is "
            "still overfitting even when nominally 'significant'."
        )
    return "\n".join(lines) + "\n"
