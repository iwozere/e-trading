"""
P21 Momentum backtest — the one-time out-of-sample check (docs/pipeline-specification.md §14.5 Rule 4).

This is the single, deliberate look at the out-of-sample window that Rule 4 permits. It is a
separate command from ``phase0_report.py`` on purpose: that orchestrator's robustness grid
never touches this window (see its own docstring/comments), so
``backtest.p21_momentum.robustness``'s single-touch guard stays armed until an operator runs
this file — once.

    python -m backtest.p21_momentum.run_oos_check
    python -m backtest.p21_momentum.run_oos_check --sequential-grid
    python -m backtest.p21_momentum.run_oos_check --acknowledge-oos-reaccess

What it does, in spec order:
  1. Loads the frozen panel (same data ``phase0_report.py`` uses — never re-fetched).
  2. Runs one default-parameter backtest confined to the out-of-sample window, to report
     track A vs. track C mechanically and after costs. This is a single point estimate, not a
     search, so on its own it carries no Rule 4 exposure.
  3. Runs the full 729-combination grid (spec §14.5 Rule 2) over the out-of-sample window —
     this is Rule 4's single permitted touch, mechanically guarded by
     ``robustness.run_grid[_parallel]()`` and logged to ``oos_access_log.md`` exactly like
     every other grid run.
  4. Writes ``results/oos_check/`` (``grid_729_oos.csv``, ``deflated_sharpe_oos.md``,
     ``marginal_surfaces_oos.png``) and ``OOS_REPORT.md``, comparing the out-of-sample findings
     against the in-sample grid already on record (``results/robustness/grid_729.csv``).

Per Rule 4 ("single evaluation, no iteration"): this script's findings are not fed back into
``ROBUSTNESS_GRID``, ``BacktestParams`` defaults, or any other tuning input, and ``OOS_REPORT.md``
does not propose retuning. If this run disagrees with the in-sample conclusion, the correct
response is to write that down and treat the strategy as not robust — not to re-run this file.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from backtest.p21_momentum.cost_sensitivity import (  # noqa: E402
    edge_survives_10bps,
    run_slippage_sweep,
)
from backtest.p21_momentum.fetch_frozen_panel import (  # noqa: E402
    load_frozen_panel,
    load_frozen_sectors,
)
from backtest.p21_momentum.metrics import compute_mechanical_metrics, compute_return_metrics  # noqa: E402
from backtest.p21_momentum.robustness import (  # noqa: E402
    IN_SAMPLE_END,
    IN_SAMPLE_START,
    OUT_OF_SAMPLE_END,
    OUT_OF_SAMPLE_START,
    deflated_sharpe_band,
    is_top_quartile_separated,
    render_deflated_sharpe_md,
    render_grid_csv,
    render_marginal_surfaces_png,
    run_grid,
    run_grid_parallel,
)
from backtest.p21_momentum.runner import run_backtest
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IN_SAMPLE_GRID_CSV = RESULTS_DIR / "robustness" / "grid_729.csv"
OOS_DIR = RESULTS_DIR / "oos_check"
OOS_REPORT_PATH = Path(__file__).resolve().parent / "OOS_REPORT.md"


@dataclass(slots=True)
class OosPointEstimate:
    """One default-parameter backtest confined to the out-of-sample window — a point estimate, not a search."""

    cagr_a: float
    sharpe_a: float
    cagr_c: float
    sharpe_c: float
    edge_cagr_a_minus_c: float
    edge_survives_10bps: Optional[bool]
    turnover_annualized_median_pct: float


@dataclass(slots=True)
class GridSummary:
    """Spec §14.5 Rule 3's deflated-Sharpe read of one grid run (in-sample or out-of-sample)."""

    n_trials: int
    n_observations: int
    band_low: float
    band_high: float
    best_sharpe: float
    separated: bool


def run_oos_point_estimate(
    panel, sector_by_ticker, start=OUT_OF_SAMPLE_START, end=OUT_OF_SAMPLE_END
) -> OosPointEstimate:
    """
    Run the strategy at default parameters, confined to the out-of-sample window only.

    Not a search over ``ROBUSTNESS_GRID`` — a single fixed-parameter run — so unlike
    :func:`~backtest.p21_momentum.robustness.run_grid`, this does not touch Rule 4's guard
    or ``oos_access_log.md``.
    """
    base_result = run_backtest(panel, sector_by_ticker, start, end)
    mechanical = compute_mechanical_metrics(base_result)
    return_a = compute_return_metrics(base_result.nav_daily["nav_a"], base_result.nav_daily["nav_c"])
    return_c = compute_return_metrics(base_result.nav_daily["nav_c"], base_result.nav_daily["nav_c"])

    slippage_results = run_slippage_sweep(panel, sector_by_ticker, start, end, slippage_levels_bps=(3.0, 10.0))
    edge_ok = edge_survives_10bps(slippage_results) if slippage_results else None

    return OosPointEstimate(
        cagr_a=return_a.cagr,
        sharpe_a=return_a.sharpe,
        cagr_c=return_c.cagr,
        sharpe_c=return_c.sharpe,
        edge_cagr_a_minus_c=return_a.cagr - return_c.cagr,
        edge_survives_10bps=edge_ok,
        turnover_annualized_median_pct=mechanical.turnover_annualized_median_pct,
    )


def summarize_grid(sharpes: Sequence[float], start, end) -> GridSummary:
    """Apply spec §14.5 Rule 3 to one grid's ``sharpe_a`` column — pure, reused for both windows."""
    n_months = (end.year - start.year) * 12 + (end.month - start.month)
    low, high = deflated_sharpe_band(len(sharpes), n_months)
    return GridSummary(
        n_trials=len(sharpes),
        n_observations=n_months,
        band_low=low,
        band_high=high,
        best_sharpe=max(sharpes) if sharpes else 0.0,
        separated=is_top_quartile_separated(sharpes),
    )


def render_oos_report_md(in_sample: GridSummary, out_of_sample: GridSummary, point: OosPointEstimate) -> str:
    """Render ``OOS_REPORT.md`` — findings only, per Rule 4's "single evaluation, no iteration"."""
    edge_word = (
        "survives" if point.edge_survives_10bps else ("does not survive" if point.edge_survives_10bps is not None else "N/A")
    )
    lines = [
        "# P21 Momentum — Out-of-Sample Check (spec §14.5 Rule 4)",
        "",
        f"Single evaluation of {OUT_OF_SAMPLE_START.isoformat()} to {OUT_OF_SAMPLE_END.isoformat()}, logged in "
        "`oos_access_log.md`. Per Rule 4 this window is touched exactly once; the findings below are reported "
        "as observed, not fed back into tuning.",
        "",
        "## Point estimate (default parameters, no search)",
        "",
        "| Metric | Track A | Track C |",
        "|---|---|---|",
        f"| CAGR | {point.cagr_a * 100:.1f}% | {point.cagr_c * 100:.1f}% |",
        f"| Sharpe | {point.sharpe_a:.2f} | {point.sharpe_c:.2f} |",
        "",
        f"Edge over C (CAGR A - CAGR C): {point.edge_cagr_a_minus_c * 100:+.1f}%. "
        f"Edge at 10 bps vs. 3 bps: **{edge_word}** (spec §14.9 B8's test, applied to this window alone).",
        f"Median annualized turnover: {point.turnover_annualized_median_pct:.0f}%.",
        "",
        "## Parameter-grid comparison: does the in-sample conclusion hold?",
        "",
        "| | In-sample (2005-2016) | Out-of-sample (2017-2026) |",
        "|---|---|---|",
        f"| Trials | {in_sample.n_trials} | {out_of_sample.n_trials} |",
        f"| Observations (months) | {in_sample.n_observations} | {out_of_sample.n_observations} |",
        f"| Expected-by-chance Sharpe band | {in_sample.band_low:.2f} - {in_sample.band_high:.2f} | "
        f"{out_of_sample.band_low:.2f} - {out_of_sample.band_high:.2f} |",
        f"| Best observed Sharpe (track A) | {in_sample.best_sharpe:.2f} | {out_of_sample.best_sharpe:.2f} |",
        f"| Top-quartile separated? | {'yes' if in_sample.separated else 'no'} | "
        f"{'yes' if out_of_sample.separated else 'no'} |",
        "",
    ]

    oos_separation_word = "separated" if out_of_sample.separated else "not separated"
    in_sample_separation_word = "separated" if in_sample.separated else "not separated"
    if in_sample.separated == out_of_sample.separated:
        in_sample_conclusion = (
            "a config was separated from the top quartile"
            if in_sample.separated
            else "flat surface, use the literature defaults"
        )
        lines.append(
            f"The out-of-sample grid's separation result ({oos_separation_word}) matches the in-sample grid's. "
            f"The in-sample conclusion ({in_sample_conclusion}) is not contradicted by this out-of-sample look."
        )
    else:
        lines.append(
            "The out-of-sample grid's separation result **disagrees** with the in-sample grid's "
            f"({in_sample_separation_word} in-sample vs. {oos_separation_word} out-of-sample). Per Rule 1, this "
            "is not a reason to retune — it is a reason to trust the parameter surface less than the in-sample "
            "grid alone suggested."
        )
    lines.append("")

    if point.edge_survives_10bps is False:
        lines.append(
            "The strategy's edge over track C does not survive realistic costs in this out-of-sample window "
            "on its own — read alongside the base-case B8 result (which covers the full 2005-2026 history "
            "including this window) before drawing a conclusion from this slice in isolation."
        )
    elif point.edge_survives_10bps is True:
        lines.append("The strategy's edge over track C survives realistic costs in this out-of-sample window.")
    lines.append("")

    lines.append("## Discipline note")
    lines.append("")
    lines.append(
        "This window must not be evaluated again. Any further work on P21 parameters should use the "
        "literature defaults, not this run's numbers, per spec §14.5 Rule 1 and Rule 4."
    )
    lines.append("")
    return "\n".join(lines)


def run_oos_check(
    acknowledge_oos_reaccess: bool = False,
    sequential_grid: bool = False,
    grid_max_workers: Optional[int] = None,
) -> Path:
    """Run the full one-time out-of-sample study and write ``OOS_REPORT.md``. See module docstring for order."""
    _logger.info("Loading frozen panel...")
    panel = load_frozen_panel()
    sector_by_ticker = load_frozen_sectors()

    _logger.info(
        "Running the out-of-sample point estimate (%s to %s, default params, no search)...",
        OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END,
    )
    point = run_oos_point_estimate(panel, sector_by_ticker)

    _logger.info("Running the out-of-sample robustness grid (spec §14.5 Rule 4's single permitted touch)...")
    OOS_DIR.mkdir(parents=True, exist_ok=True)
    if sequential_grid:
        oos_grid_rows: List = run_grid(
            panel, sector_by_ticker, OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END,
            acknowledge_oos_reaccess=acknowledge_oos_reaccess,
        )
    else:
        oos_grid_rows = run_grid_parallel(
            panel, sector_by_ticker, OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END,
            max_workers=grid_max_workers, acknowledge_oos_reaccess=acknowledge_oos_reaccess,
        )
        oos_grid_rows = sorted(
            oos_grid_rows,
            key=lambda r: (r.lookback_start, r.skip_recent, r.entry_rank, r.hold_rank, r.max_per_sector, r.vix_threshold),
        )

    render_grid_csv(oos_grid_rows).to_csv(OOS_DIR / "grid_729_oos.csv", index=False)
    oos_n_months = (
        (OUT_OF_SAMPLE_END.year - OUT_OF_SAMPLE_START.year) * 12 + (OUT_OF_SAMPLE_END.month - OUT_OF_SAMPLE_START.month)
    )
    (OOS_DIR / "deflated_sharpe_oos.md").write_text(
        render_deflated_sharpe_md(oos_grid_rows, oos_n_months), encoding="utf-8"
    )
    try:
        render_marginal_surfaces_png(oos_grid_rows, OOS_DIR / "marginal_surfaces_oos.png")
    except ImportError:
        _logger.warning("matplotlib unavailable — skipped marginal_surfaces_oos.png")

    in_sample_sharpes = pd.read_csv(IN_SAMPLE_GRID_CSV)["sharpe_a"].tolist()
    in_sample_summary = summarize_grid(in_sample_sharpes, IN_SAMPLE_START, IN_SAMPLE_END)
    oos_summary = summarize_grid([r.sharpe_a for r in oos_grid_rows], OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END)

    report_md = render_oos_report_md(in_sample_summary, oos_summary, point)
    OOS_REPORT_PATH.write_text(report_md, encoding="utf-8")
    _logger.info("Wrote %s", OOS_REPORT_PATH)
    return OOS_REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acknowledge-oos-reaccess", action="store_true",
        help="Acknowledge a repeated out-of-sample touch (spec §14.5 Rule 4) — should not normally be needed.",
    )
    parser.add_argument(
        "--sequential-grid", action="store_true",
        help="Run the robustness grid single-process (default: parallel across all CPU cores).",
    )
    parser.add_argument(
        "--grid-workers", type=int, default=None,
        help="Worker process count for the parallel robustness grid (default: os.cpu_count()).",
    )
    args = parser.parse_args()

    run_oos_check(
        acknowledge_oos_reaccess=args.acknowledge_oos_reaccess,
        sequential_grid=args.sequential_grid,
        grid_max_workers=args.grid_workers,
    )


if __name__ == "__main__":
    main()
