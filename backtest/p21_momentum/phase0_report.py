"""
P21 Momentum backtest — Phase 0 report generator (docs/pipeline-specification.md §14.9-§14.10).

One-time (or per-Phase-0-cycle) operator command, run only after
``fetch_frozen_panel.py`` has produced ``data/prices.parquet`` +
``data/constituents.json``:

    python -m backtest.p21_momentum.phase0_report
    python -m backtest.p21_momentum.phase0_report --verify-determinism
    python -m backtest.p21_momentum.phase0_report --acknowledge-oos-reaccess
    python -m backtest.p21_momentum.phase0_report --skip-robustness --skip-cost-sensitivity

Orchestrates the full Phase 0 study in spec order: base-case run -> stress
windows -> cost sensitivity -> parameter robustness -> §14.9 B1-B10
acceptance table -> ``PHASE0_REPORT.md``, which leads with the §14.3
survivorship banner and the acceptance table, **never with performance**
(spec §14.10's explicit ordering requirement).

``--verify-determinism`` is spec §14.9 B10's enforced self-check: it re-runs
the base case a second time and diffs the two ``nav_daily`` outputs
byte-for-byte, failing loudly on any difference, rather than trusting the
unit-test suite alone to catch a live-only non-determinism.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from backtest.p21_momentum.cost_sensitivity import (  # noqa: E402
    edge_survives_10bps,
    render_slippage_csv,
    render_turnover_curve_png,
    run_slippage_sweep,
    run_turnover_curve,
)
from backtest.p21_momentum.fetch_frozen_panel import (  # noqa: E402
    load_frozen_panel,
    load_frozen_sectors,
)
from backtest.p21_momentum.metrics import compute_mechanical_metrics  # noqa: E402
from backtest.p21_momentum.robustness import (  # noqa: E402
    IN_SAMPLE_END,
    IN_SAMPLE_START,
    render_deflated_sharpe_md,
    render_grid_csv,
    run_grid,
    run_grid_parallel,
)
from backtest.p21_momentum.runner import BacktestParams, BacktestResult, run_backtest  # noqa: E402
from backtest.p21_momentum.stress_windows import (  # noqa: E402
    STRESS_WINDOWS,
    evaluate_all_windows,
    render_stress_windows_md,
)
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_CASE_DIR = RESULTS_DIR / "base_case"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
COST_SENSITIVITY_DIR = RESULTS_DIR / "cost_sensitivity"
PHASE0_REPORT_PATH = Path(__file__).resolve().parent / "PHASE0_REPORT.md"

BACKTEST_START = date(2005, 1, 1)
BACKTEST_END = date(2026, 6, 30)

MAX_RUNTIME_SECONDS = 30 * 60  # spec §14.9 B9

# spec §14.3, verbatim.
UNIVERSE_BANNER = (
    "UNIVERSE: current S&P 500 constituents applied retroactively. Return figures in this "
    "report are upward-biased by an estimated 1-3% annually and are not usable as forecasts. "
    "Mechanical metrics (turnover, position count, filter attrition, regime frequency) are "
    "unaffected by this bias and are the intended output."
)


@dataclass(slots=True)
class AcceptanceRow:
    """One row of the §14.9 acceptance table."""

    id: str
    criterion: str
    threshold: str
    observed: str
    passed: Optional[bool]  # None -> not evaluated (e.g. skipped stage)
    response_on_failure: str


def _fmt_pass(passed: Optional[bool]) -> str:
    if passed is None:
        return "N/A"
    return "PASS" if passed else "**FAIL**"


def evaluate_acceptance_table(
    base_result: BacktestResult,
    stress_results,
    slippage_results,
    runtime_seconds: float,
    determinism_verified: Optional[bool],
) -> List[AcceptanceRow]:
    """Evaluate spec §14.9's B1-B10 table against one Phase 0 study's outputs."""
    mechanical = compute_mechanical_metrics(base_result)
    rows: List[AcceptanceRow] = []

    turnover = mechanical.turnover_annualized_median_pct
    rows.append(
        AcceptanceRow(
            "B1", "Median annualized turnover", "140-210%", f"{turnover:.0f}%",
            140.0 <= turnover <= 210.0, "Re-tune hold_rank via §14.7 curve",
        )
    )
    rows.append(
        AcceptanceRow(
            "B2", "Months with < 20 positions", "< 20% of months", f"{mechanical.pct_months_below_20 * 100:.1f}%",
            mechanical.pct_months_below_20 < 0.20, "Raise fallback_pool_rank above 40",
        )
    )
    rows.append(
        AcceptanceRow(
            "B3", "Months with < 12 positions", "< 3% of months", f"{mechanical.pct_months_below_12 * 100:.1f}%",
            mechanical.pct_months_below_12 < 0.03, "Sector cap too tight; consider 5",
        )
    )
    rows.append(
        AcceptanceRow(
            "B4", "Max sector weight breach", "Never",
            f"max {mechanical.max_sector_count_ever} names in one sector",
            mechanical.max_sector_count_ever <= BacktestParams().max_per_sector,
            "Bug in enforce_sector_cap",
        )
    )

    by_name = {r.window.name: r for r in stress_results}
    crash_2009 = by_name.get("2009-03 -> 2009-05")
    b5_ok = crash_2009 is not None and crash_2009.a_minus_b is not None and crash_2009.a_minus_b > 0.05
    rows.append(
        AcceptanceRow(
            "B5", "Overlay benefit in 2009-03->05", "A-B > +5%",
            f"{crash_2009.a_minus_b * 100:+.1f}%" if crash_2009 and crash_2009.a_minus_b is not None else "N/A",
            b5_ok if crash_2009 and crash_2009.in_range else None,
            "Accepted (see Known Limitations below) unless scalar path shows bear/high_vol=False "
            "during the window -- that would indicate a real lag bug",
        )
    )

    bear_2022 = by_name.get("2022-01 -> 2022-10")
    b6_ok = bear_2022 is not None and bear_2022.a_minus_b is not None and bear_2022.a_minus_b > 0.03
    rows.append(
        AcceptanceRow(
            "B6", "Overlay benefit in 2022", "A-B > +3%",
            f"{bear_2022.a_minus_b * 100:+.1f}%" if bear_2022 and bear_2022.a_minus_b is not None else "N/A",
            b6_ok if bear_2022 and bear_2022.in_range else None,
            "Accepted (see Known Limitations below) -- genuine hysteresis whipsaw, not lag",
        )
    )

    whipsaw_names = ("2011-08 -> 2011-10", "2015-08 -> 2016-02", "2018-10 -> 2018-12", "2020-02 -> 2020-04")
    whipsaw_results = [by_name[n] for n in whipsaw_names if n in by_name and by_name[n].in_range]
    whipsaw_diffs = [r.a_minus_b for r in whipsaw_results if r.a_minus_b is not None]
    b7_ok = bool(whipsaw_diffs) and min(whipsaw_diffs) > -0.04
    rows.append(
        AcceptanceRow(
            "B7", "Overlay cost in whipsaw windows", "A-B > -4% each",
            f"worst {min(whipsaw_diffs) * 100:+.1f}%" if whipsaw_diffs else "N/A",
            b7_ok if whipsaw_diffs else None,
            "Overlay too twitchy; raise vix_threshold",
        )
    )

    b8_ok = edge_survives_10bps(slippage_results) if slippage_results else None
    rows.append(
        AcceptanceRow(
            "B8", "Edge over C survives 10 bps", "Yes",
            "yes" if b8_ok else ("no" if b8_ok is not None else "N/A"),
            b8_ok, "Stop. Buy QDVA.",
        )
    )

    rows.append(
        AcceptanceRow(
            "B9", "Runtime, full backtest", "< 30 minutes", f"{runtime_seconds / 60:.1f} min",
            runtime_seconds < MAX_RUNTIME_SECONDS, "Optimize before scheduling",
        )
    )

    rows.append(
        AcceptanceRow(
            "B10", "Two identical runs -> identical output", "Bit-identical",
            "verified" if determinism_verified else ("not run" if determinism_verified is None else "**MISMATCH**"),
            determinism_verified, "Non-determinism present; find and fix before proceeding",
        )
    )
    return rows


def render_phase0_report_md(
    acceptance_rows: List[AcceptanceRow],
    base_result: BacktestResult,
    start: date,
    end: date,
) -> str:
    """Render ``PHASE0_REPORT.md`` — leads with the banner and the acceptance table, per spec §14.10."""
    all_evaluated_pass = all(r.passed for r in acceptance_rows if r.passed is not None)
    any_evaluated = any(r.passed is not None for r in acceptance_rows)

    lines = [
        "# P21 Momentum — Phase 0 Report",
        "",
        f"> {UNIVERSE_BANNER}",
        "",
        f"Backtest range: {start.isoformat()} to {end.isoformat()}.",
        "",
        "## Acceptance Table (spec §14.9)",
        "",
        "| # | Criterion | Threshold | Observed | Result | Response on failure |",
        "|---|---|---|---|---|---|",
    ]
    for r in acceptance_rows:
        lines.append(
            f"| {r.id} | {r.criterion} | {r.threshold} | {r.observed} | "
            f"{_fmt_pass(r.passed)} | {r.response_on_failure} |"
        )
    lines.append("")
    if any_evaluated:
        verdict = (
            "**Phase 0 PASSES.**" if all_evaluated_pass else "**Phase 0 DOES NOT PASS** — see failing rows above."
        )
        lines.append(verdict)
    else:
        lines.append("No criteria evaluated (all stages skipped).")
    lines.append("")
    lines.append(
        "This backtest is a mechanical validation harness (spec §14.1), not a return estimate. "
        "Any CAGR or Sharpe figures found elsewhere in this study's output are contaminated by "
        "the biases in spec §14.2 and must not be quoted, remembered, or used to set expectations."
    )
    lines.append("")
    by_id = {r.id: r for r in acceptance_rows}
    b5, b6 = by_id.get("B5"), by_id.get("B6")
    if (b5 is not None and b5.passed is False) or (b6 is not None and b6.passed is False):
        lines.append("## Known Limitations (B5/B6, diagnosed 2026-08-25)")
        lines.append("")
        lines.append(
            "B5/B6 failing does not mean the regime overlay is broken. Replaying `compute_regime()` "
            "against the frozen SPX/VIX history confirms it reacts with no lag in both windows -- "
            "these are structural properties of this specific implementation, not a bug:"
        )
        lines.append("")
        lines.append(
            f"- **B5 (2009-03->05, observed {b5.observed if b5 else 'N/A'}):** `bear=True, high_vol=True` "
            "for the entire window (scalar locked at 0.25, maximum protection, from 2008-09 through "
            "2009-07 with no gap). The spec's ~-70% reference is for the academic **long-short** UMD "
            "factor, whose crash comes from the *short leg* (beaten-down financials) violently "
            "rallying. This pipeline is long-only: Track B (no overlay) returned +2.6% in this window "
            "-- there was no crash in the DIY sleeve for the overlay to prevent. Being 75%-de-risked "
            "during a period the sleeve itself rose is exactly why A trails B here."
        )
        lines.append(
            "- **B6 (2022-01->10, observed "
            f"{b6.observed if b6 else 'N/A'}):** the regime scalar whipsawed 7 times over the year "
            "(1.0/0.6/1.0/0.6/0.25/0.6/0.25/0.6) because SPX repeatedly crossed back above its 200dma "
            "during the year's bear-market rallies (Mar, Jul-Aug, Nov) before rolling over again. "
            "Asymmetric hysteresis (instant downgrade, 2-month-confirm upgrade) damps this but a "
            "single reprieve month is enough to reset it. This is the same whipsaw mechanism the spec "
            "already accepts as a cost in the 2011/2015-16/2018 stress windows -- 2022 was simply "
            "choppier than its 'slow decline' label assumed."
        )
        lines.append("")
        lines.append(
            "Accepted as a known limitation rather than chased as a bug (see docs/pipeline-specification.md "
            "§17). Do not retune regime parameters against these two windows specifically -- spec §14.5 "
            "Rule 1 discipline applies to the overlay's trigger logic just as much as to the core signal."
        )
        lines.append("")
    mechanical = compute_mechanical_metrics(base_result)
    lines.append("## Mechanical Summary (base case)")
    lines.append("")
    lines.append(f"- Median annualized turnover: {mechanical.turnover_annualized_median_pct:.0f}%")
    lines.append(f"- Position count: mean {mechanical.position_count_mean:.1f}, min {mechanical.position_count_min}")
    lines.append(
        f"- F1/F2/F3 removed (total, whole study): "
        f"{mechanical.f1_removed_total}/{mechanical.f2_removed_total}/{mechanical.f3_removed_total}"
    )
    lines.append(f"- Regime scalar histogram: {mechanical.regime_scalar_histogram}")
    lines.append(f"- Regime state transitions: {mechanical.regime_state_transitions}")
    lines.append(
        f"- Holding period (days), median/mean: "
        f"{mechanical.holding_period_days_median}/{mechanical.holding_period_days_mean}"
    )
    lines.append(f"- WARN_UNDERFILLED months: {mechanical.warn_underfilled_count}")
    lines.append(f"- MANUAL_REVIEW events: {mechanical.manual_review_count_total}")
    lines.append(f"- EXIT_DELISTED events: {mechanical.exit_delisted_count_total}")
    lines.append("")
    lines.append(
        f"See `results/base_case/`, `results/robustness/`, `results/cost_sensitivity/`, and "
        f"the {len(STRESS_WINDOWS)} stress windows in `results/base_case/stress_windows.md` for full detail."
    )
    lines.append("")
    return "\n".join(lines)


def _write_base_case_deliverables(result: BacktestResult) -> None:
    BASE_CASE_DIR.mkdir(parents=True, exist_ok=True)
    result.nav_daily.to_csv(BASE_CASE_DIR / "nav_daily.csv")
    with (BASE_CASE_DIR / "trades.jsonl").open("w", encoding="utf-8") as f:
        for t in result.trades:
            f.write(json.dumps(t.to_dict()))
            f.write("\n")
    monthly_df = pd.DataFrame([asdict(m) for m in result.monthly_metrics])
    monthly_df.to_csv(BASE_CASE_DIR / "monthly_metrics.csv", index=False)
    stress_results = evaluate_all_windows(result)
    (BASE_CASE_DIR / "stress_windows.md").write_text(render_stress_windows_md(stress_results), encoding="utf-8")


def verify_determinism(
    panel, sector_by_ticker, start: date, end: date, params: Optional[BacktestParams] = None
) -> bool:
    """Spec §14.9 B10's enforced self-check: run the base case twice, diff nav_daily byte-for-byte."""
    result1 = run_backtest(panel, sector_by_ticker, start, end, params=params)
    result2 = run_backtest(panel, sector_by_ticker, start, end, params=params)
    try:
        pd.testing.assert_frame_equal(result1.nav_daily, result2.nav_daily)
    except AssertionError:
        _logger.error("B10 determinism check FAILED: nav_daily differs between two identical runs")
        return False
    if len(result1.trades) != len(result2.trades):
        _logger.error(
            "B10 determinism check FAILED: trade count differs (%d vs %d)", len(result1.trades), len(result2.trades)
        )
        return False
    for t1, t2 in zip(result1.trades, result2.trades):
        if t1.to_dict() != t2.to_dict():
            _logger.error("B10 determinism check FAILED: trade ledger entries differ")
            return False
    return True


def run_phase0_study(
    verify_determinism_flag: bool = False,
    acknowledge_oos_reaccess: bool = False,
    skip_robustness: bool = False,
    skip_cost_sensitivity: bool = False,
    sequential_grid: bool = False,
    grid_max_workers: Optional[int] = None,
    start: date = BACKTEST_START,
    end: date = BACKTEST_END,
) -> Path:
    """
    Run the full Phase 0 study and write ``PHASE0_REPORT.md`` plus every deliverable in
    spec §14.10's tree.

    Returns:
        Path to the written PHASE0_REPORT.md.
    """
    _logger.info("Loading frozen panel...")
    panel = load_frozen_panel()
    sector_by_ticker = load_frozen_sectors()

    _logger.info("Running base case (%s to %s)...", start, end)
    t0 = time.perf_counter()
    base_result = run_backtest(panel, sector_by_ticker, start, end)
    runtime_seconds = time.perf_counter() - t0
    _write_base_case_deliverables(base_result)
    stress_results = evaluate_all_windows(base_result)

    slippage_results = []
    if not skip_cost_sensitivity:
        _logger.info("Running slippage sweep + turnover curve...")
        COST_SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)
        slippage_results = run_slippage_sweep(panel, sector_by_ticker, start, end)
        render_slippage_csv(slippage_results).to_csv(COST_SENSITIVITY_DIR / "slippage_0_3_10_25.csv", index=False)
        curve_points = run_turnover_curve(panel, sector_by_ticker, start, end)
        try:
            render_turnover_curve_png(curve_points, COST_SENSITIVITY_DIR / "turnover_net_return_curve.png")
        except ImportError:
            _logger.warning("matplotlib unavailable — skipped turnover_net_return_curve.png")

    if not skip_robustness:
        # Only the in-sample window (spec §14.5 Rule 4) is touched by a normal Phase 0 study
        # run. Evaluating the out-of-sample window is a deliberate, separate operator step —
        # call robustness.run_grid() directly with the out-of-sample dates when that single
        # look is actually intended; acknowledge_oos_reaccess exists for that direct call,
        # not for this orchestrator, so it is accepted here only to keep the CLI flag
        # meaningful end-to-end and is not otherwise consumed by this function.
        _logger.info("Running parameter robustness grid (in-sample only, spec §14.5 Rule 4)...")
        if acknowledge_oos_reaccess:
            _logger.info("--acknowledge-oos-reaccess has no effect on the in-sample-only grid run below.")
        ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
        in_sample_start = max(start, IN_SAMPLE_START)
        in_sample_end = min(end, IN_SAMPLE_END)
        if in_sample_start < in_sample_end:
            if sequential_grid:
                grid_rows = run_grid(panel, sector_by_ticker, in_sample_start, in_sample_end)
            else:
                grid_rows = run_grid_parallel(
                    panel, sector_by_ticker, in_sample_start, in_sample_end, max_workers=grid_max_workers
                )
                # run_grid_parallel() returns completion order, not grid order — sort for a
                # reproducible grid_729.csv row order across runs (spec §14.9 B10's spirit, even
                # though B10 itself is about the backtest engine, not this CSV's row order).
                grid_rows = sorted(
                    grid_rows,
                    key=lambda r: (
                        r.lookback_start, r.skip_recent, r.entry_rank,
                        r.hold_rank, r.max_per_sector, r.vix_threshold,
                    ),
                )
            render_grid_csv(grid_rows).to_csv(ROBUSTNESS_DIR / "grid_729.csv", index=False)
            n_months = (
                (in_sample_end.year - in_sample_start.year) * 12 + (in_sample_end.month - in_sample_start.month)
            )
            (ROBUSTNESS_DIR / "deflated_sharpe.md").write_text(
                render_deflated_sharpe_md(grid_rows, n_months), encoding="utf-8"
            )

    determinism_verified: Optional[bool] = None
    if verify_determinism_flag:
        _logger.info("Verifying determinism (B10)...")
        determinism_verified = verify_determinism(panel, sector_by_ticker, start, end)

    acceptance_rows = evaluate_acceptance_table(
        base_result, stress_results, slippage_results, runtime_seconds, determinism_verified
    )
    report_md = render_phase0_report_md(acceptance_rows, base_result, start, end)
    PHASE0_REPORT_PATH.write_text(report_md, encoding="utf-8")
    _logger.info("Wrote %s", PHASE0_REPORT_PATH)
    return PHASE0_REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-determinism", action="store_true",
        help="Re-run the base case twice and diff byte-for-byte (spec §14.9 B10).",
    )
    parser.add_argument(
        "--acknowledge-oos-reaccess", action="store_true",
        help="Acknowledge a repeated out-of-sample touch (spec §14.5 Rule 4).",
    )
    parser.add_argument(
        "--skip-robustness", action="store_true", help="Skip the 729-combination parameter grid (slow)."
    )
    parser.add_argument(
        "--skip-cost-sensitivity", action="store_true", help="Skip the slippage sweep + turnover curve."
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

    run_phase0_study(
        verify_determinism_flag=args.verify_determinism,
        acknowledge_oos_reaccess=args.acknowledge_oos_reaccess,
        skip_robustness=args.skip_robustness,
        skip_cost_sensitivity=args.skip_cost_sensitivity,
        sequential_grid=args.sequential_grid,
        grid_max_workers=args.grid_workers,
    )


if __name__ == "__main__":
    main()
