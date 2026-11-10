"""
P22 — walk-forward backtest CLI (spec §0.3, §8.4, M6). Human-run, read-only. The weekly scheduler
job `jobs/run_backtest.py` runs the same backtest with an automatic `--labels-through`, then trains
the live learned model and emails the results.

    python -m src.ml.pipeline.p22_biotech_ma.cli.backtest_cli --labels-through 2026-10-30
    python -m src.ml.pipeline.p22_biotech_ma.cli.backtest_cli --start 2016-01-01 --end 2023-12-31 \
        --labels-through 2026-10-30 --step-months 3

`--labels-through` is the date the deal review is complete up to; snapshots whose 24-month window
runs past it are skipped. Until the historical candidate queue is reviewed, positives are missing
and every lift is understated. Writes `results/p22_biotech_ma/YYYY-MM-DD/backtest/`:
`metrics_all.csv`, `metrics_tier0.csv`, `metrics_learned_window.csv` (every ranking over only the
snapshots the learned model ranked), `snapshots.csv`, `return_labels.csv`, `scores.csv`, `summary.json`.
Needs the DERA quarters in the raw zone (`run_sec_universe_ingest.py`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.backtest.harness import BacktestResult, BacktestWindow, run_backtest
from src.ml.pipeline.p22_biotech_ma.backtest.labels import DealLabels, add_months, load_cvr_policy
from src.ml.pipeline.p22_biotech_ma.backtest.metrics import DEFAULT_KS
from src.ml.pipeline.p22_biotech_ma.ingest.acquirer_config import load_acquirers
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import build_universe_history
from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import all_landed_quarters
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import config_hash, load_scoring_config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Spec §0.3 success criteria.
TARGET_LIFT_AT_50 = 3.0
TARGET_PRECISION_AT_25 = 0.12
TARGET_RECALL_200_VS_RANDOM = 2.0


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))  # union, first-seen order
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(rows: Sequence[Dict[str, Any]], ranking: str, k: int, key: str) -> Optional[float]:
    for row in rows:
        if row["ranking"] == ranking and row["k"] == k:
            return row[key]
    return None


def success_criteria(rows: Sequence[Dict[str, Any]], ranking: str = "model") -> Dict[str, Any]:
    """Spec §0.3 checks for `ranking`, plus its lift@50 against naive-informed (the bar that matters)."""
    lift50 = _metric(rows, ranking, 50, "lift")
    precision25 = _metric(rows, ranking, 25, "precision")
    recall200 = _metric(rows, ranking, 200, "recall")
    random_recall200 = _metric(rows, ranking, 200, "random_recall")
    naive_lift50 = _metric(rows, "naive_informed", 50, "lift")
    recall_ratio = recall200 / random_recall200 if recall200 is not None and random_recall200 else None
    return {
        "lift_at_50": lift50,
        "lift_at_50_pass": lift50 is not None and lift50 >= TARGET_LIFT_AT_50,
        "precision_at_25": precision25,
        "precision_at_25_pass": precision25 is not None and precision25 >= TARGET_PRECISION_AT_25,
        "recall_at_200_vs_random": recall_ratio,
        "recall_at_200_pass": recall_ratio is not None and recall_ratio >= TARGET_RECALL_200_VS_RANDOM,
        "naive_informed_lift_at_50": naive_lift50,
        "beats_naive_informed_at_50": lift50 is not None and naive_lift50 is not None and lift50 > naive_lift50,
    }


def _log_table(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    _logger.info("%s", title)
    for r in rows:
        _logger.info("  %-34s k=%-3d hits=%-4d positives=%-5d precision=%s (random %s) lift=%s recall=%s (random %s)",
                     r["ranking"], r["k"], r["hits"], r["positives"], _fmt(r["precision"]), _fmt(r["random_precision"]),
                     _fmt(r["lift"]), _fmt(r["recall"]), _fmt(r["random_recall"]))


def _log_portfolio(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    _logger.info("%s", title)
    for r in rows:
        _logger.info("  %-20s k=%-3s mean=%s universe=%s excess=%s beat=%s stressed_excess=%s deal=%s price_ended=%s",
                     r["ranking"], r["k"], _fmt(r["mean_return"]), _fmt(r["universe_return"]), _fmt(r["excess"]),
                     _fmt(r["beat_share"]), _fmt(r["stressed_excess"]), _fmt(r["deal_share"]), _fmt(r["price_ended_share"]))


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_results(result: BacktestResult, out_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Write the CSVs and summary.json; returns the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, tier0_rows = result.all_names.as_rows(), result.tier0.as_rows()
    window_rows = result.learned_window.as_rows()
    _write_csv(out_dir / "metrics_all.csv", all_rows)
    _write_csv(out_dir / "metrics_tier0.csv", tier0_rows)
    _write_csv(out_dir / "metrics_learned_window.csv", window_rows)
    _write_csv(out_dir / "snapshots.csv", result.snapshots)
    _write_csv(out_dir / "return_labels.csv", result.return_rows)
    _write_csv(out_dir / "scores.csv", result.score_rows)
    portfolio_all, portfolio_window = result.portfolio_all.as_rows(), result.portfolio_learned_window.as_rows()
    _write_csv(out_dir / "portfolio_all.csv", portfolio_all)
    _write_csv(out_dir / "portfolio_learned_window.csv", portfolio_window)
    summary = {
        **meta,
        "snapshots": len(result.snapshots),
        "criteria_all_names": success_criteria(all_rows),
        "criteria_tier0": success_criteria(tier0_rows),
        "criteria_learned_window": {name: success_criteria(window_rows, name) for name in ("learned", "model")},
        "metrics_all": all_rows,
        "metrics_tier0": tier0_rows,
        "metrics_learned_window": window_rows,
        "portfolio_all": portfolio_all,
        "portfolio_learned_window": portfolio_window,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log_table("Backtest — all names", all_rows)
    _log_table("Backtest — Tier 0 names (spec §9: the honest measure of Blocks A–F)", tier0_rows)
    _log_table("Backtest — snapshots ranked by the learned model only (like-for-like)", window_rows)
    _log_portfolio("Portfolio — top k held 24 months or to the deal, learned window", portfolio_window)
    _logger.info("Success criteria, all names: %s", summary["criteria_all_names"])
    _logger.info("Success criteria, Tier 0: %s", summary["criteria_tier0"])
    _logger.info("Success criteria, learned window: %s", summary["criteria_learned_window"])
    return summary


def execute(start: date, end: Optional[date], labels_through: date, step_months: int, ks: Sequence[int],
            out_dir: Path) -> Optional[Tuple[BacktestResult, Dict[str, Any]]]:
    """
    Run the backtest and write its files to `out_dir`. Returns the result and summary, or `None`
    when no DERA quarter has landed. Shared by this CLI and the weekly job (`jobs/run_backtest.py`).
    """
    end = end or add_months(labels_through, -24)
    quarters_rows = all_landed_quarters()
    if not quarters_rows:
        _logger.error("No landed DERA quarters — run run_sec_universe_ingest.py first")
        return None
    # Exchange eligibility needs today's ticker map, which is not point in time, so it is left out.
    history = build_universe_history(quarters_rows, {})
    window = BacktestWindow(start=start, end=end, labels_complete_through=labels_through, step_months=step_months, ks=ks)

    with DatabaseService().uow() as uow:
        deals = uow.p22.list_deals()
        labels = DealLabels(deals)
        _logger.info("Backtest %s..%s every %d month(s), %d deals entered, labels complete through %s",
                     start, end, step_months, len(labels.deals), labels_through)
        result = run_backtest(
            uow.p22, history, window, labels, load_scoring_config(), load_acquirers(), load_cvr_policy(),
            on_snapshot=lambda s: _logger.info("Snapshot %s: %d targets, %d ranked, %d positives, %d in model top 50",
                                               s["as_of"], s["targets"], s["ranked_by_model"], s["positives"], s["model_hits_top50"]),
        )
    meta = {"start": start, "end": end, "labels_complete_through": labels_through, "step_months": step_months,
            "deals_entered": len(deals), "config_hash": config_hash()}
    return result, write_results(result, out_dir, meta)


def run(start: date, end: Optional[date], labels_through: date, step_months: int, ks: Sequence[int]) -> int:
    """Run the backtest and write the results; returns the exit code."""
    results_dir = setup_run_logging()
    return 0 if execute(start, end, labels_through, step_months, ks, results_dir / "backtest") is not None else 1


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1), help="First snapshot (spec: 2016)")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="Last snapshot (default: --labels-through minus 24 months)")
    parser.add_argument("--labels-through", dest="labels_through", type=date.fromisoformat, required=True,
                        help="Date the deal review is complete up to")
    parser.add_argument("--step-months", dest="step_months", type=int, default=1)
    parser.add_argument("--k", dest="ks", type=int, nargs="+", default=list(DEFAULT_KS))
    args = parser.parse_args(argv)
    return run(args.start, args.end, args.labels_through, args.step_months, args.ks)


if __name__ == "__main__":
    sys.exit(main())
