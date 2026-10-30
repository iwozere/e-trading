"""
P22 job — Phase 1 rule-based scoring (spec §5.1–§5.2, M4), 2026-10-29.

Runs on weekdays after Market Cap Compute, so Block C reads the day's market cap. Writes one
`p22_score_run` + `p22_score` rows, and the ranked watchlist (spec §6.1) as
`results/p22_biotech_ma/YYYY-MM-DD/watchlist.csv` and `watchlist.json`.

    python src/ml/pipeline/p22_biotech_ma/jobs/run_scoring.py [--as-of 2026-10-29] [--dry-run]

`--dry-run` scores and writes the files but rolls back the database writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest.acquirer_config import load_acquirers
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.ml.pipeline.p22_biotech_ma.scoring.score_run import persist_score_run, score_universe
from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import config_hash, load_scoring_config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_CSV_COLUMNS = [
    "rank_by_composite", "ticker", "name", "company_id", "composite", "tier", "tier_reason", "fundamental",
    "coverage", "likely_acquirer", "acquirer_pressure_matched", "target_asset_quality", "strategic_fit",
    "financial_attractiveness", "friction_penalty", "dilution_gate",
]


class _DryRunRollback(Exception):
    """Raised inside the unit of work to roll back a dry run."""


def write_watchlist(rows: List[Dict[str, Any]], results_dir: Path, summary: Dict[str, Any]) -> None:
    """Spec §6.1 watchlist: CSV for eyeballing, JSON with the full subscores/contributions."""
    with (results_dir / "watchlist.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            subscores = row["subscores"]
            likely = subscores.get("likely_acquirer") or {}
            writer.writerow({
                "rank_by_composite": row["rank_by_composite"],
                "ticker": row["ticker"],
                "name": row["name"],
                "company_id": row["company_id"],
                "composite": row["composite"],
                "tier": row["tier"],
                "tier_reason": row["tier_reason"],
                "fundamental": subscores["fundamental"],
                "coverage": subscores["coverage"],
                "likely_acquirer": likely.get("name"),
                "dilution_gate": subscores["dilution_gate"],
                **subscores["blocks"],
            })
    payload = {"summary": summary, "scores": rows}
    (results_dir / "watchlist.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run(as_of: date | None = None, dry_run: bool = False) -> dict:
    results_dir = setup_run_logging()
    as_of = as_of or date.today()
    config = load_scoring_config()
    acquirers = load_acquirers()
    digest = config_hash()

    summary: Dict[str, Any] = {}
    try:
        with DatabaseService().uow() as uow:
            result = score_universe(uow.p22, as_of, config, acquirers)
            summary = dict(result.summary)
            write_watchlist(result.rows, results_dir, summary)
            summary["run_id"] = persist_score_run(uow.p22, as_of, config, digest, result.rows)
            if dry_run:
                raise _DryRunRollback()
    except _DryRunRollback:
        summary["run_id"] = None
        summary["dry_run"] = True
    summary["config_hash"] = digest
    _logger.info("Scoring complete: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="P22 Phase 1 scoring")
    parser.add_argument("--as-of", dest="as_of", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = parser.parse_args(argv)
    return run(date.fromisoformat(args.as_of) if args.as_of else None, args.dry_run)


if __name__ == "__main__":
    result = main()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")
