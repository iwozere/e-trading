"""
P22 — read-only size-floor and price-quality report (2026-10-27).

Two checks of `ingest/size_floor.py` / `ingest/price_quality.py` against the live DB. Both need the
`public_float` metric, which `jobs/run_financial_facts_normalization.py` writes from the next run after
this was added — run that job once first. Nothing is written to the DB.

    # Companies whose price/share series disagrees with their own 10-K public float
    # (corrupt vintage prices, ADRs, pre-merger share counts); optional CSV of every company:
    python -m src.ml.pipeline.p22_biotech_ma.cli.size_floor_report_cli price-quality --out /tmp/p22_price_quality.csv

    # Point-in-time size floor per landed DERA quarter (met / not met / unknown counts are logged per quarter):
    python -m src.ml.pipeline.p22_biotech_ma.cli.size_floor_report_cli history --since 2015q1
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import build_universe_history
from src.ml.pipeline.p22_biotech_ma.ingest.price_quality import float_consistency_history, summarize_consistency
from src.ml.pipeline.p22_biotech_ma.ingest.size_floor import make_size_floor_lookup
from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import all_landed_quarters
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_PROGRESS_EVERY = 200
_REPORT_TOP = 40


def is_suspect(summary: Dict[str, Any]) -> bool:
    """A company is suspect when at least two float dates, or every checked one, fail the consistency check."""
    return summary["inconsistent"] >= 2 or (summary["checked"] > 0 and summary["inconsistent"] == summary["checked"])


def run_price_quality(as_of: date, out: Optional[Path]) -> int:
    """Check every company's float-date consistency and log the suspects; returns the exit code."""
    rows: List[Dict[str, Any]] = []
    with DatabaseService().uow() as uow:
        companies = uow.p22.list_companies_full()
        for i, company in enumerate(companies, 1):
            summary = summarize_consistency(float_consistency_history(uow.p22, company["company_id"], as_of))
            rows.append({"company_id": company["company_id"], "ticker": company.get("ticker"), "name": company["name"],
                         **summary, "suspect": is_suspect(summary)})
            if i % _PROGRESS_EVERY == 0:
                _logger.info("Price quality progress: %d/%d companies", i, len(companies))

    checked = [r for r in rows if r["checked"]]
    suspects = sorted((r for r in checked if r["suspect"]), key=lambda r: -r["inconsistent"])
    _logger.info("Price quality as of %s: %d companies, %d with a comparable public float, %d suspect",
                 as_of, len(rows), len(checked), len(suspects))
    for r in suspects[:_REPORT_TOP]:
        _logger.info("  suspect %s (%s, id %d): %d/%d float dates inconsistent (low %d, high %d), median ratio %.3g",
                     r["ticker"], r["name"], r["company_id"], r["inconsistent"], r["checked"], r["too_low"],
                     r["too_high"], r["median_ratio"])
    if out is not None:
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["company_id"])
            writer.writeheader()
            writer.writerows(rows)
        _logger.info("Wrote %d rows to %s", len(rows), out)
    return 0


def run_history(since: Optional[str]) -> int:
    """Build the point-in-time universe with the size floor for every landed quarter; returns the exit code."""
    quarters_rows = all_landed_quarters()
    if since:
        quarters_rows = {q: rows for q, rows in quarters_rows.items() if q >= since}
    if not quarters_rows:
        _logger.warning("No landed DERA quarters%s — run run_sec_universe_ingest.py first",
                        f" since {since}" if since else "")
        return 1
    with DatabaseService().uow() as uow:
        # Exchange eligibility is irrelevant to the size floor, so no live ticker/exchange fetch.
        build_universe_history(quarters_rows, {}, size_floor=make_size_floor_lookup(uow.p22))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run one report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="report", required=True)
    quality = sub.add_parser("price-quality", help="Companies whose prices/shares disagree with their public float")
    quality.add_argument("--as-of", type=date.fromisoformat, default=date.today(), help="Lookahead bound (default today)")
    quality.add_argument("--out", type=Path, default=None, help="Write every company's summary to this CSV")
    history = sub.add_parser("history", help="Point-in-time size floor per landed DERA quarter")
    history.add_argument("--since", default=None, help="First quarter, e.g. 2015q1")
    args = parser.parse_args(argv)
    if args.report == "price-quality":
        return run_price_quality(args.as_of, args.out)
    return run_history(args.since)


if __name__ == "__main__":
    sys.exit(main())
