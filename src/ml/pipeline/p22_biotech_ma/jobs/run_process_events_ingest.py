"""
P22 job — 8-K strategic-alternatives phrase detection (spec §2.6.1, §4.7, M5).

Runs DAILY, after the universe is resolved (reads `p22_company.cik` to scope
the universe-wide 8-K index down to in-universe filers) — see
`ingest/process_events.py`'s module docstring for the detection logic and
its disclosed EX-99-exhibit scope gap.

With `--start/--end` it backfills a date range instead: one 8-K index per
business day, each day committed on its own, so an interrupted backfill keeps
what it finished and can be re-run (already-ingested 8-Ks are skipped).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.downloader.edgar_downloader import EdgarDownloader, EftsUnavailableError
from src.ml.pipeline.p22_biotech_ma.ingest.process_events import run as run_process_events
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# EDGAR accepts filings on business days only; weekend indexes are empty.
_WEEKEND = (5, 6)


def business_days(start: date, end: date) -> List[date]:
    """
    Every Monday–Friday date from `start` to `end`, inclusive.

    Raises:
        ValueError: if `end` is before `start`.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    days = []
    day = start
    while day <= end:
        if day.weekday() not in _WEEKEND:
            days.append(day)
        day += timedelta(days=1)
    return days


def run(start: Optional[date] = None, end: Optional[date] = None) -> Dict[str, Any]:
    """
    Scan yesterday's 8-K index (no dates: the daily job), or every business day from `start` to
    `end` (a backfill).

    Returns:
        `filings_scanned`, `candidates_written`, `rejection_breakdown`; a backfill also returns
        `days_scanned` and `days_failed` (dates EFTS could not serve — re-run those).
    """
    setup_run_logging()
    edgar = EdgarDownloader()
    db_service = DatabaseService()

    if start is None and end is None:
        with db_service.uow() as uow:
            summary = run_process_events(uow.p22, edgar)
        _logger.info("Process-events ingest complete: %s", summary)
        return summary

    days = business_days(start or end or date.today(), end or date.today() - timedelta(days=1))
    totals: Counter = Counter()
    rejections: Counter = Counter()
    failed: List[str] = []
    _logger.info("Process-events backfill: %d business days, %s..%s", len(days), days[0] if days else None, days[-1] if days else None)
    for done, day in enumerate(days, start=1):
        try:
            with db_service.uow() as uow:
                summary = run_process_events(uow.p22, edgar, as_of_date=day)
        except EftsUnavailableError:
            _logger.exception("Process-events backfill: EFTS unavailable for %s — skipped, re-run this day", day)
            failed.append(str(day))
            continue
        totals.update({k: summary[k] for k in ("filings_scanned", "candidates_written")})
        rejections.update(summary["rejection_breakdown"])
        if done % 20 == 0:
            _logger.info("Process-events backfill: %d/%d days done (through %s): %s", done, len(days), day, dict(totals))

    result = {
        "days_scanned": len(days) - len(failed),
        "days_failed": failed,
        "filings_scanned": totals["filings_scanned"],
        "candidates_written": totals["candidates_written"],
        "rejection_breakdown": dict(rejections),
    }
    _logger.info("Process-events backfill complete: %s", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=None, help="Backfill from this filing date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="Backfill through this date (default: yesterday)")
    args = parser.parse_args()
    result = run(args.start, args.end)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
