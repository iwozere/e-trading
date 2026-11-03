"""
P22 job — Schedule 13D/13D-A/13G/13G-A ingest (spec §2.6.2, §4.7, M5).

Runs DAILY, after the universe is resolved — see
`ingest/activist_positions.py`'s module docstring for the EFTS-based
detection strategy and its disclosed `stated_intent` scope gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p22_biotech_ma.ingest.activist_positions import reclassify_filer_types
from src.ml.pipeline.p22_biotech_ma.ingest.activist_positions import run as run_activist_positions
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


# EFTS serves at most ~10,000 hits per query (deeper pages 500), and the uow commits only on exit:
# a long backfill window both truncates silently and loses everything on interrupt. Backfills are
# therefore split into windows, each searched and committed on its own (upserts are idempotent,
# so re-running an interrupted backfill is safe).
_BACKFILL_WINDOW_DAYS = 30


def run(lookback_days: int = 1) -> dict:
    """Scan the last `lookback_days` days of 13D/G filings (1 for the daily job; larger for a one-off backfill)."""
    setup_run_logging()

    edgar = EdgarDownloader()
    db_service = DatabaseService()
    end = date.today() - timedelta(days=1)
    totals: Counter = Counter()
    rejections: Counter = Counter()
    remaining = lookback_days
    while remaining > 0:
        window = min(remaining, _BACKFILL_WINDOW_DAYS)
        with db_service.uow() as uow:
            summary = run_activist_positions(uow.p22, edgar, as_of_date=end, lookback_days=window)
        totals.update({k: summary[k] for k in ("filings_matched", "positions_written")})
        rejections.update(summary["rejection_breakdown"])
        if lookback_days > window:
            _logger.info("Backfill window ending %s committed (%d days left): %s", end, remaining - window, summary)
        end -= timedelta(days=window)
        remaining -= window

    result = {**totals, "rejection_breakdown": dict(rejections)}
    _logger.info("Activist-positions ingest complete: %s", result)
    return result


def reclassify() -> dict:
    """Re-label `filer_type` on already-ingested positions from the current `activist_filers.yaml`
    and acquirer universe — no EDGAR calls."""
    setup_run_logging()
    with DatabaseService().uow() as uow:
        return reclassify_filer_types(uow.p22)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=1, help="Days to scan, ending yesterday (backfill: e.g. 730)")
    parser.add_argument("--reclassify", action="store_true", help="Only re-label filer_type on existing rows (no EDGAR scan)")
    args = parser.parse_args()
    result = reclassify() if args.reclassify else run(args.lookback_days)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
