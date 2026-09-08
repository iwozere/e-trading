"""
P22 job — deal-label candidate detection for the M6 backtest (spec §2.5), 2026-09-08.

Runs DAILY, after the universe is resolved. See `ingest/deal_candidates.py`'s module docstring —
this ONLY queues review items naming a filing to go read; it never writes `p22_deal` itself.
Scans a trailing window (`DEAL_CANDIDATE_LOOKBACK_DAYS`) rather than just "yesterday" since a
missed day should still get picked up on the next run, same reasoning as other EFTS-based scans
in this pipeline.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p22_biotech_ma.config import DEAL_CANDIDATE_LOOKBACK_DAYS
from src.ml.pipeline.p22_biotech_ma.ingest.deal_candidates import run as run_deal_candidates
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    end = date.today()
    start = end - timedelta(days=DEAL_CANDIDATE_LOOKBACK_DAYS)

    edgar = EdgarDownloader()
    db_service = DatabaseService()
    with db_service.uow() as uow:
        summary = run_deal_candidates(uow.p22, edgar, start_dt=str(start), end_dt=str(end))

    _logger.info("Deal-candidate ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
