"""
P22 job — 8-K strategic-alternatives phrase detection (spec §2.6.1, §4.7, M5).

Runs DAILY, after the universe is resolved (reads `p22_company.cik` to scope
the universe-wide 8-K index down to in-universe filers) — see
`ingest/process_events.py`'s module docstring for the detection logic and
its disclosed EX-99-exhibit scope gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p22_biotech_ma.ingest.process_events import run as run_process_events
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    edgar = EdgarDownloader()
    db_service = DatabaseService()
    with db_service.uow() as uow:
        summary = run_process_events(uow.p22, edgar)

    _logger.info("Process-events ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
