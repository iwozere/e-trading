"""
P22 job — Schedule 13D/13D-A/13G/13G-A ingest (spec §2.6.2, §4.7, M5).

Runs DAILY, after the universe is resolved — see
`ingest/activist_positions.py`'s module docstring for the EFTS-based
detection strategy and its disclosed `stated_intent` scope gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p22_biotech_ma.ingest.activist_positions import run as run_activist_positions
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    edgar = EdgarDownloader()
    db_service = DatabaseService()
    with db_service.uow() as uow:
        summary = run_activist_positions(uow.p22, edgar)

    _logger.info("Activist-positions ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
