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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p22_biotech_ma.ingest.activist_positions import run as run_activist_positions
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run(lookback_days: int = 1) -> dict:
    """Scan the last `lookback_days` days of 13D/G filings (1 for the daily job; larger for a one-off backfill)."""
    setup_run_logging()

    edgar = EdgarDownloader()
    db_service = DatabaseService()
    with db_service.uow() as uow:
        summary = run_activist_positions(uow.p22, edgar, lookback_days=lookback_days)

    _logger.info("Activist-positions ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=1, help="Days to scan, ending yesterday (backfill: e.g. 730)")
    result = run(parser.parse_args().lookback_days)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
