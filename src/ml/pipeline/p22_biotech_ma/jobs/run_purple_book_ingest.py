"""
P22 job — land the FDA Purple Book CSV (biologic exclusivity) (spec §2.3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.orange_book_client import fetch_and_land_purple_book
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()
    result = fetch_and_land_purple_book()
    summary = {"landed": result is not None, "was_new": result.was_new if result else False}
    _logger.info("Purple Book ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
