"""
P22 job — derive and write `market_cap` (spec §2.0.6/§2.4, resolved 2026-09-07).

Runs DAILY, after both the current-price ingest and the financial-facts
normalization jobs, so it reads the freshest raw close and shares-outstanding
fact available for each company. See `ingest/market_cap.py`'s module
docstring for why this needed no market-data vendor purchase after all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import run as run_market_cap
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    db_service = DatabaseService()
    with db_service.uow() as uow:
        company_ids = [c["company_id"] for c in uow.p22.list_companies_full()]
        summary = run_market_cap(uow.p22, company_ids)

    _logger.info("Market cap compute complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
