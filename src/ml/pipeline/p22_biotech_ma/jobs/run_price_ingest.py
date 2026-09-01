"""
P22 job — daily current-price ingest via yfinance (spec §2.0.7, M3, 2026-09-01).

Runs DAILY (unlike `ingest/fmp_backfill.py`'s one-time historical backfill,
run manually during a paid-tier month) — for every `p22_company` row with a
`ticker` on file (targets AND acquirers, since Block A needs acquirer prices
too), fetches yfinance's last few days of bars and writes
`p22_price_daily`/`p22_corporate_action`. See `ingest/yfinance_client.py`'s
docstring for why this stays a narrow trailing window rather than ever
backfilling deep history through this path.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.config import YFINANCE_REQUEST_DELAY_SECONDS
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.price_ingest import write_daily_bars
from src.ml.pipeline.p22_biotech_ma.ingest.yfinance_client import fetch_recent_daily_bars
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_RAW_SOURCE = "yfinance_daily_price"


def run() -> dict:
    setup_run_logging()

    db_service = DatabaseService()
    with db_service.uow() as uow:
        companies = [c for c in uow.p22.list_companies_full() if c.get("ticker")]

        companies_attempted = 0
        prices_written = 0
        actions_written = 0
        failed: list[str] = []

        for i, company in enumerate(companies, 1):
            companies_attempted += 1
            bars = fetch_recent_daily_bars(company["ticker"])
            if not bars:
                failed.append(company["ticker"])
                continue

            raw_zone.write(source=_RAW_SOURCE, entity=company["ticker"], as_of_date=date.today(), payload=bars)
            result = write_daily_bars(company["company_id"], bars, uow.p22)
            prices_written += result["prices_written"]
            actions_written += result["actions_written"]

            if i % 100 == 0:
                _logger.info(
                    "Daily price ingest progress: %d/%d (prices=%d actions=%d failed=%d)",
                    i, len(companies), prices_written, actions_written, len(failed),
                )
            time.sleep(YFINANCE_REQUEST_DELAY_SECONDS)

    summary = {
        "companies_attempted": companies_attempted,
        "prices_written": prices_written,
        "actions_written": actions_written,
        "failed": len(failed),
    }
    _logger.info("Daily price ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
