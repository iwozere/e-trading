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
            ticker = company["ticker"]

            bars = fetch_recent_daily_bars(ticker)
            if not bars:
                # `fetch_recent_daily_bars` never raises (module docstring) — an
                # empty result covers both "genuinely no trading days in the
                # window" and "the fetch itself failed after retries" (already
                # logged there). Either way, record it (spec §7.2) so a
                # delisted/failing ticker is queryable, not just grep-able from
                # a log file.
                failed.append(ticker)
                uow.p22.log_fetch_failure(
                    source=_RAW_SOURCE, entity=ticker, error_message="yfinance returned no bars for the lookback window"
                )
                continue

            try:
                raw_zone.write(source=_RAW_SOURCE, entity=ticker, as_of_date=date.today(), payload=bars)
                # A SAVEPOINT, not the bare uow.p22 calls directly: without it, a
                # write failure for ONE ticker (e.g. an unexpected constraint hit)
                # would leave the whole session's transaction aborted, and every
                # already-fetched company earlier in this loop would be lost when
                # the outer `with db_service.uow()` rolls back on the way out —
                # exactly the "one bad write nukes the whole run" failure mode
                # Design.md's error-handling contract exists to prevent. Rolling
                # back only this ticker's savepoint keeps the rest of the day's
                # writes intact and the session usable for the next ticker.
                with uow.s.begin_nested():
                    result = write_daily_bars(company["company_id"], bars, uow.p22)
            except Exception as exc:
                _logger.exception("Failed to persist daily bars for %s", ticker)
                failed.append(ticker)
                uow.p22.log_fetch_failure(source=_RAW_SOURCE, entity=ticker, error_message=str(exc))
                continue

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
