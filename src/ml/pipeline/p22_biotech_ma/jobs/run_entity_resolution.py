"""
P22 job — M2 entity resolution: build/refresh `p22_company` from the latest
landed SEC DERA universe snapshot (spec §2.0.2, §2.0.3).

Reads the most recently landed DERA partition (`run_sec_universe_ingest.py`
must have run first), resolves current ticker/exchange, applies the
eligibility filters `ingest/entity_resolution.py` can compute at this stage,
and writes the result to `p22_company` via `P22Repo`. See that module's
docstring for exactly which spec §2.0.3 filters this does and does not cover
yet.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import (
    build_universe,
    fetch_ticker_exchange_map,
    write_universe,
)
from src.ml.pipeline.p22_biotech_ma.ingest.review_queue import queue_depth_report
from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import latest_universe_rows
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    dera_rows = latest_universe_rows()
    if not dera_rows:
        _logger.warning("No DERA universe rows available — run run_sec_universe_ingest.py first")
        return {"companies_written": 0, "spac_flagged_for_review": 0, "total_candidates": 0}

    ticker_exchange_map = fetch_ticker_exchange_map()
    as_of = date.today()
    candidates = build_universe(dera_rows, ticker_exchange_map, as_of=as_of)

    db_service = DatabaseService()
    with db_service.uow() as uow:
        stats = write_universe(candidates, uow.p22)
        pending = uow.p22.get_pending_review_items()

    # Spec §3.4: "Queue depth and median age by item_type are reported in every run."
    depth_report = queue_depth_report(pending, now=datetime.now(timezone.utc))
    _logger.info("Review queue depth: %s", depth_report)
    result: dict = {**stats, "review_queue": depth_report}

    _logger.info("Entity resolution complete: %s", result)
    return result


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
