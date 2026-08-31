"""
P22 job — land SEC submissions + XBRL company facts for the current universe
snapshot (spec §2.1).

M1 placeholder: iterates the interim `latest_universe_rows()` snapshot
(`ingest/universe_snapshot.py`) rather than a real `p22_company` table, which
doesn't exist until M2. Switch this to reading from `P22Repo` once M2 lands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.sec_raw_ingest import land_submissions_and_facts
from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import latest_universe_rows
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()
    universe = latest_universe_rows()
    ciks = sorted({row["cik"] for row in universe if row.get("cik")})
    if not ciks:
        _logger.warning("No universe CIKs available — run run_sec_universe_ingest.py first")
        return {"ciks_attempted": 0, "ciks_landed": 0}

    outcomes = land_submissions_and_facts(ciks)
    landed = sum(1 for o in outcomes.values() if o["submissions"] or o["company_facts"])
    summary = {"ciks_attempted": len(ciks), "ciks_landed": landed}
    _logger.info("SEC ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
