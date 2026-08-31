"""
P22 job — land SEC DERA Financial Statement Data Sets (spec §2.0).

Scheduled quarterly; idempotent (raw-zone content-hash dedup means a re-run
against already-landed quarters is a fast no-op), so a stray extra run does
no harm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.sec_universe_ingest import land_all_quarters
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()
    results = land_all_quarters()
    new_writes = sum(1 for r in results.values() if r.was_new)
    summary = {"quarters_landed": len(results), "quarters_newly_written": new_writes}
    _logger.info("SEC DERA universe ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
