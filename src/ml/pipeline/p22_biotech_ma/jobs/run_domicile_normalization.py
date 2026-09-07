"""
P22 job — normalize landed SEC submissions payloads into `is_foreign_domiciled`
(spec §4.5, Block E), 2026-09-08.

Reads the most recently landed `sec_submissions` raw-zone partition
(`run_sec_ingest.py` must have run first), resolves each payload's CIK to a
`company_id` via `p22_company` (`run_entity_resolution.py` must have run
first), and writes `p22_financial_fact` rows. See
`ingest/domicile_normalization.py` for the extraction logic and its
disclosed `isForeignLocation`-alone-isn't-reliable finding.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.pipeline.dependency_status import deferred_result, require_dependencies_or_defer
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.domicile_normalization import METRIC, SOURCE_ID, extract_is_foreign_domiciled
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    ready, statuses = require_dependencies_or_defer("P22 Domicile Normalization")
    if not ready:
        return deferred_result(statuses)

    submissions_by_cik = raw_zone.read_latest_partition_with_manifest("sec_submissions")
    if not submissions_by_cik:
        _logger.warning("No SEC submissions payloads landed yet — run run_sec_ingest.py first")
        return {"ciks_attempted": 0, "ciks_matched": 0, "facts_written": 0, "unknown": 0}

    ciks_matched = 0
    facts_written = 0
    unknown = 0

    db_service = DatabaseService()
    with db_service.uow() as uow:
        for submissions, manifest in submissions_by_cik:
            cik = manifest.get("entity")
            if not cik or not isinstance(submissions, dict):
                continue

            company = uow.p22.get_company_by_cik(cik)
            if company is None:
                _logger.warning("No resolved p22_company for CIK %s — skipping (run run_entity_resolution.py?)", cik)
                continue
            ciks_matched += 1

            is_foreign = extract_is_foreign_domiciled(submissions)
            if is_foreign is None:
                unknown += 1
                continue

            uow.p22.upsert_financial_fact_bitemporal(
                company_id=company["company_id"],
                metric=METRIC,
                value=1.0 if is_foreign else 0.0,
                known_from=datetime.fromisoformat(manifest["known_from"]),
                source_id=SOURCE_ID,
            )
            facts_written += 1

    summary = {
        "ciks_attempted": len(submissions_by_cik), "ciks_matched": ciks_matched,
        "facts_written": facts_written, "unknown": unknown,
    }
    _logger.info("Domicile normalization complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
