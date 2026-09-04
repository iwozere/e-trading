"""
P22 job — normalize landed ClinicalTrials.gov study payloads into `p22_trial`
(spec §2.2, §3.2, M3 Block B input).

Reads the most recently landed `clinicaltrials_studies` raw-zone partition
(`run_clinicaltrials_ingest.py` must have run first), resolves each payload's
CIK (the raw-zone manifest's `entity` — `cik or name` per that job, so a
company with no CIK on file lands under its name and is skipped here rather
than guessed at) to a `company_id` via `p22_company`, and writes every
`p22_trial` row via `ingest.trial_normalization`. See that module's docstring
for exactly which columns are populated this pass, and
`ingest.asset_normalization`'s docstring for how single-intervention trials
get `asset_id` linked (multi-intervention trials still don't).
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
from src.ml.pipeline.p22_biotech_ma.ingest.trial_normalization import extract_trial_records, write_trial_records
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    ready, statuses = require_dependencies_or_defer("P22 Trial Normalization")
    if not ready:
        return deferred_result(statuses)

    studies_by_entity = raw_zone.read_latest_partition_with_manifest("clinicaltrials_studies")
    if not studies_by_entity:
        _logger.warning("No ClinicalTrials.gov payloads landed yet — run run_clinicaltrials_ingest.py first")
        return {"entities_attempted": 0, "companies_matched": 0, "trials_written": 0}

    companies_matched = 0
    trials_written = 0

    db_service = DatabaseService()
    with db_service.uow() as uow:
        for studies, manifest in studies_by_entity:
            cik = manifest.get("entity")
            known_from = manifest.get("known_from")
            if not cik or not isinstance(studies, list) or not known_from:
                continue

            company = uow.p22.get_company_by_cik(cik)
            if company is None:
                # Either the entity is a name (no CIK on file for this company
                # at ingest time) or the company hasn't been resolved yet —
                # either way, skip rather than guess. See module docstring.
                _logger.debug("No resolved p22_company for CT.gov entity %s — skipping", cik)
                continue
            companies_matched += 1

            records = extract_trial_records(studies, datetime.fromisoformat(known_from))
            trials_written += write_trial_records(records, uow.p22, company_id=company["company_id"])

    summary = {
        "entities_attempted": len(studies_by_entity),
        "companies_matched": companies_matched,
        "trials_written": trials_written,
    }
    _logger.info("Trial normalization complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
