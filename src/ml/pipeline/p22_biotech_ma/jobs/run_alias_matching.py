"""
P22 job — resolve landed CT.gov/openFDA sponsor strings against the resolved
`p22_company` roster (spec §3.3).

Reads the most recently landed `clinicaltrials_studies` and `openfda_drugsfda`
raw-zone partitions (both jobs must have run first, as must
`run_entity_resolution.py` — there's nothing to match against otherwise),
extracts candidate sponsor/applicant names, and calls
`alias_matching.resolve_aliases` per source. See that module's docstring for
the live-verified field paths and the deterministic/fuzzy/unresolved routing.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.pipeline.dependency_status import deferred_result, require_dependencies_or_defer
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.alias_matching import (
    extract_ctgov_sponsor_names,
    extract_openfda_sponsor_names,
    resolve_aliases,
)
from src.ml.pipeline.p22_biotech_ma.ingest.review_queue import queue_depth_report
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    # Cron fires this 60 minutes after ClinicalTrials Ingest *starts* — not a
    # guarantee it finished. That gap has already failed once in production
    # (ClinicalTrials Ingest timed out at 7200s having covered only 215/1705
    # companies; see specs/p22_specs.py) — check both ingests actually
    # completed today before reading their raw-zone output.
    ready, statuses = require_dependencies_or_defer("P22 Alias Matching")
    if not ready:
        return deferred_result(statuses)

    db_service = DatabaseService()
    with db_service.uow() as uow:
        known_companies = uow.p22.list_companies()
        if not known_companies:
            _logger.warning("No resolved companies in p22_company — run run_entity_resolution.py first")
            return {"clinicaltrials": {}, "openfda": {}}

        ctgov_candidates: list[tuple[str, datetime]] = []
        for studies, manifest in raw_zone.read_latest_partition_with_manifest("clinicaltrials_studies"):
            if isinstance(studies, list):
                known_from = datetime.fromisoformat(manifest["known_from"])
                ctgov_candidates.extend((name, known_from) for name in extract_ctgov_sponsor_names(studies))

        openfda_candidates: list[tuple[str, datetime]] = []
        for applications, manifest in raw_zone.read_latest_partition_with_manifest("openfda_drugsfda"):
            if isinstance(applications, list):
                known_from = datetime.fromisoformat(manifest["known_from"])
                openfda_candidates.extend((name, known_from) for name in extract_openfda_sponsor_names(applications))

        ctgov_stats = resolve_aliases(ctgov_candidates, known_companies, uow.p22, source="clinicaltrials")
        openfda_stats = resolve_aliases(openfda_candidates, known_companies, uow.p22, source="openfda")
        pending = uow.p22.get_pending_review_items()

    # Spec §3.4: "Queue depth and median age by item_type are reported in every run."
    depth_report = queue_depth_report(pending, now=datetime.now(timezone.utc))
    _logger.info("Review queue depth: %s", depth_report)

    summary = {"clinicaltrials": ctgov_stats, "openfda": openfda_stats, "review_queue": depth_report}
    _logger.info("Alias matching complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
