"""
P22 job — load `config/pipeline/p22_acquirers.yaml` into `p22_company` (spec §2.0.4).

Writes acquirer *identity* only (name/ticker/cik/role) via
`ingest.acquirer_config.upsert_acquirer_roster` — see that module's docstring
for why `bloc`/`entry_date`/`exit_date` are deliberately not persisted to any
DB column. Must run before `run_patent_expiry_normalization.py`, which needs
an acquirer roster to resolve Orange Book `Applicant_Full_Name` strings
against. Should run after `run_entity_resolution.py` so an acquirer that's
also in the DERA target universe merges into its existing `cik`-bearing row
instead of creating a duplicate identity (see `P22Repo.upsert_acquirer_company`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.pipeline.dependency_status import deferred_result, require_dependencies_or_defer
from src.ml.pipeline.p22_biotech_ma.ingest.acquirer_config import load_acquirers, upsert_acquirer_roster
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    ready, statuses = require_dependencies_or_defer("P22 Acquirer Roster Load")
    if not ready:
        return deferred_result(statuses)

    acquirers = load_acquirers()
    if not acquirers:
        _logger.warning("No acquirer entries loaded from config — check config/pipeline/p22_acquirers.yaml")
        return {"acquirers_loaded": 0, "companies_written": 0}

    db_service = DatabaseService()
    with db_service.uow() as uow:
        written = upsert_acquirer_roster(acquirers, uow.p22)

    summary = {"acquirers_loaded": len(acquirers), "companies_written": written}
    _logger.info("Acquirer roster load complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
