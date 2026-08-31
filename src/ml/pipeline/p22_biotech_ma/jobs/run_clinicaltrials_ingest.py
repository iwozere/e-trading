"""
P22 job — land ClinicalTrials.gov studies + version history for the current
universe snapshot (spec §2.2).

M1 placeholder: queries CT.gov by company `name` from `latest_universe_rows()`
as a best-effort sponsor-name match. Real alias resolution (sponsor string ->
verified company) is M2/M3's job (spec §2.2: "match on sponsor.leadSponsor.
name, then hand-verify aliases into an override table") — this job only
lands raw data, it does not claim to have resolved sponsor identity.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.clinicaltrials_client import ClinicalTrialsClient
from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import latest_universe_rows
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()
    universe = latest_universe_rows()
    if not universe:
        _logger.warning("No universe rows available — run run_sec_universe_ingest.py first")
        return {"companies_attempted": 0, "studies_landed": 0}

    today = date.today()
    studies_landed = 0

    with ClinicalTrialsClient() as client:
        for row in universe:
            name = row.get("name")
            cik = row.get("cik")
            if not name:
                continue

            studies = client.fetch_studies_for_sponsor(name)
            if studies:
                raw_zone.write(source="clinicaltrials_studies", entity=cik or name, as_of_date=today, payload=studies)
                studies_landed += len(studies)

            for study in studies:
                nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                if not nct_id:
                    continue
                history = client.fetch_study_version_history(nct_id)
                if history:
                    raw_zone.write(
                        source="clinicaltrials_history", entity=nct_id, as_of_date=today, payload=history
                    )

    summary = {"companies_attempted": len(universe), "studies_landed": studies_landed}
    _logger.info("ClinicalTrials.gov ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
