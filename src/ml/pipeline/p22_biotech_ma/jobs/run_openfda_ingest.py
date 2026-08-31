"""
P22 job — land openFDA Drugs@FDA applications for the current universe
snapshot (spec §2.3).

Same M1 caveat as run_clinicaltrials_ingest.py: queries by company `name`
as a best-effort sponsor match, not a resolved alias (M2/M3).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.openfda_client import OpenFDAClient
from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import latest_universe_rows
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()
    universe = latest_universe_rows()
    if not universe:
        _logger.warning("No universe rows available — run run_sec_universe_ingest.py first")
        return {"companies_attempted": 0, "applications_landed": 0}

    today = date.today()
    applications_landed = 0

    with OpenFDAClient() as client:
        for row in universe:
            name = row.get("name")
            cik = row.get("cik")
            if not name:
                continue

            applications = client.fetch_applications_for_sponsor(name)
            if applications:
                raw_zone.write(
                    source="openfda_drugsfda", entity=cik or name, as_of_date=today, payload=applications
                )
                applications_landed += len(applications)

    summary = {"companies_attempted": len(universe), "applications_landed": applications_landed}
    _logger.info("openFDA ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
