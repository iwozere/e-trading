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
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.clinicaltrials_client import ClinicalTrialsClient
from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import latest_universe_rows
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _load_last_known_updates(before_date: date, *, root: Path | None = None) -> Dict[str, str]:
    """
    Build an `{nct_id: lastUpdatePostDate}` map from the most recent
    `clinicaltrials_studies` partition landed strictly before `before_date`
    (i.e. the previous time this job ran, not today's in-progress partition).

    Used to skip the expensive history re-fetch (spec §2.2's rate-limited
    internal endpoint, see `clinicaltrials_client.py`) for any study whose
    `lastUpdatePostDate` hasn't moved since we last recorded its history —
    both cheaper and more correct for a daily "what changed" job. Returns an
    empty map on the very first run (no prior partition), which is a correct
    "treat every study as new" fallback, not an error.

    Args:
        before_date: Only partitions strictly earlier than this date count.
        root: Override the raw-zone root (used by tests).
    """
    last_known: Dict[str, str] = {}
    for payload in raw_zone.read_partition_before("clinicaltrials_studies", before_date, root=root):
        if not isinstance(payload, list):
            continue
        for study in payload:
            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            last_update = (
                study.get("protocolSection", {})
                .get("statusModule", {})
                .get("lastUpdatePostDateStruct", {})
                .get("date")
            )
            if nct_id and last_update:
                last_known[nct_id] = last_update
    return last_known


def run() -> dict:
    setup_run_logging()
    universe = latest_universe_rows()
    if not universe:
        _logger.warning("No universe rows available — run run_sec_universe_ingest.py first")
        return {"companies_attempted": 0, "studies_landed": 0}

    today = date.today()
    studies_landed = 0
    history_fetched = 0
    history_skipped_unchanged = 0

    last_known_updates = _load_last_known_updates(today)
    _logger.info("Loaded %d known NCT-ID last-update dates from the prior partition", len(last_known_updates))

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
                last_update = (
                    study.get("protocolSection", {})
                    .get("statusModule", {})
                    .get("lastUpdatePostDateStruct", {})
                    .get("date")
                )
                if last_update and last_known_updates.get(nct_id) == last_update:
                    history_skipped_unchanged += 1
                    continue

                history = client.fetch_study_version_history(nct_id)
                history_fetched += 1
                if history:
                    raw_zone.write(
                        source="clinicaltrials_history", entity=nct_id, as_of_date=today, payload=history
                    )

    summary = {
        "companies_attempted": len(universe),
        "studies_landed": studies_landed,
        "history_fetched": history_fetched,
        "history_skipped_unchanged": history_skipped_unchanged,
    }
    _logger.info("ClinicalTrials.gov ingest complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
