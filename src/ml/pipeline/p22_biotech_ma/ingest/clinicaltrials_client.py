"""
P22 — ClinicalTrials.gov API v2 client (spec §2.2).

Pulls study records for a sponsor name, plus each NCT ID's *version history* —
primary-endpoint changes, enrollment changes, and completion-date slips are
only visible in the diff between versions, and the spec calls this out as
high-signal (§2.2, "Critical").

Two live-verified corrections vs. a literal reading of the spec (2026-08-30):

1. **Field names must be fully qualified.** The documented `/api/v2/studies`
   `fields` param only accepts bare names for a couple of top-level fields
   (`NCTId`, `hasResults`); every other field in the spec's list (`briefTitle`,
   `overallStatus`, etc.) 400s unless given as its full
   `protocolSection.<module>.<field>` path — see `CLINICALTRIALS_FIELDS` in
   config.py.
2. **Version history has no documented public endpoint.** `/api/v2/studies/
   {nctId}/history` (a natural guess, and arguably what the spec implies)
   404s. The data only exists behind `/api/int/studies/{nctId}/history` — the
   undocumented endpoint CT.gov's own history-viewer UI calls. It returns a
   `changes` array with `version`/`date`/`moduleLabels` per revision (which
   modules changed, not a full field-level diff — extracting the actual
   primary-endpoint-text/enrollment/completion-date changes from that is M3
   feature-engineering work, not this client's job). Being undocumented and
   internal, this endpoint could change or disappear without notice — same
   risk class as P20's pdufa.bio dependency (see that pipeline's
   docs/Tasks.md); there is no officially documented alternative for this
   spec-required data.

This module owns request/parse/retry for clinicaltrials.gov only. It does not
write to the raw zone or the DB — callers (job scripts) do that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import (
    CLINICALTRIALS_BASE_URL,
    CLINICALTRIALS_FIELDS,
    CLINICALTRIALS_HISTORY_BASE_URL,
)
from src.ml.pipeline.p22_biotech_ma.ingest.http_retry import get_with_retry
from src.ml.pipeline.p22_biotech_ma.ingest.rate_limits import clinicaltrials_limiter
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_PAGE_SIZE = 100


class ClinicalTrialsClient:
    """Thin client over ClinicalTrials.gov API v2 (+ the internal history endpoint)."""

    def __init__(self, base_url: str = CLINICALTRIALS_BASE_URL, timeout: float = 30.0) -> None:
        self._base_url = base_url
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ClinicalTrialsClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        del _exc_info
        self.close()

    def fetch_studies_for_sponsor(self, sponsor_name: str) -> List[Dict[str, Any]]:
        """
        Fetch all studies whose lead sponsor matches `sponsor_name`, paginating
        through the full result set.

        Args:
            sponsor_name: Exact or near-exact sponsor string (matched against
                `sponsor.leadSponsor.name` per spec §2.2). Aliasing/fuzzy match
                is entity resolution's job (M2), not this client's.

        Returns:
            List of study records (raw API JSON `study` objects), possibly empty.
        """
        studies: List[Dict[str, Any]] = []
        page_token: Optional[str] = None

        while True:
            params: Dict[str, Any] = {
                "query.spons": sponsor_name,
                "fields": ",".join(CLINICALTRIALS_FIELDS),
                "pageSize": _PAGE_SIZE,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = get_with_retry(self._client, self._base_url, params=params, rate_limiter=clinicaltrials_limiter)
            if resp is None or resp.status_code != 200:
                if resp is not None:
                    _logger.error(
                        "ClinicalTrials.gov request for sponsor=%s failed: status %d",
                        sponsor_name,
                        resp.status_code,
                    )
                break

            data = resp.json()
            page_studies = data.get("studies", [])
            studies.extend(page_studies)

            page_token = data.get("nextPageToken")
            if not page_token or not page_studies:
                break

        _logger.info("Fetched %d studies for sponsor=%s", len(studies), sponsor_name)
        return studies

    def fetch_study_version_history(self, nct_id: str) -> List[Dict[str, Any]]:
        """
        Fetch the version history for a single NCT ID (spec §2.2, "Critical").
        See the module docstring for why this hits an undocumented endpoint.

        Returns:
            List of version-change entries (`{"version", "date", "moduleLabels",
            ...}`), oldest first, possibly empty if the study has no recorded
            history or the request failed.
        """
        url = f"{CLINICALTRIALS_HISTORY_BASE_URL}/{nct_id}/history"
        resp = get_with_retry(self._client, url, rate_limiter=clinicaltrials_limiter)
        if resp is None or resp.status_code != 200:
            if resp is not None:
                _logger.error(
                    "ClinicalTrials.gov history fetch for %s failed: status %d", nct_id, resp.status_code
                )
            return []
        data = resp.json()
        return data.get("changes", []) if isinstance(data, dict) else []
