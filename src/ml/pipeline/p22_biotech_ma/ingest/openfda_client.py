"""
P22 — openFDA Drugs@FDA client (spec §2.3).

Approvals, application numbers, and sponsor for drug applications. Pairs with
Orange Book patent/exclusivity data (`orange_book_client.py`) for the
revenue-at-risk join described in spec §2.3.

**Live-verified correction (2026-08-30):** openFDA's `sponsor_name` search
field is case-sensitive against the stored (uppercase) values —
`sponsor_name:Pfizer` 404s ("no matches") while `sponsor_name:PFIZER`
returns real results. The search term is uppercased before every query.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import (
    OPENFDA_API_KEY,
    OPENFDA_DRUGSFDA_URL,
)
from src.ml.pipeline.p22_biotech_ma.ingest.http_retry import get_with_retry
from src.ml.pipeline.p22_biotech_ma.ingest.rate_limits import openfda_limiter
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_PAGE_LIMIT = 100  # openFDA's max per-request limit


class OpenFDAClient:
    """Thin client over openFDA's Drugs@FDA endpoint."""

    def __init__(self, base_url: str = OPENFDA_DRUGSFDA_URL, timeout: float = 30.0) -> None:
        self._base_url = base_url
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenFDAClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        del _exc_info
        self.close()

    def fetch_applications_for_sponsor(self, sponsor_name: str) -> List[Dict[str, Any]]:
        """
        Fetch all Drugs@FDA application records for a given sponsor name,
        paginating through the full result set.

        Args:
            sponsor_name: Applicant/sponsor name as it appears in Drugs@FDA.
                Uppercased internally (openFDA's `sponsor_name` field is
                case-sensitive); entity resolution (M2) is responsible for
                mapping this to a resolved company in the first place.

        Returns:
            List of application records, possibly empty.
        """
        results: List[Dict[str, Any]] = []
        skip = 0
        search_term = sponsor_name.upper()

        while True:
            params: Dict[str, Any] = {
                "search": f'sponsor_name:"{search_term}"',
                "limit": _PAGE_LIMIT,
                "skip": skip,
            }
            if OPENFDA_API_KEY:
                params["api_key"] = OPENFDA_API_KEY

            resp = get_with_retry(self._client, self._base_url, params=params, rate_limiter=openfda_limiter)
            if resp is None:
                break
            if resp.status_code == 404:
                # openFDA returns 404 for "no matches" rather than an empty list.
                break
            if resp.status_code != 200:
                _logger.error(
                    "openFDA request for sponsor=%s failed: status %d", sponsor_name, resp.status_code
                )
                break

            data = resp.json()
            page_results = data.get("results", [])
            results.extend(page_results)

            if len(page_results) < _PAGE_LIMIT:
                break
            skip += _PAGE_LIMIT

        _logger.info("Fetched %d Drugs@FDA applications for sponsor=%s", len(results), sponsor_name)
        return results
