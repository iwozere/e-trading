"""
P22 — SEC DERA Financial Statement Data Sets ingest (spec §2.0, added v0.5).

The universe is built from EDGAR, never from a market-data vendor's current
roster: DERA's quarterly Financial Statement Data Sets give one record per
XBRL submission (`sub.txt`), and because EDGAR filings are immutable, a
delisted company's filing history stays permanently retrievable by CIK. This
is what makes the universe survivorship-free.

This module lands each quarter's submission index in the raw zone, filtered
to the biotech SIC codes this pipeline cares about (spec §2.0.1's
`universe_as_of` pseudocode filters on the same codes immediately, so landing
the full unfiltered multi-market `sub.txt` — tens of thousands of rows most
of which are never touched again — would just be dead weight in the raw
zone). Turning this into an actual `p22_company` roster with eligibility
filters (§2.0.3) and historical ticker resolution (§2.0.2) is M2 entity-
resolution work, not this module's job.

Archive URLs are derived from the DERA landing page rather than hardcoded —
the spec explicitly warns the path has moved before (§2.0.1).
"""

from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import (
    BIOTECH_SIC_CODES,
    EDGAR_USER_AGENT,
    SEC_DERA_LANDING_PAGE,
)
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.http_retry import get_with_retry
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# e.g. "2019q3.zip" — DERA's naming convention for quarterly archives.
_ARCHIVE_LINK_RE = re.compile(r'href="([^"]*?(\d{4}q[1-4])\.zip)"', re.IGNORECASE)


def _headers() -> Dict[str, str]:
    return {"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _get_ok(client: httpx.Client, url: str) -> Optional[httpx.Response]:
    """GET with retry, returning the response only on 2xx (None otherwise)."""
    resp = get_with_retry(client, url, headers=_headers(), follow_redirects=True)
    if resp is None:
        return None
    if resp.status_code != 200:
        _logger.error("SEC DERA request to %s failed: status %d", url, resp.status_code)
        return None
    return resp


def discover_quarterly_archive_urls(client: Optional[httpx.Client] = None) -> Dict[str, str]:
    """
    Fetch the DERA landing page and extract every quarterly archive URL.

    Returns:
        Mapping of quarter string (e.g. "2019q3") -> absolute archive URL.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = _get_ok(client, SEC_DERA_LANDING_PAGE)
        if resp is None:
            return {}

        urls: Dict[str, str] = {}
        for href, quarter in _ARCHIVE_LINK_RE.findall(resp.text):
            absolute = href if href.startswith("http") else f"https://www.sec.gov{href}"
            urls[quarter] = absolute

        _logger.info("Discovered %d SEC DERA quarterly archives", len(urls))
        return urls
    finally:
        if owns_client:
            client.close()


def fetch_quarter_submissions(quarter: str, archive_url: str, client: Optional[httpx.Client] = None) -> List[Dict[str, Any]]:
    """
    Download one quarter's DERA archive, extract `sub.txt`, and return the
    rows whose SIC code is in `BIOTECH_SIC_CODES`.

    Args:
        quarter: Quarter string, e.g. "2019q3" (used only for logging).
        archive_url: Absolute URL to the quarter's ZIP archive.
        client: Optional shared httpx.Client.

    Returns:
        List of filtered `sub.txt` rows as dicts, possibly empty on failure.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=120.0)
    try:
        resp = _get_ok(client, archive_url)
        if resp is None:
            return []

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                with zf.open("sub.txt") as f:
                    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    reader = csv.DictReader(text, delimiter="\t")
                    rows = [row for row in reader if row.get("sic") in BIOTECH_SIC_CODES]
        except (zipfile.BadZipFile, KeyError) as exc:
            _logger.error("SEC DERA archive for %s malformed or missing sub.txt: %s", quarter, exc)
            return []

        _logger.info("Fetched %d biotech-SIC submissions for quarter=%s", len(rows), quarter)
        return rows
    finally:
        if owns_client:
            client.close()


def land_all_quarters(since_quarter: str = "2010q1") -> Dict[str, raw_zone.RawZoneWriteResult]:
    """
    Discover and land every DERA quarterly archive from `since_quarter` to the
    present, filtered to biotech SIC codes, in the raw zone.

    Args:
        since_quarter: Earliest quarter to land, inclusive (spec §2.0.1: "Walk
            every quarter from 2010 to present").

    Returns:
        Mapping of quarter -> RawZoneWriteResult, one entry per quarter that
        was successfully fetched (failures are logged and omitted).
    """
    today = date.today()
    results: Dict[str, raw_zone.RawZoneWriteResult] = {}

    with httpx.Client(timeout=120.0) as client:
        archive_urls = discover_quarterly_archive_urls(client)
        for quarter in sorted(q for q in archive_urls if q >= since_quarter):
            rows = fetch_quarter_submissions(quarter, archive_urls[quarter], client)
            if not rows:
                continue
            result = raw_zone.write(
                source="sec_dera_universe",
                entity=quarter,
                as_of_date=today,
                payload=rows,
            )
            results[quarter] = result

    _logger.info("Landed %d/%d SEC DERA quarters", len(results), len(archive_urls))
    return results
