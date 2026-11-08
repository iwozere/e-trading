"""
P22 — point-in-time `is_foreign_domiciled` history from filing-time business addresses (spec §4.5, §3.1), 2026-11-08.

`jobs/run_domicile_normalization.py` writes the CURRENT address from SEC submissions, stamped with the day
it ran, so a backtest snapshot before 2026-09 sees no domicile at all and Block E scored nobody in
2016-2024. EDGAR full-text search returns each filing's business address as filed (`biz_states`), and it
does change over time: Horizon Pharma's 10-Ks show Deerfield, IL up to 2014 and Dublin from 2015, after
its inversion. This module scans annual reports (10-K, 20-F, 40-F) of every universe company and writes
one fact per change of domicile, known from the filing date, through
`P22Repo.insert_financial_fact_history` (only before the earliest stored row, so live rows stay as they are).

The foreign/domestic test is the same as `domicile_normalization.extract_is_foreign_domiciled`: a
`stateOrCountry` code outside the US states and territories is foreign.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.domicile_normalization import METRIC, extract_is_foreign_domiciled
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

SOURCE_ID = "sec_efts_filing_address"
ANNUAL_FORMS = ("10-K", "20-F", "40-F")


def _state_for_cik(source: Dict[str, Any], cik: str) -> Optional[str]:
    """The filing's business state/country code for `cik`; None when the hit does not say unambiguously."""
    ciks = source.get("ciks") or []
    states = source.get("biz_states") or []
    if len(states) == len(ciks) and cik in ciks:
        return states[ciks.index(cik)] or None
    if len(states) == 1 and len(ciks) == 1:
        return states[0] or None
    return None


def observations_from_hits(hits: Iterable[Dict[str, Any]], companies_by_cik: Dict[str, int]) -> Dict[int, List[Tuple[date, bool, str]]]:
    """
    `{company_id: [(file_date, is_foreign, accession), ...]}` sorted by date, from EFTS hits.

    Args:
        hits: EFTS search hits (`_source` with `ciks`, `biz_states`, `file_date`, `adsh`).
        companies_by_cik: Universe CIK (10 digits) -> company_id.
    """
    observed: Dict[int, List[Tuple[date, bool, str]]] = {}
    for hit in hits:
        source = hit.get("_source") or {}
        filed = source.get("file_date")
        if not filed:
            continue
        for cik in source.get("ciks") or []:
            company_id = companies_by_cik.get(cik)
            if company_id is None:
                continue
            state = _state_for_cik(source, cik)
            if state is None:
                continue
            is_foreign = extract_is_foreign_domiciled({"addresses": {"business": {"stateOrCountry": state}}})
            if is_foreign is None:
                continue
            observed.setdefault(company_id, []).append((date.fromisoformat(filed), is_foreign, str(source.get("adsh") or "")))
    for rows in observed.values():
        rows.sort()
    return observed


def changes_only(observations: List[Tuple[date, bool, str]]) -> List[Tuple[date, bool, str]]:
    """The first observation and every one whose domicile differs from the one before."""
    kept: List[Tuple[date, bool, str]] = []
    for row in observations:
        if not kept or kept[-1][1] != row[1]:
            kept.append(row)
    return kept


def _year_windows(start: date, end: date) -> List[Tuple[str, str]]:
    """Calendar-year `(start, end)` strings covering `[start, end]` — one EFTS query per year stays under its result cap."""
    windows: List[Tuple[str, str]] = []
    current = start
    while current <= end:
        window_end = min(date(current.year, 12, 31), end)
        windows.append((current.isoformat(), window_end.isoformat()))
        current = date(current.year + 1, 1, 1)
    return windows


def _filing_url(cik: Optional[str], accession: str) -> Optional[str]:
    if not cik or not accession:
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"


def run(repo: Any, edgar: Any, start_dt: str, end_dt: str) -> Dict[str, Any]:
    """
    Scan annual reports filed in `[start_dt, end_dt]` and backfill `is_foreign_domiciled` history.

    Args:
        repo: A `P22Repo`-shaped object.
        edgar: An `EdgarDownloader`-shaped object (`efts_filings_search`).
        start_dt: `"YYYY-MM-DD"`.
        end_dt: `"YYYY-MM-DD"`.

    Returns:
        `{"filings": int, "companies_observed": int, "facts_inserted": int, "companies_changed_domicile": int}`.
    """
    companies = [c for c in repo.list_companies_full() if c.get("cik")]
    companies_by_cik = {c["cik"]: c["company_id"] for c in companies}
    cik_by_company = {c["company_id"]: c["cik"] for c in companies}

    hits: Dict[str, Dict[str, Any]] = {}
    for year_start, year_end in _year_windows(date.fromisoformat(start_dt), date.fromisoformat(end_dt)):
        for form in ANNUAL_FORMS:
            for hit in edgar.efts_filings_search(ciks=list(companies_by_cik), forms=form, start_dt=year_start, end_dt=year_end):
                hits.setdefault(str(hit.get("_id")), hit)
        _logger.info("Domicile history: %s..%s scanned, %d filings so far", year_start, year_end, len(hits))

    observed = observations_from_hits(hits.values(), companies_by_cik)
    inserted = 0
    changed: Counter[str] = Counter()
    for company_id, rows in observed.items():
        kept = changes_only(rows)
        if len(kept) > 1:
            changed["changed"] += 1
        facts = [
            {
                "known_from": datetime.combine(filed, time.min, tzinfo=timezone.utc),
                "valid_from": filed,
                "period_end": filed,
                "value": 1.0 if is_foreign else 0.0,
                "source_id": SOURCE_ID,
                "source_url": _filing_url(cik_by_company.get(company_id), accession),
            }
            for filed, is_foreign, accession in kept
        ]
        inserted += repo.insert_financial_fact_history(company_id=company_id, metric=METRIC, facts=facts)

    summary = {
        "filings": len(hits), "companies_observed": len(observed), "facts_inserted": inserted,
        "companies_changed_domicile": changed["changed"],
    }
    _logger.info("Domicile history backfill complete: %s", summary)
    return summary
