"""
P22 — deal-label candidate detection for the M6 backtest (spec §2.5), 2026-09-08.

Detection ONLY. Spec is explicit that this dataset "must be manually reviewed; automated
extraction alone will be too noisy for labels" — `announcement_date`, `acquirer`, `consideration
per share`, `CVR presence`, `premium to prior close`, and especially `deal_type` (spec's mandatory
`reverse_merger`/`shell_transaction`/etc. exclusions, which "look like acquisitions in the
filings") all need a human reading the actual document, not a keyword heuristic. This module does
NOT write `p22_deal` rows at all — only `p22_review_item` candidates naming which filing to go
read. Writing the hand-verified `p22_deal` row itself is a manual/future step (no CLI for it yet;
building one that could safely auto-populate `deal_type` from a payload would risk exactly the
noise spec warns against).

**Live-verified 2026-09-08, real EFTS form strings**: `"SC 14D9"` and `"S-4"` (unlike Schedule
13D/G's `edgar_downloader._13DG_FORM_TYPE_ALIASES` bug fixed this session, these did NOT need
correction). Also discovered: querying the base form alone already returns amendments too (`"SC
14D9"` returned both `SC 14D9` and `SC 14D9/A` hits; `"S-4"` returned both `S-4` and `S-4/A`) —
different from the documented "comma-list" exact-match quirk (`efts_filings_search`'s own
docstring) — so this module queries only the three base forms, not six.

**CIK-scoped like `activist_positions.py`, not SIC-scoped like spec's own §2.5 suggestion.** Spec
says "EDGAR full-text search... by SIC codes 2836, 8731, 2834" — a universe-wide sweep. Querying
by the P22 universe's own CIK list instead (`efts_filings_search`, chunked at 100) is equivalent
coverage for companies already in `p22_company` (which is itself SIC-filtered at DERA-ingest time,
`sec_universe_ingest.py`) and reuses the exact same efficient pattern already proven in
`activist_positions.py`, without a second SIC-based sweep that would surface companies never
otherwise resolved into this repo.

**Neither side of the transaction is assumed** — a hit's `ciks` may include the P22 universe
member as either the target (the usual case for SC 14D9, filed BY the target in response to a
tender offer) or a party named for other reasons (e.g. an S-4 the ACQUIRER files to register
merger shares); this module surfaces the match without guessing which role it plays — that's
exactly the kind of judgment spec routes to hand-verification.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.activist_positions import parse_source_url
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# DEFM14C added 2026-11-07: deals approved by written consent of a majority holder (PPD/Thermo Fisher 2021)
# file an information statement instead of a proxy, and the 2010-2026 backfill missed every one of them.
_DEAL_FORMS = ("SC 14D9", "DEFM14A", "DEFM14C", "S-4")
_ACCESSION_PATTERN = re.compile(r"^(\d{10})-?(\d{2})-?(\d{6})$")


class ManualCandidateError(ValueError):
    """A hand-added deal candidate cannot be queued (unknown company, bad accession, duplicate)."""


def _normalize_accession(accession: str) -> str:
    m = _ACCESSION_PATTERN.match((accession or "").strip())
    if not m:
        raise ManualCandidateError(f"Not an EDGAR accession number: {accession!r}")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def add_manual_candidate(
    repo: Any,
    *,
    cik: str,
    evidence_url: str,
    form_type: str,
    filed_date: str,
    accession_no: Optional[str] = None,
    note: Optional[str] = None,
) -> int:
    """
    Queue a `deal_candidate` the EFTS scan cannot find, typed in by hand from a filing URL.

    The scan only sees filings that list the universe company's own CIK. Deals where only the acquirer
    filed (its S-4 does not name the target as a filer: Assertio/Zyla, Rafael/Cyclo) and foreign schemes
    of arrangement are invisible to it; `deal_entry_cli coverage` lists such companies, and this queues
    one of them so the normal `enter`/`link`/`reject` flow can label it.

    Args:
        repo: A `P22Repo`-shaped object.
        cik: The TARGET company's CIK (it must be in `p22_company`); the URL may be any filer's.
        evidence_url: The filing document on sec.gov (`.../Archives/edgar/data/{cik}/{accession}/{file}`).
        form_type: The filing's form, e.g. `"S-4"`, `"8-K"`.
        filed_date: `"YYYY-MM-DD"`.
        accession_no: Defaults to the accession in `evidence_url`.
        note: Why the candidate was added by hand.

    Returns:
        The new review `item_id`.

    Raises:
        ManualCandidateError: unknown company, unparseable accession/date, or the same (company,
            filing) is already queued in any status.
    """
    company = repo.get_company_by_cik(cik)
    if company is None:
        raise ManualCandidateError(f"CIK {cik} is not a P22 universe company")
    if accession_no is None:
        parsed = parse_source_url(evidence_url)
        if parsed is None:
            raise ManualCandidateError(f"Cannot read an accession from {evidence_url!r}; pass it explicitly")
        accession_no = parsed[1]
    accession = _normalize_accession(accession_no)
    try:
        date.fromisoformat(filed_date)
    except ValueError as exc:
        raise ManualCandidateError(f"filed_date must be YYYY-MM-DD, got {filed_date!r}") from exc

    company_id = int(company["company_id"])
    for item in repo.get_review_items(item_type="deal_candidate"):
        payload = item.get("payload") or {}
        if payload.get("company_id") == company_id and payload.get("accession_no") == accession:
            raise ManualCandidateError(f"Already queued as item {item['item_id']} ({item['status']})")

    item_id = repo.add_review_item(
        item_type="deal_candidate",
        payload={
            "reason": "deal_candidate",
            "company_id": company_id,
            "cik": company.get("cik") or cik,
            "form_type": form_type,
            "accession_no": accession,
            "filed_date": filed_date,
            "all_ciks_in_filing": [],
            "all_names_in_filing": [],
            "source": "manual",
            "manual_note": note,
        },
        evidence_url=evidence_url,
        priority=0,
    )
    _logger.info("Queued manual deal candidate %s: company %s, %s %s", item_id, company_id, form_type, accession)
    return int(item_id)


def run(repo: Any, edgar: Any, start_dt: str, end_dt: str) -> Dict[str, Any]:
    """
    Scan `[start_dt, end_dt]` for SC 14D9/DEFM14A/DEFM14C/S-4 filings involving any P22 universe company
    and queue a `deal_candidate` review item for each (company, filing) pair not already pending.

    Args:
        repo: A `P22Repo`-shaped object.
        edgar: An `EdgarDownloader`-shaped object (`efts_filings_search`).
        start_dt: `"YYYY-MM-DD"`.
        end_dt: `"YYYY-MM-DD"`.

    Returns:
        `{"filings_matched": int, "candidates_queued": int, "already_queued": int}`.
    """
    companies_by_cik = {c["cik"]: c["company_id"] for c in repo.list_companies_full() if c.get("cik")}
    universe_ciks = list(companies_by_cik.keys())

    # Every status, not just pending: a candidate a human already confirmed or rejected must not come
    # back when the trailing window (or a historical backfill) rescans the same filing.
    already_queued: Set[Tuple[int, str]] = {
        (item["payload"]["company_id"], item["payload"]["accession_no"])
        for item in repo.get_review_items(item_type="deal_candidate")
        if "company_id" in item.get("payload", {}) and "accession_no" in item.get("payload", {})
    }

    seen_ids: Set[str] = set()
    hits: List[Dict[str, Any]] = []
    for form in _DEAL_FORMS:
        for hit in edgar.efts_filings_search(ciks=universe_ciks, forms=form, start_dt=start_dt, end_dt=end_dt):
            hit_id = hit.get("_id")
            if hit_id and hit_id not in seen_ids:
                seen_ids.add(hit_id)
                hits.append(hit)

    candidates_queued = 0
    skipped_already_queued = 0

    for hit in hits:
        src = hit.get("_source", {})
        accession = str(src.get("adsh") or "")
        form = str(src.get("form") or "")
        filed_date = str(src.get("file_date") or "")
        ciks_in_hit = src.get("ciks") or []
        display_names = src.get("display_names") or []
        if not accession:
            continue

        matched_ciks = [cik for cik in ciks_in_hit if cik in companies_by_cik]
        for cik in matched_ciks:
            company_id = companies_by_cik[cik]
            if (company_id, accession) in already_queued:
                skipped_already_queued += 1
                continue

            efts_id = str(hit.get("_id") or "")
            filename = efts_id.split(":", 1)[1] if ":" in efts_id else ""
            source_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{filename}"
                if filename else None
            )
            repo.add_review_item(
                item_type="deal_candidate",
                payload={
                    "reason": "deal_candidate",
                    "company_id": company_id,
                    "cik": cik,
                    "form_type": form,
                    "accession_no": accession,
                    "filed_date": filed_date,
                    "all_ciks_in_filing": ciks_in_hit,
                    "all_names_in_filing": display_names,
                },
                evidence_url=source_url,
                # Spec: "Expect 400-700 events" total, "manually reviewed" one at a time —
                # no strength/priority signal exists at detection time (unlike process_events.py's
                # strong/moderate phrase match), so every candidate queues at the same priority.
                priority=0,
            )
            already_queued.add((company_id, accession))
            candidates_queued += 1

    summary = {
        "filings_matched": len(hits), "candidates_queued": candidates_queued,
        "already_queued": skipped_already_queued,
    }
    _logger.info("Deal-candidate scan complete: %s", summary)
    return summary
