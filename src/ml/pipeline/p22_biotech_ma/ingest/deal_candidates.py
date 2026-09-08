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

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_DEAL_FORMS = ("SC 14D9", "DEFM14A", "S-4")


def run(repo: Any, edgar: Any, start_dt: str, end_dt: str) -> Dict[str, Any]:
    """
    Scan `[start_dt, end_dt]` for SC 14D9/DEFM14A/S-4 filings involving any P22 universe company
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

    already_queued: Set[Tuple[int, str]] = {
        (item["payload"]["company_id"], item["payload"]["accession_no"])
        for item in repo.get_pending_review_items(item_type="deal_candidate")
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
