"""
P22 — 8-K strategic-alternatives phrase detection (spec §2.6.1, §4.7, M5).

Detection source: `EdgarDownloader.download_8k_filings`, the universe-wide
daily 8-K index (already used by P17's CatalystAgent) filtered to Item 7.01
(Reg FD) or Item 8.01 (Other Events) — **never** Item 1.01, which only fires
once a deal is already signed (spec's own framing: "by that point the trade
is over"). Phrase matching is a CANDIDATE GENERATOR, not a classifier (spec
§2.6.1): every hit is written as an unverified `p22_corporate_process_event`
row (visible in the dossier as "pending verification" per spec §4.7) plus a
`p22_review_item` for a human to confirm or reject — nothing here writes a
verified row.

**Scope disclosed, not an oversight**: only the 8-K's PRIMARY document is
scanned. Spec also names the EX-99.1 press-release exhibit as a detection
surface ("plus the press-release exhibit"), but `download_8k_filings`'s
universe-wide index gives one `primary_document` filename per filing, not
the exhibit list — resolving that needs a second per-filing index fetch this
pass doesn't add (see `docs/Tasks.md`). Many Item 7.01/8.01 8-Ks quote the
press release directly in the primary document body, so this still catches
a real share of true positives; it does not catch every one.

**`"{ADVISOR}"` placeholder collapsed to a substring match.** Spec's
`"engaged {ADVISOR} as financial advisor"` phrase names a template variable
this module doesn't resolve (advisor names aren't a maintained list
anywhere in this repo) — matching on `"engaged"` ... `"as financial
advisor"` as two required substrings, in order, is the pragmatic
candidate-generator equivalent: still routed to human review like every
other match, so a false positive here costs a wasted review-queue glance,
not a bad score.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import STRATEGIC_PROCESS_PHRASES_YAML
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# spec §2.6.1: only these two item codes count — Item 1.01 (deal already signed) is explicitly excluded.
_RELEVANT_8K_ITEMS = frozenset({"7.01", "8.01"})

_ADVISOR_PHRASE_MARKERS = ("engaged", "as financial advisor")


def load_strategic_process_phrases(path: Path = STRATEGIC_PROCESS_PHRASES_YAML) -> Dict[str, List[str]]:
    """Parse `p22_strategic_process_phrases.yaml` into `{"strong": [...], "moderate": [...],
    "negative": [...]}`. Raises `ValueError` if any of the three required keys is missing —
    this drives a scoring-relevant classification, so a malformed file should fail loudly."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    phrases = raw.get("strategic_process_phrases", {})
    missing = {"strong", "moderate", "negative"} - phrases.keys()
    if missing:
        raise ValueError(f"{path} missing required phrase categories: {sorted(missing)}")
    return phrases


def classify_filing_text(text: str, phrases: Dict[str, List[str]]) -> Optional[Dict[str, Optional[str]]]:
    """
    Classify one filing's text against the phrase list. Checked in this order — negative first —
    because a filing announcing a CONCLUDED process may still contain the word sequence
    "strategic alternatives" in a way that would otherwise false-match a `strong`/`moderate`
    phrase (e.g. "concluded its review of strategic alternatives" contains "strategic
    alternatives" but is the opposite signal).

    Args:
        text: Raw filing document text (HTML or plain text — matched case-insensitively, so no
            HTML-stripping is required for phrase presence, only for a human-readable snippet).
        phrases: From `load_strategic_process_phrases`.

    Returns:
        `{"state": ..., "strength": ...(or None), "matched_phrase": ...}` for the first match, or
        `None` if nothing in any category matched. `state` is `"concluded_no_deal"` for a
        `negative` match, `"disclosed_open"` for `strong`/`moderate` (spec §2.6.1's state machine —
        `rumored` isn't reachable from phrase-matching alone; it needs a weaker, unconfirmed-rumor
        source this module doesn't have).
    """
    lowered = text.lower()

    for phrase in phrases.get("negative", []):
        if phrase.lower() in lowered:
            return {"state": "concluded_no_deal", "strength": None, "matched_phrase": phrase}

    for phrase in phrases.get("strong", []):
        if phrase == "engaged {ADVISOR} as financial advisor":
            if all(marker in lowered for marker in _ADVISOR_PHRASE_MARKERS):
                return {"state": "disclosed_open", "strength": "strong", "matched_phrase": phrase}
            continue
        if phrase.lower() in lowered:
            return {"state": "disclosed_open", "strength": "strong", "matched_phrase": phrase}

    for phrase in phrases.get("moderate", []):
        if phrase.lower() in lowered:
            return {"state": "disclosed_open", "strength": "moderate", "matched_phrase": phrase}

    return None


def run(repo: Any, edgar: Any, as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Scan one day's 8-K index for strategic-alternatives candidates and write unverified
    `p22_corporate_process_event` + `p22_review_item` rows for every match.

    Args:
        repo: A `P22Repo`-shaped object.
        edgar: An `EdgarDownloader`-shaped object (`download_8k_filings`, `fetch_filing_document`).
        as_of_date: Filing date to scan. Defaults to `download_8k_filings`'s own default (yesterday).

    Returns:
        Summary dict — `filings_scanned` (item 7.01/8.01, in-universe filings whose primary
        document was actually fetched), `candidates_written`, and `rejection_breakdown` (why a
        filing was skipped before reaching the phrase check). A high `candidates_written` count
        most days would itself be suspicious (spec: "the single strongest observable signal" is
        supposed to be rare) — unlike P20's empty-funnel diagnostics, zero matches here is the
        expected, healthy outcome, so this does NOT warn on zero.
    """
    phrases = load_strategic_process_phrases()
    companies_by_cik = {c["cik"]: c["company_id"] for c in repo.list_companies_full() if c.get("cik")}

    index_df = edgar.download_8k_filings(as_of_date)

    filings_scanned = 0
    candidates_written = 0
    skip_reasons: Counter[str] = Counter()

    for _, row in index_df.iterrows():
        items = {i.strip() for i in str(row.get("items") or "").split(",") if i.strip()}
        if not items & _RELEVANT_8K_ITEMS:
            skip_reasons["item_not_relevant"] += 1
            continue

        cik = str(row.get("cik") or "").zfill(10)
        company_id = companies_by_cik.get(cik)
        if company_id is None:
            skip_reasons["cik_not_in_universe"] += 1
            continue

        accession_number = str(row.get("accession_number") or "")
        primary_document = str(row.get("primary_document") or "")
        if not accession_number or not primary_document:
            skip_reasons["missing_accession_or_document"] += 1
            continue

        doc_text = edgar.fetch_filing_document(cik, accession_number, primary_document)
        if doc_text is None:
            skip_reasons["document_fetch_failed"] += 1
            continue
        filings_scanned += 1

        match = classify_filing_text(doc_text, phrases)
        if match is None:
            continue

        filed_date_str = str(row.get("filed_date") or "")
        try:
            event_date = date.fromisoformat(filed_date_str) if filed_date_str else (as_of_date or date.today())
        except ValueError:
            event_date = as_of_date or date.today()

        known_from = datetime.now(timezone.utc)
        event_id = repo.upsert_corporate_process_event(
            company_id=company_id,
            event_date=event_date,
            state=match["state"],
            scope="unclear",  # spec §2.6.1: scope (whole_company vs asset_only) needs human review
            strength=match["strength"],
            accession_no=accession_number,
            matched_phrase=match["matched_phrase"],
            is_verified=False,
            known_from=known_from,
            source_url=(
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession_number.replace('-', '')}/{primary_document}"
            ),
        )
        repo.add_review_item(
            item_type="strategic_alternatives_candidate",
            payload={
                "reason": "strategic_alternatives_candidate",
                "event_id": event_id,
                "company_id": company_id,
                "state": match["state"],
                "strength": match["strength"],
                "matched_phrase": match["matched_phrase"],
                "accession_no": accession_number,
            },
            evidence_url=(
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession_number.replace('-', '')}/{primary_document}"
            ),
            # Strong phrases carry more weight downstream (spec §5.2's Tier 3) — surface them
            # first in a human reviewer's queue.
            priority=2 if match["strength"] == "strong" else 1,
        )
        candidates_written += 1

    summary = {
        "filings_scanned": filings_scanned,
        "candidates_written": candidates_written,
        "rejection_breakdown": dict(skip_reasons),
    }
    _logger.info("Strategic-alternatives scan complete: %s", summary)
    return summary
