"""
P22 — Schedule 13D/13D-A/13G/13G-A ingest (spec §2.6.2, §4.7, M5).

**Detection source, live-verified 2026-09-08 — corrects `download_13dg_filings`'s own docstring**:
that method's claim "EDGAR's full-text search (EFTS) does not index SC 13D/G filings" is only
true for the WRONG form-type string it was built against (`"SC 13D"`/`"SC 13G"` — see the real bug
just fixed in `edgar_downloader._13DG_FORM_TYPE_ALIASES`). Queried with the REAL form strings
(`"SCHEDULE 13D"`, `"SCHEDULE 13D/A"`, `"SCHEDULE 13G"`, `"SCHEDULE 13G/A"`), EFTS returns real
hits, and its `ciks` field is populated with EVERY entity associated with the filing — both the
subject company AND the filer(s) — which is what makes `EdgarDownloader.efts_filings_search`
usable here at all: querying with the WHOLE P22 universe's CIK list finds every 13D/G filing that
mentions any of them, whether as subject or as filer, in one targeted pass instead of scanning
every 13D/G filed industry-wide (the `download_13dg_filings`/quarterly-form.idx approach, which is
filer-centric and would need a document fetch per filing just to learn the subject company).

**Why the document still has to be fetched anyway**: EFTS's `_source` doesn't tag which `ciks`
entry is the subject vs. the filer(s) — resolving that (and getting `pct_of_class`) needs the
actual filing. `_id` (`"{accession}:primary_doc.xml"`) already names the exact document, so no
filename-guessing is needed (unlike `process_events.py`'s use of a similar pattern for 8-Ks).
That document is the structured-XML Schedule 13D/G, which has NO SGML `SUBJECT COMPANY:` header —
until 2026-11-01 the parser only understood that header, so every filing was rejected as
`header_parse_failed`. Subject/filer CIKs now come from the XML; filer names from EFTS
`display_names` (EDGAR conformed names), falling back to the 13D reporting-person name.

**`pct_of_class` is populated ONLY when exactly one distinct `<percentOfClass>` value appears
in the whole document.** Real filings (live-verified) often carry several cover-page rows with
DIFFERENT percentages for different reporting persons even within what the header calls one
"FILED BY" company (affiliated funds/individuals filing jointly) — attributing the wrong one to
the wrong filer would be a worse error than reporting `None`.

**`stated_intent` is always `None` — deliberately not attempted this pass.** Spec says to
"classify intent... via the review queue," but Item 4 (Purpose of Transaction) text is dense,
boilerplate-heavy legal prose (live-verified against a real filing) that a keyword heuristic
cannot reliably reduce to one of `passive | engagement | board_seats | sale_demand` — and unlike
`process_events.py`'s phrase list, spec gives no candidate phrases for this classification at all.
Doing it properly also needs `review_queue.py`'s confirm dispatch to accept a multi-valued
classification, not just confirm/reject — real, separate work, not attempted here (see
`docs/Tasks.md`). Every other field (`filer_type` via mechanical membership checks,
`pct_of_class`, `filed_date`) is populated without a review gate — spec's own framing is that a
real SEC filing already IS the verification, unlike `process_events.py`'s keyword candidates,
which is why `p22_activist_position` has no `is_verified` column at all (see
`P22Repo.get_verified_activist_positions`'s docstring).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import ACTIVIST_FILERS_YAML
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_13DG_FORMS = ("SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A")
# Real EFTS/quarterly-index form_type -> the short form p22_activist_position.form_type's CHECK
# constraint expects (spec §3.2) — same normalization as edgar_downloader's _13DG_FORM_TYPE_ALIASES,
# duplicated here rather than imported since that map also carries the legacy "SC 13D" aliases this
# module never needs to accept (EFTS only ever returns the real "SCHEDULE..." strings).
_FORM_TYPE_CANONICAL = {
    "SCHEDULE 13D": "SC 13D",
    "SCHEDULE 13D/A": "SC 13D/A",
    "SCHEDULE 13G": "SC 13G",
    "SCHEDULE 13G/A": "SC 13G/A",
}

_FILER_BLOCK_MARKER = re.compile(r"FILED BY:|REPORTING-OWNER:")
_CIK_PATTERN = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)")
_NAME_PATTERN = re.compile(r"COMPANY CONFORMED NAME:\s*(.+)")
# 13D uses <percentOfClass>, 13G uses <classPercent> (live-verified 2026-11-01).
_PCT_OF_CLASS_PATTERN = re.compile(r"<(?:percentOfClass|classPercent)>\s*([\d.]+)\s*</(?:percentOfClass|classPercent)>")

# Structured-XML Schedule 13D/G (`primary_doc.xml`, mandatory since Dec 2024) — no SGML header at
# all. Live-verified 2026-11-01: subject is `<issuerCIK>` (13D) / `<issuerCik>` (13G); each filer is
# a `<filerCredentials><cik>` in `<headerData>`; 13D `<reportingPersonInfo>` blocks pair a CIK with a
# name, 13G reporting persons carry a name only.
_XML_ROOT_MARKER = "<edgarSubmission"
_XML_SUBJECT_CIK = re.compile(r"<issuerC[iI][kK]>\s*(\d+)\s*</issuerC[iI][kK]>")
_XML_FILER_CIK = re.compile(r"<filerCredentials>.*?<cik>\s*(\d+)\s*</cik>", re.DOTALL)
_XML_REPORTING_PERSON = re.compile(r"<reportingPersonInfo>(.*?)</reportingPersonInfo>", re.DOTALL)
_XML_REPORTING_CIK = re.compile(r"<reportingPersonCIK>\s*(\d+)\s*</reportingPersonCIK>")
_XML_REPORTING_NAME = re.compile(r"<reportingPersonName>\s*(.*?)\s*</reportingPersonName>", re.DOTALL)
# EFTS `display_names` entries look like "SenesTech, Inc.  (SNES)  (CIK 0001680378)".
_DISPLAY_NAME_PATTERN = re.compile(r"^(.*?)\s{2}.*\(CIK (\d+)\)\s*$")


def load_activist_filers(path: Path = ACTIVIST_FILERS_YAML) -> set:
    """CIKs from `config/activist_filers.yaml` (spec §2.6.2), zero-padded to 10 digits to match
    the header-parsed CIK format. Missing file -> empty set (not an error — this list starts
    genuinely incomplete pending domain review, see that file's header)."""
    if not path.exists():
        return set()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(cik).zfill(10) for cik in raw.get("activist_filer_ciks", [])}


def parse_13dg_header(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the subject company's CIK and every filer's (CIK, name) from a 13D/G filing — the
    structured-XML `primary_doc.xml` (what EFTS `_id` names today) or a legacy SGML header.
    Returns `None` if neither shape is recognised (logged as `header_parse_failed` by the caller,
    not silently treated as "no filers").

    Returns:
        `{"subject_cik": str, "filers": [{"cik": str, "name": Optional[str]}, ...]}` — CIKs
        zero-padded to 10 digits. `filers` may be empty if no filer CIK could be parsed. An XML
        filer's `name` is `None` when no reporting person carries its CIK (always so for 13G).
    """
    if _XML_ROOT_MARKER in text:
        return _parse_13dg_xml(text)
    return _parse_13dg_sgml(text)


def _parse_13dg_xml(text: str) -> Optional[Dict[str, Any]]:
    """Structured-XML branch of `parse_13dg_header`."""
    subject_match = _XML_SUBJECT_CIK.search(text)
    if subject_match is None:
        return None
    names_by_cik: Dict[str, str] = {}
    for block in _XML_REPORTING_PERSON.findall(text):
        cik_match = _XML_REPORTING_CIK.search(block)
        name_match = _XML_REPORTING_NAME.search(block)
        if cik_match and name_match:
            names_by_cik[cik_match.group(1).zfill(10)] = unescape(name_match.group(1))

    filers: List[Dict[str, Optional[str]]] = []
    seen: set = set()
    for raw_cik in _XML_FILER_CIK.findall(text):
        cik = raw_cik.zfill(10)
        if cik not in seen:
            seen.add(cik)
            filers.append({"cik": cik, "name": names_by_cik.get(cik)})
    return {"subject_cik": subject_match.group(1).zfill(10), "filers": filers}


def _parse_13dg_sgml(text: str) -> Optional[Dict[str, Any]]:
    """Legacy SGML-header branch of `parse_13dg_header` (`SUBJECT COMPANY:` / `FILED BY:` blocks)."""
    subject_idx = text.find("SUBJECT COMPANY:")
    if subject_idx == -1:
        return None

    first_filer_match = _FILER_BLOCK_MARKER.search(text, subject_idx)
    subject_block_end = first_filer_match.start() if first_filer_match else len(text)
    subject_block = text[subject_idx:subject_block_end]
    subject_cik_match = _CIK_PATTERN.search(subject_block)
    if subject_cik_match is None:
        return None
    subject_cik = subject_cik_match.group(1).zfill(10)

    filers: List[Dict[str, str]] = []
    if first_filer_match is not None:
        filer_section = text[first_filer_match.start():]
        # Split on every filer-block marker; the first chunk (before the first marker) is empty.
        for chunk in _FILER_BLOCK_MARKER.split(filer_section)[1:]:
            cik_match = _CIK_PATTERN.search(chunk)
            name_match = _NAME_PATTERN.search(chunk)
            if cik_match is None or name_match is None:
                continue
            filers.append({"cik": cik_match.group(1).zfill(10), "name": name_match.group(1).strip()})

    return {"subject_cik": subject_cik, "filers": filers}


def efts_names_by_cik(display_names: List[str]) -> Dict[str, str]:
    """Map zero-padded CIK -> EDGAR conformed name from an EFTS hit's `display_names` list —
    the filer-name source for XML filings, whose header carries only the filer's CIK."""
    names: Dict[str, str] = {}
    for entry in display_names:
        match = _DISPLAY_NAME_PATTERN.match(str(entry))
        if match:
            names[match.group(2).zfill(10)] = match.group(1).strip()
    return names


def extract_single_pct_of_class(text: str) -> Optional[float]:
    """`pct_of_class`, only when every `<percentOfClass>` tag in the document agrees on one
    value — see module docstring for why a document with several DIFFERENT values (multiple
    reporting persons on the same filing) is left `None` rather than guessed."""
    values = {m.group(1) for m in _PCT_OF_CLASS_PATTERN.finditer(text)}
    if len(values) == 1:
        return float(next(iter(values)))
    return None


def classify_filer_type(filer_cik: str, repo: Any, activist_filer_ciks: set) -> Optional[str]:
    """
    Mechanical filer-type classification (spec §2.6.2) — both checks are set-membership/DB-lookup,
    not judgment calls, so unlike `stated_intent` these ARE safe to auto-populate:
      - `'activist'` if `filer_cik` is in the curated `activist_filers.yaml` list.
      - `'strategic_corporate'` if `filer_cik` matches a `p22_company` row with
        `role in ('acquirer', 'both')` — spec: "13D filed by a strategic (corporate) filer in the
        acquirer universe."
      - `None` otherwise (not a guess at `'crossover_fund'`/`'other'` — genuinely unclassified).
    """
    if filer_cik in activist_filer_ciks:
        return "activist"
    company = repo.get_company_by_cik(filer_cik)
    if company is not None and company.get("role") in ("acquirer", "both"):
        return "strategic_corporate"
    return None


def run(repo: Any, edgar: Any, as_of_date: Optional[date] = None, lookback_days: int = 1) -> Dict[str, Any]:
    """
    Scan `lookback_days` of Schedule 13D/13D-A/13G/13G-A filings for every company in the P22
    universe (as subject) and write `p22_activist_position` rows.

    Args:
        repo: A `P22Repo`-shaped object.
        edgar: An `EdgarDownloader`-shaped object (`efts_filings_search`, `fetch_filing_document`).
        as_of_date: End of the scan window (inclusive). Defaults to yesterday.
        lookback_days: Window width — a daily job only needs 1, but this allows a wider backfill
            call without a separate code path.

    Returns:
        Summary dict — `filings_matched`, `positions_written`, `rejection_breakdown`.
    """
    end = as_of_date or (date.today() - timedelta(days=1))
    start = end - timedelta(days=lookback_days - 1)

    companies_by_cik = {c["cik"]: c["company_id"] for c in repo.list_companies_full() if c.get("cik")}
    activist_filer_ciks = load_activist_filers()
    universe_ciks = list(companies_by_cik.keys())

    seen_ids: set = set()
    hits: List[Dict[str, Any]] = []
    for form in _13DG_FORMS:
        for hit in edgar.efts_filings_search(ciks=universe_ciks, forms=form, start_dt=str(start), end_dt=str(end)):
            hit_id = hit.get("_id")
            if hit_id and hit_id not in seen_ids:
                seen_ids.add(hit_id)
                hits.append(hit)

    positions_written = 0
    skip_reasons: Counter = Counter()

    _logger.info("Activist-position scan %s..%s: %d filings to fetch", start, end, len(hits))
    for done, hit in enumerate(hits, start=1):
        if done % 250 == 0:
            _logger.info("Activist-position scan %s..%s: %d/%d filings processed", start, end, done, len(hits))
        src = hit.get("_source", {})
        accession = str(src.get("adsh") or "")
        canonical_form = _FORM_TYPE_CANONICAL.get(str(src.get("form") or ""))
        efts_id = str(hit.get("_id") or "")
        filename = efts_id.split(":", 1)[1] if ":" in efts_id else ""
        ciks_in_hit = src.get("ciks") or []
        if not accession or canonical_form is None or not filename or not ciks_in_hit:
            skip_reasons["malformed_hit"] += 1
            continue

        doc_text = edgar.fetch_filing_document(ciks_in_hit[0], accession, filename)
        if doc_text is None:
            skip_reasons["document_fetch_failed"] += 1
            continue

        header = parse_13dg_header(doc_text)
        if header is None:
            skip_reasons["header_parse_failed"] += 1
            continue

        company_id = companies_by_cik.get(header["subject_cik"])
        if company_id is None:
            skip_reasons["subject_not_in_universe"] += 1
            continue

        pct_of_class = extract_single_pct_of_class(doc_text)
        # Spec §4.7/§8.3: known_from is the filing date, never the 5%-crossing date or the ingest time.
        try:
            filed_date = date.fromisoformat(str(src.get("file_date")))
            known_from = datetime.combine(filed_date, time.min, tzinfo=timezone.utc)
        except (TypeError, ValueError):
            filed_date = end
            known_from = datetime.now(timezone.utc)
        source_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(ciks_in_hit[0])}/"
            f"{accession.replace('-', '')}/{filename}"
        )

        if not header["filers"]:
            skip_reasons["no_filers_parsed"] += 1
            continue

        efts_names = efts_names_by_cik(src.get("display_names") or [])
        for filer in header["filers"]:
            filer_type = classify_filer_type(filer["cik"], repo, activist_filer_ciks)
            repo.upsert_activist_position(
                company_id=company_id,
                filer_cik=filer["cik"],
                filer_name=efts_names.get(filer["cik"]) or filer["name"],
                filer_type=filer_type,
                form_type=canonical_form,
                pct_of_class=pct_of_class,
                stated_intent=None,
                filed_date=filed_date,
                known_from=known_from,
                source_url=source_url,
            )
            positions_written += 1

    summary = {
        "filings_matched": len(hits),
        "positions_written": positions_written,
        "rejection_breakdown": dict(skip_reasons),
    }
    _logger.info("Activist-position scan complete: %s", summary)
    return summary
