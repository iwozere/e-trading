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

**`pct_of_class` is the HIGHEST `<percentOfClass>`/`<classPercent>` value in the document.**
Joint filings carry one cover-page row per reporting person (the fund, its adviser, the controlling
individual), often with different percentages (live-verified 2026-11-03: Avoro on WHWK 19.99/4.1/
19.99, Sarissa on IRWD 5.6/5.8). The highest is the group's beneficial ownership, which is what
Block G's stake threshold asks about. Until 2026-11-03 such filings were stored as `None`; rows
written then are back-filled by `repair_missing_pct` (`run_activist_positions_ingest.py
--repair-pct`).

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
from typing import Any, Dict, List, Optional, Tuple

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
# The `source_url` this module writes (see `run`).
_SOURCE_URL_PATTERN = re.compile(r"^https://www\.sec\.gov/Archives/edgar/data/(\d+)/(\d+)/([^/]+)$")
# EFTS `display_names` entries look like "SenesTech, Inc.  (SNES)  (CIK 0001680378)".
_DISPLAY_NAME_PATTERN = re.compile(r"^(.*?)\s{2}.*\(CIK (\d+)\)\s*$")


def _load_cik_list(path: Path, key: str) -> set:
    """One CIK list from `config/activist_filers.yaml`, zero-padded to 10 digits to match the
    header-parsed CIK format. Missing file or key -> empty set."""
    if not path.exists():
        return set()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(cik).zfill(10) for cik in raw.get(key) or []}


def load_activist_filers(path: Path = ACTIVIST_FILERS_YAML) -> set:
    """Activist-fund CIKs from `config/activist_filers.yaml` (spec §2.6.2). Missing file -> empty
    set (not an error — this list starts genuinely incomplete pending domain review, see that
    file's header)."""
    return _load_cik_list(path, "activist_filer_ciks")


def load_crossover_funds(path: Path = ACTIVIST_FILERS_YAML) -> set:
    """Healthcare crossover-fund CIKs from `config/activist_filers.yaml` — spec §2.6.2: a 13D
    from a crossover fund "is not the same event as a 13D from an activist with a campaign
    history, and the model must distinguish them"."""
    return _load_cik_list(path, "crossover_fund_ciks")


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


def extract_pct_of_class(text: str) -> Optional[float]:
    """The highest `<percentOfClass>`/`<classPercent>` value in a 13D/G document — the group's
    beneficial ownership on a joint filing (see module docstring). `None` if the document has none."""
    values = []
    for m in _PCT_OF_CLASS_PATTERN.finditer(text):
        try:
            values.append(float(m.group(1)))
        except ValueError:
            continue  # e.g. a lone "." — not a percentage
    return max(values) if values else None


def parse_source_url(url: str) -> Optional[Tuple[str, str, str]]:
    """`(cik, accession_no_dashes, filename)` from a stored `source_url`
    (`https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}`), or `None`."""
    m = _SOURCE_URL_PATTERN.match(url or "")
    return (m.group(1), m.group(2), m.group(3)) if m else None


def repair_missing_pct(repo: Any, edgar: Any) -> Dict[str, Any]:
    """
    Re-fetch every filing whose positions have `pct_of_class` NULL and fill it with
    `extract_pct_of_class`. Each filing is fetched once even when it produced several rows (one
    per filer).

    Returns:
        `{"filings_checked", "filings_repaired", "rows_updated", "still_missing": {reason: count}}`.
    """
    rows = repo.list_activist_positions_missing_pct()
    by_url: Dict[str, List[int]] = {}
    for position_id, source_url in rows:
        by_url.setdefault(source_url or "", []).append(position_id)

    still_missing: Counter = Counter()
    filings_repaired = 0
    rows_updated = 0
    _logger.info("pct_of_class repair: %d rows over %d filings", len(rows), len(by_url))
    for done, (url, position_ids) in enumerate(by_url.items(), start=1):
        if done % 100 == 0:
            _logger.info("pct_of_class repair: %d/%d filings processed", done, len(by_url))
        parts = parse_source_url(url)
        if parts is None:
            still_missing["bad_source_url"] += 1
            continue
        doc_text = edgar.fetch_filing_document(*parts)
        if doc_text is None:
            still_missing["document_fetch_failed"] += 1
            continue
        pct = extract_pct_of_class(doc_text)
        if pct is None:
            still_missing["no_pct_in_document"] += 1
            continue
        rows_updated += repo.set_activist_position_pct(position_ids, pct)
        filings_repaired += 1

    summary = {
        "filings_checked": len(by_url),
        "filings_repaired": filings_repaired,
        "rows_updated": rows_updated,
        "still_missing": dict(still_missing),
    }
    _logger.info("pct_of_class repair complete: %s", summary)
    return summary


def classify_filer_type(
    filer_cik: str, repo: Any, activist_filer_ciks: set, crossover_fund_ciks: Optional[set] = None
) -> Optional[str]:
    """
    Mechanical filer-type classification (spec §2.6.2) — every check is set-membership/DB-lookup,
    not a judgment call, so unlike `stated_intent` these ARE safe to auto-populate:
      - `'activist'` if `filer_cik` is in the curated `activist_filer_ciks` list.
      - `'crossover_fund'` if `filer_cik` is in the curated `crossover_fund_ciks` list.
      - `'strategic_corporate'` if `filer_cik` matches a `p22_company` row with
        `role in ('acquirer', 'both')` — spec: "13D filed by a strategic (corporate) filer in the
        acquirer universe."
      - `None` otherwise (not a guess at `'other'` — genuinely unclassified).
    """
    if filer_cik in activist_filer_ciks:
        return "activist"
    if crossover_fund_ciks and filer_cik in crossover_fund_ciks:
        return "crossover_fund"
    company = repo.get_company_by_cik(filer_cik)
    if company is not None and company.get("role") in ("acquirer", "both"):
        return "strategic_corporate"
    return None


def reclassify_filer_types(repo: Any) -> Dict[str, Any]:
    """
    Re-run `classify_filer_type` over every filer already in `p22_activist_position` and rewrite
    `filer_type` where it changed — needed after `activist_filers.yaml` or the acquirer universe
    changes, since `filer_type` is stamped at ingest time.

    Returns:
        `{"filers_checked": int, "filers_changed": int, "rows_updated": int, "changes": {...}}`,
        where `changes` maps `"old->new"` to the number of filers that moved.
    """
    activist_filer_ciks = load_activist_filers()
    crossover_fund_ciks = load_crossover_funds()
    filers = repo.list_activist_position_filer_types()
    changes: Counter = Counter()
    rows_updated = 0
    for filer_cik, current in filers:
        new = classify_filer_type(filer_cik, repo, activist_filer_ciks, crossover_fund_ciks)
        if new == current:
            continue
        rows_updated += repo.set_activist_position_filer_type(filer_cik, new)
        changes[f"{current}->{new}"] += 1
    summary = {
        "filers_checked": len(filers),
        "filers_changed": sum(changes.values()),
        "rows_updated": rows_updated,
        "changes": dict(changes),
    }
    _logger.info("Activist-position filer reclassification complete: %s", summary)
    return summary


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
    crossover_fund_ciks = load_crossover_funds()
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

        pct_of_class = extract_pct_of_class(doc_text)
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
            filer_type = classify_filer_type(filer["cik"], repo, activist_filer_ciks, crossover_fund_ciks)
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
