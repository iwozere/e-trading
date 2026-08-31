"""
P22 — SEC XBRL financial-fact normalizer (spec §2.1, §3.1, M3 Block C input).

Turns landed `sec_company_facts` raw-zone payloads (the full XBRL
companyfacts JSON, already landed by `sec_raw_ingest.land_submissions_and_facts`
via `EdgarDownloader.load_company_facts`) into `p22_financial_fact` bitemporal
rows via `P22Repo.upsert_financial_fact_bitemporal`.

**Scope of this pass**, tags live-verified 2026-08-30 against 3 real biotech
filers (Moderna, Sarepta, Alnylam):

- `cash_and_equivalents` <- `us-gaap:CashAndCashEquivalentsAtCarryingValue` (USD, instant)
- `shares_outstanding` <- `dei:EntityCommonStockSharesOutstanding` (shares, instant)
- `short_term_investments` <- `us-gaap:ShortTermInvestments` (USD, instant) — confirmed present
  for only 1 of the 3 filers checked (Sarepta); Moderna and Alnylam report neither this tag nor
  spec §2.1's suggested alternative (`MarketableSecuritiesCurrent`, checked and absent for all
  three) in the period ranges checked. Not extended with a fallback tag this pass — a company
  simply not reporting this line is indistinguishable, from this data alone, from "uses a tag
  nobody's checked yet," and only tags actually observed live are in `FACT_TAG_MAP` (see below).
  `None` for a company without a live-verified tag match is correct, not a gap.
- `total_debt` <- fall back through `us-gaap:LongTermDebtNoncurrent`, `us-gaap:LongTermDebt`,
  `us-gaap:ConvertibleDebtNoncurrent` (all USD, instant) — all three live-verified, and live data
  showed *why* a fallback chain (not a single "best" tag) is necessary: Alnylam used `LongTermDebt`
  in 2021-2022 filings and switched to `ConvertibleDebtNoncurrent` from 2025 onward, with no
  overlap between the two tags' reporting periods. **Correction vs. spec §2.1's own suggested
  list**, which names `ConvertibleNotesPayable` as the second debt tag: that tag was checked and is
  absent from all three filers; `ConvertibleDebtNoncurrent` is what's actually in use. `extract_fact_series`
  merges entries from every candidate tag for a metric (not "first tag with any data, stop") —
  the Alnylam tag-migration case is exactly why: stopping at the first non-empty candidate would
  have silently discarded its post-2025 debt history entirely.
- `quarterly_opex_burn` <- derived from `us-gaap:NetCashProvidedByUsedInOperatingActivities` (USD,
  duration) via `extract_quarterly_delta_series`, live-verified present (and cumulative-YTD-shaped,
  confirming the derivation below is actually needed) across all three filers.

**Deliberately not added this pass:** `rd_expense` (`ResearchAndDevelopmentExpense` — live-verified
present for all three filers, easy to add) has no consumer yet (no built Block C/B feature reads
R&D spend) — adding a metric nothing reads is speculative work with no way to test it's actually
useful, the same reasoning `docs/Design.md` gives for not persisting `build_universe_history`
speculatively. Add it once a feature needs it, live-verifying the quarter-delta shape at that point
too (duration concepts need the same cumulative-vs-standalone check `quarterly_opex_burn` needed).

**A real correctness trap this module guards against:** one XBRL tag's raw
entries include the SAME `end` (period_end) repeated across MULTIPLE later
filings — each 10-Q/10-K re-reports the prior period's balance as a
comparative column, not because the fact changed. Naively re-processing every
entry as if it were new would treat that repetition as a fact first known on
the LATER filing's date, corrupting the bitemporal history with a
false-lookahead-safe-looking but wrong `known_from`. This module dedupes by
`end`, keeping only the earliest-filed entry per period — the true first
public disclosure. `extract_quarterly_delta_series` applies the same trap-guard
to duration facts, deduping by `(start, end)` instead of `end` alone.

**Known limitation — restatements.** `upsert_financial_fact_bitemporal`'s
uniqueness is per `(company_id, metric)`, not per
`(company_id, metric, period_end)` — it tracks one current-value time series
per metric, not a value indexed by fiscal period. A genuine restatement (a
LATER filing reporting a DIFFERENT value for an ALREADY-seen period_end)
doesn't fit that model cleanly: naively appending it out of period_end order
risks setting `valid_from` earlier than an already-inserted later-period row.
Restatements of cash/shares-outstanding specifically are rare (far more
common for revenue/earnings). Not handled in this pass —
`extract_fact_series` detects a changed value for an already-seen period_end
and logs a warning instead of silently writing something wrong or silently
dropping it.

**Quarter-delta derivation for duration concepts.** SEC's XBRL companyfacts
API reports flow (duration) concepts like operating cash flow as running
totals from each fiscal year's start — `start` is always that year's first
day, `end` moves forward with each quarterly/annual filing — not as discrete
per-quarter figures (live-verified 2026-08-30: Moderna's
`NetCashProvidedByUsedInOperatingActivities` entries for FY2022 show
`start=2022-01-01` paired with `end=2022-03-31`, `2022-06-30`, `2022-09-30`,
`2022-12-31` in sequence, each a cumulative total, not a quarterly one).
`extract_quarterly_delta_series` groups entries by `start` (== fiscal year),
sorts by `end`, and takes consecutive differences to recover the
quarter-standalone value. The stored value is the raw signed XBRL delta
(negative = cash used by operations, for this specific tag) — this module
does not reinterpret the sign as a "burn magnitude"; that's Block C's job.
**Known limitation, not solved here:** a fiscal-year redefinition or an
amended filing reporting a slightly different `start` for what's conceptually
the same fiscal year would start a new group instead of continuing the
existing one, silently losing that quarter's delta (it becomes its own
group's baseline entry instead of a true delta). Not observed in the 3
filers checked.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# metric name -> list of (XBRL taxonomy, tag, unit) candidates, tried in order and MERGED (not
# "first non-empty, stop") — see module docstring's `total_debt`/Alnylam example for why a
# tag-migration case needs every candidate's entries, not just the first that has any data. Add
# a candidate only after live-verifying it against several real filers, the same way these were.
FACT_TAG_MAP: Dict[str, List[Tuple[str, str, str]]] = {
    "cash_and_equivalents": [("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD")],
    "shares_outstanding": [("dei", "EntityCommonStockSharesOutstanding", "shares")],
    "short_term_investments": [("us-gaap", "ShortTermInvestments", "USD")],
    "total_debt": [
        ("us-gaap", "LongTermDebtNoncurrent", "USD"),
        ("us-gaap", "LongTermDebt", "USD"),
        ("us-gaap", "ConvertibleDebtNoncurrent", "USD"),
    ],
}

# Duration (flow) concepts needing quarter-delta derivation — see
# `extract_quarterly_delta_series` and the module docstring section above.
DURATION_DELTA_TAG_MAP: Dict[str, Tuple[str, str, str]] = {
    "quarterly_opex_burn": ("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD"),
}


@dataclass(frozen=True)
class NormalizedFact:
    """One (company, metric, period) fact, ready for `P22Repo.upsert_financial_fact_bitemporal`."""

    metric: str
    value: float
    unit: str
    period_end: date
    known_from: datetime
    source_id: str  # SEC accession number
    source_url: str


def filing_index_url(cik: str, accession_number: str) -> str:
    """
    The standard EDGAR filing-index URL for one accession number
    (`https://www.sec.gov/Archives/edgar/data/<cik>/<accn-no-dashes>/<accn>-index.htm`),
    live-verified 2026-08-30 (200 OK against a real Sarepta filing).
    """
    cik_int = str(int(cik))  # strips any zero-padding
    accn_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{accession_number}-index.htm"


def extract_fact_series(companyfacts: Dict[str, Any], cik: str, metric: str) -> List[NormalizedFact]:
    """
    Extract one `NormalizedFact` per distinct `period_end` for `metric` (see
    `FACT_TAG_MAP`), keeping only the earliest-filed entry per period — see
    the module docstring's "correctness trap" note for why.

    When `metric` has more than one candidate tag, every candidate's entries
    are MERGED before deduping — not "use the first candidate with any data,
    stop" — because a filer can migrate from one tag name to another mid-
    history (see module docstring's `total_debt`/Alnylam example) and
    stopping early would silently drop the newer tag's entries entirely. A
    genuine same-period value conflict between two candidate tags is caught
    by the same "possible restatement" check below that guards against the
    comparative-column trap — it doesn't know or care which tag an entry
    came from.

    Args:
        companyfacts: The raw XBRL companyfacts JSON (landed under
            `sec_company_facts` in the raw zone).
        cik: The company's CIK, for building `source_url`.
        metric: A key in `FACT_TAG_MAP`.

    Returns:
        Facts sorted by `known_from` ascending — the order
        `write_financial_facts` needs to preserve the bitemporal history
        correctly. Possibly empty if no candidate tag has data for this
        company (not every filer reports every tag).

    Raises:
        ValueError: if `metric` isn't in `FACT_TAG_MAP`.
    """
    if metric not in FACT_TAG_MAP:
        raise ValueError(f"Unknown metric {metric!r}; add it to FACT_TAG_MAP first (see module docstring)")
    candidates = FACT_TAG_MAP[metric]
    unit = candidates[0][2]  # every current metric's candidates share one unit

    entries: List[Dict[str, Any]] = []
    for taxonomy, tag, tag_unit in candidates:
        entries.extend(companyfacts.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {}).get(tag_unit, []))
    if not entries:
        return []

    entries_by_filed = sorted(entries, key=lambda e: e.get("filed") or "")

    by_period: Dict[date, Dict[str, Any]] = {}
    for entry in entries_by_filed:
        end_raw, filed_raw, val = entry.get("end"), entry.get("filed"), entry.get("val")
        if end_raw is None or filed_raw is None or val is None:
            continue
        period_end = date.fromisoformat(end_raw)

        existing = by_period.get(period_end)
        if existing is not None:
            if existing["val"] != val:
                _logger.warning(
                    "Possible restatement for metric=%s cik=%s period_end=%s: value changed from %s "
                    "(filed %s) to %s (filed %s) — not written, see module docstring's known limitation",
                    metric, cik, period_end, existing["val"], existing["filed"], val, filed_raw,
                )
            continue  # earliest-filed entry for this period already kept
        by_period[period_end] = entry

    facts = [
        NormalizedFact(
            metric=metric,
            value=float(entry["val"]),
            unit=unit,
            period_end=period_end,
            known_from=datetime.combine(date.fromisoformat(entry["filed"]), time.min, tzinfo=timezone.utc),
            source_id=entry.get("accn", ""),
            source_url=filing_index_url(cik, entry["accn"]) if entry.get("accn") else "",
        )
        for period_end, entry in by_period.items()
    ]
    facts.sort(key=lambda f: f.known_from)
    return facts


def extract_quarterly_delta_series(companyfacts: Dict[str, Any], cik: str, metric: str) -> List[NormalizedFact]:
    """
    Derive quarter-standalone values for a duration (flow) concept from its
    cumulative year-to-date XBRL entries — see the module docstring's
    "Quarter-delta derivation" section for the mechanics and the known
    fiscal-year-redefinition limitation.

    Args:
        companyfacts: The raw XBRL companyfacts JSON.
        cik: The company's CIK, for building `source_url`.
        metric: A key in `DURATION_DELTA_TAG_MAP`.

    Returns:
        Facts sorted by `known_from` ascending. Possibly empty if the tag has
        no data for this company.

    Raises:
        ValueError: if `metric` isn't in `DURATION_DELTA_TAG_MAP`.
    """
    if metric not in DURATION_DELTA_TAG_MAP:
        raise ValueError(f"Unknown duration-delta metric {metric!r}; add it to DURATION_DELTA_TAG_MAP first")
    taxonomy, tag, unit = DURATION_DELTA_TAG_MAP[metric]

    entries = companyfacts.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {}).get(unit, [])
    if not entries:
        return []

    entries_by_filed = sorted(entries, key=lambda e: e.get("filed") or "")

    # Dedupe by (start, end), keeping earliest-filed — the same comparative-column trap as
    # extract_fact_series, applied to duration facts (keyed on the whole period, not just `end`).
    by_period: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in entries_by_filed:
        start_raw, end_raw, filed_raw, val = entry.get("start"), entry.get("end"), entry.get("filed"), entry.get("val")
        if start_raw is None or end_raw is None or filed_raw is None or val is None:
            continue
        key = (start_raw, end_raw)
        if key in by_period:
            continue  # earliest-filed entry for this period already kept
        by_period[key] = entry

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in by_period.values():
        groups.setdefault(entry["start"], []).append(entry)

    facts: List[NormalizedFact] = []
    for group_entries in groups.values():
        group_entries.sort(key=lambda e: e["end"])
        prior_val: Optional[float] = None
        for entry in group_entries:
            val = float(entry["val"])
            delta = val if prior_val is None else val - prior_val
            prior_val = val
            facts.append(
                NormalizedFact(
                    metric=metric,
                    value=delta,
                    unit=unit,
                    period_end=date.fromisoformat(entry["end"]),
                    known_from=datetime.combine(date.fromisoformat(entry["filed"]), time.min, tzinfo=timezone.utc),
                    source_id=entry.get("accn", ""),
                    source_url=filing_index_url(cik, entry["accn"]) if entry.get("accn") else "",
                )
            )

    facts.sort(key=lambda f: f.known_from)
    return facts


def write_financial_facts(company_id: int, facts: List[NormalizedFact], repo: Any) -> int:
    """
    Write a list of `NormalizedFact`s via `P22Repo.upsert_financial_fact_bitemporal`,
    in `known_from` order (re-sorted here defensively — callers should
    already pass them sorted, from `extract_fact_series`, but the bitemporal
    correctness of this write depends entirely on chronological order, so
    this does not trust the caller).

    Returns:
        Number of facts written.
    """
    for fact in sorted(facts, key=lambda f: f.known_from):
        repo.upsert_financial_fact_bitemporal(
            company_id=company_id,
            metric=fact.metric,
            value=fact.value,
            known_from=fact.known_from,
            source_id=fact.source_id,
            unit=fact.unit,
            period_end=fact.period_end,
            valid_from=fact.period_end,
            source_url=fact.source_url or None,
        )
    return len(facts)
