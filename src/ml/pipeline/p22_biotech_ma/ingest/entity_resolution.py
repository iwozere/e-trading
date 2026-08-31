"""
P22 — M2 entity resolution: turn the landed SEC DERA rosters into a real
`p22_company` table (spec §2.0.2, §2.0.3, §3.3).

This is the first M2 slice, not the complete spec §2.0.3 filter set. What's
implemented here and why:

- **Reporting status** (filed a 10-K/10-Q in the trailing 6 months) — computable
  directly from the landed DERA rows.
- **Filer type** (exclude SPACs) — a name-based heuristic only, so hits are
  routed to the review queue rather than silently dropped (spec §3.3's
  "unresolved... do not silently drop" principle extended to this filter,
  since the heuristic isn't authoritative).
- **Exchange** (NYSE/NYSE American/Nasdaq) — resolved via SEC's current-snapshot
  `company_tickers_exchange.json`. This is **current names only**; it will not
  resolve a delisted ticker's historical exchange. Spec §2.0.2's suggested answer
  — parsing `dei:TradingSymbol`/`dei:SecurityExchangeName` off each company's own
  10-K/10-Q cover pages — turns out NOT to be reachable via the SEC XBRL
  `companyfacts` aggregation API this repo already lands (`EdgarDownloader.
  load_company_facts`): live-verified 2026-08-30 against three CIKs, including
  Meta (a known FB->META ticker change), that API's `dei` facts never include
  `TradingSymbol`, `SecurityExchangeName`, or even `EntityRegistrantName` — only
  numeric facts like `EntityCommonStockSharesOutstanding` are aggregated there.
  Those cover-page values exist only as inline XBRL in each filing's own HTML
  document, which would need per-filing document fetch + iXBRL parsing —
  genuinely new scraping infrastructure, not a read of already-landed data — so
  this is NOT built in this slice (tracked in docs/Tasks.md as needing a
  redesigned approach, not a quick follow-up). `eligible_exchange` is `None`,
  not `False`, when the current snapshot has no entry for a CIK, so a delisted
  name isn't misclassified as ineligible for a reason the code never actually
  checked.

**Not implemented at all in this slice** (spec §2.0.3), and explicitly left
`None` on `UniverseCandidate` rather than guessed at:

- **Size floor** (market cap > $25M) — blocked on the deferred market-data
  vendor decision (spec §2.0.6/§2.4).
- **Asset floor** (>=1 Phase I+ program) — needs `p22_trial` populated from
  CT.gov data cross-referenced to the resolved company, which is a further
  M2/M3 entity-resolution step (§3.3's alias matching, see
  `alias_matching.py`) not built yet either.
- **Roster-disappearance cross-reference against `deal`** (spec §2.0.1) —
  `p22_deal` isn't populated until M6.

`build_universe_history` (added 2026-08-30) is the point-in-time
re-computation spec §2.0.3 requires ("applied per `as_of`, not once"): it
walks every landed DERA quarter and computes eligibility for that quarter's
own `as_of` from a cumulative union of everything filed up to and including
it, not from today's roster. It is a pure function over already-assembled
`{quarter: rows}` data (see `ingest/universe_snapshot.all_landed_quarters`
for the raw-zone read) — it does **not** persist per-quarter snapshots
anywhere. That's deliberately left to M6: no consumer of a persisted
per-quarter universe exists yet (the backtest harness that would read it
isn't built), and the spec doesn't define a storage shape for it beyond "the
per-quarter set is the eligible universe for that `as_of`" — inventing a
`p22_company_history` schema now, before M6 defines what it needs to answer,
risks building the wrong shape. `eligible_exchange` in each historical
quarter's candidates still comes from the **current** ticker/exchange
snapshot (the same limitation noted above for the live roster) — this
function does not solve that, it only makes the reporting-status and
SPAC-heuristic filters genuinely point-in-time.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import EDGAR_USER_AGENT
from src.ml.pipeline.p22_biotech_ma.ingest.http_retry import get_with_retry
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

_REPORTING_FORMS = frozenset({"10-K", "10-Q"})
_ELIGIBLE_EXCHANGES = frozenset({"NYSE", "NYSE American", "Nasdaq"})
_DEFAULT_REPORTING_LOOKBACK_DAYS = 183  # "trailing 6 months" (spec §2.0.3)

# Spec §3.3's exact normalization token list.
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|corp|ltd|plc|holdings|therapeutics|pharmaceuticals|biosciences)\b\.?",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[.,]")
_WHITESPACE_RE = re.compile(r"\s+")

# Name-based SPAC heuristic (spec §2.0.3: "SPACs carry biotech SIC codes and
# pollute the universe"). Not authoritative — see module docstring.
_SPAC_RE = re.compile(
    r"\b(acquisition\s+(corp|corporation|company|holdings)|special\s+purpose\s+acquisition|blank\s+check)\b",
    re.IGNORECASE,
)


def normalize_company_name(name: str) -> str:
    """
    Deterministic name normalization (spec §3.3): lowercase, strip the
    listed legal/business suffixes, collapse whitespace. Used both for the
    deterministic-match step in `alias_matching.py` and, indirectly, as a
    stable dedup key.
    """
    normalized = _PUNCT_RE.sub("", name.lower())
    normalized = _LEGAL_SUFFIX_RE.sub("", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def is_likely_spac(name: str) -> bool:
    """Name-based SPAC heuristic — a candidate generator for the review queue, not a classifier."""
    return bool(_SPAC_RE.search(name))


def normalize_cik(raw: str) -> str:
    """Zero-pad to EDGAR's canonical 10-digit CIK string, regardless of source formatting."""
    return raw.strip().lstrip("0").zfill(10) if raw.strip() else raw


def _parse_dera_filed(raw: str) -> Optional[date]:
    """DERA's `filed` column is `YYYYMMDD`; malformed/blank values return None rather than raising."""
    if not raw or len(raw) != 8 or not raw.isdigit():
        return None
    try:
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return None


@dataclass
class UniverseCandidate:
    """One CIK's resolved-so-far universe row. See module docstring for what's NOT computed yet."""

    cik: str
    name: str
    sic: Optional[str]
    ticker: Optional[str]
    exchange: Optional[str]
    most_recent_form: Optional[str]
    most_recent_filed: Optional[date]
    eligible_reporting: bool
    eligible_exchange: Optional[bool]  # None = unresolved (current-snapshot map has no entry), not "ineligible"
    flagged_spac: bool
    size_floor_met: Optional[bool] = None  # not computed at M2 — spec §2.0.6, blocked on vendor decision
    asset_floor_met: Optional[bool] = None  # not computed at M2 — needs p22_trial linkage


def fetch_ticker_exchange_map(client: Optional[httpx.Client] = None) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    CIK -> (ticker, exchange) from SEC's current-snapshot mapping (spec
    §2.0.2). **Current names only** — see module docstring for the delisted-
    name limitation.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = get_with_retry(
            client,
            COMPANY_TICKERS_EXCHANGE_URL,
            headers={"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        )
        if resp is None or resp.status_code != 200:
            _logger.error("Failed to fetch company_tickers_exchange.json")
            return {}
        payload = resp.json()
        mapping: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for row in payload.get("data", []):
            if len(row) < 4:
                continue
            cik_raw, ticker, exchange = row[0], row[2], row[3]
            mapping[normalize_cik(str(cik_raw))] = (ticker, exchange)
        _logger.info("Loaded %d CIK->ticker/exchange mappings", len(mapping))
        return mapping
    finally:
        if owns_client:
            client.close()


def build_universe(
    dera_rows: List[Dict[str, Any]],
    ticker_exchange_map: Mapping[str, Tuple[Optional[str], Optional[str]]],
    as_of: date,
    reporting_lookback_days: int = _DEFAULT_REPORTING_LOOKBACK_DAYS,
) -> List[UniverseCandidate]:
    """
    Build one `UniverseCandidate` per distinct CIK from landed DERA `sub.txt`
    rows, applying the eligibility filters this slice can compute (see module
    docstring for the ones it can't yet).
    """
    first_row_by_cik: Dict[str, Dict[str, Any]] = {}
    latest_filing_by_cik: Dict[str, Tuple[date, Optional[str]]] = {}

    for row in dera_rows:
        raw_cik = row.get("cik")
        if not raw_cik:
            continue
        cik = normalize_cik(str(raw_cik))
        first_row_by_cik.setdefault(cik, row)

        if row.get("form") in _REPORTING_FORMS:
            filed = _parse_dera_filed(row.get("filed", ""))
            if filed is None:
                continue
            current = latest_filing_by_cik.get(cik)
            if current is None or filed > current[0]:
                latest_filing_by_cik[cik] = (filed, row.get("form"))

    candidates: List[UniverseCandidate] = []
    for cik, row in first_row_by_cik.items():
        name = row.get("name", "")
        filed, form = latest_filing_by_cik.get(cik, (None, None))
        eligible_reporting = filed is not None and (as_of - filed).days <= reporting_lookback_days

        ticker, exchange = ticker_exchange_map.get(cik, (None, None))
        eligible_exchange = (exchange in _ELIGIBLE_EXCHANGES) if exchange else None

        candidates.append(
            UniverseCandidate(
                cik=cik,
                name=name,
                sic=row.get("sic"),
                ticker=ticker,
                exchange=exchange,
                most_recent_form=form,
                most_recent_filed=filed,
                eligible_reporting=eligible_reporting,
                eligible_exchange=eligible_exchange,
                flagged_spac=is_likely_spac(name),
            )
        )

    _logger.info("Built %d universe candidates from %d DERA rows", len(candidates), len(dera_rows))
    return candidates


_QUARTER_RE = re.compile(r"^(\d{4})q([1-4])$")
_QUARTER_END_MONTH_DAY = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def _quarter_end_date(quarter: str) -> Optional[date]:
    """"2019q3" -> 2019-09-30. Returns None for a malformed quarter string."""
    match = _QUARTER_RE.match(quarter)
    if not match:
        return None
    year, q = int(match.group(1)), int(match.group(2))
    month, day = _QUARTER_END_MONTH_DAY[q]
    return date(year, month, day)


def build_universe_history(
    quarters_rows: Mapping[str, List[Dict[str, Any]]],
    ticker_exchange_map: Mapping[str, Tuple[Optional[str], Optional[str]]],
    reporting_lookback_days: int = _DEFAULT_REPORTING_LOOKBACK_DAYS,
) -> Dict[str, List[UniverseCandidate]]:
    """
    Point-in-time eligibility, re-computed per quarter (spec §2.0.3). See the
    module docstring for what this does and does not do.

    Args:
        quarters_rows: `{quarter: [DERA sub.txt row dicts]}`, e.g. from
            `universe_snapshot.all_landed_quarters()`. Malformed quarter keys
            (not `YYYYqN`) are skipped with a warning rather than raising,
            since a single bad key shouldn't abort the whole history build.
        ticker_exchange_map: Same current-snapshot map `build_universe` takes
            — see the module docstring's caveat about its use here.
        reporting_lookback_days: Passed through to `build_universe` per quarter.

    Returns:
        `{quarter: [UniverseCandidate, ...]}`, one entry per valid quarter key
        present in `quarters_rows`, each computed against a cumulative union
        of every quarter up to and including it (chronological order).
    """
    valid_quarters = sorted((q for q in quarters_rows if _quarter_end_date(q) is not None))
    skipped = set(quarters_rows) - set(valid_quarters)
    if skipped:
        _logger.warning("Skipping malformed DERA quarter keys: %s", sorted(skipped))

    history: Dict[str, List[UniverseCandidate]] = {}
    cumulative_rows: List[Dict[str, Any]] = []
    for quarter in valid_quarters:
        cumulative_rows = cumulative_rows + quarters_rows[quarter]
        as_of = _quarter_end_date(quarter)
        assert as_of is not None  # guaranteed by valid_quarters filter above
        history[quarter] = build_universe(
            cumulative_rows, ticker_exchange_map, as_of=as_of, reporting_lookback_days=reporting_lookback_days
        )

    _logger.info("Built point-in-time universe history for %d quarters", len(history))
    return history


def write_universe(candidates: List[UniverseCandidate], repo: Any) -> Dict[str, int]:
    """
    Persist candidates via `P22Repo`. SPAC-flagged names are NOT written to
    `p22_company` — they go to the review queue (`item_type='entity_match'`)
    pending human confirmation, per spec §2.0.3's filer-type exclusion, since
    the SPAC heuristic is name-based and not authoritative.

    Args:
        candidates: From `build_universe`.
        repo: A `P22Repo`-shaped object (duck-typed for test doubles).

    Returns:
        Counters: `companies_written`, `spac_flagged_for_review`, `total_candidates`.
    """
    written = 0
    flagged = 0
    for c in candidates:
        if c.flagged_spac:
            # Payload carries everything a confirmation (`ingest/review_queue.py`) needs to call
            # upsert_company() without re-deriving it — spec §3.4 payload is "a candidate record
            # awaiting confirmation," not just enough to display it.
            repo.add_review_item(
                item_type="entity_match",
                payload={
                    "reason": "spac_name_heuristic",
                    "cik": c.cik,
                    "name": c.name,
                    "sic": c.sic,
                    "ticker": c.ticker,
                    "exchange": c.exchange,
                    "eligible_reporting": c.eligible_reporting,
                },
                evidence_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={c.cik}",
                priority=1,
            )
            flagged += 1
            continue

        repo.upsert_company(
            cik=c.cik,
            name=c.name,
            ticker=c.ticker,
            exchange=c.exchange,
            sic_code=c.sic,
            is_active=c.eligible_reporting,
            role="target",
        )
        written += 1

    stats = {"companies_written": written, "spac_flagged_for_review": flagged, "total_candidates": len(candidates)}
    _logger.info("Wrote universe: %s", stats)
    return stats
