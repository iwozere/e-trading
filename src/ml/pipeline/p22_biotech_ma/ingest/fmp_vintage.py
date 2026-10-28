"""
P22 — historical raw prices and corporate actions from the FMP vintage (spec §2.0.6/§2.0.7, M3, 2026-10-27).

Replaces `ingest/fmp_backfill.py`'s per-ticker API download for historical prices: the shared FMP data
pipeline (`src/data/fmp_pipeline/`) already downloaded the whole US market, delisted securities included,
into a read-only vintage (`DATA_CACHE_DIR/fmp-data/vintages/<id>/`). This module maps `p22_company` rows to
vintage securities and turns their raw daily bars, splits and dividends into `p22_price_daily` /
`p22_corporate_action` rows (vendor/source `'fmp'`).

**Raw-on-raw, verified against the vintage** (resolves `docs/Tasks.md` item 1's open raw-vs-adjusted
question): `prices/1d_raw` is as traded — NVDA closes 1208.90 on 2024-06-07 and 121.79 on 2024-06-10, the
day of its 10-for-1 split — and `events/splits.ratio` is numerator/denominator, the same convention as
`p22_corporate_action.ratio` (4.0 for a 4:1 forward split, 0.05 for a 1:20 reverse split).

**Company -> security matching.** CIK first. A CIK often owns several vintage securities — preferreds,
warrants, units, other share classes, and FMP keeps a ticker change as two securities (CECE until 2022,
CECO after) — so only plain common listings count (`is_common_listing`), and they are stitched: the
listing with the latest last price is primary, an older listing only fills dates before the primary's
first bar, never gaps inside it. A company with no CIK match falls back to an exact normalized-name match,
and only against listings without a conflicting CIK. Anything else stays unmatched — no fuzzy guessing.

**Coverage caveat, measured 2026-10-27:** FMP's delisted list is essentially 2021+. Of the 1,705 DERA
biotech CIKs, ~91% of still-filing companies are covered, ~72% of those that stopped filing in 2021 or
later, but only ~9% of those that stopped before 2021 (Kite, Juno, Loxo, MyoKardia, Immunomedics, Spark
are absent). See `docs/Tasks.md`.

**`known_from`:** spec §2.4 treats vendor prices as known at period end + `VENDOR_PRICE_LAG_DAYS` (0) —
stamping the load date instead would hide every backfilled bar from any backtest `as_of` before today.
Bars get the trade date at `_PRICE_KNOWN_UTC_TIME` (after the US close); corporate actions get their
ex-date at midnight UTC — later than the real announcement, so conservative, never a lookahead.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fmp_pipeline.builders.flags import OHLC_VIOLATION, SPIKE_REVERTED
from src.data.fmp_pipeline.builders.validate import SECURITIES
from src.ml.pipeline.p22_biotech_ma.config import VENDOR_PRICE_LAG_DAYS
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import normalize_cik
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

VENDOR = "fmp"
# 21:00 UTC is after the 16:00 New York close in both EST and EDT.
_PRICE_KNOWN_UTC_TIME = time(21, 0)
# Bars the FMP builder flagged as bad prints (high < low etc., or a one-day spike that reverted) are not
# written: p22_price_daily is never rewritten, so a bad print would stay forever. STALE / ZERO_VOLUME /
# JUMP_NO_SPLIT bars are real trading days and are kept.
_SKIP_FLAGS = OHLC_VIOLATION | SPIKE_REVERTED
_COMMON_INSTRUMENT_TYPES = frozenset({"stock", "adr"})
_PLAIN_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")
# Nasdaq's fifth-letter suffixes for warrants, rights and units (ABCDW / ABCDR / ABCDU).
_DERIVATIVE_SUFFIX_RE = re.compile(r"^[A-Z]{4}[WRU]$")
# Legal-form words only. entity_resolution.normalize_company_name also strips "therapeutics" /
# "pharmaceuticals" / "biosciences", which is right for alias matching but would let "Acme Therapeutics"
# take "Acme Pharmaceuticals"' price history here.
_LEGAL_FORM_RE = re.compile(r"\b(the|inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|sa|nv|ag)\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class VintageSecurity:
    """The vintage `reference/securities` fields matching needs."""

    security_id: int
    symbol: str
    name: Optional[str]
    cik: Optional[str]
    instrument_type: Optional[str]
    first_price_date: Optional[date]
    last_price_date: Optional[date]
    identity_uncertain: bool = False


@dataclass(frozen=True)
class CompanyMatch:
    """The vintage securities chosen for one company, primary first."""

    company_id: int
    security_ids: Tuple[int, ...]
    method: str  # 'cik' | 'name'


def load_vintage_securities(vintage_dir: Path) -> List[VintageSecurity]:
    """
    Read the vintage's security reference table.

    Args:
        vintage_dir: `<data_root>/vintages/<id>` (`FmpStore.dir`).

    Returns:
        One `VintageSecurity` per row.
    """
    columns = ["security_id", "symbol", "name", "cik", "instrument_type", "first_price_date", "last_price_date",
               "identity_uncertain"]
    out: List[VintageSecurity] = []
    for row in pq.read_table(vintage_dir / SECURITIES, columns=columns).to_pylist():
        cik = row["cik"]
        out.append(VintageSecurity(
            security_id=int(row["security_id"]),
            symbol=row["symbol"],
            name=row["name"],
            cik=normalize_cik(str(cik)) if cik else None,
            instrument_type=row["instrument_type"],
            first_price_date=row["first_price_date"],
            last_price_date=row["last_price_date"],
            identity_uncertain=bool(row["identity_uncertain"]),
        ))
    return out


def name_key(name: str) -> str:
    """Exact-match key for the name fallback: lowercase, punctuation and legal-form words removed."""
    words = _NON_ALNUM_RE.sub(" ", name.lower())
    return " ".join(_LEGAL_FORM_RE.sub(" ", words).split())


def is_common_listing(security: VintageSecurity) -> bool:
    """
    True for a plain common-stock (or ADR) listing with price history.

    Excludes preferreds (`AIG-PA`), warrants (`AP-WT`, `ACHRW`), units, rights, other share-class symbols,
    securities whose identity the FMP builder could not settle, and listings that never traded.
    """
    if security.identity_uncertain or security.first_price_date is None:
        return False
    if security.instrument_type not in _COMMON_INSTRUMENT_TYPES:
        return False
    symbol = security.symbol or ""
    return bool(_PLAIN_SYMBOL_RE.match(symbol)) and not _DERIVATIVE_SUFFIX_RE.match(symbol)


def _order_listings(listings: Iterable[VintageSecurity]) -> Tuple[int, ...]:
    """Primary first: latest last price, then longest history, then lowest id (stable)."""
    def key(s: VintageSecurity) -> Tuple[date, int, int]:
        last = s.last_price_date or date.min
        span = (last - s.first_price_date).days if s.first_price_date else 0
        return (last, span, -s.security_id)

    return tuple(s.security_id for s in sorted(listings, key=key, reverse=True))


def match_companies(
    companies: Sequence[Dict[str, Any]], securities: Sequence[VintageSecurity]
) -> Tuple[Dict[int, CompanyMatch], List[int]]:
    """
    Map `p22_company` rows to vintage common listings.

    Args:
        companies: `P22Repo.list_companies_full()` rows (`company_id`, `cik`, `name`).
        securities: `load_vintage_securities()` output.

    Returns:
        `({company_id: CompanyMatch}, [unmatched company_id])`.
    """
    common = [s for s in securities if is_common_listing(s)]
    by_cik: Dict[str, List[VintageSecurity]] = {}
    by_name: Dict[str, List[VintageSecurity]] = {}
    for s in common:
        if s.cik:
            by_cik.setdefault(s.cik, []).append(s)
        if s.name:
            by_name.setdefault(name_key(s.name), []).append(s)

    matches: Dict[int, CompanyMatch] = {}
    unmatched: List[int] = []
    for company in companies:
        company_id = int(company["company_id"])
        cik = normalize_cik(company["cik"]) if company.get("cik") else None
        if cik and cik in by_cik:
            matches[company_id] = CompanyMatch(company_id, _order_listings(by_cik[cik]), "cik")
            continue
        key = name_key(company.get("name") or "")
        # A listing that carries a different CIK belongs to another filer, whatever its name says.
        candidates = [s for s in by_name.get(key, []) if s.cik is None or s.cik == cik] if key else []
        if candidates:
            matches[company_id] = CompanyMatch(company_id, _order_listings(candidates), "name")
        else:
            unmatched.append(company_id)
    return matches, unmatched


def stitch_daily_bars(frames: Sequence[Tuple[int, Optional[pd.DataFrame]]]) -> pd.DataFrame:
    """
    Combine a company's listings into one daily series.

    Args:
        frames: `(security_id, FmpStore.prices(...) frame or None)`, primary first.

    Returns:
        Bars on a `DatetimeIndex` with a `security_id` column. Each later listing contributes only the dates
        before everything already taken — the ticker-change case — so two listings never interleave.
    """
    parts: List[pd.DataFrame] = []
    earliest: Optional[pd.Timestamp] = None
    for security_id, frame in frames:
        if frame is None or frame.empty:
            continue
        part = frame if earliest is None else frame[frame.index < earliest]
        if part.empty:
            continue
        part = part.assign(security_id=security_id)
        parts.append(part)
        earliest = part.index.min()
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "flags", "security_id"])
    return pd.concat(parts).sort_index()


def price_known_from(trade_date: date) -> datetime:
    """`known_from` for a backfilled daily bar (spec §2.4: period end + `VENDOR_PRICE_LAG_DAYS`)."""
    return datetime.combine(trade_date + timedelta(days=VENDOR_PRICE_LAG_DAYS), _PRICE_KNOWN_UTC_TIME,
                            tzinfo=timezone.utc)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None or pd.isna(value) else float(value)


def build_price_rows(company_id: int, bars: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    """
    Turn stitched bars into `p22_price_daily` rows.

    Returns:
        `(rows, skipped)` — `skipped` counts bad-print bars (`_SKIP_FLAGS`) and bars without a close.
    """
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for ts, bar in zip(pd.DatetimeIndex(bars.index), bars.to_dict("records")):
        flags = int(bar.get("flags") or 0)
        if flags & _SKIP_FLAGS or pd.isna(bar["close"]):
            skipped += 1
            continue
        trade_date = ts.date()
        volume = bar.get("volume")
        rows.append({
            "company_id": company_id,
            "trade_date": trade_date,
            "vendor": VENDOR,
            "open_raw": _optional_float(bar.get("open")),
            "high_raw": _optional_float(bar.get("high")),
            "low_raw": _optional_float(bar.get("low")),
            "close_raw": float(bar["close"]),
            "volume_raw": None if volume is None or pd.isna(volume) else int(volume),
            "known_from": price_known_from(trade_date),
        })
    return rows, skipped


def _in_span(when: date, spans: Dict[int, Tuple[date, date]], security_id: int) -> bool:
    span = spans.get(security_id)
    return span is not None and span[0] <= when <= span[1]


def build_corporate_action_rows(
    company_id: int,
    events: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
    bars: pd.DataFrame,
    existing: Set[Tuple[date, str]],
) -> List[Dict[str, Any]]:
    """
    Turn the stitched listings' splits and dividends into `p22_corporate_action` rows.

    Args:
        company_id: The company.
        events: `FmpStore.events_many(...)` for the company's listings.
        bars: `stitch_daily_bars` output — an event counts only inside the date range its own listing
            contributed (plus the day after, so a split effective on the first bar of the next listing is
            kept), so a stitched-out listing's events do not leak in.
        existing: `(ex_date, action_type)` keys the company already has. They are left alone — a row may
            come from the daily yfinance ingest or be `is_verified` by SEC reconciliation, which an FMP
            re-load must not overwrite.

    Returns:
        New rows only.
    """
    spans: Dict[int, Tuple[date, date]] = {}
    listing_ids = bars["security_id"].tolist()
    for security_id in set(listing_ids):
        index = pd.DatetimeIndex(bars.index[[sid == security_id for sid in listing_ids]])
        spans[int(security_id)] = (index.min().date(), index.max().date() + timedelta(days=1))

    rows: Dict[Tuple[date, str], Dict[str, Any]] = {}
    for security_id, (splits, dividends) in events.items():
        for _, split in splits.iterrows():
            ex_date = pd.Timestamp(split["date"]).date()
            if not split["denominator"] or not _in_span(ex_date, spans, security_id):
                continue
            ratio = float(split["numerator"]) / float(split["denominator"])
            if ratio == 1.0:
                continue
            action_type = "split" if ratio > 1 else "reverse_split"
            rows.setdefault((ex_date, action_type), _action_row(company_id, ex_date, action_type, ratio=ratio))
        for _, dividend in dividends.iterrows():
            ex_date = pd.Timestamp(dividend["date"]).date()
            if not _in_span(ex_date, spans, security_id):
                continue
            rows.setdefault((ex_date, "dividend"),
                            _action_row(company_id, ex_date, "dividend", cash_amount=float(dividend["amount"])))
    return [row for key, row in sorted(rows.items()) if key not in existing]


def _action_row(company_id: int, ex_date: date, action_type: str, *, ratio: Optional[float] = None,
                cash_amount: Optional[float] = None) -> Dict[str, Any]:
    return {
        "company_id": company_id,
        "ex_date": ex_date,
        "action_type": action_type,
        "ratio": ratio,
        "cash_amount": cash_amount,
        "source": VENDOR,
        "is_verified": False,
        "known_from": datetime.combine(ex_date, time.min, tzinfo=timezone.utc),
    }


def load_company(store: Any, repo: Any, match: CompanyMatch, replace_prices: bool = False) -> Dict[str, int]:
    """
    Write one company's vintage history to `p22_price_daily` / `p22_corporate_action`.

    Args:
        store: An `FmpStore` (`prices`, `events_many`).
        repo: A `P22Repo`-shaped object (`insert_price_daily_bulk`, `get_corporate_action_keys`,
            `upsert_corporate_action`, and `delete_price_daily` for `replace_prices`).
        match: The company's listings from `match_companies`.
        replace_prices: Delete the company's existing FMP bars first. Existing keys are otherwise left alone, so
            a corrected vintage read (price-basis override) never reaches rows already loaded.

    Returns:
        `{"prices_deleted", "prices_written", "bars_skipped", "actions_written", "listings_used"}`.
    """
    bars = stitch_daily_bars([(sid, store.prices(sid, "1d")) for sid in match.security_ids])
    price_rows, skipped = build_price_rows(match.company_id, bars)
    prices_deleted = repo.delete_price_daily(match.company_id, VENDOR) if replace_prices else 0
    prices_written = repo.insert_price_daily_bulk(price_rows) if price_rows else 0

    action_rows: List[Dict[str, Any]] = []
    if not bars.empty:
        used = sorted({int(sid) for sid in bars["security_id"].unique()})
        events = store.events_many(used)
        action_rows = build_corporate_action_rows(match.company_id, events, bars,
                                                  repo.get_corporate_action_keys(match.company_id))
    for row in action_rows:
        repo.upsert_corporate_action(**row)

    return {
        "prices_deleted": prices_deleted,
        "prices_written": prices_written,
        "bars_skipped": skipped,
        "actions_written": len(action_rows),
        "listings_used": int(bars["security_id"].nunique()) if not bars.empty else 0,
    }
