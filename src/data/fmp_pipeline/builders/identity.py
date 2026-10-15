"""
Security identity (spec §7.3): stable ``security_id`` allocation, ticker-reuse segmentation, symbol history.

**Allocation.** Every security has an *identity key*: its symbol, or ``SYMBOL|<delisting date>`` /
``SYMBOL|active`` for one segment of a reused ticker. ``IdRegistry`` maps keys to integers in
``<data_root>/security_ids.json`` and is only ever appended to: a key keeps its id forever, an id is never given
to another key, and new keys are numbered in sorted order. A rebuild from the same archive therefore yields the
same ids, and a later dump keeps the ids of the symbols it shares with this one.

**Segmentation** (§7.3 rule 1) applies when the listing entries of one symbol (``delisted-companies`` rows plus
the active listing) name **different companies**. With ``build.segment_reused_tickers`` on and clean evidence
(delisting dates strictly increasing, every later entry's IPO on or after the previous delisting) each entry
becomes its own security, and the timeline is cut at the delisting dates so every bar belongs to exactly one
segment. Otherwise, and always while the setting is off (the default until the paid probe shows that FMP serves
merged series for reused tickers, spec §4.2 item 8), the symbol stays one security with
``identity_uncertain = true`` and the reason (rule 2: never guess).

**Symbol history.** ``symbol-change`` rows (``date``, ``oldSymbol``, ``newSymbol``; field names [VERIFY]) give
the renamed security one row per earlier symbol. All ranges are inclusive dates; ``None`` is open-ended.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from src.data.fmp_pipeline.builders.records import parse_date
from src.data.fmp_pipeline.fsutil import atomic_write_bytes
from src.data.fmp_pipeline.planner.universe import Security, Universe
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

ACTIVE = "active"
_NAME_STOPWORDS = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "the",
                   "holdings", "holding", "group", "sa", "nv", "llc", "lp", "ag", "se", "class", "common", "stock",
                   "shares", "ordinary", "new"}
_CLASS_SHARE_RE = re.compile(r"^[A-Z]+\.[A-Z]{1,2}$")


class IdRegistry:
    """Append-only ``identity key -> security_id`` map persisted as JSON."""

    def __init__(self, path: Path) -> None:
        """
        Args:
            path: ``<data_root>/security_ids.json``; created on the first ``save``.

        Raises:
            ValueError: If the file exists but is not a valid registry.
        """
        self.path = path
        self.ids: dict[str, int] = {}
        self.next_id = 1
        self.added: list[str] = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("ids"), dict):
                raise ValueError(f"{path} is not a security id registry")
            self.ids = {str(k): int(v) for k, v in data["ids"].items()}
            self.next_id = int(data.get("next_id", max(self.ids.values(), default=0) + 1))
            if self.ids and self.next_id <= max(self.ids.values()):
                raise ValueError(f"{path}: next_id {self.next_id} would reuse an allocated id")

    def allocate(self, keys: list[str]) -> dict[str, int]:
        """
        Return ids for ``keys``, allocating new ones in sorted key order.

        Args:
            keys: Identity keys.

        Returns:
            ``{key: security_id}`` for every key.
        """
        for key in sorted(set(keys) - set(self.ids)):
            self.ids[key] = self.next_id
            self.next_id += 1
            self.added.append(key)
        return {key: self.ids[key] for key in keys}

    def save(self) -> None:
        """Write the registry atomically (no-op when nothing was added)."""
        if not self.added and self.path.exists():
            return
        payload = {"version": 1, "next_id": self.next_id, "ids": dict(sorted(self.ids.items()))}
        atomic_write_bytes(self.path, json.dumps(payload, indent=1).encode("utf-8"))
        self.added = []


@dataclass
class SecurityRecord:
    """One row of ``securities.parquet`` (spec §7.3) plus the builder's bookkeeping."""

    key: str
    symbol: str
    security_id: int = 0
    yf_symbol: Optional[str] = None
    exchange: Optional[str] = None
    name: Optional[str] = None
    instrument_type: str = "stock"
    cik: Optional[str] = None
    isin: Optional[str] = None
    cusip: Optional[str] = None
    ipo_date: Optional[date] = None
    delisted_date: Optional[date] = None
    is_actively_trading: Optional[bool] = None
    first_price_date: Optional[date] = None
    last_price_date: Optional[date] = None
    source_flags: str = ""
    tier: str = "E"
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    identity_uncertain: bool = False
    uncertain_reason: Optional[str] = None
    rename_only: bool = False

    def covers(self, when: Optional[date]) -> bool:
        """True if ``when`` lies in the segment's range (always true for an unsegmented security)."""
        if when is None:
            return self.valid_from is None and self.valid_to is None
        return (self.valid_from is None or when >= self.valid_from) and (self.valid_to is None or when <= self.valid_to)

    def mark_uncertain(self, reason: str) -> None:
        """Set ``identity_uncertain`` and append ``reason``."""
        self.identity_uncertain = True
        self.uncertain_reason = reason if not self.uncertain_reason else f"{self.uncertain_reason}; {reason}"


@dataclass(frozen=True)
class HistoryRow:
    """One row of ``symbol_history.parquet``."""

    security_id: int
    symbol: str
    valid_from: Optional[date]
    valid_to: Optional[date]
    reason: str


@dataclass(frozen=True)
class _Entry:
    name: Optional[str]
    ipo: Optional[date]
    delisted: Optional[date]


def normalize_name(name: Optional[str]) -> str:
    """Company name reduced to its distinctive words (case, punctuation and legal suffixes removed)."""
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return " ".join(w for w in words if w not in _NAME_STOPWORDS)


def yf_symbol(symbol: str) -> str:
    """Derive the yfinance symbol: indices keep ``^``; class shares ``BRK.B`` become ``BRK-B`` (spec §9.6)."""
    return symbol.replace(".", "-") if _CLASS_SHARE_RE.match(symbol) else symbol


def instrument_type(sec: Security, name: Optional[str]) -> str:
    """
    Tag the instrument (spec §7.2): index, etf, fund, warrant, unit, right, preferred, adr or stock.

    Warrants, units, rights and preferreds are recognised from the company name (FMP has no type field for them),
    so the tag is a heuristic; the rows are kept either way.
    """
    if sec.is_index:
        return "index"
    if sec.is_etf:
        return "etf"
    if sec.is_fund:
        return "fund"
    lowered = f" {(name or '').lower()} "
    for tag, words in (("warrant", (" warrant", " wts ")), ("unit", (" unit ", " units ")),
                       ("right", (" right ", " rights ")), ("preferred", (" preferred", " pfd ", " depositary sh"))):
        if any(w in lowered for w in words):
            return tag
    return "adr" if sec.is_adr else "stock"


@dataclass
class IdentityResult:
    """Securities, their symbol history and the reports the builder writes."""

    securities: list[SecurityRecord]
    history: list[HistoryRow] = field(default_factory=list)
    by_symbol: dict[str, list[SecurityRecord]] = field(default_factory=dict)
    _history_index: Optional[dict[str, list[HistoryRow]]] = field(default=None, repr=False)

    def resolve(self, symbol: Optional[str], when: Optional[date] = None) -> Optional[int]:
        """
        Map a symbol (and optionally a date) to a ``security_id``.

        The security's own symbol wins; a historical symbol resolves through ``symbol_history`` when ``when``
        falls in its range. Returns ``None`` when unknown or ambiguous (several segments and no date).
        """
        if symbol is None:
            return None
        candidates = self.by_symbol.get(symbol)
        if candidates:
            if len(candidates) == 1:
                return candidates[0].security_id
            matches = [c for c in candidates if when is not None and c.covers(when)]
            return matches[0].security_id if len(matches) == 1 else None
        if when is None:
            return None
        owners = {h.security_id for h in self._history_by_symbol().get(symbol, [])
                  if (h.valid_from is None or when >= h.valid_from) and (h.valid_to is None or when <= h.valid_to)}
        return owners.pop() if len(owners) == 1 else None

    def _history_by_symbol(self) -> dict[str, list[HistoryRow]]:
        if self._history_index is None:
            index: dict[str, list[HistoryRow]] = defaultdict(list)
            for row in self.history:
                index[row.symbol].append(row)
            self._history_index = index
        return self._history_index

    def drop(self, security_ids: set[int]) -> None:
        """Remove securities (and their history rows), e.g. rename-only symbols that have no bars."""
        self.securities = [s for s in self.securities if s.security_id not in security_ids]
        self.history = [h for h in self.history if h.security_id not in security_ids]
        self.by_symbol = {}
        for record in self.securities:
            self.by_symbol.setdefault(record.symbol, []).append(record)
        self._history_index = None


def _rows(value: Any) -> list[dict[str, Any]]:
    return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []


def _listing_entries(sec: Security, delisted_rows: list[dict[str, Any]], profile: Optional[dict[str, Any]],
                     as_of: date) -> list[_Entry]:
    """Distinct listing entries of one symbol: past delistings (by date) and the active listing, if any."""
    entries: dict[tuple[str, Optional[date]], _Entry] = {}
    for row in delisted_rows:
        delisted = parse_date(row.get("delistedDate"))
        if delisted is None or delisted > as_of:
            continue  # scheduled delisting (probe F13)
        entry = _Entry(row.get("companyName"), parse_date(row.get("ipoDate")), delisted)
        entries.setdefault((normalize_name(entry.name), delisted), entry)
    if sec.actively_trading:
        name = (profile or {}).get("companyName")
        entries[(normalize_name(name), None)] = _Entry(name, parse_date((profile or {}).get("ipoDate")), None)
    return sorted(entries.values(), key=lambda e: (e.delisted is None, e.delisted or date.min))


def _segmentable(entries: list[_Entry]) -> Optional[str]:
    """Return why ``entries`` cannot be segmented cleanly, or ``None`` if they can."""
    for prev, cur in zip(entries, entries[1:]):
        assert prev.delisted is not None
        if cur.delisted is not None and cur.delisted <= prev.delisted:
            return "delisting dates not increasing"
        if cur.ipo is None:
            return f"no IPO date for the listing after {prev.delisted}"
        if cur.ipo < prev.delisted:
            return f"listings overlap ({cur.name!r} IPO {cur.ipo} before {prev.delisted})"
    return None


class IdentityBuilder:
    """Turns the universe and the reference lists into securities with stable ids."""

    def __init__(self, registry: IdRegistry, segment_reused_tickers: bool = False) -> None:
        """
        Args:
            registry: The persistent id registry.
            segment_reused_tickers: Split reused tickers into one security per listing (see module doc).
        """
        self.registry = registry
        self.segment = segment_reused_tickers

    def build(self, universe: Universe, sources: dict[str, Optional[list[Any]]],
              profiles: dict[str, Optional[dict[str, Any]]]) -> IdentityResult:
        """
        Build the securities of a vintage.

        Args:
            universe: The assembled universe (only resolved securities become securities).
            sources: Bootstrap lists as read from the archive.
            profiles: ``{symbol: profile row}`` for every archived profile.

        Returns:
            Securities (ids allocated, not yet saved), symbol history and a resolver.
        """
        delisted_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _rows(sources.get("delisted-companies")):
            if isinstance(row.get("symbol"), str):
                delisted_by_symbol[row["symbol"].strip()].append(row)
        records: list[SecurityRecord] = []
        for sec in universe.resolved:
            records += self._records(sec, delisted_by_symbol.get(sec.symbol, []), profiles.get(sec.symbol),
                                     universe.as_of)
        ids = self.registry.allocate([r.key for r in records])
        result = IdentityResult(records)
        for record in records:
            record.security_id = ids[record.key]
            result.by_symbol.setdefault(record.symbol, []).append(record)
        result.history = self._history(result, _rows(sources.get("symbol-change")))
        return result

    def _records(self, sec: Security, delisted_rows: list[dict[str, Any]], profile: Optional[dict[str, Any]],
                 as_of: date) -> list[SecurityRecord]:
        profile = profile or {}
        entries = _listing_entries(sec, delisted_rows, profile, as_of)
        base = SecurityRecord(
            key=sec.symbol, symbol=sec.symbol, yf_symbol=yf_symbol(sec.symbol), exchange=sec.exchange,
            name=profile.get("companyName") or next((e.name for e in reversed(entries) if e.name), None),
            cik=profile.get("cik") or None, isin=profile.get("isin") or None, cusip=profile.get("cusip") or None,
            ipo_date=sec.ipo_date, delisted_date=sec.delisted_date if sec.is_delisted(as_of) else None,
            is_actively_trading=sec.actively_trading, source_flags=",".join(sorted(sec.sources)), tier=sec.tier,
            rename_only=sec.sources == {"symbol-change"})
        base.instrument_type = instrument_type(sec, base.name)
        names = {normalize_name(e.name) for e in entries if e.name}
        if len(names) < 2:
            return [base]
        reason = _segmentable(entries)
        if not self.segment or reason is not None:
            base.mark_uncertain("ticker reuse suspected: " + "; ".join(
                f"{e.name} ({e.ipo} - {e.delisted or 'active'})" for e in entries)
                + ("" if self.segment else " (segmentation off until the paid probe, §4.2 item 8)")
                + (f"; not segmented: {reason}" if reason and self.segment else ""))
            return [base]
        segments = []
        start: Optional[date] = None
        for entry in entries:
            is_active = entry.delisted is None
            segment = SecurityRecord(
                key=f"{sec.symbol}|{ACTIVE if is_active else entry.delisted}", symbol=sec.symbol,
                yf_symbol=base.yf_symbol, exchange=base.exchange if is_active else None, name=entry.name,
                instrument_type=base.instrument_type, cik=base.cik if is_active else None,
                isin=base.isin if is_active else None, cusip=base.cusip if is_active else None, ipo_date=entry.ipo,
                delisted_date=entry.delisted, is_actively_trading=is_active, source_flags=base.source_flags,
                tier=base.tier, valid_from=start, valid_to=entry.delisted)
            segments.append(segment)
            start = entry.delisted + timedelta(days=1) if entry.delisted else None
        _logger.info("Segmented reused ticker %s into %d securities", sec.symbol, len(segments))
        return segments

    def _history(self, result: IdentityResult, changes: list[dict[str, Any]]) -> list[HistoryRow]:
        renames: dict[str, list[tuple[date, str]]] = defaultdict(list)
        for row in changes:
            when, old, new = parse_date(row.get("date")), row.get("oldSymbol"), row.get("newSymbol")
            if when is not None and isinstance(old, str) and isinstance(new, str) and old != new:
                renames[new.strip()].append((when, old.strip()))
        history: list[HistoryRow] = []
        for record in result.securities:
            into = [d for d, _ in renames.get(record.symbol, []) if record.covers(d)]
            history.append(HistoryRow(record.security_id, record.symbol, max(into, default=record.valid_from),
                                      record.valid_to, "current"))
            self._walk(record, record.symbol, None, renames, result, history, {record.symbol})
        return sorted(history, key=lambda h: (h.security_id, h.valid_from or date.min, h.symbol))

    def _walk(self, record: SecurityRecord, symbol: str, before: Optional[date],
              renames: dict[str, list[tuple[date, str]]], result: IdentityResult, history: list[HistoryRow],
              seen: set[str]) -> None:
        """Add a history row for every earlier symbol of ``symbol`` (renamed into it before ``before``)."""
        for when, old in sorted(renames.get(symbol, []), reverse=True):
            if (before is not None and when >= before) or old in seen or not record.covers(when):
                continue
            seen.add(old)
            earlier = [d for d, _ in renames.get(old, []) if d < when]
            history.append(HistoryRow(record.security_id, old, max(earlier, default=None), when - timedelta(days=1),
                                      "symbol-change"))
            for other in result.by_symbol.get(old, []):
                other.mark_uncertain(f"renamed to {record.symbol} on {when}; its prices may duplicate "
                                     f"security {record.security_id}")
            self._walk(record, old, when, renames, result, history, seen)
