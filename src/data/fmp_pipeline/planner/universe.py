"""
Universe assembly (spec §7.2) and download tiers (spec §6.3.3).

The universe is the union of ``stock-list``, ``delisted-companies``, ``etf-list`` and both sides of
``symbol-change``, plus the explicit tier-A list (index ETFs and indices, spec §5.3), filtered by exchange and type.
Index constituents (current and historical) mark tier-B membership and are candidates too.

What each source tells us is uneven, and the field names of the lists the free key could not fetch are
``[VERIFY]``, so parsing accepts the common FMP spellings and the report shows what was seen:

- ``delisted-companies`` gives exchange, IPO and delisting dates (probe F13). Future delisting dates are scheduled
  delistings and are ignored until they pass.
- ``stock-list`` may carry only symbol and name. A symbol whose exchange is unknown stays an **unresolved**
  candidate: only its ``profile`` is planned, and the next ``fmp plan`` after the profiles are fetched decides
  (D36). Profiles override list attributes.

Every exclusion is counted per rule for ``universe_report.md``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from src.data.fmp_pipeline.planner.config import UniverseConfig

SYMBOL_KEYS = ("symbol", "removedTicker", "addedTicker")
CONSTITUENT_ENDPOINTS = ("historical-sp500-constituent", "historical-nasdaq-constituent",
                         "historical-dowjones-constituent", "sp500-constituent", "nasdaq-constituent",
                         "dowjones-constituent")
#: Exclusion rules in the order they are applied.
RULES = ("otc", "exchange", "fund", "no_profile")
TIERS = ("A", "B", "C", "D", "E")


def normalize_exchange(value: Any) -> Optional[str]:
    """Return an exchange code upper-cased with spaces, ``-``, ``_`` and ``.`` removed; ``None`` if blank."""
    if not isinstance(value, str) or not value.strip():
        return None
    return "".join(ch for ch in value.upper() if ch not in " -_.")


def _date(value: Any) -> Optional[date]:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


@dataclass
class Security:
    """One candidate symbol and everything the sources said about it."""

    symbol: str
    sources: set[str] = field(default_factory=set)
    exchange: Optional[str] = None
    instrument: Optional[str] = None
    ipo_date: Optional[date] = None
    delisted_date: Optional[date] = None
    is_etf: bool = False
    is_fund: bool = False
    is_adr: bool = False
    actively_trading: Optional[bool] = None
    index_member: bool = False
    explicit: bool = False
    profile: str = "pending"
    tier: str = "E"
    resolved: bool = True

    @property
    def is_index(self) -> bool:
        """Indices keep FMP's ``^`` prefix (probe F12)."""
        return self.symbol.startswith("^")

    @property
    def is_stock(self) -> bool:
        """Common stocks, ADRs and other non-fund instruments (statements, earnings)."""
        return not (self.is_etf or self.is_fund or self.is_index)

    def is_delisted(self, as_of: date) -> bool:
        """Delisted: not actively trading, and a past delisting date or absent from ``actively-trading-list``."""
        if self.actively_trading:
            return False
        if self.delisted_date is not None and self.delisted_date <= as_of:
            return True
        return self.actively_trading is False


@dataclass
class Universe:
    """The planned universe plus everything ``universe_report.md`` shows."""

    securities: dict[str, Security]
    excluded: Counter[str]
    excluded_examples: dict[str, list[str]]
    exchanges: Counter[str]
    source_rows: dict[str, int]
    source_status: dict[str, str]
    statement_symbols: Optional[set[str]]
    as_of: date

    @property
    def resolved(self) -> list[Security]:
        """Included securities whose inclusion is decided, sorted by symbol."""
        return sorted((s for s in self.securities.values() if s.resolved), key=lambda s: s.symbol)

    @property
    def unresolved(self) -> list[Security]:
        """Candidates waiting for their profile, sorted by symbol."""
        return sorted((s for s in self.securities.values() if not s.resolved), key=lambda s: s.symbol)

    def tier_counts(self) -> Counter[str]:
        """Return resolved securities per tier."""
        return Counter(s.tier for s in self.resolved)

    def sample(self, fixed: Iterable[str], size: int) -> list[str]:
        """
        Return the reconciliation sample (spec D8): the fixed symbols present, then the stocks with the lowest
        ``sha256(symbol)`` until ``size`` — deterministic and spread over active and delisted names.
        """
        stocks = [s.symbol for s in self.resolved if s.is_stock]
        chosen = [sym for sym in fixed if sym in self.securities and self.securities[sym].resolved]
        ranked = sorted((sym for sym in stocks if sym not in chosen),
                        key=lambda sym: hashlib.sha256(sym.encode("utf-8")).hexdigest())
        return chosen + ranked[: max(0, size - len(chosen))]


def _rows(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, dict):
        body = [body]
    return [r for r in body if isinstance(r, dict)] if isinstance(body, list) else []


class _Builder:
    def __init__(self, config: UniverseConfig, as_of: date) -> None:
        self.config = config
        self.as_of = as_of
        self.candidates: dict[str, Security] = {}

    def get(self, symbol: Any, source: str) -> Optional[Security]:
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        symbol = symbol.strip()
        sec = self.candidates.setdefault(symbol, Security(symbol))
        sec.sources.add(source)
        return sec

    def add_listing(self, row: dict[str, Any], source: str) -> Optional[Security]:
        sec = self.get(row.get("symbol"), source)
        if sec is None:
            return None
        sec.exchange = sec.exchange or normalize_exchange(row.get("exchangeShortName") or row.get("exchange"))
        kind = row.get("type")
        if isinstance(kind, str) and kind:
            sec.instrument = kind.lower()
            sec.is_etf = sec.is_etf or sec.instrument == "etf"
            sec.is_fund = sec.is_fund or sec.instrument == "fund"
        sec.ipo_date = sec.ipo_date or _date(row.get("ipoDate"))
        delisted = _date(row.get("delistedDate"))
        if delisted is not None and delisted <= self.as_of:
            sec.delisted_date = max(sec.delisted_date or delisted, delisted)
        return sec

    def apply_profile(self, sec: Security, profile: Optional[dict[str, Any]]) -> None:
        if profile is None:
            sec.profile = "missing"
            return
        sec.profile = "ok"
        sec.exchange = normalize_exchange(profile.get("exchange") or profile.get("exchangeShortName")) or sec.exchange
        sec.is_etf = bool(profile.get("isEtf", sec.is_etf))
        sec.is_fund = bool(profile.get("isFund", sec.is_fund))
        sec.is_adr = bool(profile.get("isAdr", sec.is_adr))
        if isinstance(profile.get("isActivelyTrading"), bool):
            sec.actively_trading = profile["isActivelyTrading"]
        sec.ipo_date = _date(profile.get("ipoDate")) or sec.ipo_date


def assemble_universe(
    sources: Mapping[str, Any],
    profiles: Mapping[str, Optional[dict[str, Any]]],
    config: UniverseConfig,
    as_of: date,
) -> Universe:
    """
    Build the universe from fetched source lists and profiles.

    Args:
        sources: ``{endpoint: rows}`` for each bootstrap list (all pages concatenated). ``None`` means the list
            could not be fetched (plan-restricted, failed, or not fetched yet).
        profiles: ``{symbol: profile row}`` for finished ``profile`` tasks; ``None`` for a finished task that
            returned nothing usable (empty, restricted). Symbols without a finished profile task are absent.
        config: Universe rules.
        as_of: The dump date: delisting dates after it are scheduled, not past.

    Returns:
        The universe.
    """
    b = _Builder(config, as_of)
    for endpoint in ("delisted-companies", "stock-list", "etf-list"):
        for row in _rows(sources.get(endpoint)):
            sec = b.add_listing(row, endpoint)
            if sec is not None and endpoint == "etf-list":
                sec.is_etf = True
    for row in _rows(sources.get("symbol-change")):
        for key in ("oldSymbol", "newSymbol"):
            b.get(row.get(key), "symbol-change")
    for endpoint in CONSTITUENT_ENDPOINTS:
        for row in _rows(sources.get(endpoint)):
            for key in SYMBOL_KEYS:
                sec = b.get(row.get(key), endpoint)
                if sec is not None:
                    sec.index_member = True
    if sources.get("actively-trading-list") is not None:
        active = {r.get("symbol") for r in _rows(sources["actively-trading-list"])}
        for sec in b.candidates.values():
            sec.actively_trading = sec.symbol in active
    for symbol in dict.fromkeys(config.tier_a + config.indices):
        sec = b.get(symbol, "config")
        assert sec is not None
        sec.explicit = True
    for symbol, profile in profiles.items():
        if symbol in b.candidates:
            b.apply_profile(b.candidates[symbol], profile)

    universe = Universe({}, Counter(), {}, Counter(), {}, {}, None, as_of)
    for endpoint, rows in sources.items():
        universe.source_rows[endpoint] = len(_rows(rows))
    listed = sources.get("financial-statement-symbol-list")
    universe.statement_symbols = None if listed is None else {
        r["symbol"] for r in _rows(listed) if isinstance(r.get("symbol"), str)}
    include = {normalize_exchange(x) for x in config.include_exchanges}
    otc = {normalize_exchange(x) for x in config.otc_exchanges}
    for sec in sorted(b.candidates.values(), key=lambda s: s.symbol):
        if sec.exchange is not None:
            universe.exchanges[sec.exchange] += 1
        rule = None if sec.explicit else _exclusion(sec, include, otc, config)
        if rule is not None:
            universe.excluded[rule] += 1
            universe.excluded_examples.setdefault(rule, [])
            if len(universe.excluded_examples[rule]) < 10:
                universe.excluded_examples[rule].append(sec.symbol)
            continue
        sec.resolved = sec.explicit or sec.exchange is not None or sec.profile != "pending"
        sec.tier = _tier(sec, config, as_of)
        universe.securities[sec.symbol] = sec
    return universe


def _exclusion(sec: Security, include: set[Optional[str]], otc: set[Optional[str]],
               config: UniverseConfig) -> Optional[str]:
    if sec.exchange is not None and sec.exchange in otc:
        return None if config.include_otc else "otc"
    if sec.exchange is not None and sec.exchange not in include:
        return "exchange"
    if sec.is_fund and not sec.is_etf:
        return "fund"
    if sec.exchange is None and sec.profile == "missing" and not config.keep_unresolved:
        return "no_profile"
    return None


def _tier(sec: Security, config: UniverseConfig, as_of: date) -> str:
    if sec.symbol in config.tier_a or sec.is_index:
        return "A"
    if sec.index_member:
        return "B"
    if sec.is_delisted(as_of):
        return "D"
    if sec.is_etf or (sec.instrument is None and sec.profile != "ok"):
        return "E"
    return "C"
