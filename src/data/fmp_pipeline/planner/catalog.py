"""
The endpoint catalogue (spec §5) as data: what to request, for which symbols, in which block, at what expected size.

Blocks follow the strict order of spec §6.3.5. A task's manifest priority is ``block * 100 + sub``: ``sub`` 0 is
the bootstrap lists ``fmp plan`` fetches itself, then delisted symbols before active ones, and for intraday the tier
order (A … E) with delisted before active inside each tier.

Byte rates per row are **measured** where the free-key probe saw the endpoint (``docs/fmp_probe_report.md``, F10)
and **assumed** otherwise; ``rows`` is always an assumption for per-symbol endpoints (the probe saw one symbol,
AAPL, which is not typical). The plan report says which numbers are assumed. Re-planning after a run replaces the
estimate with the real bytes of every finished task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

BLOCKS = {
    0: "reference",
    1: "events",
    2: "index membership, calendars, macro",
    3: "daily raw",
    4: "reconciliation sample",
    5: "fundamentals",
    6: "intraday",
}

#: How tasks are generated for an endpoint.
#: list: one request, no params. paged: page 0 … (follow-ups until an empty page). symbol: one per symbol.
#: statement: one per symbol and period. daily: per symbol, ``from``/``to`` year windows. calendar: global
#: ``from``/``to`` windows. exchange: one per exchange. indicator: one per indicator name. intraday: per symbol,
#: windows sized from the row cap.
KINDS = ("list", "paged", "symbol", "statement", "daily", "calendar", "exchange", "indicator", "intraday")

#: Which symbols a per-symbol endpoint covers. universe: every included security (and the explicit indices).
#: stocks: non-ETF, non-fund securities. statements: ``financial-statement-symbol-list`` ∩ stocks when that list was
#: fetched, else stocks. sample: the reconciliation sample. etfs: ETFs only.
SCOPES = ("universe", "stocks", "statements", "sample", "etfs")


@dataclass(frozen=True)
class EndpointSpec:
    """One catalogue entry."""

    endpoint: str
    block: int
    priority: str
    kind: str
    scope: Optional[str] = None
    wire_per_row: float = 50.0
    decoded_per_row: float = 300.0
    rows: float = 1.0
    measured: bool = False
    bootstrap: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r} for {self.endpoint}")
        if self.scope is not None and self.scope not in SCOPES:
            raise ValueError(f"unknown scope {self.scope!r} for {self.endpoint}")


def _e(endpoint: str, block: int, priority: str, kind: str, scope: Optional[str] = None,
       rate: Optional[tuple[float, float]] = None, rows: float = 1.0, bootstrap: bool = False,
       measured: bool = True) -> EndpointSpec:
    """
    Shorthand for an entry.

    ``rate`` is ``(wire, decoded)`` bytes per row, probe-measured unless ``measured=False``; ``None`` keeps the
    generic assumption.
    """
    if rate is None:
        return EndpointSpec(endpoint, block, priority, kind, scope, rows=rows, bootstrap=bootstrap)
    return EndpointSpec(endpoint, block, priority, kind, scope, rate[0], rate[1], rows, measured, bootstrap)


_STATEMENT_RATES = {
    "income-statement": (265, 1406), "balance-sheet-statement": (380, 2181), "cash-flow-statement": (331, 1818),
    "key-metrics": (580, 1982), "ratios": (868, 3064), "enterprise-values": (80, 280),
    "income-statement-as-reported": (458, 1827),
}

CATALOG: tuple[EndpointSpec, ...] = (
    # --- block 0: reference. Bootstrap lists are fetched by `fmp plan` itself (universe + tier B membership).
    _e("stock-list", 0, "P0", "list", rows=90_000, bootstrap=True),
    _e("actively-trading-list", 0, "P0", "list", rows=60_000, bootstrap=True),
    _e("delisted-companies", 0, "P0", "paged", rate=(28, 166), rows=100, bootstrap=True),
    _e("symbol-change", 0, "P0", "paged", rows=100, bootstrap=True),
    _e("etf-list", 0, "P0", "list", rows=13_000, bootstrap=True),
    _e("available-exchanges", 0, "P1", "list", rows=80, bootstrap=True),
    _e("financial-statement-symbol-list", 0, "P1", "list", rows=50_000, bootstrap=True),
    _e("historical-sp500-constituent", 0, "P0", "list", rows=1_500, bootstrap=True),
    _e("historical-nasdaq-constituent", 0, "P0", "list", rows=1_000, bootstrap=True),
    _e("historical-dowjones-constituent", 0, "P0", "list", rows=100, bootstrap=True),
    _e("sp500-constituent", 0, "P0", "list", rows=503, bootstrap=True),
    _e("nasdaq-constituent", 0, "P0", "list", rows=101, bootstrap=True),
    _e("dowjones-constituent", 0, "P0", "list", rows=30, bootstrap=True),
    _e("profile", 0, "P0", "symbol", "universe", rate=(1593, 3148)),
    _e("cik-list", 0, "P1", "paged", rows=100),
    _e("available-sectors", 0, "P1", "list", rows=11),
    _e("available-industries", 0, "P1", "list", rows=160),
    _e("available-countries", 0, "P1", "list", rows=120),
    _e("shares-float-all", 0, "P1", "paged", rate=(32, 162), rows=100),
    _e("mergers-acquisitions-latest", 0, "P1", "paged", rows=100),
    _e("holidays-by-exchange", 0, "P1", "exchange", rate=(28, 168), rows=12),
    _e("exchange-market-hours", 0, "P1", "exchange", rate=(196, 196)),
    _e("employee-count", 0, "P2", "symbol", "stocks", rate=(42, 367), rows=10),
    _e("historical-employee-count", 0, "P2", "symbol", "stocks", rate=(42, 367), rows=10),
    _e("index-list", 0, "P2", "list", rate=(16, 128), rows=430),
    _e("etf/info", 0, "P2", "symbol", "etfs"),
    _e("etf/holdings", 0, "P2", "symbol", "etfs", rows=100),
    # --- block 1: corporate events
    _e("splits", 1, "P0", "symbol", "universe", rate=(33, 130), rows=1),
    _e("dividends", 1, "P0", "symbol", "universe", rate=(34, 269), rows=25),
    _e("earnings", 1, "P1", "symbol", "stocks", rate=(18, 207), rows=60),
    # --- block 2: calendars and macro (index membership is bootstrap, above)
    _e("splits-calendar", 2, "P1", "calendar", rows=150),
    _e("dividends-calendar", 2, "P1", "calendar", rows=4_000),
    _e("earnings-calendar", 2, "P1", "calendar", rows=6_000),
    _e("ipos-calendar", 2, "P2", "calendar", rows=150),
    _e("treasury-rates", 2, "P1", "calendar", rows=63),
    _e("economic-indicators", 2, "P1", "indicator", rows=300),
    _e("economic-calendar", 2, "P2", "calendar", rows=3_000),
    _e("market-risk-premium", 2, "P2", "list", rows=200),
    # --- block 3: daily raw (primary price source, spec D8)
    _e("historical-price-eod/non-split-adjusted", 3, "P0", "daily", "universe", rate=(25, 171)),
    # --- block 4: reconciliation sample only (spec D8)
    _e("historical-price-eod/full", 4, "P1", "daily", "sample", rate=(38, 230)),
    _e("historical-price-eod/dividend-adjusted", 4, "P1", "daily", "sample", rate=(25, 172)),
    # --- block 5: fundamentals
    *(_e(name, 5, "P1", "statement", "statements", rate=rate, rows=20) for name, rate in _STATEMENT_RATES.items()
      if not name.endswith("as-reported")),
    _e("historical-market-capitalization", 5, "P1", "daily", "statements", rate=(10, 70), measured=False),
    _e("financial-reports-dates", 5, "P1", "symbol", "statements", rate=(14, 344), rows=60),
    _e("income-statement-as-reported", 5, "P2", "statement", "statements",
       rate=_STATEMENT_RATES["income-statement-as-reported"], rows=20),
    _e("balance-sheet-statement-as-reported", 5, "P2", "statement", "statements", rows=20),
    _e("cash-flow-statement-as-reported", 5, "P2", "statement", "statements", rows=20),
    _e("financial-statement-full-as-reported", 5, "P2", "statement", "statements", rows=20),
    _e("income-statement-growth", 5, "P2", "statement", "statements", rows=20),
    _e("balance-sheet-statement-growth", 5, "P2", "statement", "statements", rows=20),
    _e("cash-flow-statement-growth", 5, "P2", "statement", "statements", rows=20),
    _e("financial-scores", 5, "P2", "symbol", "statements"),
    _e("owner-earnings", 5, "P2", "symbol", "statements", rows=40),
    _e("revenue-product-segmentation", 5, "P2", "symbol", "statements", rows=10),
    _e("revenue-geographic-segmentation", 5, "P2", "symbol", "statements", rows=10),
    _e("analyst-estimates", 5, "P2", "statement", "statements", rows=10),
    _e("grades-historical", 5, "P2", "symbol", "statements", rows=50),
    _e("ratings-historical", 5, "P2", "symbol", "statements", rows=50),
    # --- block 6: intraday (endpoint name is completed with the configured interval)
    _e("historical-chart/{interval}", 6, "P0", "intraday", "universe"),
)

#: Endpoints whose responses feed universe assembly and tiering.
BOOTSTRAP = tuple(e for e in CATALOG if e.bootstrap)
DAILY_RAW = "historical-price-eod/non-split-adjusted"
#: Market-cap history is per symbol with a date range; its rows are one per trading day.
DAILY_ROW_ENDPOINTS = ("historical-price-eod/non-split-adjusted", "historical-price-eod/full",
                       "historical-price-eod/dividend-adjusted", "historical-market-capitalization")


def selected(priorities: tuple[str, ...], intervals: Union[str, tuple[str, ...]]) -> list[EndpointSpec]:
    """
    Return the catalogue entries to plan.

    Args:
        priorities: Priorities to include (``P0``, ``P1``, ``P2``). Bootstrap lists are always included.
        intervals: Intraday interval(s), substituted into ``historical-chart/{interval}``: one entry per interval.

    Returns:
        Entries in catalogue order.
    """
    chosen = []
    for spec in CATALOG:
        if spec.priority not in priorities and not spec.bootstrap:
            continue
        if "{interval}" not in spec.endpoint:
            chosen.append(spec)
            continue
        for interval in (intervals,) if isinstance(intervals, str) else intervals:
            chosen.append(EndpointSpec(spec.endpoint.format(interval=interval), spec.block, spec.priority, spec.kind,
                                       spec.scope, spec.wire_per_row, spec.decoded_per_row, spec.rows, spec.measured))
    return chosen


def priority_of(block: int, sub: int) -> int:
    """Return the manifest priority for a block and an order inside it."""
    return block * 100 + sub
