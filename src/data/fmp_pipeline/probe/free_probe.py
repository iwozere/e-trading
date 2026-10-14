"""
The free-key probe plan (spec §4.2, plan WP-1 §4.1–§4.2): which calls to make, in which order.

Stages, each depending only on earlier ones:

1. **Entitlement sweep** (Finding C): one tiny request per candidate symbol, so later stages only use symbols
   the free key can actually see.
2. **Schema sweep**: every §5 endpoint once, on the best entitled symbol.
3. **Parameters**: ``limit`` and ``from``/``to`` behaviour, calendar window caps, default history depth.
4. **Pagination**: page size caps and the empty-page terminator for the paged list endpoints.
5. **Error shapes**: invalid symbol, invalid key, bad parameter, empty result, unknown and Ultimate-only endpoints.
6. **Adjustment facts**: dividend basis (AAPL 2020-08-07, NVDA 2024-03-05) and the three daily endpoints around
   known splits.
7. **Ticker reuse**: candidates found in stage 2/4 data, then their profile and price history.

Total ≈ 130 calls, under the 200 budget. Every response is cached by ``ProbeClient``, so re-runs are free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.data.fmp_pipeline.probe.analysis import ReusedTicker, find_reused_tickers
from src.data.fmp_pipeline.probe.probe_client import (
    AuthMode,
    BudgetExhausted,
    ProbeClient,
    ProbeResponse,
    QuotaExhausted,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

ENTITLEMENT_WINDOW = {"from": "2025-06-02", "to": "2025-06-06"}
YEAR_WINDOW = {"from": "2025-01-01", "to": "2025-12-31"}
MONTH_WINDOW = {"from": "2025-06-01", "to": "2025-06-30"}
ENTITLEMENT_CANDIDATES = (
    "AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL", "META", "JPM", "XOM", "KO",
    "BRK-B", "BRK.B", "SPY", "QQQ", "^GSPC", "^VIX", "MRNA", "AMGN", "ETSY", "TWTR",
)
PRIMARY_PREFERENCE = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM", "KO", "XOM", "MRNA")
DAILY_PRICE_ENDPOINTS = (
    "historical-price-eod/non-split-adjusted",
    "historical-price-eod/full",
    "historical-price-eod/dividend-adjusted",
)
# (label, symbol, split date, ratio, request window)
SPLIT_WINDOWS = (
    ("AAPL 4:1 2020-08-31", "AAPL", "2020-08-31", 4.0, {"from": "2020-08-24", "to": "2020-09-04"}),
    ("AAPL 7:1 2014-06-09", "AAPL", "2014-06-09", 7.0, {"from": "2014-06-02", "to": "2014-06-13"}),
    ("NVDA 10:1 2024-06-10", "NVDA", "2024-06-10", 10.0, {"from": "2024-06-03", "to": "2024-06-14"}),
)
# (symbol, ex-date, known raw-basis amount)
DIVIDEND_FACTS = (("AAPL", "2020-08-07", 0.82), ("NVDA", "2024-03-05", 0.04))


@dataclass(frozen=True)
class ProbeCall:
    """One planned request."""

    group: str
    label: str
    endpoint: str
    params: dict[str, Any] = field(default_factory=dict)
    auth: AuthMode = "header"


@dataclass
class ProbeResult:
    """A planned request and its outcome; ``response`` is ``None`` when it was skipped."""

    call: ProbeCall
    response: Optional[ProbeResponse]
    skipped: str = ""


@dataclass
class ProbeRun:
    """Everything one probe run produced."""

    results: list[ProbeResult] = field(default_factory=list)
    primary_symbol: str = ""
    entitled: list[str] = field(default_factory=list)
    reused: list[ReusedTicker] = field(default_factory=list)
    stopped: str = ""

    def by_label(self, label: str) -> Optional[ProbeResponse]:
        """Return the response recorded under ``label``, if any."""
        for result in self.results:
            if result.call.label == label:
                return result.response
        return None

    def rows(self, label: str) -> list[dict[str, Any]]:
        """Return the list-of-records body recorded under ``label``, or an empty list."""
        response = self.by_label(label)
        body = response.json() if response is not None else None
        if isinstance(body, dict) and isinstance(body.get("historical"), list):
            body = body["historical"]
        return [r for r in body if isinstance(r, dict)] if isinstance(body, list) else []


def _schema_calls(symbol: str) -> list[ProbeCall]:
    """Every §5 endpoint once (spec §4.2.1)."""
    s = {"symbol": symbol}
    page = {"page": 0, "limit": 100}
    rows: list[tuple[str, dict[str, Any]]] = [
        # §5.1 universe and reference
        ("stock-list", {}), ("actively-trading-list", {}), ("delisted-companies", page), ("symbol-change", {}),
        ("etf-list", {}), ("profile", s), ("cik-list", page), ("available-exchanges", {}),
        ("available-sectors", {}), ("available-industries", {}), ("available-countries", {}),
        ("shares-float-all", page), ("mergers-acquisitions-latest", page),
        ("mergers-acquisitions-search", {"name": "Activision"}), ("holidays-by-exchange", {"exchange": "NYSE"}),
        ("exchange-market-hours", {"exchange": "NASDAQ"}), ("employee-count", s), ("historical-employee-count", s),
        ("index-list", {}), ("etf/info", {"symbol": "SPY"}), ("etf/holdings", {"symbol": "SPY"}),
        # §5.2 corporate events
        ("splits", s), ("dividends", s), ("splits-calendar", MONTH_WINDOW), ("dividends-calendar", MONTH_WINDOW),
        ("earnings", s), ("earnings-calendar", MONTH_WINDOW), ("ipos-calendar", MONTH_WINDOW),
        # §5.3 prices (one calendar year each, doubling as the bytes-per-row measurement, §4.2.2)
        *((endpoint, {**s, **YEAR_WINDOW}) for endpoint in DAILY_PRICE_ENDPOINTS),
        ("historical-chart/1hour", {**s, **ENTITLEMENT_WINDOW}),
        ("historical-chart/5min", {**s, "from": "2025-06-02", "to": "2025-06-02"}),
        # §5.4 index membership
        ("historical-sp500-constituent", {}), ("historical-nasdaq-constituent", {}),
        ("historical-dowjones-constituent", {}), ("sp500-constituent", {}), ("nasdaq-constituent", {}),
        ("dowjones-constituent", {}),
        # §5.4a fundamentals
        *((endpoint, {**s, "period": "annual", "limit": 5}) for endpoint in (
            "income-statement", "balance-sheet-statement", "cash-flow-statement", "key-metrics", "ratios",
            "enterprise-values",
        )),
        ("historical-market-capitalization", {**s, **MONTH_WINDOW}), ("financial-statement-symbol-list", {}),
        ("financial-reports-dates", s),
        *((endpoint, {**s, "period": "annual", "limit": 2}) for endpoint in (
            "income-statement-as-reported", "balance-sheet-statement-as-reported",
            "cash-flow-statement-as-reported", "financial-statement-full-as-reported",
        )),
        ("financial-scores", s), ("owner-earnings", s), ("income-statement-growth", {**s, "limit": 5}),
        ("revenue-product-segmentation", s), ("revenue-geographic-segmentation", s),
        ("analyst-estimates", {**s, "period": "annual", "limit": 5}), ("grades-historical", s),
        ("ratings-historical", s),
        # §5.5 macro
        ("treasury-rates", MONTH_WINDOW), ("economic-indicators", {"name": "GDP"}),
        ("economic-calendar", MONTH_WINDOW), ("market-risk-premium", {}),
    ]
    return [ProbeCall("schema", f"schema:{endpoint}", endpoint, params) for endpoint, params in rows]


def _param_calls(symbol: str) -> list[ProbeCall]:
    """``limit``, ``period``, window caps and default depth (spec §4.2.1 [VERIFY] parameters)."""
    s = {"symbol": symbol}
    return [
        ProbeCall("params", "params:statement-quarter-limit-2", "income-statement",
                  {**s, "period": "quarter", "limit": 2}),
        ProbeCall("params", "params:price-default-depth", "historical-price-eod/non-split-adjusted", s),
        ProbeCall("params", "params:splits-calendar-1y", "splits-calendar", YEAR_WINDOW),
        ProbeCall("params", "params:dividends-calendar-1y", "dividends-calendar", YEAR_WINDOW),
        ProbeCall("params", "params:earnings-calendar-1y", "earnings-calendar", YEAR_WINDOW),
        ProbeCall("params", "params:treasury-rates-1y", "treasury-rates", YEAR_WINDOW),
        ProbeCall("params", "params:price-pre-5y", "historical-price-eod/non-split-adjusted",
                  {**s, "from": "2015-01-02", "to": "2015-01-09"}),
    ]


def _pagination_calls() -> list[ProbeCall]:
    """Page size caps and the empty-page terminator (spec §4.2.3)."""
    calls = []
    for endpoint in ("delisted-companies", "cik-list", "shares-float-all"):
        calls += [
            ProbeCall("pagination", f"page:{endpoint}:p1", endpoint, {"page": 1, "limit": 100}),
            ProbeCall("pagination", f"page:{endpoint}:limit1000", endpoint, {"page": 0, "limit": 1000}),
            ProbeCall("pagination", f"page:{endpoint}:beyond", endpoint, {"page": 100000, "limit": 100}),
        ]
    calls.append(ProbeCall("pagination", "page:symbol-change:paged", "symbol-change", {"page": 0, "limit": 100}))
    calls.append(ProbeCall("pagination", "page:mergers-acquisitions-latest:beyond", "mergers-acquisitions-latest",
                           {"page": 100000, "limit": 100}))
    return calls


def _error_calls(symbol: str) -> list[ProbeCall]:
    """Error shapes (spec §4.2.4)."""
    price = "historical-price-eod/non-split-adjusted"
    return [
        ProbeCall("errors", "error:invalid-symbol", price, {"symbol": "ZZZZQX", **ENTITLEMENT_WINDOW}),
        ProbeCall("errors", "error:invalid-key", price, {"symbol": symbol, **ENTITLEMENT_WINDOW}, auth="invalid"),
        ProbeCall("errors", "error:bad-date-param", price,
                  {"symbol": symbol, "from": "not-a-date", "to": "2025-06-06"}),
        ProbeCall("errors", "error:empty-weekend", price, {"symbol": symbol, "from": "2025-06-07", "to": "2025-06-08"}),
        ProbeCall("errors", "error:unknown-endpoint", "this-endpoint-does-not-exist", {}),
        ProbeCall("errors", "error:ultimate-only-bulk", "eod-bulk", {"date": "2025-06-02"}),
    ]


def _adjustment_calls() -> list[ProbeCall]:
    """Dividend basis (spec §4.2.5) and split cross-checks (spec §4.2.6)."""
    calls = [ProbeCall("adjust", f"dividends:{symbol}", "dividends", {"symbol": symbol})
             for symbol, _, _ in DIVIDEND_FACTS]
    calls += [ProbeCall("adjust", f"splits:{symbol}", "splits", {"symbol": symbol}) for symbol in ("AAPL", "NVDA")]
    for label, symbol, _, _, window in SPLIT_WINDOWS:
        calls += [ProbeCall("adjust", f"split:{label}:{endpoint}", endpoint, {"symbol": symbol, **window})
                  for endpoint in DAILY_PRICE_ENDPOINTS]
    return calls


def _reuse_calls(candidates: list[ReusedTicker]) -> list[ProbeCall]:
    """Profile and default-depth raw prices for each reused-ticker candidate (spec §4.2.8)."""
    calls = []
    for candidate in candidates:
        calls.append(ProbeCall("reuse", f"reuse:{candidate.symbol}:profile", "profile", {"symbol": candidate.symbol}))
        calls.append(ProbeCall("reuse", f"reuse:{candidate.symbol}:prices", "historical-price-eod/non-split-adjusted",
                               {"symbol": candidate.symbol}))
    return calls


class FreeProbe:
    """Runs the probe stages against a ``ProbeClient``."""

    def __init__(self, client: ProbeClient) -> None:
        """
        Args:
            client: Budgeted, cached client; its cache makes the whole run idempotent.
        """
        self._client = client
        self._run = ProbeRun()

    def run(self) -> ProbeRun:
        """
        Execute every stage in order, stopping early only on quota exhaustion.

        Returns:
            All results, including skipped calls (budget exhausted or quota hit).
        """
        sweep = [ProbeCall("entitlement", f"entitlement:{symbol}", "historical-price-eod/non-split-adjusted",
                           {"symbol": symbol, **ENTITLEMENT_WINDOW}) for symbol in ENTITLEMENT_CANDIDATES]
        self._execute(sweep)
        self._run.entitled = [c.params["symbol"] for c in sweep if self._classification(c.label) == "ok"]
        self._run.primary_symbol = next((s for s in PRIMARY_PREFERENCE if s in self._run.entitled), "AAPL")
        _logger.info("Entitled symbols: %s; primary: %s", self._run.entitled, self._run.primary_symbol)

        primary = self._run.primary_symbol
        self._execute(_schema_calls(primary))
        self._execute(_param_calls(primary))
        self._execute(_pagination_calls())
        self._execute(_error_calls(primary))
        self._execute(_adjustment_calls())

        delisted = (self._run.rows("schema:delisted-companies") + self._run.rows("page:delisted-companies:p1")
                    + self._run.rows("page:delisted-companies:limit1000"))
        active = [str(r.get("symbol")) for r in self._run.rows("schema:actively-trading-list")]
        changes = self._run.rows("schema:symbol-change")
        self._run.reused = find_reused_tickers(delisted, active, changes)
        self._execute(_reuse_calls(self._run.reused))
        return self._run

    def _classification(self, label: str) -> str:
        response = self._run.by_label(label)
        return response.classification if response is not None else ""

    def _execute(self, calls: list[ProbeCall]) -> None:
        for call in calls:
            if self._run.stopped:
                self._run.results.append(ProbeResult(call, None, self._run.stopped))
                continue
            try:
                response = self._client.get(call.endpoint, call.params, call.auth)
            except BudgetExhausted as exc:
                self._run.results.append(ProbeResult(call, None, str(exc)))
                continue
            except QuotaExhausted as exc:
                _logger.error("Stopping the probe: %s", exc)
                self._run.stopped = str(exc)
                self._run.results.append(ProbeResult(call, None, str(exc)))
                continue
            self._run.results.append(ProbeResult(call, response))
