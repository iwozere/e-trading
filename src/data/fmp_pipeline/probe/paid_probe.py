"""
``fmp probe-paid`` (spec §4.4, Phase 2): answer every "requires paid probe" item in one unattended run.

Stages (each only uses what earlier stages found):

1. **Account and coverage**: a known symbol, symbols the free key could not see, indices, both class-share
   spellings, foreign listings, an unknown symbol, and well-known delisted symbols in the price endpoints.
2. **Intraday intervals**: one day of each interval (``1min`` … ``4hour``).
3. **Intraday conventions** (spec §8.4): AAPL bars around both DST changes of two years plus two early closes,
   which settle time zone, bar label, bars per day and extended hours.
4. **Row cap**: 1-, 3- and 10-year windows; capped responses give ``max_rows`` and which end is kept, and all
   responses together give the measured bytes per row.
5. **Depth**: the earliest year with intraday bars, then month-level bisection; the daily depth of AAPL.
6. **Intraday split adjustment**: NVDA bars across its 2024-06-10 10:1 split.

Stages 3–6 run for the configured ``intraday.interval`` and for ``1hour`` (the fallback), so the owner chooses
between them on measured numbers. Every measured interval's settings are kept (``settings_by_interval``).
7. **Lists, paging and windows**: the reference lists the free key could not read (field names, exchange codes),
   page-size caps, constituent endpoints, calendar maximum windows, market-cap ranges, indicator names.
8. **Ticker reuse** (spec §4.2.8): candidates from the full delisted list, then each one's whole price history.

Everything goes through ``ProbeClient``: responses are cached under ``<data_root>/probe-paid/`` (never committed,
D5), the call budget holds across runs, and a re-run costs no calls.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from src.data.fmp_pipeline.planner.config import INTRADAY_MINUTES, SESSION_MINUTES
from src.data.fmp_pipeline.probe.analysis import find_reused_tickers
from src.data.fmp_pipeline.probe.free_probe import ProbeCall, ProbeResult, ProbeRun
from src.data.fmp_pipeline.probe.paid_analysis import (
    HourlyConventions,
    RowCap,
    RowCapProbe,
    expected_bars,
    first_date,
    hourly_conventions,
    intraday_split_verdict,
    last_date,
    reuse_verdict,
    row_cap,
    window_complete,
)
from src.data.fmp_pipeline.probe.probe_client import BudgetExhausted, ProbeClient, QuotaExhausted
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

RAW = "historical-price-eod/non-split-adjusted"
HOURLY = "1hour"
INTERVALS = tuple(INTRADAY_MINUTES)
COVERAGE_SYMBOLS = ("AAPL", "AMGN", "QQQ", "BRK-B", "BRK.B", "^GSPC", "^IXIC", "^DJI", "^VIX", "^RUT",
                    "VOD.L", "SHOP.TO", "ZZZZQX")
DELISTED_CHECKS = (("TWTR", "2021-06-01", "2021-06-30"), ("SIVB", "2022-06-01", "2022-06-30"),
                   ("BBBY", "2022-06-01", "2022-06-30"), ("ATVI", "2022-06-01", "2022-06-30"),
                   ("LEH", "2007-06-01", "2007-06-30"))
#: Friday before and Monday after each US DST change of two years, plus two early-close days (spec §8.4).
CONVENTION_WINDOWS = (("2024-03-08", "2024-03-11"), ("2024-11-01", "2024-11-04"), ("2025-03-07", "2025-03-10"),
                      ("2025-10-31", "2025-11-03"), ("2024-11-27", "2024-12-02"), ("2024-12-23", "2024-12-26"))
CAP_WINDOWS = (("2025-01-01", "2025-12-31"), ("2023-01-01", "2025-12-31"), ("2016-01-01", "2025-12-31"))
DEPTH_YEARS = (1998, 2003, 2008, 2013, 2018, 2023)
LISTS = ("stock-list", "actively-trading-list", "etf-list", "available-exchanges", "financial-statement-symbol-list",
         "symbol-change", "historical-sp500-constituent", "historical-nasdaq-constituent",
         "historical-dowjones-constituent", "sp500-constituent", "nasdaq-constituent", "dowjones-constituent")
PAGED = ("delisted-companies", "cik-list", "symbol-change", "mergers-acquisitions-latest")
PAGE_LIMITS = (100, 1000, 10000)
CALENDARS = ("splits-calendar", "dividends-calendar", "earnings-calendar", "ipos-calendar", "economic-calendar")
CALENDAR_WINDOWS = (30, 90, 180, 365)
INDICATORS = ("GDP", "realGDP", "CPI", "inflationRate", "federalFunds", "unemploymentRate", "initialClaims",
              "retailSales", "consumerSentiment", "industrialProductionTotalIndex")


@dataclass
class IntradayFindings:
    """Stages 3–6 for one intraday interval."""

    interval: str
    conventions: Optional[HourlyConventions] = None
    cap: Optional[RowCap] = None
    cap_probes: list[RowCapProbe] = field(default_factory=list)
    earliest: Optional[date] = None
    split: str = ""
    wire_bytes_per_row: Optional[float] = None
    decoded_bytes_per_row: Optional[float] = None


@dataclass
class PaidFindings:
    """
    What the paid probe concluded.

    ``settings`` is what ``fmp plan`` and ``fmp build`` will use with ``interval`` (the configured one);
    ``settings_by_interval`` holds the intraday part for every measured interval.
    """

    intervals: dict[str, str] = field(default_factory=dict)
    interval: str = HOURLY
    intraday: dict[str, IntradayFindings] = field(default_factory=dict)
    earliest_daily: Optional[date] = None
    coverage: dict[str, str] = field(default_factory=dict)
    delisted: dict[str, str] = field(default_factory=dict)
    lists: dict[str, dict[str, Any]] = field(default_factory=dict)
    page_sizes: dict[str, int] = field(default_factory=dict)
    page_full: dict[str, int] = field(default_factory=dict)
    calendar_windows: dict[str, int] = field(default_factory=dict)
    market_cap_years: int = 0
    indicators: dict[str, int] = field(default_factory=dict)
    exchanges: dict[str, int] = field(default_factory=dict)
    reuse: dict[str, str] = field(default_factory=dict)
    rate_headers: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    settings_by_interval: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class PaidRun(ProbeRun):
    """A probe run plus its findings."""

    findings: PaidFindings = field(default_factory=PaidFindings)


def _window(start: str, end: str) -> dict[str, str]:
    return {"from": start, "to": end}


def _chart(interval: str) -> str:
    return f"historical-chart/{interval}"


def session_bars(interval: str) -> int:
    """Bars of a full regular session (09:30–16:00 ET) at ``interval``: 7 for ``1hour``, 78 for ``5min``."""
    return math.ceil(SESSION_MINUTES / INTRADAY_MINUTES[interval])


class PaidProbe:
    """Runs the paid-probe stages against a ``ProbeClient`` holding the Premium key."""

    def __init__(self, client: ProbeClient, today: Optional[date] = None, interval: str = HOURLY) -> None:
        """
        Args:
            client: Budgeted, cached client (cache dir ``<data_root>/probe-paid``).
            today: Reference date for "to today" windows (default: today).
            interval: The configured ``intraday.interval``; measured in depth together with ``1hour``.

        Raises:
            ValueError: ``interval`` is not an FMP intraday interval.
        """
        if interval not in INTRADAY_MINUTES:
            raise ValueError(f"unknown intraday interval {interval!r}; expected one of {', '.join(INTERVALS)}")
        self._client = client
        self._today = today or date.today()
        self._run = PaidRun()
        self.f = self._run.findings
        self.f.interval = interval

    def run(self) -> PaidRun:
        """Execute every stage; stops early only when FMP reports the quota exhausted."""
        self._coverage()
        self._intervals()
        for interval in dict.fromkeys((self.f.interval, HOURLY)):
            if self.f.intervals.get(interval) != "ok":
                self.f.notes.append(f"{interval} bars are not served on this plan: no {interval} tasks can be planned")
                continue
            m = self.f.intraday[interval] = IntradayFindings(interval)
            self._conventions(m)
            self._row_cap(m)
            self._depth(m)
            self._intraday_split(m)
        self._daily_depth()
        self._lists()
        self._paging()
        self._windows()
        self._indicators()
        self._reuse()
        self.f.settings_by_interval = {i: intraday_settings(m) for i, m in self.f.intraday.items()}
        self.f.settings = derive_settings(self.f)
        return self._run

    # --- plumbing -------------------------------------------------------------------------------------------

    def _get(self, group: str, label: str, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """One call; returns its rows (empty on error, restriction, budget or quota stop)."""
        call = ProbeCall(group, label, endpoint, params)
        if self._run.stopped:
            self._run.results.append(ProbeResult(call, None, self._run.stopped))
            return []
        try:
            response = self._client.get(endpoint, params)
        except BudgetExhausted as exc:
            self._run.results.append(ProbeResult(call, None, str(exc)))
            return []
        except QuotaExhausted as exc:
            _logger.error("Stopping the paid probe: %s", exc)
            self._run.stopped = str(exc)
            self._run.results.append(ProbeResult(call, None, str(exc)))
            return []
        self._run.results.append(ProbeResult(call, response))
        if not self.f.rate_headers:
            self.f.rate_headers = {k: v for k, v in response.headers.items() if "limit" in k or "retry" in k}
        return self._run.rows(label)

    def _status(self, label: str) -> str:
        response = self._run.by_label(label)
        return response.classification if response is not None else "skipped"

    # --- stages ---------------------------------------------------------------------------------------------

    def _coverage(self) -> None:
        for symbol in COVERAGE_SYMBOLS:
            label = f"coverage:{symbol}"
            self._get("coverage", label, RAW, {"symbol": symbol, **_window("2025-06-02", "2025-06-06")})
            self.f.coverage[symbol] = self._status(label)
        for symbol, start, end in DELISTED_CHECKS:
            label = f"delisted:{symbol}"
            rows = self._get("coverage", label, RAW, {"symbol": symbol, **_window(start, end)})
            self.f.delisted[symbol] = f"{len(rows)} bars" if rows else self._status(label)

    def _intervals(self) -> None:
        for interval in INTERVALS:
            label = f"interval:{interval}"
            self._get("intervals", label, f"historical-chart/{interval}",
                      {"symbol": "AAPL", **_window("2025-06-02", "2025-06-02")})
            self.f.intervals[interval] = self._status(label)

    def _conventions(self, m: IntradayFindings) -> None:
        rows: list[dict[str, Any]] = []
        for start, end in CONVENTION_WINDOWS:
            rows += self._get("conventions", f"conventions:{m.interval}:{start}", _chart(m.interval),
                              {"symbol": "AAPL", **_window(start, end)})
        m.conventions = hourly_conventions(rows, minutes=INTRADAY_MINUTES[m.interval])

    @staticmethod
    def _bars_per_day(m: IntradayFindings) -> int:
        conventions = m.conventions
        return conventions.bars_per_day if conventions and conventions.bars_per_day else session_bars(m.interval)

    def _row_cap(self, m: IntradayFindings) -> None:
        rows_seen = wire = decoded = 0
        for start, end in CAP_WINDOWS:
            label = f"cap:{m.interval}:{start}:{end}"
            rows = self._get("row-cap", label, _chart(m.interval), {"symbol": "AAPL", **_window(start, end)})
            lo, hi = date.fromisoformat(start), min(date.fromisoformat(end), self._today)
            m.cap_probes.append(RowCapProbe(lo, hi, len(rows), first_date(rows), last_date(rows),
                                            expected_bars(lo, hi, self._bars_per_day(m))))
            response = self._run.by_label(label)
            if rows and response is not None:
                rows_seen += len(rows)
                wire += response.bytes_wire
                decoded += response.bytes_decoded
        m.cap = row_cap(m.cap_probes)
        if rows_seen:
            m.wire_bytes_per_row = round(wire / rows_seen, 1)
            m.decoded_bytes_per_row = round(decoded / rows_seen, 1)

    def _depth_probe(self, m: IntradayFindings, start: date) -> Optional[date]:
        rows = self._get("depth", f"depth:{m.interval}:{start}", _chart(m.interval),
                         {"symbol": "AAPL", **_window(start.isoformat(), (start + timedelta(days=13)).isoformat())})
        return first_date(rows)

    def _depth(self, m: IntradayFindings) -> None:
        found: Optional[date] = None
        previous: Optional[date] = None
        for year in DEPTH_YEARS:
            start = date(year, 1, 5)
            found = self._depth_probe(m, start)
            if found is not None:
                break
            previous = start
        if found is None:
            self.f.notes.append(f"no {m.interval} bars found in any probed year")
            return
        if previous is None:
            m.earliest = None  # data already at the oldest probe: depth at least back to it
            self.f.notes.append(f"{m.interval} bars exist at the oldest probe ({found}); depth is at least that")
            return
        lo, hi = previous, date(found.year, found.month, 1)
        while (hi - lo).days > 31:  # month-level bisection between "no data" and "data"
            mid = lo + (hi - lo) / 2
            mid = date(mid.year, mid.month, 1)
            if mid <= lo:
                break
            if self._depth_probe(m, mid) is not None:
                hi = mid
            else:
                lo = mid
        m.earliest = hi

    def _intraday_split(self, m: IntradayFindings) -> None:
        rows = self._get("split", f"split:NVDA:{m.interval}", _chart(m.interval),
                         {"symbol": "NVDA", **_window("2024-06-06", "2024-06-11")})
        m.split = intraday_split_verdict(rows, "2024-06-10", 10.0)

    def _daily_depth(self) -> None:
        rows = self._get("depth", "depth:daily:AAPL", RAW, {"symbol": "AAPL", **_window("1980-01-01", "1981-12-31")})
        self.f.earliest_daily = first_date(rows)

    def _lists(self) -> None:
        exchanges: Counter[str] = Counter()
        for endpoint in LISTS:
            rows = self._get("lists", f"list:{endpoint}", endpoint, {})
            fields = sorted({k for r in rows[:200] for k in r})
            self.f.lists[endpoint] = {"status": self._status(f"list:{endpoint}"), "rows": len(rows), "fields": fields}
            if endpoint in ("stock-list", "available-exchanges"):
                for row in rows:
                    code = row.get("exchangeShortName") or row.get("exchange")
                    if isinstance(code, str):
                        exchanges[code] += 1
        self.f.exchanges = dict(exchanges.most_common())

    def _paging(self) -> None:
        for endpoint in PAGED:
            for limit in PAGE_LIMITS:
                rows = self._get("paging", f"page:{endpoint}:{limit}", endpoint, {"page": 0, "limit": limit})
                if len(rows) == limit:
                    self.f.page_full[endpoint] = limit  # a full page: the cap is at least this
                    continue
                if len(rows) in PAGE_LIMITS:
                    self.f.page_sizes[endpoint] = len(rows)  # e.g. 1000 rows for limit 10000: a cap
                break  # a cap, or the end of a short list (which says nothing about the cap)
            self._get("paging", f"page:{endpoint}:p1", endpoint, {"page": 1, "limit": 100})

    def _windows(self) -> None:
        start = date(2025, 1, 1)
        for endpoint in CALENDARS:
            best = 0
            for days in CALENDAR_WINDOWS:
                end = start + timedelta(days=days - 1)
                rows = self._get("windows", f"window:{endpoint}:{days}", endpoint, _window(str(start), str(end)))
                if not window_complete(rows, start, end):
                    break
                best = days
            self.f.calendar_windows[endpoint] = best
        for years in (1, 10):
            rows = self._get("windows", f"window:market-cap:{years}y", "historical-market-capitalization",
                             {"symbol": "AAPL", **_window(f"{2026 - years}-01-01", "2025-12-31")})
            if window_complete(rows, date(2026 - years, 1, 1), date(2025, 12, 31)):
                self.f.market_cap_years = years

    def _indicators(self) -> None:
        for name in INDICATORS:
            rows = self._get("indicators", f"indicator:{name}", "economic-indicators",
                             {"name": name, **_window("2015-01-01", "2025-12-31")})
            self.f.indicators[name] = len(rows)

    def _reuse(self) -> None:
        limit = self.f.page_sizes.get("delisted-companies") or self.f.page_full.get("delisted-companies", 100)
        delisted: list[dict[str, Any]] = []
        for page in range(3):  # the most recent delistings: enough to find candidates without paging everything
            rows = self._get("reuse", f"reuse:delisted:{page}", "delisted-companies", {"page": page, "limit": limit})
            if not rows:
                break
            delisted += rows
        active = [str(r.get("symbol")) for r in self._run.rows("list:actively-trading-list")]
        candidates = find_reused_tickers(delisted, active, self._run.rows("list:symbol-change"))
        for candidate in candidates:
            prices = self._get("reuse", f"reuse:{candidate.symbol}:prices", RAW,
                               {"symbol": candidate.symbol, **_window("1985-01-01", self._today.isoformat())})
            self.f.reuse[candidate.symbol] = f"{candidate.reason}: {reuse_verdict(list(candidate.entries), prices)}"


def intraday_settings(m: IntradayFindings) -> dict[str, Any]:
    """
    The ``fmp.yaml``-shaped settings one interval's measurements support (only what was measured conclusively).

    Returns:
        A partial ``fmp:`` mapping: ``intraday`` (``max_rows``, ``bars_per_day``, ``earliest``, bytes per row) and
        ``build`` (``intraday_tz``, ``intraday_label``); either may be missing.
    """
    settings: dict[str, Any] = {}
    intraday: dict[str, Any] = {}
    if m.cap is not None and m.cap.max_rows:
        intraday["max_rows"] = int(m.cap.max_rows)
    if m.conventions is not None and m.conventions.bars_per_day:
        intraday["bars_per_day"] = int(m.conventions.bars_per_day)
    if m.earliest is not None:
        intraday["earliest"] = m.earliest.isoformat()
    if m.wire_bytes_per_row and m.decoded_bytes_per_row:
        intraday["wire_bytes_per_row"] = m.wire_bytes_per_row
        intraday["decoded_bytes_per_row"] = m.decoded_bytes_per_row
    if intraday:
        settings["intraday"] = intraday
    if m.conventions is not None and m.conventions.conclusive:
        settings["build"] = {"intraday_tz": m.conventions.timezone, "intraday_label": m.conventions.label}
    return settings


def derive_settings(f: PaidFindings) -> dict[str, Any]:
    """
    Turn findings into ``fmp.yaml``-shaped settings for the configured interval (only what was measured).

    Returns:
        A partial ``fmp:`` mapping: ``intraday`` and ``build`` from ``f.interval``'s measurements (see
        ``intraday_settings``), ``page_size``, ``calendar_window_days``, ``economic_indicators``.
    """
    measured = f.intraday.get(f.interval)
    settings: dict[str, Any] = intraday_settings(measured) if measured is not None else {}
    sizes = list(f.page_sizes.values()) or list(f.page_full.values())
    if sizes:
        settings["page_size"] = int(min(sizes))
    windows = [d for d in f.calendar_windows.values() if d > 0]
    if windows:
        settings["calendar_window_days"] = int(min(windows))
    names = [name for name, rows in f.indicators.items() if rows > 0]
    if names:
        settings["economic_indicators"] = names
    return settings
