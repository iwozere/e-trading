"""
Local mock of the FMP ``/stable`` API for offline downloader tests (spec §6.7, plan WP-3).

Standard library only (``http.server`` on a background thread). The default behaviour reproduces what the free-key
probe actually observed (``docs/fmp_probe_report.md``):

- The key is read from the ``apikey`` header; a wrong key gets HTTP 401 with a JSON ``Error Message`` (F5).
- Plan restrictions are HTTP 402 with a **plain-text** body: ``Premium Query Parameter`` for a symbol,
  ``Restricted Endpoint`` for an endpoint (F5).
- Unknown endpoints are HTTP 404 with body ``[]``; empty results are HTTP 200 ``[]``.
- Lists come back newest first. Daily price fields follow the endpoint: ``adjOpen``…``adjClose`` on
  ``non-split-adjusted`` and ``dividend-adjusted``, ``open``…``close`` plus ``change``/``changePercent``/``vwap``
  on ``full`` (F3).
- An unparseable ``from``/``to`` is silently ignored; with no window the last five years are served (F7, F8).
- Bodies over 200 bytes are gzip-encoded and sent **chunked, without ``Content-Length``** (F10). No rate-limit
  headers (F11).

Everything the free key could not show is configurable rather than guessed: page size cap, intraday row cap,
the quota response, unknown-symbol behaviour. Faults (429 with ``Retry-After``, 5xx bursts, 200 with an error
body, slow responses) are scripted per test with ``Fault``.

Every value served is synthetic. Field names come from the synthetic fixtures in ``tests/fixtures/``.
"""

from __future__ import annotations

import gzip
import json
import socket
import threading
import time
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

PLAN_SYMBOL_TEXT = (
    "Premium Query Parameter: 'Special Endpoint : This value set for 'symbol' is not available under your current "
    "subscription please visit our subscription page to upgrade your plan at https://financialmodelingprep.com/"
)
PLAN_ENDPOINT_TEXT = (
    "Restricted Endpoint: This endpoint is not available under your current subscription please visit our "
    "subscription page to upgrade your plan at https://financialmodelingprep.com/"
)
INVALID_KEY_BODY = {"Error Message": "Invalid API KEY. Feel free to create a Free API Key or visit "
                                     "https://site.financialmodelingprep.com/faqs?search=why-is-my-api-key-invalid "
                                     "for more information."}
# Not observed on the free key; the shape is a [VERIFY] item. The classifier keys on "Limit Reach".
QUOTA_BODY = {"Error Message": "Limit Reach . Please upgrade your plan or visit our documentation for more details "
                               "at https://site.financialmodelingprep.com/"}

DAILY_PRICE_ENDPOINTS = (
    "historical-price-eod/non-split-adjusted",
    "historical-price-eod/full",
    "historical-price-eod/dividend-adjusted",
)
PAGED_ENDPOINTS = ("delisted-companies", "cik-list", "shares-float-all", "mergers-acquisitions-latest",
                   "symbol-change")
INTRADAY_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1hour": 60, "4hour": 240}
GZIP_MIN_BYTES = 200
_SESSION_OPEN = (9, 30)
_SESSION_MINUTES = 390


@dataclass
class Fault:
    """
    A scripted response override.

    ``status=None`` means "delay only": the normal response is served after ``delay_s``. A ``str`` body is sent as
    ``text/plain``, anything else as JSON. ``times=None`` keeps the fault active forever.
    """

    endpoint: str = "*"
    status: Optional[int] = None
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0
    symbol: Optional[str] = None
    times: Optional[int] = 1

    def matches(self, endpoint: str, symbol: Optional[str]) -> bool:
        """Return whether this fault applies to a request and still has activations left."""
        if self.times is not None and self.times <= 0:
            return False
        if self.endpoint not in ("*", endpoint):
            return False
        return self.symbol is None or self.symbol == symbol


@dataclass
class MockConfig:
    """Behaviour knobs; defaults match the free-key probe observations where those exist."""

    api_key: str = "test-key"
    today: date = date(2026, 10, 14)
    default_history_years: int = 5
    earliest: date = date(1990, 1, 2)
    restricted_symbols: set[str] = field(default_factory=set)
    restricted_endpoints: set[str] = field(default_factory=set)
    unknown_symbols: set[str] = field(default_factory=set)
    listings: dict[str, tuple[date, date]] = field(default_factory=dict)
    paged_totals: dict[str, int] = field(default_factory=lambda: {e: 250 for e in PAGED_ENDPOINTS})
    max_page_size: int = 100
    intraday_max_rows: int = 1000
    quota_after: Optional[int] = None
    send_content_length: bool = False
    gzip_enabled: bool = True


@dataclass(frozen=True)
class RecordedRequest:
    """One request as the server saw it, plus what it sent back."""

    endpoint: str
    params: dict[str, str]
    headers: dict[str, str]
    query_has_apikey: bool
    status: int
    bytes_wire: int
    bytes_decoded: int


@dataclass
class _Response:
    status: int
    body: bytes
    content_type: str
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _weekdays(start: date, end: date) -> list[date]:
    """Weekdays from ``start`` to ``end`` inclusive (holidays are not modelled)."""
    days, day = [], start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _price(symbol: str, when: date, minute: int = 0) -> float:
    """Deterministic synthetic price: a per-symbol level plus a slow wave."""
    level = 20 + zlib.crc32(symbol.encode("utf-8")) % 480
    t = when.toordinal() + minute / _SESSION_MINUTES
    return round(level * (1 + 0.1 * ((t % 97) / 97 - 0.5)), 2)


def _bar(symbol: str, when: date, names: tuple[str, str, str, str], minute: int = 0) -> dict[str, Any]:
    close = _price(symbol, when, minute)
    open_ = round(close * 0.995, 2)
    o, h, lo, c = names
    return {o: open_, h: round(close * 1.01, 2), lo: round(close * 0.985, 2), c: close,
            "volume": 100_000 + zlib.crc32(f"{symbol}{when}{minute}".encode("utf-8")) % 900_000}


class MockFmpServer:
    """
    Threaded mock FMP server. Use as a context manager::

        with MockFmpServer() as fmp:
            httpx.get(fmp.base_url + "profile", params={"symbol": "AAA"}, headers={"apikey": fmp.config.api_key})
    """

    def __init__(self, config: Optional[MockConfig] = None, fixtures_dir: Path = FIXTURES_DIR) -> None:
        """
        Args:
            config: Behaviour knobs; defaults to ``MockConfig()``.
            fixtures_dir: Directory of synthetic fixtures named ``<endpoint with / as __>.json``.
        """
        self.config = config or MockConfig()
        self.requests: list[RecordedRequest] = []
        self.key_revoked = False
        self._faults: list[Fault] = []
        self._lock = threading.Lock()
        self._fixtures = {
            path.stem.replace("__", "/"): json.loads(path.read_text(encoding="utf-8"))
            for path in fixtures_dir.glob("*.json") if not path.name.startswith("_")
        }
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # --- lifecycle -------------------------------------------------------------------------------------------

    def start(self) -> "MockFmpServer":
        """Bind to a free localhost port and serve on a daemon thread."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                super().setup()
                # Headers and body are separate small writes; without this, Nagle + delayed ACK add ~0.4 s each.
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            def do_GET(self) -> None:  # noqa: N802 - http.server naming
                server._handle(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature from base class
                del format, args

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="mock-fmp", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Shut the server down and wait for its thread."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "MockFmpServer":
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        del _exc_info
        self.stop()

    @property
    def base_url(self) -> str:
        """Base URL ending in ``/stable/``, like ``https://financialmodelingprep.com/stable/``."""
        if self._httpd is None:
            raise RuntimeError("server not started")
        host, port = self._httpd.server_address[:2]
        return f"http://{host!s}:{port}/stable/"

    # --- scripting -------------------------------------------------------------------------------------------

    def add_fault(self, fault: Fault) -> None:
        """Queue a scripted fault; the first matching fault with activations left wins."""
        with self._lock:
            self._faults.append(fault)

    def revoke_key(self) -> None:
        """Make every following request fail with HTTP 401, as if the key were revoked or the plan lapsed."""
        self.key_revoked = True

    @property
    def bytes_wire(self) -> int:
        """Total body bytes sent on the wire (after gzip)."""
        return sum(r.bytes_wire for r in self.requests)

    @property
    def bytes_decoded(self) -> int:
        """Total body bytes before gzip."""
        return sum(r.bytes_decoded for r in self.requests)

    # --- request handling ------------------------------------------------------------------------------------

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        url = urlsplit(handler.path)
        endpoint = url.path.removeprefix("/stable/").strip("/")
        query = parse_qsl(url.query, keep_blank_values=True)
        params = {k: v for k, v in query if k != "apikey"}
        headers = {k.lower(): v for k, v in handler.headers.items() if k.lower() != "apikey"}

        with self._lock:
            served = len(self.requests)
            fault = next((f for f in self._faults if f.matches(endpoint, params.get("symbol"))), None)
            if fault is not None and fault.times is not None:
                fault.times -= 1

        response = self._route(endpoint, params, handler.headers.get("apikey"), served, fault)
        if response.delay_s:
            time.sleep(response.delay_s)
        wire, decoded = self._send(handler, response)
        with self._lock:
            self.requests.append(RecordedRequest(endpoint, params, headers, any(k == "apikey" for k, _ in query),
                                                 response.status, wire, decoded))

    def _route(self, endpoint: str, params: dict[str, str], key: Optional[str], served: int,
               fault: Optional[Fault]) -> _Response:
        cfg = self.config
        if self.key_revoked or key != cfg.api_key:
            return self._json(401, INVALID_KEY_BODY)
        if cfg.quota_after is not None and served >= cfg.quota_after:
            return self._json(429, QUOTA_BODY)
        delay = 0.0
        if fault is not None:
            delay = fault.delay_s
            if fault.status is not None:
                response = (self._text(fault.status, fault.body) if isinstance(fault.body, str)
                            else self._json(fault.status, fault.body))
                response.headers.update(fault.headers)
                response.delay_s = delay
                return response

        response = self._serve(endpoint, params)
        response.delay_s = delay
        return response

    def _serve(self, endpoint: str, params: dict[str, str]) -> _Response:
        cfg = self.config
        symbol = params.get("symbol")
        if endpoint in cfg.restricted_endpoints:
            return self._text(402, PLAN_ENDPOINT_TEXT)
        if symbol is not None and symbol in cfg.restricted_symbols:
            return self._text(402, PLAN_SYMBOL_TEXT)
        if symbol is not None and symbol in cfg.unknown_symbols:
            return self._json(200, [])
        if endpoint in DAILY_PRICE_ENDPOINTS:
            return self._json(200, self._daily(endpoint, symbol or "", params))
        if endpoint.startswith("historical-chart/") and endpoint.split("/", 1)[1] in INTRADAY_MINUTES:
            return self._json(200, self._intraday(INTRADAY_MINUTES[endpoint.split("/", 1)[1]], symbol or "", params))
        if endpoint in cfg.paged_totals and endpoint in self._fixtures:
            return self._json(200, self._paged(endpoint, params))
        if endpoint in self._fixtures:
            return self._json(200, self._fixture(endpoint, symbol))
        return self._json(404, [])

    # --- bodies ----------------------------------------------------------------------------------------------

    def _window(self, symbol: str, params: dict[str, str]) -> tuple[date, date]:
        cfg = self.config
        start, end = _parse_date(params.get("from")), _parse_date(params.get("to"))
        end = min(end or cfg.today, cfg.today)
        if start is None:
            # Anchored to today, not to `to`: FMP served 2021-10-15 … `to` for a bad `from` (probe F7).
            years_back = cfg.today.year - cfg.default_history_years
            start = date(years_back, cfg.today.month, min(cfg.today.day, 28)) + timedelta(days=1)
        first, last = cfg.listings.get(symbol, (cfg.earliest, cfg.today))
        return max(start, first), min(end, last)

    def _daily(self, endpoint: str, symbol: str, params: dict[str, str]) -> list[dict[str, Any]]:
        start, end = self._window(symbol, params)
        raw_names = ("adjOpen", "adjHigh", "adjLow", "adjClose")
        rows = []
        for day in reversed(_weekdays(start, end)):
            if endpoint.endswith("/full"):
                bar = _bar(symbol, day, ("open", "high", "low", "close"))
                prev = _price(symbol, day - timedelta(days=1))
                bar |= {"change": round(bar["close"] - prev, 2),
                        "changePercent": round((bar["close"] / prev - 1) * 100, 4),
                        "vwap": round((bar["high"] + bar["low"] + bar["close"]) / 3, 4)}
            else:
                bar = _bar(symbol, day, raw_names)
            rows.append({"symbol": symbol, "date": day.isoformat(), **bar})
        return rows

    def _intraday(self, minutes: int, symbol: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if "from" not in params and "to" not in params:
            params = {**params, "from": (self.config.today - timedelta(days=7)).isoformat()}
        start, end = self._window(symbol, params)
        rows = []
        for day in reversed(_weekdays(start, end)):
            for offset in reversed(range(0, _SESSION_MINUTES, minutes)):
                stamp = datetime(day.year, day.month, day.day, *_SESSION_OPEN) + timedelta(minutes=offset)
                bar = _bar(symbol, day, ("open", "high", "low", "close"), offset)
                rows.append({"date": stamp.strftime("%Y-%m-%d %H:%M:%S"), **bar})
        # Row cap: the newest rows are kept, so a too-wide window silently loses its start (spec §6.5).
        return rows[: self.config.intraday_max_rows]

    def _paged(self, endpoint: str, params: dict[str, str]) -> list[dict[str, Any]]:
        cfg = self.config
        try:
            page = max(int(params.get("page", 0)), 0)
            limit = int(params.get("limit", cfg.max_page_size))
        except ValueError:
            page, limit = 0, cfg.max_page_size
        size = max(1, min(limit, cfg.max_page_size))
        template = self._fixtures[endpoint][0]
        rows = []
        for i in range(page * size, min(cfg.paged_totals[endpoint], (page + 1) * size)):
            row = dict(template)
            for key, value in template.items():
                if key.lower().endswith("symbol"):
                    row[key] = f"SYM{i:05d}"
                elif isinstance(value, str) and _parse_date(value) is not None and len(value) == 10:
                    row[key] = (self.config.today - timedelta(days=i)).isoformat()
            rows.append(row)
        return rows

    def _fixture(self, endpoint: str, symbol: Optional[str]) -> Any:
        body = json.loads(json.dumps(self._fixtures[endpoint]))
        if symbol is not None:
            for row in body if isinstance(body, list) else [body]:
                if isinstance(row, dict) and "symbol" in row:
                    row["symbol"] = symbol
        return body

    # --- encoding --------------------------------------------------------------------------------------------

    @staticmethod
    def _json(status: int, body: Any) -> _Response:
        return _Response(status, json.dumps(body).encode("utf-8"), "application/json; charset=utf-8")

    @staticmethod
    def _text(status: int, body: str) -> _Response:
        return _Response(status, body.encode("utf-8"), "text/plain; charset=utf-8")

    def _send(self, handler: BaseHTTPRequestHandler, response: _Response) -> tuple[int, int]:
        """Write the response, gzip-encoded and chunked as configured; return (wire, decoded) body bytes."""
        payload = response.body
        accepts_gzip = "gzip" in handler.headers.get("Accept-Encoding", "")
        gzipped = self.config.gzip_enabled and accepts_gzip and len(payload) > GZIP_MIN_BYTES
        if gzipped:
            payload = gzip.compress(payload, compresslevel=6)

        handler.send_response(response.status)
        handler.send_header("Content-Type", response.content_type)
        if gzipped:
            handler.send_header("Content-Encoding", "gzip")
        for name, value in response.headers.items():
            handler.send_header(name, value)
        if self.config.send_content_length:
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
        else:
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            for offset in range(0, len(payload), 8192):
                chunk = payload[offset:offset + 8192]
                handler.wfile.write(f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n")
            handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
        return len(payload), len(response.body)
