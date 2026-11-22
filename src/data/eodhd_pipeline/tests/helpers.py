"""
A mock EODHD API over invented data for the tests. Nothing here is downloaded data.

Symbol lists:

- live: ``ABC`` (NYSE common stock), ``ABC-P`` (preferred), ``OTCZ`` (PINK common stock), ``FUNDX`` (mutual fund)
- delisted: ``XYZ_old`` (NASDAQ common stock, ticker since reused), ``GONE`` (NYSE common stock EODHD has no bars for)

``ABC`` has six bars, a 2-for-1 split on day 4 and a 1.00 cash dividend (0.50 split-adjusted) with ex-date day 2.
Raw closes 100, 102, 104, then 53, 54, 55; raw volume 1,000 a day, which EODHD reports split-adjusted (2,000 before
the split). ``adjusted_close`` = split-adjusted close times the dividend factor 1 - 1.00 / 100 before the ex-date.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import httpx

from src.data.eodhd_pipeline.client import EodhdClient

DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]

LIVE = [
    {"Code": "ABC", "Name": "ABC Corp", "Country": "USA", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock", "Isin": "US0000000101"},
    {"Code": "ABC-P", "Name": "ABC Corp Pref", "Country": "USA", "Exchange": "NYSE", "Currency": "USD", "Type": "Preferred Stock", "Isin": None},
    {"Code": "OTCZ", "Name": "Otc Z Inc", "Country": "USA", "Exchange": "PINK", "Currency": "USD", "Type": "Common Stock", "Isin": None},
    {"Code": "FUNDX", "Name": "Fund X", "Country": "USA", "Exchange": "NMFQS", "Currency": "USD", "Type": "FUND", "Isin": None},
]
DELISTED = [
    {"Code": "XYZ_old", "Name": "XYZ Inc", "Country": "USA", "Exchange": "NASDAQ", "Currency": "USD", "Type": "Common Stock", "Isin": None},
    {"Code": "GONE", "Name": "Gone Co", "Country": "USA", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock", "Isin": ""},
]

_ABC_RAW = [100.0, 102.0, 104.0, 53.0, 54.0, 55.0]
_ABC_ADJ = [49.5, 51.0, 52.0, 53.0, 54.0, 55.0]
_XYZ = [12.0, 12.5, 13.0, 13.5, 14.0]


def _abc_bars() -> list[dict[str, Any]]:
    return [{"date": day, "open": close - 1, "high": close + 1, "low": close - 2, "close": close, "adjusted_close": adj,
             "volume": 2000 if i < 3 else 1000} for i, (day, close, adj) in enumerate(zip(DAYS, _ABC_RAW, _ABC_ADJ))]


def _xyz_bars() -> list[dict[str, Any]]:
    return [{"date": day, "open": close, "high": close, "low": close, "close": close, "adjusted_close": close, "volume": 500}
            for day, close in zip(DAYS, _XYZ)]


BARS = {"ABC": _abc_bars(), "XYZ_old": _xyz_bars()}
SPLITS = {"ABC": [{"date": "2024-01-05", "split": "2.000000/1.000000"}], "XYZ_old": []}
DIVIDENDS = {"ABC": [{"date": "2024-01-03", "declarationDate": "2023-12-20", "recordDate": "2024-01-04", "paymentDate": "2024-01-10",
                      "period": "Quarterly", "value": 0.5, "unadjustedValue": 1.0, "currency": "USD"}], "XYZ_old": []}


class FakeEodhd:
    """Request handler with switches for the free plan, quota and failures; counts requests per path kind."""

    def __init__(self, free: bool = False, used: int = 0, daily: int = 100_000, quota_after: Optional[int] = None,
                 fail_codes: tuple[str, ...] = ()) -> None:
        self.free = free
        self.used = used
        self.daily = daily
        self.quota_after = quota_after
        self.fail_codes = set(fail_codes)
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assert request.url.params["api_token"] == "k"
        if path.endswith("/user"):
            return httpx.Response(200, json={"subscriptionType": "free" if self.free else "commercial", "dailyRateLimit": self.daily,
                                             "apiRequests": self.used, "extraLimit": 0, "apiRequestsDate": "2026-11-22"})
        self.requests.append(path)
        if self.quota_after is not None and len(self.requests) > self.quota_after:
            return httpx.Response(402, text="You exceeded your daily API requests limit")
        if "exchange-symbol-list" in path:
            return httpx.Response(200, json=DELISTED if request.url.params.get("delisted") == "1" else LIVE)
        kind, symbol = path.split("/")[-2], path.rsplit("/", 1)[-1].removesuffix(".US")
        if symbol in self.fail_codes:
            return httpx.Response(500, text="boom")
        if kind in ("splits", "div") and self.free:
            return httpx.Response(403, text="Only EOD data allowed for free users.")
        if symbol not in BARS:
            return httpx.Response(404, text="Ticker Not Found.")
        if kind == "eod":
            return httpx.Response(200, json=BARS[symbol] + ([{"warning": "Data is limited by one year"}] if self.free else []))
        return httpx.Response(200, json=SPLITS[symbol] if kind == "splits" else DIVIDENDS[symbol])


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> EodhdClient:
    """EODHD client on a mock transport that never sleeps."""
    return EodhdClient("k", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None)
