"""
EODHD HTTP client: symbol lists, end-of-day bars, splits, dividends and the account's quota.

Every request costs one API call, a 404 included. The client counts calls, keeps below the per-minute limit and
turns EODHD's answers into three exceptions the downloader acts on:

- ``QuotaExhausted`` (HTTP 402, or 429 after the retries): stop for the day.
- ``PlanRestricted`` (HTTP 403): the plan does not include this endpoint (the free plan serves bars only).
- ``EodhdError``: anything else that is not a success.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Optional

import httpx
import pandas as pd

from src.data.eodhd_pipeline.config import API_BASE, EXCHANGE

_RETRY_STATUS = (429, 500, 502, 503, 504)
_ATTEMPTS = 4


class EodhdError(RuntimeError):
    """EODHD answered with an error."""


class QuotaExhausted(EodhdError):
    """The daily (or minute) call quota is used up."""


class PlanRestricted(EodhdError):
    """The plan of this key does not include the endpoint."""


@dataclass(frozen=True)
class EodHistory:
    """Daily bars of one EODHD symbol."""

    code: str
    bars: pd.DataFrame
    plan_limited: bool


class _Pacer:
    """Allow at most ``per_minute`` calls in any 60-second window (thread-safe)."""

    def __init__(self, per_minute: Optional[int], sleep: Callable[[float], None], clock: Callable[[], float]) -> None:
        self._per_minute = per_minute
        self._sleep = sleep
        self._clock = clock
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self._per_minute:
            return
        while True:
            with self._lock:
                now = self._clock()
                while self._stamps and now - self._stamps[0] >= 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self._per_minute:
                    self._stamps.append(now)
                    return
                delay = 60.0 - (now - self._stamps[0])
            self._sleep(max(delay, 0.01))


class EodhdClient:
    """Thread-safe EODHD client (one ``httpx.Client`` shared by all threads)."""

    def __init__(self, api_key: str, client: Optional[httpx.Client] = None, sleep: Callable[[float], None] = time.sleep,
                 calls_per_minute: Optional[int] = None, clock: Callable[[], float] = time.monotonic) -> None:
        """
        Args:
            api_key: EODHD API token.
            client: Existing ``httpx.Client`` (tests pass a mock transport).
            sleep: Wait function used between retries and by the pacer.
            calls_per_minute: Pace requests to at most this many a minute; None does not pace.
            clock: Monotonic clock of the pacer.
        """
        self._key = api_key
        self._client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0))
        self._sleep = sleep
        self._pacer = _Pacer(calls_per_minute, sleep, clock)
        self._lock = threading.Lock()
        self.calls = 0

    def close(self) -> None:
        """Close the connection pool."""
        self._client.close()

    def _count(self) -> None:
        with self._lock:
            self.calls += 1

    def _get(self, path: str, params: Optional[dict[str, Any]] = None, counted: bool = True) -> Any:
        """Return the JSON payload, or None on 404."""
        for attempt in range(_ATTEMPTS):
            self._pacer.wait()
            if counted:
                self._count()
            response = self._client.get(f"{API_BASE}{path}", params={"api_token": self._key, "fmt": "json", **(params or {})})
            status = response.status_code
            if status == 404:
                return None
            if status == 402:
                raise QuotaExhausted(f"{path}: HTTP 402 {response.text[:200]}")
            if status == 403:
                raise PlanRestricted(f"{path}: HTTP 403 {response.text[:200]}")
            if status in _RETRY_STATUS and attempt < _ATTEMPTS - 1:
                self._sleep(2.0 ** attempt * (5.0 if status == 429 else 1.0))
                continue
            if status == 429:
                raise QuotaExhausted(f"{path}: HTTP 429 after {_ATTEMPTS} attempts")
            if status >= 400:
                raise EodhdError(f"{path}: HTTP {status} {response.text[:200]}")
            return response.json()
        raise EodhdError(f"{path}: retries exhausted")

    def user(self) -> dict[str, Any]:
        """Return the account record (``subscriptionType``, ``dailyRateLimit``, ``apiRequests``, ``extraLimit``)."""
        return dict(self._get("/user", counted=False) or {})

    def calls_left(self) -> Optional[int]:
        """Return the calls left today (daily limit plus extra calls minus calls made), or None if unknown."""
        record = self.user()
        try:
            return int(record["dailyRateLimit"]) + int(record.get("extraLimit") or 0) - int(record["apiRequests"])
        except (KeyError, TypeError, ValueError):
            return None

    def symbols(self, exchange: str = EXCHANGE, delisted: bool = False) -> pd.DataFrame:
        """Return an exchange's symbol list (``Code``, ``Name``, ``Country``, ``Exchange``, ``Currency``, ``Type``, ``Isin``)."""
        return pd.DataFrame(self._get(f"/exchange-symbol-list/{exchange}", {"delisted": 1} if delisted else {}) or [])

    def delisted_symbols(self, exchange: str = EXCHANGE) -> pd.DataFrame:
        """Return an exchange's delisted symbols."""
        return self.symbols(exchange, delisted=True)

    def eod_rows(self, code: str, start: Optional[date] = None, end: Optional[date] = None,
                 exchange: str = EXCHANGE) -> tuple[Optional[list[dict[str, Any]]], bool]:
        """
        Return a symbol's daily bars as delivered, and whether the plan cut the history.

        Returns:
            ``(rows, plan_limited)``; rows is None when EODHD does not know the symbol.
        """
        params = {key: value.isoformat() for key, value in (("from", start), ("to", end)) if value is not None}
        payload = self._get(f"/eod/{code}.{exchange}", params)
        if payload is None:
            return None, False
        rows = payload if isinstance(payload, list) else []
        limited = any(isinstance(row, dict) and "warning" in row for row in rows)
        return [row for row in rows if isinstance(row, dict) and "date" in row], limited

    def eod(self, code: str, start: Optional[date] = None, end: Optional[date] = None, exchange: str = EXCHANGE) -> EodHistory:
        """
        Return a symbol's daily bars between ``start`` and ``end`` (whole history when both are None).

        Returns:
            Bars with ``date``, ``open``, ``high``, ``low``, ``close`` (as traded), ``adjusted_close``, ``volume``;
            empty when EODHD has none. ``plan_limited`` is True when EODHD says the plan cut the history.
        """
        rows, limited = self.eod_rows(code, start, end, exchange)
        bars = pd.DataFrame(rows or [])
        if len(bars):
            bars["date"] = pd.to_datetime(bars["date"]).dt.date
        return EodHistory(code, bars, limited)

    def splits(self, code: str, exchange: str = EXCHANGE) -> list[dict[str, Any]]:
        """Return a symbol's splits as delivered (``date``, ``split`` = ``"new/old"``); empty if none or unknown."""
        payload = self._get(f"/splits/{code}.{exchange}")
        return [row for row in payload or [] if isinstance(row, dict)]

    def dividends(self, code: str, exchange: str = EXCHANGE) -> list[dict[str, Any]]:
        """Return a symbol's dividends as delivered (``date`` = ex-date, ``value``, ``unadjustedValue``, ``currency``, …)."""
        payload = self._get(f"/div/{code}.{exchange}")
        return [row for row in payload or [] if isinstance(row, dict)]
