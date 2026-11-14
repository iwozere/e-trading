"""
Measure how deep EODHD's delisted price history is, using Sharadar's free ticker list as the yardstick.

EODHD's terms let a private user keep downloaded data, so it is a candidate for a kept delisted-price layer
(docs/Findings.md §7). Its delisted symbol list is long, but a symbol in a list proves nothing about its history.
This check takes a stratified sample of Sharadar's delisted common stocks (their first and last price dates come
with the free ``tickers`` table), finds each one in EODHD by ticker and company name, downloads its daily bars and
compares the date range and bar count.

A paid EODHD plan is required: the free plan serves one year of prices only. One sampled stock costs one API call
per EODHD candidate (at most ``MAX_CANDIDATES``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Optional

import httpx
import numpy as np
import pandas as pd

from src.data.sharadar_pipeline.convert import securities_frame
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

EODHD_BASE = "https://eodhd.com/api"
HISTORY_START = date(1997, 12, 1)
#: Sharadar prices start here; a stock listed earlier can only be compared from this date on.
YARDSTICK_START = date(1997, 12, 31)
#: Last-price-year buckets of the sample.
BUCKETS = ((1998, 2002), (2003, 2007), (2008, 2011), (2012, 2016), (2017, 2020), (2021, 2100))
MAX_CANDIDATES = 2
#: A range end counts as matching when it is within this many calendar days.
DATE_TOLERANCE_DAYS = 10
#: Share of the expected sessions that must have a bar for a history to count as full.
MIN_BAR_SHARE = 0.9
#: Company names agree when their token overlap is at least this.
MIN_NAME_SIMILARITY = 0.5

_NAME_NOISE = {"inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "holdings", "holding", "group", "the", "sa",
               "nv", "ag", "lp", "llc", "trust", "class", "a", "b", "common", "stock", "new", "de", "of", "and"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EodhdError(RuntimeError):
    """EODHD answered with an error."""


@dataclass(frozen=True)
class EodHistory:
    """Daily bars of one EODHD symbol."""

    code: str
    bars: pd.DataFrame
    plan_limited: bool


class EodhdClient:
    """Minimal EODHD client: the delisted symbol list and end-of-day bars."""

    def __init__(self, api_key: str, client: Optional[httpx.Client] = None, sleep: Callable[[float], None] = lambda _: None) -> None:
        """
        Args:
            api_key: EODHD API token.
            client: Existing ``httpx.Client`` (tests pass a mock transport).
            sleep: Wait function used between retries.
        """
        self._key = api_key
        self._client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0))
        self._sleep = sleep
        self.calls = 0

    def close(self) -> None:
        """Close the connection pool."""
        self._client.close()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        for attempt in range(4):
            self.calls += 1
            response = self._client.get(f"{EODHD_BASE}{path}", params={"api_token": self._key, "fmt": "json", **params})
            if response.status_code == 404:
                return None
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                self._sleep(2.0 ** attempt)
                continue
            if response.status_code >= 400:
                raise EodhdError(f"{path}: HTTP {response.status_code} {response.text[:200]}")
            return response.json()
        raise EodhdError(f"{path}: retries exhausted")

    def delisted_symbols(self, exchange: str = "US") -> pd.DataFrame:
        """Return EODHD's delisted symbols of an exchange (``Code``, ``Name``, ``Exchange``, ``Type``, …)."""
        return pd.DataFrame(self._get(f"/exchange-symbol-list/{exchange}", {"delisted": 1}) or [])

    def eod(self, code: str, start: date, end: date, exchange: str = "US") -> EodHistory:
        """
        Return a symbol's daily bars between ``start`` and ``end``.

        Returns:
            Bars with ``date``, ``open``, ``high``, ``low``, ``close`` (as traded), ``adjusted_close``, ``volume``;
            empty when EODHD has none. ``plan_limited`` is True when EODHD says the plan cut the history.
        """
        payload = self._get(f"/eod/{code}.{exchange}", {"from": start.isoformat(), "to": end.isoformat()}) or []
        limited = any(isinstance(row, dict) and "warning" in row for row in payload)
        rows = [row for row in payload if isinstance(row, dict) and "date" in row]
        bars = pd.DataFrame(rows)
        if len(bars):
            bars["date"] = pd.to_datetime(bars["date"]).dt.date
        return EodHistory(code, bars, limited)


def name_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Return the token overlap (Jaccard) of two company names after dropping legal-form words; 0 if either is empty."""
    left = {t for t in _TOKEN_RE.findall((a or "").lower()) if t not in _NAME_NOISE}
    right = {t for t in _TOKEN_RE.findall((b or "").lower()) if t not in _NAME_NOISE}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def delisted_universe(tickers: pd.DataFrame) -> pd.DataFrame:
    """
    Return Sharadar's delisted common stocks with a known price range.

    Args:
        tickers: Sharadar ``tickers`` table (free sample).

    Returns:
        ``security_id``, ``symbol``, ``name``, ``first_price_date``, ``last_price_date``, ``bucket``.
    """
    securities = securities_frame(tickers)
    rows = securities[(securities["instrument_type"] == "stock") & ~securities["is_actively_trading"]].copy()
    rows = rows.dropna(subset=["first_price_date", "last_price_date"])
    year = pd.to_datetime(rows["last_price_date"]).dt.year
    rows["bucket"] = ""
    for low, high in BUCKETS:
        rows.loc[year.between(low, high), "bucket"] = f"{low}-{min(high, 2099)}" if high < 2100 else f"{low}+"
    rows = rows[rows["bucket"] != ""]
    return rows[["security_id", "symbol", "name", "first_price_date", "last_price_date", "bucket"]].reset_index(drop=True)


def sample_universe(universe: pd.DataFrame, per_bucket: Optional[int], seed: int = 7) -> pd.DataFrame:
    """Draw up to ``per_bucket`` stocks from every bucket (all of them when ``per_bucket`` is None), reproducibly."""
    if per_bucket is None:
        return universe.reset_index(drop=True)
    parts = [group.sample(n=min(per_bucket, len(group)), random_state=seed) for _, group in universe.groupby("bucket", sort=True)]
    return pd.concat(parts).sort_values(["bucket", "symbol"]).reset_index(drop=True)


def find_candidates(symbol: str, name: Optional[str], listed: pd.DataFrame) -> list[tuple[str, float]]:
    """
    Return EODHD codes that could be the stock: same ticker (EODHD marks reuse as ``ABC_old``), best name match first.

    Args:
        symbol: Ticker as traded (Sharadar ``.`` class separator is ``-`` at EODHD).
        name: Sharadar company name.
        listed: EODHD delisted list with a ``base`` column (code before ``_``, upper case).

    Returns:
        Up to ``MAX_CANDIDATES`` ``(code, name_similarity)`` pairs.
    """
    rows = listed[listed["base"] == symbol.replace(".", "-").upper()]
    scored = sorted(((code, name_similarity(name, other)) for code, other in zip(rows["Code"], rows["Name"])),
                    key=lambda pair: (-pair[1], pair[0]))
    return scored[:MAX_CANDIDATES]


def _expected_sessions(first: date, last: date) -> int:
    return max(int(np.busday_count(first, last + timedelta(days=1)) * 0.965), 1)  # about 9 market holidays a year


def grade(first: date, last: date, bars: pd.DataFrame) -> dict[str, Any]:
    """
    Compare an EODHD history with the expected price range.

    Returns:
        ``bars``, ``first``, ``last``, ``start_gap_days`` (EODHD starts this many days late), ``end_gap_days``
        (EODHD ends this many days early), ``bar_share`` and ``status``: ``full``, ``partial`` or ``none``.
    """
    start = max(first, YARDSTICK_START)
    if bars.empty:
        return {"bars": 0, "first": None, "last": None, "start_gap_days": None, "end_gap_days": None, "bar_share": 0.0, "status": "none"}
    inside = bars[(bars["date"] >= start) & (bars["date"] <= last)]
    got_first, got_last = bars["date"].min(), bars["date"].max()
    start_gap, end_gap = (got_first - start).days, (last - got_last).days
    share = min(len(inside) / _expected_sessions(start, last), 1.0)
    full = start_gap <= DATE_TOLERANCE_DAYS and end_gap <= DATE_TOLERANCE_DAYS and share >= MIN_BAR_SHARE
    return {"bars": len(inside), "first": got_first, "last": got_last, "start_gap_days": start_gap, "end_gap_days": end_gap,
            "bar_share": round(share, 3), "status": "full" if full else "partial" if len(inside) else "none"}


def run_check(client: EodhdClient, tickers: pd.DataFrame, per_bucket: Optional[int] = 100, seed: int = 7,
              max_calls: Optional[int] = None, fmp_closes: Optional[Callable[[int, date, date], Optional[pd.Series]]] = None) -> pd.DataFrame:
    """
    Check EODHD's history for a sample of Sharadar's delisted common stocks.

    Args:
        client: EODHD client holding a paid key.
        tickers: Sharadar ``tickers`` table.
        per_bucket: Stocks per last-price-year bucket; None checks every delisted common stock.
        seed: Sampling seed.
        max_calls: Stop when the client has made this many API calls.
        fmp_closes: Optional ``(sharadar_security_id, start, end) -> raw closes by date`` used to compare price
            levels where the FMP vintage has the stock.

    Returns:
        One row per sampled stock: Sharadar range, the chosen EODHD code, name similarity, ``grade`` fields and, when
        FMP closes exist, ``fmp_days`` and ``fmp_close_ok`` (share of common days within 1%).
    """
    listed = client.delisted_symbols()
    if listed.empty:
        raise EodhdError("EODHD returned no delisted symbols")
    listed = listed[listed["Type"] == "Common Stock"].copy()
    listed["base"] = listed["Code"].str.split("_").str[0].str.upper()
    sample = sample_universe(delisted_universe(tickers), per_bucket, seed)
    rows: list[dict[str, Any]] = []
    limited = False
    for record in sample.to_dict("records"):
        if max_calls is not None and client.calls >= max_calls:
            _logger.warning("Stopped at %d API calls; %d of %d stocks checked", client.calls, len(rows), len(sample))
            break
        first, last = record["first_price_date"], record["last_price_date"]
        best: dict[str, Any] = {"eodhd_code": None, "name_similarity": 0.0, **grade(first, last, pd.DataFrame())}
        best_bars = pd.DataFrame()
        for code, similarity in find_candidates(record["symbol"], record["name"], listed):
            history = client.eod(code, HISTORY_START, last + timedelta(days=30))
            limited = limited or history.plan_limited
            result = {"eodhd_code": code, "name_similarity": round(similarity, 2), **grade(first, last, history.bars)}
            if best["eodhd_code"] is None or (result["status"] == "full", result["bars"]) > (best["status"] == "full", best["bars"]):
                best, best_bars = result, history.bars
            if best["status"] == "full":
                break
        row = {**record, **best, "name_ok": best["name_similarity"] >= MIN_NAME_SIMILARITY}
        if fmp_closes is not None and len(best_bars):
            row.update(_compare_fmp(best_bars, fmp_closes(int(record["security_id"]), max(first, YARDSTICK_START), last)))
        rows.append(row)
    if limited:
        _logger.warning("EODHD cut the history to the plan's limit: this key is on a free or restricted plan and the result is not valid")
    out = pd.DataFrame(rows)
    out.attrs["plan_limited"] = limited
    return out


def _compare_fmp(bars: pd.DataFrame, closes: Optional[pd.Series]) -> dict[str, Any]:
    if closes is None or closes.empty:
        return {}
    ours = pd.Series(pd.to_numeric(bars["close"], errors="coerce").to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(bars["date"])))
    joined = pd.concat([ours.rename("eodhd"), closes.rename("fmp")], axis=1, join="inner").dropna()
    joined = joined[(joined["eodhd"] > 0) & (joined["fmp"] > 0)]
    if joined.empty:
        return {}
    return {"fmp_days": len(joined), "fmp_close_ok": round(float(((joined["eodhd"] / joined["fmp"] - 1).abs() < 0.01).mean()), 3)}


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the check to one row per bucket.

    Returns:
        ``stocks``, ``found`` (an EODHD candidate exists), ``full``, ``partial``, ``none`` (shares of ``stocks``),
        ``full_name_ok`` (full history **and** matching company name — the number to trust), ``median_bar_share``
        and, when present, ``fmp_pairs`` and ``fmp_close_ok``.
    """
    if result.empty:
        return pd.DataFrame()
    rows = []
    groups = list(result.groupby("bucket", sort=True)) + [("all", result)]
    for bucket, part in groups:
        row = {"bucket": bucket, "stocks": len(part), "found": round(float(part["eodhd_code"].notna().mean()), 3)}
        for status in ("full", "partial", "none"):
            row[status] = round(float((part["status"] == status).mean()), 3)
        row["full_name_ok"] = round(float(((part["status"] == "full") & part["name_ok"]).mean()), 3)
        row["median_bar_share"] = round(float(part["bar_share"].median()), 3)
        if "fmp_close_ok" in part.columns and part["fmp_close_ok"].notna().any():
            compared = part["fmp_close_ok"].dropna()
            row["fmp_pairs"] = len(compared)
            row["fmp_close_ok"] = round(float((compared >= 0.99).mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)
