"""
``SharadarStore``: read-only access to the built Sharadar store, shaped like ``FmpStore``.

``bulk_daily`` / ``events`` / ``events_many`` return the same frames as ``FmpStore`` and ``dir`` holds a
``reference/securities.parquet`` with the vintage's column names, so the P24 helpers (``load_securities``,
``load_daily_batch``) and the event bench run on either store. Security ids are Sharadar ``permaticker`` values and
are **not** FMP ids; use ``fmp_security_id`` to join FMP-only data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.data.sharadar_pipeline import config
from src.data.sharadar_pipeline.convert import delisting_return

SOURCE = "sharadar"


class AmbiguousSymbol(ValueError):
    """A symbol belongs to more than one security and no date picks one."""


@dataclass(frozen=True)
class SecurityInfo:
    """One Sharadar security."""

    security_id: int
    symbol: str
    sharadar_ticker: str
    name: Optional[str]
    instrument_type: str
    cik: Optional[int]
    first_price_date: Optional[date]
    last_price_date: Optional[date]
    delisted_date: Optional[date]
    is_actively_trading: bool
    fmp_security_id: Optional[int]

    @property
    def alive(self) -> bool:
        """Still trading as far as the download knows."""
        return self.is_actively_trading

    def covers(self, when: date) -> bool:
        """True if ``when`` lies inside the price history."""
        return (self.first_price_date is None or when >= self.first_price_date) and (
            self.last_price_date is None or when <= self.last_price_date)


@dataclass(frozen=True)
class Delisting:
    """How a security stopped trading."""

    security_id: int
    date: date
    reason: str
    acquirer: Optional[str]
    cash_per_share: Optional[float]
    stock_ratio: Optional[float]
    stock_ticker: Optional[str]
    other_consideration: bool


def _none(value: Any) -> Any:
    return None if value is None or (not isinstance(value, str) and pd.isna(value)) else value


def _no_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.DataFrame({"date": pd.Series([], dtype="object"), "numerator": pd.Series([], dtype=float),
                          "denominator": pd.Series([], dtype=float)}),
            pd.DataFrame({"date": pd.Series([], dtype="object"), "amount": pd.Series([], dtype=float)}))


class SharadarStore:
    """Read-only view of ``<data_root>/store``."""

    def __init__(self, data_root: Path) -> None:
        """
        Args:
            data_root: Sharadar data root (holds ``store/``).

        Raises:
            FileNotFoundError: The store has not been built.
        """
        self.data_root = data_root
        self.dir = data_root / config.STORE_DIR
        if not (self.dir / config.SECURITIES).exists():
            raise FileNotFoundError(f"no Sharadar store under {self.dir}; run the build first")
        info_path = self.dir / config.BUILD_INFO
        self.build_info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        self._securities: dict[int, SecurityInfo] = {}
        self._by_symbol: dict[str, list[int]] = {}
        for row in pq.read_table(self.dir / config.SECURITIES).to_pylist():
            info = SecurityInfo(int(row["security_id"]), row["symbol"], row["sharadar_ticker"], row["name"],
                                row["instrument_type"], _none(row["cik"]), row["first_price_date"], row["last_price_date"],
                                row["delisted_date"], bool(row["is_actively_trading"]), _none(row["fmp_security_id"]))
            self._securities[info.security_id] = info
            for name in {info.symbol, info.sharadar_ticker}:
                self._by_symbol.setdefault(name, []).append(info.security_id)
        self._former: dict[str, list[int]] = {}
        changes = self.dir / config.SYMBOL_CHANGES
        if changes.exists():
            for row in pq.read_table(changes, columns=["security_id", "old_symbol"]).to_pylist():
                self._former.setdefault(row["old_symbol"], []).append(int(row["security_id"]))

    def security(self, security_id: int) -> SecurityInfo:
        """Return a security by id."""
        return self._securities[security_id]

    def securities(self) -> list[SecurityInfo]:
        """Return every security, by id."""
        return [self._securities[sid] for sid in sorted(self._securities)]

    @property
    def as_of(self) -> Optional[date]:
        """Latest price date in the store."""
        dates = [s.last_price_date for s in self._securities.values() if s.last_price_date is not None]
        return max(dates) if dates else None

    def resolve(self, symbol: str, as_of: Optional[date] = None) -> Optional[SecurityInfo]:
        """
        Map a symbol to a security: current or as-traded symbol first, then former symbols.

        Args:
            symbol: Symbol as traded (without Sharadar's reuse number) or the Sharadar ticker.
            as_of: Date that picks one of several securities that used the symbol.

        Raises:
            AmbiguousSymbol: Several securities match and ``as_of`` does not pick exactly one.
        """
        for index in (self._by_symbol, self._former):
            candidates = [self._securities[sid] for sid in dict.fromkeys(index.get(symbol, []))]
            if not candidates:
                continue
            if len(candidates) == 1:
                return candidates[0]
            matches = [c for c in candidates if as_of is not None and c.covers(as_of)]
            if len(matches) == 1:
                return matches[0]
            raise AmbiguousSymbol(f"{symbol!r} matches securities {[c.security_id for c in candidates]}; pass as_of to choose")
        return None

    def _price_files(self, security_ids: Iterable[int]) -> list[str]:
        buckets = sorted({sid % config.PRICE_BUCKETS for sid in security_ids})
        paths = [self.dir / config.PRICES_DIR / f"bucket={bucket:02d}.parquet" for bucket in buckets]
        return [str(path) for path in paths if path.exists()]

    def bulk_daily(self, security_ids: Iterable[int], start: date, end: date, vendor_adjusted: bool = False) -> dict[int, pd.DataFrame]:
        """
        Read many securities' raw daily bars between ``start`` and ``end``.

        Args:
            security_ids: Securities to read; those without bars are absent from the result.
            start: First date (inclusive).
            end: Last date (inclusive).
            vendor_adjusted: Also return ``closeadj_vendor`` (Sharadar's own fully adjusted close).

        Returns:
            ``{security_id: frame}`` on a ``DatetimeIndex`` named ``timestamp`` with raw ``open`` … ``volume``,
            ``flags`` (always 0) and ``source``.
        """
        import duckdb  # deferred, as in FmpStore

        wanted = list(dict.fromkeys(int(sid) for sid in security_ids))
        files = self._price_files(wanted)
        if not files:
            return {}
        extra = ", closeadj_vendor" if vendor_adjusted else ""
        query = (f"SELECT security_id, date, open, high, low, close, volume{extra} FROM read_parquet(?) "
                 "WHERE security_id IN (SELECT unnest(?)) AND date BETWEEN ? AND ? ORDER BY security_id, date")
        with duckdb.connect() as con:
            df = con.execute(query, [files, wanted, start, end]).df()
        out: dict[int, pd.DataFrame] = {}
        for _, part in df.groupby("security_id", sort=True):
            sid = int(part["security_id"].iloc[0])
            frame = part.drop(columns=["security_id"])
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"))).rename("timestamp")
            frame["flags"] = 0
            frame["source"] = SOURCE
            out[sid] = frame
        return out

    def prices(self, security_id: int, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """Return one security's full raw daily history, or None (only ``1d`` exists)."""
        if timeframe != "1d":
            return None
        return self.bulk_daily([security_id], date(1900, 1, 1), date(2999, 1, 1)).get(security_id)

    def events_many(self, security_ids: Iterable[int], spinoffs: bool = True) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Return the events of many securities in ``adjust()`` form.

        Args:
            security_ids: Securities to read; those without events are absent.
            spinoffs: Include spinoff dividends in the dividend frame (total return then covers spinoffs).

        Returns:
            ``{security_id: (splits[date, numerator, denominator], dividends[date, amount])}``.
        """
        wanted = sorted(set(int(sid) for sid in security_ids))
        if not wanted:
            return {}
        filters = [("security_id", "in", wanted)]
        splits = pq.read_table(self.dir / config.SPLITS, columns=["security_id", "date", "numerator", "denominator"],
                               filters=filters).to_pandas()
        dividends = pq.read_table(self.dir / config.DIVIDENDS, columns=["security_id", "date", "amount", "kind"],
                                  filters=filters).to_pandas()
        if not spinoffs:
            dividends = dividends[dividends["kind"] == "cash"]
        dividends = dividends.groupby(["security_id", "date"], as_index=False)["amount"].sum()
        split_groups = {int(k): g for k, g in splits.groupby("security_id")}
        dividend_groups = {int(k): g for k, g in dividends.groupby("security_id")}
        out = {}
        for sid in split_groups.keys() | dividend_groups.keys():
            empty_s, empty_d = _no_events()
            s, d = split_groups.get(sid), dividend_groups.get(sid)
            out[sid] = (empty_s if s is None else s.drop(columns=["security_id"]).reset_index(drop=True),
                        empty_d if d is None else d.drop(columns=["security_id"]).reset_index(drop=True))
        return out

    def events(self, security_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return one security's ``(splits, dividends)`` in ``adjust()`` form."""
        return self.events_many([security_id]).get(security_id) or _no_events()

    def fundamentals(self, security_ids: Optional[Iterable[int]] = None, dimension: str = "ARQ",
                     columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Return as-reported fundamentals.

        Args:
            security_ids: Securities to read; None reads all.
            dimension: ``ARQ``, ``ART`` or ``ARY``.
            columns: Indicator columns to return; keys and dates are always included.

        Returns:
            Rows sorted by ``security_id`` and filing date. Use ``available_from`` as the first usable day.
        """
        path = self.dir / config.FUNDAMENTALS
        if not path.exists():
            return pd.DataFrame()
        filters: list[tuple[str, str, Any]] = [("dimension", "==", dimension)]
        if security_ids is not None:
            filters.append(("security_id", "in", sorted(set(int(sid) for sid in security_ids))))
        keys = ["security_id", "dimension", "date", "reportperiod", "calendardate", "available_from"]
        wanted = None if columns is None else keys + [c for c in columns if c not in keys]
        frame = pq.read_table(path, columns=wanted, filters=filters).to_pandas()
        return frame.sort_values(["security_id", "date", "reportperiod"]).reset_index(drop=True)

    def delisting(self, security_id: int) -> Optional[Delisting]:
        """Return how a security stopped trading, or None if it is active or has no delisting row."""
        rows = pq.read_table(self.dir / config.DELISTINGS, filters=[("security_id", "==", int(security_id))]).to_pylist()
        if not rows:
            return None
        row = rows[-1]
        return Delisting(int(row["security_id"]), row["date"], row["reason"], _none(row["acquirer"]), _none(row["cash_per_share"]),
                         _none(row["stock_ratio"]), _none(row["stock_ticker"]), bool(row["other_consideration"]))

    def delisting_return(self, security_id: int) -> Optional[float]:
        """
        Return the holder's return from the last raw close to the deal consideration.

        Cash, and stock priced at the acquirer's raw close on the last trade date. None when the security has no
        delisting row, the consideration is unknown or of a kind this store does not interpret, or the acquirer has
        no price that day.
        """
        event = self.delisting(security_id)
        if event is None or event.other_consideration:
            return None
        bars = self.bulk_daily([security_id], date(1900, 1, 1), event.date).get(security_id)
        if bars is None or bars.empty:
            return None
        stock_price = None
        if event.stock_ratio is not None and event.stock_ticker:
            try:
                acquirer = self.resolve(event.stock_ticker, event.date)
            except AmbiguousSymbol:
                acquirer = None
            if acquirer is not None:
                quote = self.bulk_daily([acquirer.security_id], event.date, event.date).get(acquirer.security_id)
                stock_price = float(quote["close"].iloc[-1]) if quote is not None and len(quote) else None
        return delisting_return(float(bars["close"].iloc[-1]), event.cash_per_share, event.stock_ratio, stock_price)
