"""
``FmpStore``: read-only access to the active FMP vintage (spec §9.2 layer 1).

Reads ``current.txt`` once at construction and stays on that vintage, so one store never mixes two vintages.
Securities and symbol history are small and loaded eagerly; prices and events are read per security on demand
(one Parquet file per security, events filtered by ``security_id`` with row-group statistics), and many
securities' daily bars at once with a single DuckDB query.

**Symbol resolution** (spec §9.6). In order: the security's own ``symbol``, its ``yf_symbol``, the same two with
the class-share separator swapped (``BRK-B`` ↔ ``BRK.B``), then ``symbol_history``. More than one security
without an ``as_of`` that picks exactly one raises ``AmbiguousSymbol`` — never a silent guess.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.builder import HISTORY, read_current
from src.data.fmp_pipeline.builders.events import DIVIDENDS, SPLITS
from src.data.fmp_pipeline.builders.prices import INTRADAY_DIR, bucket_path
from src.data.fmp_pipeline.builders.validate import SECURITIES

#: DataManager timeframe -> vintage price folder.
TIMEFRAME_KINDS = {"1d": "1d_raw", **{v: f"{v}_raw" for v in INTRADAY_DIR.values()}}
PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "flags", "source"]


class AmbiguousSymbol(ValueError):
    """A symbol maps to more than one security and nothing disambiguates it."""


@dataclass(frozen=True)
class SecurityInfo:
    """What the adapter needs to know about one security."""

    security_id: int
    symbol: str
    first_price_date: Optional[date]
    last_price_date: Optional[date]
    valid_from: Optional[date]
    valid_to: Optional[date]

    def covers(self, when: date) -> bool:
        """True if ``when`` lies in the security's validity range (segments of a reused ticker)."""
        return (self.valid_from is None or when >= self.valid_from) and (self.valid_to is None or when <= self.valid_to)


def _date(value: Any) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    return value if isinstance(value, date) else None


def _no_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.DataFrame({"date": pd.Series([], dtype="object"), "numerator": pd.Series([], dtype=float),
                          "denominator": pd.Series([], dtype=float)}),
            pd.DataFrame({"date": pd.Series([], dtype="object"), "amount": pd.Series([], dtype=float)}))


def _swap_separator(symbol: str) -> Optional[str]:
    if "-" in symbol:
        return symbol.replace("-", ".")
    if "." in symbol:
        return symbol.replace(".", "-")
    return None


class FmpStore:
    """Read-only view of one vintage."""

    def __init__(self, data_root: Path, vintage: Optional[str] = None) -> None:
        """
        Args:
            data_root: Pipeline data root (holds ``current.txt`` and ``vintages/``).
            vintage: Dump id to open; default: the one ``current.txt`` names.

        Raises:
            FileNotFoundError: No active vintage, or its files are missing.
        """
        vintage = vintage or read_current(data_root)
        if vintage is None:
            raise FileNotFoundError(f"no active FMP vintage ({data_root / 'current.txt'} missing or empty)")
        self.vintage = vintage
        self.dir = data_root / "vintages" / vintage
        if not (self.dir / SECURITIES).exists():
            raise FileNotFoundError(f"vintage {vintage} has no {SECURITIES} under {self.dir}")
        info_path = self.dir / "_meta" / "build_info.json"
        self.build_info: dict[str, Any] = (json.loads(info_path.read_text(encoding="utf-8"))
                                           if info_path.exists() else {})
        self.intraday_tz: str = self.build_info.get("build_config", {}).get("intraday_tz", "America/New_York")
        self._securities: dict[int, SecurityInfo] = {}
        self._by_symbol: dict[str, list[int]] = {}
        self._history: dict[str, list[tuple[int, Optional[date], Optional[date]]]] = {}
        self._load_reference()

    def _load_reference(self) -> None:
        columns = ["security_id", "symbol", "yf_symbol", "first_price_date", "last_price_date", "valid_from",
                   "valid_to"]
        for row in pq.read_table(self.dir / SECURITIES, columns=columns).to_pylist():
            info = SecurityInfo(int(row["security_id"]), row["symbol"], _date(row["first_price_date"]),
                                _date(row["last_price_date"]), _date(row["valid_from"]), _date(row["valid_to"]))
            self._securities[info.security_id] = info
            for name in {row["symbol"], row["yf_symbol"]} - {None}:
                self._by_symbol.setdefault(name, []).append(info.security_id)
        if (self.dir / HISTORY).exists():
            for row in pq.read_table(self.dir / HISTORY).to_pylist():
                if row["reason"] != "current":
                    self._history.setdefault(row["symbol"], []).append(
                        (int(row["security_id"]), _date(row["valid_from"]), _date(row["valid_to"])))

    def security(self, security_id: int) -> SecurityInfo:
        """Return a security by id."""
        return self._securities[security_id]

    def resolve(self, symbol: str, as_of: Optional[date] = None) -> Optional[SecurityInfo]:
        """
        Map a caller's symbol to a security.

        Args:
            symbol: Symbol as the caller passes it (yfinance convention, ``^`` for indices).
            as_of: Date that picks one of several securities (reused ticker, renamed symbol).

        Returns:
            The security, or ``None`` if the vintage does not know the symbol.

        Raises:
            AmbiguousSymbol: Several securities match and ``as_of`` does not pick exactly one.
        """
        for name in (symbol, _swap_separator(symbol)):
            if name is None or name not in self._by_symbol:
                continue
            candidates = [self._securities[sid] for sid in dict.fromkeys(self._by_symbol[name])]
            if len(candidates) == 1:
                return candidates[0]
            matches = [c for c in candidates if as_of is not None and c.covers(as_of)]
            if len(matches) == 1:
                return matches[0]
            raise AmbiguousSymbol(f"{symbol!r} matches {len(candidates)} securities in FMP vintage {self.vintage} "
                                  f"({', '.join(str(c.security_id) for c in candidates)}); pass as_of to choose")
        rows = self._history.get(symbol, [])
        if as_of is not None:
            rows = [r for r in rows if (r[1] is None or as_of >= r[1]) and (r[2] is None or as_of <= r[2])]
        owners = sorted({r[0] for r in rows})
        if len(owners) > 1:
            raise AmbiguousSymbol(f"former symbol {symbol!r} belonged to securities {owners} in FMP vintage "
                                  f"{self.vintage}; pass as_of to choose")
        return self._securities[owners[0]] if owners else None

    def price_path(self, security_id: int, timeframe: str) -> Optional[Path]:
        """Return the security's price file for ``timeframe``, or ``None`` if the vintage has none."""
        kind = TIMEFRAME_KINDS.get(timeframe)
        if kind is None:
            return None
        path = self.dir / bucket_path(kind, security_id)
        return path if path.exists() else None

    def prices(self, security_id: int, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Read a security's raw bars.

        Returns:
            Frame on a tz-naive ``DatetimeIndex`` (daily: the date; intraday: the UTC bar start) with ``open`` …
            ``volume``, ``flags``, ``source``, plus ``date_et`` for intraday; ``None`` if there is no file.
        """
        path = self.price_path(security_id, timeframe)
        if path is None:
            return None
        df = pq.read_table(path).to_pandas()
        key = "date" if timeframe == "1d" else "ts_utc"
        index = pd.DatetimeIndex(pd.to_datetime(df.pop(key)))
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        df.index = index.rename("timestamp")
        return df.drop(columns=["security_id", "vintage"])

    def events(self, security_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return the security's events in ``adjust()`` form.

        Returns:
            ``(splits[date, numerator, denominator], dividends[date, amount])``; rows without a usable ratio or
            amount are left out.
        """
        return self.events_many([security_id]).get(security_id) or _no_events()

    def events_many(self, security_ids: Iterable[int]) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Return the events of many securities with one read per table (``events`` shape per security).

        Securities without any event are absent.
        """
        wanted = sorted(set(security_ids))
        if not wanted:
            return {}
        filters = [("security_id", "in", wanted)]
        splits = pq.read_table(self.dir / SPLITS.path,
                               columns=["security_id", "date", "numerator", "denominator", "ratio"],
                               filters=filters).to_pandas()
        splits = splits[splits["ratio"].notna()].drop(columns=["ratio"])
        dividends = pq.read_table(self.dir / DIVIDENDS.path, columns=["security_id", "ex_date", "amount_raw"],
                                  filters=filters).to_pandas()
        dividends = dividends[dividends["amount_raw"].notna() & dividends["ex_date"].notna()]
        dividends = dividends.rename(columns={"ex_date": "date", "amount_raw": "amount"})
        split_groups = {int(k): g for k, g in splits.groupby("security_id")}
        dividend_groups = {int(k): g for k, g in dividends.groupby("security_id")}
        out = {}
        for sid in split_groups.keys() | dividend_groups.keys():
            s, d = split_groups.get(sid), dividend_groups.get(sid)
            empty_s, empty_d = _no_events()
            out[sid] = (empty_s if s is None else s.drop(columns=["security_id"]).reset_index(drop=True),
                        empty_d if d is None else d.drop(columns=["security_id"]).reset_index(drop=True))
        return out

    def bulk_daily(self, security_ids: Iterable[int], start: date, end: date) -> dict[int, pd.DataFrame]:
        """
        Read many securities' raw daily bars between ``start`` and ``end`` with one DuckDB query (spec §9.5).

        Args:
            security_ids: Securities to read; those without a daily file are skipped.
            start: First date (inclusive).
            end: Last date (inclusive).

        Returns:
            ``{security_id: frame}`` in the ``prices`` shape.
        """
        import duckdb  # deferred: only readers with the adapter enabled need it

        paths = [(sid, self.price_path(sid, "1d")) for sid in dict.fromkeys(security_ids)]
        files = [str(p) for _, p in paths if p is not None]
        if not files:
            return {}
        query = ("SELECT security_id, date, open, high, low, close, volume, flags, source "
                 "FROM read_parquet(?) WHERE date BETWEEN ? AND ? ORDER BY security_id, date")
        with duckdb.connect() as con:
            df = con.execute(query, [files, start, end]).df()
        out: dict[int, pd.DataFrame] = {}
        for _, part in df.groupby("security_id", sort=True):
            sid = int(part["security_id"].iloc[0])
            frame = part.drop(columns=["security_id"])
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"))).rename("timestamp")
            out[sid] = frame
        return out

    def events_hash(self, splits: pd.DataFrame, dividends: pd.DataFrame) -> str:
        """Identify an event set together with the vintage (part of the adjusted-cache key, spec §9.5)."""
        digest = hashlib.sha256(self.vintage.encode("utf-8"))
        for frame in (splits, dividends):
            digest.update(frame.sort_values("date").to_csv(index=False).encode("utf-8"))
        return digest.hexdigest()[:16]
