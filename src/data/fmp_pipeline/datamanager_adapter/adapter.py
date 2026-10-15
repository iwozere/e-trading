"""
``FmpOhlcvAdapter``: OHLCV from the FMP vintage (plus increments) in the ``DataManager`` contract (spec §9).

**Resolution policy ``fmp_first``** (spec §9.2): a symbol the vintage knows, in a timeframe the vintage has a file
for, is served from the vintage's raw bars up to its last bar, then increment bars strictly after it; adjustment
is applied to the combined raw series with the union of both event sets (FMP wins on a shared date), always by
``adjust()`` — FMP's or yfinance's adjusted values are never spliced. Anything else returns ``None`` so the
caller falls through to the legacy cache and the existing cache-on-miss loader, unchanged.

**Output** (``src.data.ohlcv_contract``): tz-naive ``DatetimeIndex`` named ``timestamp`` (daily: the date;
intraday: the UTC bar start, like the legacy files), columns ``open``, ``high``, ``low``, ``close``, ``volume``.
``adjust="legacy"`` is ``total`` (Finding A, D12/Q5): what the legacy files hold. ``source`` and ``flags`` columns
and ``exclude_suspect`` are opt-in (spec §9.3). The range is inclusive at both ends, like ``UnifiedCache.get``.

**Caching** (spec §9.5): adjusted full histories are kept in a small in-memory LRU keyed by vintage, security,
timeframe, mode, a hash of the events used and the increment tail, so a new dividend or a new vintage can never
be served from a stale entry. Nothing FMP-derived is ever written into the legacy cache.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.fmp_pipeline.adjust.adjust import AdjustMode, adjust
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore, SecurityInfo
from src.data.fmp_pipeline.datamanager_adapter.increments import IncrementSource, NoIncrements
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

ADJUST_MODES = ("legacy", "raw", "split", "total")
OHLCV = ["open", "high", "low", "close", "volume"]
#: Bars before ``start`` that a bulk read also fetches: the ``C_prev`` of a dividend just after ``start``.
BULK_LOOKBACK = timedelta(days=10)


def adjust_mode(adjust_value: str) -> AdjustMode:
    """Map a ``DataManager`` ``adjust`` value to an ``adjust()`` mode (``legacy`` → ``total``, D12)."""
    if adjust_value not in ADJUST_MODES:
        raise ValueError(f"adjust must be one of {ADJUST_MODES}, got {adjust_value!r}")
    return "total" if adjust_value == "legacy" else adjust_value  # type: ignore[return-value]


def _as_date(value: datetime | date) -> date:
    return value.date() if isinstance(value, datetime) else value


def _union_events(fmp: tuple[pd.DataFrame, pd.DataFrame], extra: tuple[pd.DataFrame, pd.DataFrame]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union of two event sets, deduplicated by date; the FMP row wins (spec §9.2, §10.2)."""
    out = []
    for ours, theirs in zip(fmp, extra):
        ours = ours.assign(date=pd.to_datetime(ours["date"]))
        if theirs is None or theirs.empty:
            out.append(ours)
            continue
        theirs = theirs.assign(date=pd.to_datetime(theirs["date"]))
        new = theirs[~theirs["date"].isin(set(ours["date"]))]
        if not new.empty:
            _logger.warning("Using %d increment event(s) not in the FMP vintage", len(new))
        out.append(pd.concat([ours, new], ignore_index=True) if not new.empty else ours)
    return out[0], out[1]


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    """Contract dtypes: float prices, int64 volume when it has no gaps (as the legacy files)."""
    out = frame.copy()
    for column in OHLCV[:4]:
        out[column] = out[column].astype(float)
    volume = out["volume"].astype(float)
    out["volume"] = volume.round().astype("int64") if volume.notna().all() else volume
    out.index = pd.DatetimeIndex(out.index).as_unit("us").rename("timestamp")
    return out


class FmpOhlcvAdapter:
    """Serves ``get_ohlcv`` requests from an ``FmpStore`` and an increment layer."""

    def __init__(self, store: FmpStore, increments: Optional[IncrementSource] = None,
                 timeframes: Iterable[str] = ("1d", "1h"), cache_entries: int = 256) -> None:
        """
        Args:
            store: The active vintage.
            increments: Increment layer (default: none, until WP-9).
            timeframes: Timeframes to serve from FMP; others always fall through to the legacy path.
            cache_entries: Adjusted full histories kept in memory.
        """
        self.store = store
        self.increments: IncrementSource = increments or NoIncrements()
        self.timeframes = tuple(timeframes)
        self.cache_entries = cache_entries
        self._cache: OrderedDict[tuple[Any, ...], pd.DataFrame] = OrderedDict()

    @property
    def vintage(self) -> str:
        """The vintage this adapter serves."""
        return self.store.vintage

    def resolve(self, symbol: str, timeframe: str, as_of: Optional[date] = None) -> Optional[SecurityInfo]:
        """
        Return the security that would serve this request, or ``None`` (legacy path).

        Raises:
            AmbiguousSymbol: The symbol matches several securities and ``as_of`` does not pick one.
        """
        if timeframe not in self.timeframes:
            return None
        info = self.store.resolve(symbol, as_of)
        if info is None or self.store.price_path(info.security_id, timeframe) is None:
            return None
        return info

    def get_ohlcv(self, symbol: str, timeframe: str, start: datetime | date, end: datetime | date,
                  adjust_value: str = "legacy", include_source: bool = False, include_flags: bool = False,
                  exclude_suspect: bool = False, as_of: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        Return the bars of ``symbol`` in ``[start, end]`` from FMP, or ``None`` if FMP does not serve it.

        Args:
            symbol: Caller's symbol.
            timeframe: ``1d`` or an intraday timeframe the vintage holds (``1h``).
            start: First bar time (inclusive).
            end: Last bar time (inclusive).
            adjust_value: ``legacy`` (= ``total``), ``raw``, ``split`` or ``total``.
            include_source: Add the ``source`` column.
            include_flags: Add the ``flags`` column (spec §7.5).
            exclude_suspect: Drop bars with any flag set.
            as_of: Disambiguates a reused or renamed symbol.

        Returns:
            Frame in the contract, possibly empty; ``None`` means "not FMP's: use the legacy path".

        Raises:
            AmbiguousSymbol: See ``resolve``.
            ValueError: Unknown ``adjust_value``.
        """
        mode = adjust_mode(adjust_value)
        info = self.resolve(symbol, timeframe, as_of)
        if info is None:
            return None
        full = self._full(info, timeframe, mode)
        assert full is not None  # resolve() checked the file exists
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        if lo.tzinfo is not None:
            lo = lo.tz_convert("UTC").tz_localize(None)
        if hi.tzinfo is not None:
            hi = hi.tz_convert("UTC").tz_localize(None)
        return self._select(full[(full.index >= lo) & (full.index <= hi)], include_source, include_flags,
                            exclude_suspect)

    def get_ohlcv_batch(self, symbols: Iterable[str], timeframe: str, start: datetime | date, end: datetime | date,
                        adjust_value: str = "legacy") -> dict[str, pd.DataFrame]:
        """
        Serve many symbols at once; daily raw bars come from one DuckDB query (spec §9.5).

        Returns:
            ``{symbol: frame}`` for the symbols FMP serves; the others are absent (legacy path).
        """
        mode = adjust_mode(adjust_value)
        served = {s: info for s in dict.fromkeys(symbols) if (info := self.resolve(s, timeframe)) is not None}
        if timeframe != "1d":
            return {s: frame for s in served if (frame := self.get_ohlcv(s, timeframe, start, end, adjust_value))
                    is not None}
        first, last = _as_date(start), _as_date(end)
        ids = [i.security_id for i in served.values()]
        raw = self.store.bulk_daily(ids, first - BULK_LOOKBACK, date.max)
        events = self.store.events_many(ids)
        out = {}
        for symbol, info in served.items():
            frame = raw.get(info.security_id)
            if frame is None:
                continue
            full = self._adjusted(info, "1d", mode, frame, events.get(info.security_id))
            mask = (full.index >= pd.Timestamp(first)) & (full.index <= pd.Timestamp(last))
            out[symbol] = self._select(full[mask], False, False, False)
        return out

    # --- internals -------------------------------------------------------------------------------------------

    @staticmethod
    def _select(frame: pd.DataFrame, include_source: bool, include_flags: bool,
                exclude_suspect: bool) -> pd.DataFrame:
        if exclude_suspect:
            frame = frame[frame["flags"] == 0]
        columns = OHLCV + (["source"] if include_source else []) + (["flags"] if include_flags else [])
        return frame[columns].copy()

    def _full(self, info: SecurityInfo, timeframe: str, mode: AdjustMode) -> Optional[pd.DataFrame]:
        raw = self.store.prices(info.security_id, timeframe)
        if raw is None:
            return None
        return self._adjusted(info, timeframe, mode, raw)

    def _adjusted(self, info: SecurityInfo, timeframe: str, mode: AdjustMode, raw: pd.DataFrame,
                  fmp_events: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Append increments, adjust, cache. ``raw`` may start late (bulk read) but must run to the last bar;
        ``fmp_events`` are the security's vintage events when already read (bulk), else they are read here.
        """
        last = raw.index.max() if len(raw) else None
        after = None if last is None else last.date()
        tail = self.increments.bars(info.security_id, info.symbol, timeframe, after)
        if tail is not None and len(tail):
            tail = tail[tail.index > last] if last is not None else tail
            raw = pd.concat([raw, tail[raw.columns.intersection(tail.columns)]])
        if fmp_events is None:
            fmp_events = self.store.events(info.security_id)
        splits, dividends = _union_events(fmp_events,
                                          self.increments.events(info.security_id, info.symbol))
        tail_key = (len(tail), str(tail.index.max())) if tail is not None and len(tail) else (0, "")
        key = (self.vintage, info.security_id, timeframe, mode, self.store.events_hash(splits, dividends), tail_key,
               str(raw.index.min()) if len(raw) else "")
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        adjusted = self._apply(info, timeframe, mode, raw, splits, dividends)
        self._cache[key] = adjusted
        if len(self._cache) > self.cache_entries:
            self._cache.popitem(last=False)
        return adjusted

    def _apply(self, info: SecurityInfo, timeframe: str, mode: AdjustMode, raw: pd.DataFrame,
               splits: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
        frame = raw[OHLCV + ["flags", "source"]].copy()
        if mode == "raw" or frame.empty:
            return _finish(frame)
        if timeframe == "1d":
            return _finish(adjust(frame, splits, dividends, mode))
        # Intraday: factors by exchange-local date, C_prev from the daily session closes (spec §9.4).
        utc_index = frame.index
        frame.index = pd.DatetimeIndex(utc_index).tz_localize("UTC").tz_convert(self.store.intraday_tz)
        daily = self.store.prices(info.security_id, "1d")
        reference = daily["close"] if daily is not None else None
        adjusted = adjust(frame, splits, dividends, mode, reference_closes=reference)
        adjusted.index = utc_index
        return _finish(adjusted)


def empty_frame() -> pd.DataFrame:
    """An empty frame in the contract."""
    frame = pd.DataFrame({c: np.array([], dtype=float) for c in OHLCV})
    frame.index = pd.DatetimeIndex([], name="timestamp")
    return _finish(frame)
