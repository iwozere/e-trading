"""
Signal Ledger — daily open/close price cache.

A wide parquet frame (columns ``(field, ticker)`` with field in ``open``/``close``,
index = session date) under ``results/signal_ledger/_state/``. Prices are
split/dividend-adjusted, and adjusted history shifts after every new split or
dividend, so a ticker's columns are always replaced as a whole from one fetch,
never stitched from two fetches.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, cast

import pandas as pd

from src.data.downloader.data_downloader_factory import DataDownloaderFactory
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.signal_ledger.config import CALENDAR_TICKER, PRICE_BATCH_SIZE, PRICE_CACHE_PATH
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

FIELDS = ("open", "close")

# (tickers, start, end_exclusive) -> {ticker: frame with timestamp/open/close}
Fetcher = Callable[[List[str], datetime, datetime], Dict[str, pd.DataFrame]]


def yahoo_symbol(ticker: str) -> str:
    """Map a ledger ticker to its Yahoo symbol (``BRK.B`` -> ``BRK-B``)."""
    return ticker.replace(".", "-")


def _yahoo_fetcher(tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Fetch adjusted daily bars through the project's Yahoo downloader."""
    downloader = cast(Optional[YahooDataDownloader], DataDownloaderFactory.create_downloader("yahoo"))
    if downloader is None:
        raise RuntimeError("DataDownloaderFactory could not create a 'yahoo' downloader")
    by_symbol = {yahoo_symbol(t): t for t in tickers}
    frames = downloader.get_ohlcv_batch(list(by_symbol), interval="1d", start_date=start, end_date=end)
    return {by_symbol.get(symbol, symbol): frame for symbol, frame in frames.items()}


class PriceCache:
    """Wide open/close panel persisted to parquet."""

    def __init__(self, path: Path = PRICE_CACHE_PATH, fetcher: Optional[Fetcher] = None) -> None:
        self.path = path
        self._fetcher = fetcher or _yahoo_fetcher
        self.frame = self._load()

    def _load(self) -> pd.DataFrame:
        """Read the cache, or return an empty panel."""
        if self.path.exists():
            frame = pd.read_parquet(self.path)
            frame.index = pd.DatetimeIndex(frame.index)
            return frame
        return pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=["field", "ticker"]))

    def save(self) -> None:
        """Persist the cache."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.sort_index().to_parquet(self.path)

    @property
    def tickers(self) -> set[str]:
        """Tickers that have a column in the cache."""
        if self.frame.empty and len(self.frame.columns) == 0:
            return set()
        return set(self.frame.columns.get_level_values("ticker"))

    def first_date(self, ticker: str) -> Optional[date]:
        """First date with a cached close for ticker, or None."""
        if ("close", ticker) not in self.frame.columns:
            return None
        closes = self.frame[("close", ticker)].dropna()
        return pd.Timestamp(closes.index[0]).date() if len(closes) else None

    def refresh(self, tickers: Iterable[str], start: date, end: Optional[date] = None) -> int:
        """
        Re-download whole histories for the given tickers and replace their columns.

        Args:
            tickers: Tickers to refresh.
            start: First date to fetch.
            end: Last date to fetch (inclusive, default today).

        Returns:
            Number of tickers that came back with data.
        """
        wanted = sorted(set(tickers))
        end_exclusive = datetime.combine((end or date.today()) + timedelta(days=1), datetime.min.time())
        start_dt = datetime.combine(start, datetime.min.time())
        fresh: Dict[str, pd.DataFrame] = {}
        for i in range(0, len(wanted), PRICE_BATCH_SIZE):
            chunk = wanted[i : i + PRICE_BATCH_SIZE]
            for ticker, bars in self._fetcher(chunk, start_dt, end_exclusive).items():
                wide = _to_wide(bars)
                if wide is not None:
                    fresh[ticker] = wide
            _logger.info("Price refresh: %d/%d tickers fetched", min(i + PRICE_BATCH_SIZE, len(wanted)), len(wanted))
        self._replace(fresh)
        return len(fresh)

    def _replace(self, fresh: Dict[str, pd.DataFrame]) -> None:
        """Drop cached columns of refreshed tickers and join the new ones."""
        if not fresh:
            return
        new = pd.concat({t: f for t, f in fresh.items()}, axis=1)  # (ticker, field)
        new = new.swaplevel(axis=1)
        new.columns = new.columns.set_names(["field", "ticker"])
        if len(self.frame.columns):
            keep = self.frame.loc[:, ~self.frame.columns.get_level_values("ticker").isin(list(fresh))]
            if len(keep.columns):
                new = keep.join(new, how="outer")
        self.frame = new.sort_index(axis=1)

    def sessions(self, before: Optional[date] = None) -> pd.DatetimeIndex:
        """
        Trading sessions = dates on which the calendar ticker has a close.

        Args:
            before: Drop sessions on/after this date (e.g. today's unfinished bar).

        Returns:
            Sorted session dates.
        """
        if ("close", CALENDAR_TICKER) not in self.frame.columns:
            return pd.DatetimeIndex([])
        closes = self.frame[("close", CALENDAR_TICKER)].dropna()
        index = pd.DatetimeIndex(closes.index).normalize()
        if before is not None:
            index = index[index < pd.Timestamp(before)]
        return index.sort_values()

    def panel(self, field: str, before: Optional[date] = None) -> pd.DataFrame:
        """
        Return one field (``open``/``close``) restricted to trading sessions.

        Args:
            field: ``"open"`` or ``"close"``.
            before: Drop sessions on/after this date.

        Returns:
            Frame indexed by session, one column per ticker.
        """
        sessions = self.sessions(before)
        if field not in self.frame.columns.get_level_values("field"):
            return pd.DataFrame(index=sessions)
        frame = cast(pd.DataFrame, self.frame[field]).copy()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        return frame.reindex(sessions)


def _to_wide(bars: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Turn a downloader frame (timestamp/open/close...) into a date-indexed open/close frame."""
    if bars is None or bars.empty or not {"timestamp", "open", "close"} <= set(bars.columns):
        return None
    frame = bars[["timestamp", *FIELDS]].copy()
    stamps = pd.to_datetime(frame["timestamp"])
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_localize(None)
    frame.index = pd.DatetimeIndex(stamps).normalize()
    frame = frame[list(FIELDS)].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame if not frame.empty else None
