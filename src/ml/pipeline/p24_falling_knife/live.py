"""
P24 Falling Knife — live data: universe and daily bars for the scan (spec §9).

The universe is the NASDAQ Trader symbol directory (``shared.universe_downloader``, NASDAQ + NYSE/AMEX listings,
test issues and ETFs excluded) narrowed to plain common stock by symbol and security name. Bars come from
``DataManager.get_ohlcv_batch`` (split-adjusted). For the live scan the split-adjusted price of recent bars *is* the
as-traded price (adjustment only rewrites bars before a split), so the price floor and cost tier use it directly;
only a split inside the lookback makes older bars differ, which affects eligibility of those old bars, not of today.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.ml.pipeline.p24_falling_knife.universe import is_plain_symbol, is_test_listing
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

UNIVERSE_FILE = "01_nasdaq_universe.csv"
# Security names that are not plain common stock (NASDAQ Trader "Security Name").
_NOT_COMMON_RE = re.compile(r"(?i)\b(warrants?|rights?|units?|preferred|depositary|notes?|debentures?|etf|etn|fund|"
                            r"trust\s+units|acquisition\s+corp|subordinated|perpetual|cumulative)\b")
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def is_live_common_stock(symbol: str, name: Optional[str]) -> bool:
    """
    True for a plain common-stock listing in the NASDAQ Trader directory.

    Args:
        symbol: Ticker.
        name: Security name, or None if unknown (then only the symbol rules apply).

    Returns:
        True if the listing belongs in the live universe.
    """
    if not is_plain_symbol(symbol) or is_test_listing(symbol, name):
        return False
    return name is None or not _NOT_COMMON_RE.search(name)


def filter_universe(tickers: Iterable[str], names: Dict[str, str]) -> List[str]:
    """
    Keep plain common stock.

    Args:
        tickers: Directory tickers.
        names: ``ticker -> security name`` (may be partial).

    Returns:
        Sorted unique tickers.
    """
    return sorted({t for t in tickers if is_live_common_stock(t, names.get(t))})


def load_universe(out_dir: Path) -> List[str]:
    """
    Download (or reuse today's cache of) the NASDAQ Trader directory and keep plain common stock.

    Args:
        out_dir: The run's result folder (the directory file is saved there).

    Returns:
        Tickers; empty if the directory could not be downloaded.
    """
    from src.ml.pipeline.shared.config import UniverseConfig
    from src.ml.pipeline.shared.universe_downloader import create_universe_downloader

    config = UniverseConfig(exclude_etfs=True)
    tickers = create_universe_downloader(config, results_dir=out_dir).download_universe()
    names: Dict[str, str] = {}
    path = out_dir / UNIVERSE_FILE
    if path.exists():
        frame = pd.read_csv(path, usecols=lambda c: c in ("Symbol", "Security Name"))
        if "Security Name" in frame.columns:
            names = {str(s): str(n) for s, n in zip(frame["Symbol"], frame["Security Name"]) if isinstance(n, str)}
    else:
        _logger.warning("P24 live universe: %s missing, security-name filter skipped", path)
    kept = filter_universe(tickers, names)
    _logger.info("P24 live universe: %d directory tickers, %d plain common stock", len(tickers), len(kept))
    return kept


def prepare_live_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Turn one DataManager daily frame into P24 daily bars (raw columns = split-adjusted, see module docstring).

    Args:
        frame: Normalized OHLCV frame (tz-naive ``DatetimeIndex``).

    Returns:
        Bars sorted by date with ``open_raw``, ``close_raw``, ``volume_raw`` and ``jump_no_split``; empty if unusable.
    """
    if frame is None or frame.empty or any(c not in frame.columns for c in PRICE_COLUMNS):
        return pd.DataFrame(columns=PRICE_COLUMNS)
    bars = frame[PRICE_COLUMNS].astype(float).sort_index()
    bars.index = pd.DatetimeIndex(bars.index).normalize()
    bars = bars[~bars.index.duplicated(keep="last")]
    bars = bars[bars["close"].notna() & (bars["close"] > 0)]
    bars["open_raw"] = bars["open"]
    bars["close_raw"] = bars["close"]
    bars["volume_raw"] = bars["volume"]
    bars["jump_no_split"] = False
    return bars


def merge_fresh(cached: Dict[str, pd.DataFrame], fresh: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Overlay freshly downloaded recent bars on the cached history (fresh rows win on the same date).

    Args:
        cached: ``symbol -> bars`` from the cache-first batch.
        fresh: ``symbol -> bars`` from the forced recent download.

    Returns:
        ``symbol -> merged bars``, sorted by date, for every symbol in either input.
    """
    out: Dict[str, pd.DataFrame] = {}
    for symbol in dict.fromkeys(list(cached) + list(fresh)):
        frames = [f for f in (cached.get(symbol), fresh.get(symbol)) if f is not None and not f.empty]
        if not frames:
            continue
        merged = pd.concat(frames) if len(frames) > 1 else frames[0]
        out[symbol] = merged[~merged.index.duplicated(keep="last")].sort_index()
    return out


def fetch_daily(tickers: List[str], session: date, lookback_days: int, refresh_days: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily bars through ``DataManager`` and prepare them.

    ``DataManager`` treats a daily cache as current while it is at most 8 days behind the request (its holiday
    tolerance), so a cache-first batch alone can return last week's bars. The history therefore comes from the cache
    and the last ``refresh_days`` calendar days are always downloaded again (``force_refresh``) and overlaid.

    Args:
        tickers: Symbols.
        session: Last session to include.
        lookback_days: Calendar days of history.
        refresh_days: Calendar days re-downloaded on every run.

    Returns:
        ``symbol -> P24 daily bars`` for the symbols with data.
    """
    from src.data.data_manager import DataManager

    manager = DataManager()
    start = datetime.combine(session - timedelta(days=lookback_days), time.min)
    end = datetime.combine(session + timedelta(days=1), time.min)
    cached = manager.get_ohlcv_batch(tickers, "1d", start, end)
    recent = datetime.combine(session - timedelta(days=refresh_days), time.min)
    fresh = manager.get_ohlcv_batch(tickers, "1d", recent, end, force_refresh=True)
    _logger.info("P24 live data: %d symbols from cache, %d refreshed for the last %d days", len(cached), len(fresh),
                 refresh_days)
    raw = merge_fresh(cached, fresh)
    out: Dict[str, pd.DataFrame] = {}
    for symbol, frame in raw.items():
        bars = prepare_live_daily(frame)
        bars = bars[bars.index <= pd.Timestamp(session)]
        if not bars.empty:
            out[symbol] = bars
    _logger.info("P24 live data: %d of %d symbols with daily bars", len(out), len(tickers))
    return out
