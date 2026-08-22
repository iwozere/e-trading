"""
P21 Momentum — Price panel fetch, and TTL-cached fundamentals/sectors.

Thin wrapper over ``YahooDataDownloader.get_ohlcv_batch()`` /
``get_fundamentals_batch()`` (docs/pipeline-specification.md §2). Per §2
"Fetching": ``get_ohlcv_batch()`` already batches and falls back to
individual downloads on batch failure; this module must not add a second
retry loop on top of it, but it does not raise on an empty result, so
non-emptiness is checked explicitly here (§13 "Tickers with complete price
data").
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from src.data.downloader.data_downloader_factory import DataDownloaderFactory
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.p21_momentum.config import (
    FUNDAMENTALS_CACHE_PATH,
    FUNDAMENTALS_CACHE_TTL_DAYS,
    MIN_PRICE_COVERAGE_PCT,
    SECTORS_CACHE_PATH,
    SECTORS_CACHE_TTL_DAYS,
)
from src.ml.pipeline.p21_momentum.quality.gates import PipelineAbort
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _get_yahoo_downloader() -> YahooDataDownloader:
    downloader = DataDownloaderFactory.create_downloader("yahoo")
    if downloader is None:
        raise RuntimeError("DataDownloaderFactory could not create a 'yahoo' downloader")
    return cast(YahooDataDownloader, downloader)


def fetch_price_panel(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    min_coverage_pct: float = MIN_PRICE_COVERAGE_PCT,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch adjusted OHLCV for every ticker via YahooDataDownloader.get_ohlcv_batch().

    Args:
        tickers: Symbols to fetch (already yfinance-normalized).
        start_date: Batch start.
        end_date: Batch end.
        min_coverage_pct: Minimum fraction of tickers required to come back
            non-empty before this raises PipelineAbort (spec §13:
            "Tickers with complete price data >= 95% of universe" -> ABORT).

    Returns:
        Dict mapping ticker -> OHLCV DataFrame (columns: timestamp, open,
        high, low, close, volume; close is split/dividend-adjusted, §10.1).
        Tickers with no data are still present in the dict, mapped to an
        empty DataFrame — callers must not assume every key is populated.

    Raises:
        PipelineAbort: if fewer than min_coverage_pct of tickers returned
            non-empty data.
    """
    downloader = _get_yahoo_downloader()
    _logger.info("Fetching price panel for %d tickers (%s to %s)", len(tickers), start_date.date(), end_date.date())
    panel = downloader.get_ohlcv_batch(tickers, interval="1d", start_date=start_date, end_date=end_date)

    non_empty = [t for t, df in panel.items() if df is not None and not df.empty]
    coverage = len(non_empty) / len(tickers) if tickers else 0.0
    _logger.info("Price panel coverage: %d/%d (%.1f%%)", len(non_empty), len(tickers), coverage * 100)

    if coverage < min_coverage_pct:
        raise PipelineAbort(
            check="Tickers with complete price data",
            message=f"Coverage {coverage:.1%} below required {min_coverage_pct:.1%}",
            context={"requested": len(tickers), "returned_non_empty": len(non_empty)},
        )
    return panel


# ---------------------------------------------------------------------------
# TTL-cached fundamentals / sectors (§5 F4, §3 cache/fundamentals.json + cache/sectors.json)
# ---------------------------------------------------------------------------


def _read_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        _logger.exception("Failed to read cache %s — treating as empty", path)
        return {}


def _write_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _is_stale(fetched_at_iso: str, ttl_days: int) -> bool:
    try:
        fetched_at = datetime.fromisoformat(fetched_at_iso)
    except ValueError:
        return True
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
    return age_days > ttl_days


def fetch_fundamentals_cached(
    tickers: List[str],
    cache_path: Path = FUNDAMENTALS_CACHE_PATH,
    ttl_days: int = FUNDAMENTALS_CACHE_TTL_DAYS,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Return {ticker: {fcf_ttm, net_income_ttm, fetched_at}}, TTL-cached 90 days.

    Only fetches tickers missing from the cache or past TTL; everything else
    is served from ``cache_path`` unchanged. Backed by
    YahooDataDownloader.get_fundamentals_batch() (spec §2, §5 F4).

    Args:
        tickers: Symbols needing fundamentals.
        cache_path: Path to results/p21_momentum/_state/cache/fundamentals.json.
        ttl_days: Cache lifetime in days (spec: 90).

    Returns:
        Dict keyed by ticker, values {"fcf_ttm": float|None, "net_income_ttm": float|None,
        "fetched_at": iso str}. A ticker whose fetch fails entirely maps to
        {"fcf_ttm": None, "net_income_ttm": None, "fetched_at": <now>} so F4
        can apply its "pass on missing data" rule (spec §5 F4).
    """
    cache = _read_cache(cache_path)
    stale_or_missing = [t for t in tickers if t not in cache or _is_stale(cache[t].get("fetched_at", ""), ttl_days)]

    if stale_or_missing:
        _logger.info("Fetching fundamentals for %d/%d tickers (cache miss/stale)", len(stale_or_missing), len(tickers))
        downloader = _get_yahoo_downloader()
        fetched = downloader.get_fundamentals_batch(stale_or_missing)
        now_iso = datetime.now(timezone.utc).isoformat()
        for t in stale_or_missing:
            f = fetched.get(t)
            cache[t] = {
                "fcf_ttm": getattr(f, "free_cash_flow", None) if f is not None else None,
                "net_income_ttm": getattr(f, "net_income", None) if f is not None else None,
                "fetched_at": now_iso,
            }
        _write_cache(cache_path, cache)

    return {t: cache[t] for t in tickers if t in cache}


def fetch_sectors_cached(
    tickers: List[str],
    universe_sectors: Optional[Dict[str, str]] = None,
    cache_path: Path = SECTORS_CACHE_PATH,
    ttl_days: int = SECTORS_CACHE_TTL_DAYS,
) -> Dict[str, str]:
    """
    Return {ticker: sector}, TTL-cached 30 days.

    ``universe_sectors`` (from the current universe.json, already fetched
    via data/universe.py in the same run) is used to refresh the cache when
    provided — avoids a second network round-trip for sector data the
    universe fetch already retrieved. When a ticker is missing from both the
    cache and universe_sectors, it is simply omitted (caller decides
    fallback behavior).

    Args:
        tickers: Symbols needing a sector.
        universe_sectors: Optional {ticker: sector} from this run's universe
            fetch, used to populate/refresh the cache.
        cache_path: Path to results/p21_momentum/_state/cache/sectors.json.
        ttl_days: Cache lifetime in days (spec: 30).

    Returns:
        Dict keyed by ticker -> sector, only for tickers with a known sector.
    """
    cache = _read_cache(cache_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    if universe_sectors:
        for t, sector in universe_sectors.items():
            cache[t] = {"sector": sector, "fetched_at": now_iso}
        _write_cache(cache_path, cache)

    return {
        t: cache[t]["sector"]
        for t in tickers
        if t in cache and "sector" in cache[t] and not _is_stale(cache[t].get("fetched_at", ""), ttl_days)
    }


def non_empty_symbols(panel: Dict[str, pd.DataFrame]) -> List[str]:
    """Return tickers whose OHLCV frame in panel is present and non-empty."""
    return [t for t, df in panel.items() if df is not None and not df.empty]
