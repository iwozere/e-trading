"""
FINRA TRF Downloader Wrapper

Thin wrapper around FinraTRFDownloader (src.data.downloader.finra_trf_downloader),
which stores data in DATA_CACHE_DIR/trf/{YYYY-MM-DD}.csv.gz — the project-wide
single cache for FINRA regShoDaily data.  All pipelines (P06, P10, P17) share
this cache; no duplicate downloads or duplicate CSV files in results/.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.donotshare.donotshare import DATA_CACHE_DIR
from src.data.downloader.finra_trf_downloader import FinraTRFDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _is_cache_fresh(path: Path, max_age_days: int = 1) -> bool:
    """Return True if the cached file exists and is younger than max_age_days."""
    if not path.exists():
        return False
    age_days = (datetime.now().timestamp() - path.stat().st_mtime) / 86400.0
    return age_days <= max_age_days


def get_previous_trading_day(dt: Optional[datetime] = None) -> datetime:
    """Return the most recent weekday before today (or before dt)."""
    if dt is None:
        dt = datetime.now()
    if dt.weekday() == 0:   # Monday → Friday
        return dt - timedelta(days=3)
    if dt.weekday() == 6:   # Sunday → Friday
        return dt - timedelta(days=2)
    return dt - timedelta(days=1)


def download_trf(target_date: Optional[datetime] = None, force_download: bool = False) -> Path:
    """
    Ensure FINRA TRF data for target_date is available in DATA_CACHE_DIR/trf/.

    Args:
        target_date: Date to download TRF data for. If None, uses previous trading day.
        force_download: If True, re-download even if the cache file is fresh.

    Returns:
        Path to DATA_CACHE_DIR/trf/{YYYY-MM-DD}.csv.gz
    """
    if target_date is None or (isinstance(target_date, datetime) and target_date.date() == date.today()):
        trf_date = get_previous_trading_day()
    else:
        trf_date = target_date

    date_str = trf_date.strftime("%Y-%m-%d")
    cache_path = Path(DATA_CACHE_DIR) / "trf" / f"{date_str}.csv.gz"

    if cache_path.exists() and not force_download:
        if _is_cache_fresh(cache_path):
            _logger.info("TRF cache is fresh for %s: %s", date_str, cache_path)
            return cache_path
        _logger.info("TRF cache is stale for %s — refreshing", date_str)

    _logger.info("Downloading TRF data for %s", date_str)
    downloader = FinraTRFDownloader(date=date_str, fetch_yfinance_data=False)
    downloader.run()

    if not cache_path.exists():
        # Market was closed / FINRA returned no data — write an empty sentinel so
        # subsequent calls don't trigger re-downloads for the same date.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": [], "ticker": [], "short_volume": [], "total_volume": []}).to_csv(
            cache_path, index=False, compression="gzip"
        )
        _logger.info("No TRF data for %s (market closed?) — wrote empty sentinel", date_str)

    return cache_path


def get_trf_correction_factor(ticker: str, dt: datetime) -> float:
    """
    Return the TRF volume correction factor for a ticker on a given date.

    Correction factor = total_volume / (total_volume - short_volume).
    Returns 1.0 (no correction) if data is unavailable.

    Args:
        ticker: Stock ticker symbol (empty string → just ensure the cache exists).
        dt: Date to look up.
    """
    date_str = dt.strftime("%Y-%m-%d")
    cache_path = Path(DATA_CACHE_DIR) / "trf" / f"{date_str}.csv.gz"

    if not cache_path.exists():
        try:
            download_trf(dt)
        except Exception as e:
            _logger.warning("Failed to download TRF data for %s: %s", date_str, e)

    # Fall back up to 5 trading days if still missing (e.g. holiday gap)
    current_dt = dt
    days_back = 5
    while not cache_path.exists() and days_back > 0:
        current_dt = current_dt - timedelta(days=1)
        date_str = current_dt.strftime("%Y-%m-%d")
        cache_path = Path(DATA_CACHE_DIR) / "trf" / f"{date_str}.csv.gz"
        days_back -= 1

    if not cache_path.exists():
        return 1.0

    try:
        df = pd.read_csv(cache_path)
        if not ticker or "ticker" not in df.columns:
            return 1.0
        row = df[df["ticker"] == ticker.upper()]
        if row.empty:
            return 1.0
        total_vol = row["total_volume"].values[0]  # type: ignore[union-attr]
        short_vol = row["short_volume"].values[0]  # type: ignore[union-attr]
        if total_vol > 0 and short_vol < total_vol:
            factor = total_vol / (total_vol - short_vol)
            _logger.debug(
                "TRF correction factor for %s: %.4f (date=%s, total=%s, short=%s)",
                ticker, factor, cache_path.stem, total_vol, short_vol,
            )
            return factor
    except Exception as e:
        _logger.error("Error reading TRF file %s: %s", cache_path, e)

    return 1.0


def main() -> None:
    """Command-line entry point: download TRF data for a given date."""
    import argparse

    parser = argparse.ArgumentParser(description="Download FINRA TRF data to DATA_CACHE_DIR/trf/")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format (default: previous trading day)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cache is fresh")
    args = parser.parse_args()

    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
        download_trf(target_date, args.force)
    except Exception as e:
        _logger.error("Error: %s", e)
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
