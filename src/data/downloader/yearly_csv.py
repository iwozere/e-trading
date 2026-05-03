"""
Yearly CSV.gz cache utilities for P15 pipeline downloaders.

Implements the same per-year YYYY.csv.gz convention used by the OHLCV
cache system (symbol/timeframe/YYYY.csv.gz), applied to all macro and
alternative-data downloaders so that every cached dataset is human-readable
and consistently structured.

Cache layout produced:
    <cache_dir>/
        2010.csv.gz
        2011.csv.gz
        ...
        2025.csv.gz

Rules:
  - The DataFrame index must be a DatetimeIndex named "date".
  - For sources with a unique date index (CBOE, Fear & Greed, AAII, FRED,
    yfinance) dedup retains the last row per date.
  - For sources with a non-unique date index (GDELT themes/events, FINRA
    tickers — many rows per date) dedup is date-level: all rows for dates
    present in the new data replace the corresponding rows in the existing
    year file.  This handles both incremental appends and force-overwrites.
"""

import sys
from datetime import date
from pathlib import Path
from typing import Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def save(df: pd.DataFrame, cache_dir: Path) -> None:
    """
    Write df into per-year YYYY.csv.gz files under cache_dir.

    df must carry a DatetimeIndex.  Each year's data is merged with any
    existing file for that year: rows whose normalised date appears in
    the new data replace the existing rows; all other rows are kept.

    Args:
        df:        DataFrame with DatetimeIndex to persist.
        cache_dir: Directory that will hold the YYYY.csv.gz files.
                   Created automatically if absent.
    """
    if df.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    for year_val, group in df.groupby(df.index.year):  # type: ignore[attr-defined]
        path = cache_dir / f"{year_val}.csv.gz"
        if path.exists():
            try:
                existing = pd.read_csv(
                    path, index_col=0, parse_dates=True, compression="gzip"
                )
                # Date-level dedup: remove rows from existing whose normalised
                # date collides with any date in the incoming group.
                new_day_set = set(pd.DatetimeIndex(group.index).normalize())  # type: ignore[attr-defined]
                keep = ~pd.DatetimeIndex(existing.index).normalize().isin(new_day_set)  # type: ignore[attr-defined]
                group = pd.concat([existing.loc[keep], group]).sort_index()
            except Exception:
                _logger.warning("Could not merge with existing %s — overwriting", path)
        group.to_csv(path, compression="gzip")


def load(cache_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all YYYY.csv.gz files from cache_dir.

    Args:
        cache_dir: Directory containing YYYY.csv.gz files.

    Returns:
        Sorted DataFrame with DatetimeIndex, or an empty DataFrame if
        cache_dir is absent or contains no readable files.
    """
    if not cache_dir.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(cache_dir.glob("????.csv.gz")):
        try:
            frames.append(
                pd.read_csv(f, index_col=0, parse_dates=True, compression="gzip")
            )
        except Exception:
            _logger.warning("Could not read %s — skipping", f)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def watermark(cache_dir: Path) -> Optional[date]:
    """
    Return the most recent date in the latest YYYY.csv.gz under cache_dir.

    Reads only the index column of the newest year file for efficiency.

    Args:
        cache_dir: Directory containing YYYY.csv.gz files.

    Returns:
        Most recent date present, or None if no readable files exist.
    """
    if not cache_dir.exists():
        return None
    for f in sorted(cache_dir.glob("????.csv.gz"), reverse=True):
        try:
            idx = pd.read_csv(  # type: ignore
                f, index_col=0, parse_dates=True,
                compression="gzip", usecols=[0],
            ).index
            if not idx.empty:
                return pd.Timestamp(idx.max()).date()  # type: ignore[arg-type]
        except Exception:
            continue
    return None


def cached_dates(cache_dir: Path) -> Set[date]:
    """
    Return the set of all normalised dates across all YYYY.csv.gz files.

    Used by range downloaders (GDELT) to decide which days to skip
    without making a network request.  Only the index column is read.

    Args:
        cache_dir: Directory containing YYYY.csv.gz files.

    Returns:
        Set of datetime.date objects (empty if cache_dir is absent).
    """
    result: Set[date] = set()
    if not cache_dir.exists():
        return result
    for f in cache_dir.glob("????.csv.gz"):
        try:
            idx = pd.read_csv(  # type: ignore
                f, index_col=0, parse_dates=True,
                compression="gzip", usecols=[0],
            ).index
            result.update(pd.DatetimeIndex(idx).normalize().date)  # type: ignore[attr-defined, union-attr]
        except Exception:
            pass
    return result
