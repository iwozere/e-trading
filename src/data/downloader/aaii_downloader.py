"""
AAII Investor Sentiment Survey Downloader

Downloads the AAII (American Association of Individual Investors) weekly
sentiment survey data from the official AAII XLS file and caches it as
per-year CSV.gz files.

The XLS is published weekly (every Thursday) and contains the full history
from 1987 onward — there is no incremental API, so each run replaces the
cache with a fresh full download.

Source:
    https://www.aaii.com/files/surveys/sentiment.xls
    Sheet: "Sentiment Survey", header at row 4 (skiprows=3), columns A:F.

Cache layout:
    DATA_CACHE_DIR/aaii/
        2010.csv.gz   ← one file per calendar year
        2011.csv.gz
        ...           (DatetimeIndex weekly series)

Output columns:
    bullish         — % respondents bullish (0–100 scale)
    neutral         — % respondents neutral (0–100 scale)
    bearish         — % respondents bearish (0–100 scale)
    total           — sum (should be ~100; may be NaN in some rows)
    bull_bear_spread — bullish minus bearish spread

Classes:
- AaiiDownloader: Main downloader class for AAII sentiment survey data
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.data.downloader.yearly_csv import load as ycsv_load  # type: ignore[import]
from src.data.downloader.yearly_csv import save as ycsv_save  # type: ignore[import]
from src.data.downloader.yearly_csv import watermark as ycsv_watermark  # type: ignore[import]
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
_SHEET_NAME = "Sentiment Survey"
_SKIP_ROWS = 3
_USE_COLS = "A:F"
_RAW_COLUMNS = ["date", "bullish", "neutral", "bearish", "total", "bull_bear_spread"]
_PCT_COLUMNS = ["bullish", "neutral", "bearish"]


class AaiiDownloader(BaseDataDownloader):
    """
    AAII Investor Sentiment Survey Downloader.

    Downloads the full AAII weekly sentiment history from the official XLS file,
    normalises percentage columns, and saves as per-year CSV.gz files under
    DATA_CACHE_DIR/aaii/.

    Because AAII publishes a complete history file (not a delta), each run
    overwrites the cache.  Pass ``force=False`` to skip the download when a
    fresh cache already exists.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        request_timeout: int = 60,
    ):
        """
        Initialize the AAII downloader.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       Output is stored under <cache_dir>/aaii/.
            request_timeout: HTTP request timeout in seconds. Default: 60
                             (the XLS file is ~1 MB and AAII servers can be slow).
        """
        super().__init__()
        root = Path(cache_dir) if cache_dir else Path(DATA_CACHE_DIR)
        self._aaii_dir = root / "aaii"
        self._timeout = request_timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; e-trading-research; akossyrev@gmail.com)",
        })

    # ------------------------------------------------------------------
    # BaseDataDownloader interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return "aaii"

    def get_supported_intervals(self) -> List[str]:
        """AAII provides weekly sentiment data — no OHLCV intervals."""
        return []

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Not supported — returns an empty DataFrame.

        Args:
            symbol: Unused.
            interval: Unused.
            start_date: Unused.
            end_date: Unused.
            **kwargs: Unused.

        Returns:
            Empty DataFrame.
        """
        del symbol, interval, start_date, end_date, kwargs
        _logger.warning("AaiiDownloader does not provide OHLCV data. Use download() instead.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def download(self, force: bool = False) -> Optional[Path]:
        """
        Download the AAII sentiment XLS, parse, and cache as YYYY.csv.gz files.

        Args:
            force: If True, re-download even when the cache file already exists.
                   The default (False) skips the download only if the file is
                   present; the scheduler should always call with force=True to
                   ensure weekly data is never stale.

        Returns:
            Path to the saved CSV.gz file, or None if the download failed.
        """
        if ycsv_watermark(self._aaii_dir) is not None and not force:
            _logger.info("AAII sentiment already cached at %s", self._aaii_dir)
            return self._aaii_dir

        raw_bytes = self._fetch_xls()
        if raw_bytes is None:
            return None

        df = self._parse_xls(raw_bytes)
        if df is None or df.empty:
            return None

        ycsv_save(df, self._aaii_dir)

        _logger.info(
            "Saved AAII sentiment: %d weeks, %s → %s → %s",
            len(df),
            str(df.index.min())[:10],
            str(df.index.max())[:10],
            self._aaii_dir,
        )
        return self._aaii_dir

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load the cached AAII sentiment data, downloading if absent.

        Args:
            force_refresh: If True, re-download before loading.

        Returns:
            DataFrame with DatetimeIndex and columns bullish, neutral, bearish,
            total, bull_bear_spread.  Returns an empty DataFrame on failure.
        """
        result = self.download(force=force_refresh)
        if result is None:
            return pd.DataFrame()
        return ycsv_load(self._aaii_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_xls(self) -> Optional[bytes]:
        """
        Download the AAII XLS file and return raw bytes.

        Returns:
            Raw response bytes, or None on HTTP / network error.
        """
        try:
            _logger.debug("GET %s", _AAII_URL)
            response = self._session.get(_AAII_URL, timeout=self._timeout)
            response.raise_for_status()
            return response.content
        except requests.HTTPError as exc:
            _logger.warning("HTTP error fetching AAII XLS: %s", exc)
            return None
        except Exception:
            _logger.exception("Failed to fetch AAII XLS")
            return None

    def _parse_xls(self, raw_bytes: bytes) -> Optional[pd.DataFrame]:
        """
        Parse the AAII XLS bytes into a clean DataFrame.

        Handles:
        - Multi-row header (skiprows=3)
        - Trailing summary / blank rows (dropped by date coercion)
        - Percentage columns stored as decimals (0.38 → 38.0)

        Args:
            raw_bytes: Raw bytes of the downloaded XLS file.

        Returns:
            Clean DataFrame with DatetimeIndex, or None on parse error.
        """
        try:
            df: pd.DataFrame = pd.read_excel(  # type: ignore[call-overload]
                raw_bytes,
                sheet_name=_SHEET_NAME,
                skiprows=_SKIP_ROWS,
                usecols=_USE_COLS,
                engine="xlrd",
            )
            df.columns = _RAW_COLUMNS

            # Drop rows where date is not a recognisable date (summary rows, blanks)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = pd.DataFrame(df[df["date"].notna()])

            # Normalise percentage columns: AAII stores them as decimals (0.38)
            # in older rows and as whole numbers (38.0) in newer ones.
            for col in _PCT_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].dropna()
                if not non_null.empty and non_null.abs().max() < 2.0:
                    df[col] = df[col] * 100.0

            df["total"] = pd.to_numeric(df["total"], errors="coerce")
            df["bull_bear_spread"] = pd.to_numeric(df["bull_bear_spread"], errors="coerce")

            df = df.sort_values("date").set_index("date")
            df = pd.DataFrame(df[~df.index.duplicated(keep="last")])

            return df

        except Exception:
            _logger.exception("Failed to parse AAII XLS")
            return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download AAII investor sentiment survey data to local cache."
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"Cache root directory (default: {DATA_CACHE_DIR})",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="HTTP request timeout in seconds (default: 60)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_dl = subparsers.add_parser("download", help="Download and cache AAII sentiment data")
    p_dl.add_argument(
        "--force", action="store_true",
        help="Re-download even if the cache file already exists",
    )

    args = parser.parse_args()
    dl = AaiiDownloader(cache_dir=args.cache_dir, request_timeout=args.timeout)

    if args.command == "download":
        directory = dl.download(force=args.force)
        rows = len(dl.load()) if directory is not None else 0
        result = {
            "success": directory is not None,
            "path": str(directory) if directory else None,
            "rows": rows,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
