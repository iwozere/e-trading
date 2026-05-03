"""
CBOE Put/Call Ratio Downloader

Downloads CBOE equity, index, and total put/call ratio data from the CBOE
CDN and caches the merged result as a single CSV.gz file.

CBOE publishes six CSV files (three current + three archives).  The current
files are overwritten daily with the full dataset up to today; the archive
files hold the longer historical series.  Both are merged with archive
filling any gaps left by the current file.

Cache layout:
    DATA_CACHE_DIR/cboe/
        cboe_putcall.csv.gz   ← single file, fully replaced on each run

Output columns:
    pc_total   — total options put/call ratio
    pc_equity  — equity options put/call ratio
    pc_index   — index options put/call ratio

Classes:
- CboeDownloader: Main downloader class for CBOE put/call ratio data
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

# Each entry: url to fetch, target column name in the merged output.
# Current files (suffix *_current) take priority; archive fills gaps.
_CBOE_SOURCES: Dict[str, Dict[str, str]] = {
    "total_current":   {
        "url": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
        "col": "pc_total",
    },
    "equity_current":  {
        "url": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
        "col": "pc_equity",
    },
    "index_current":   {
        "url": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
        "col": "pc_index",
    },
    "total_archive":   {
        "url": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpcarchive.csv",
        "col": "pc_total_archive",
    },
    "equity_archive":  {
        "url": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypcarchive.csv",
        "col": "pc_equity_archive",
    },
    "index_archive":   {
        "url": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv",
        "col": "pc_index_archive",
    },
}

# Possible date column names across different CBOE file versions
_DATE_COLUMN_CANDIDATES = ["Trade Date", "DATE", "Date", "date"]
# Possible P/C ratio column names
_PC_COLUMN_CANDIDATES = ["P/C Ratio", "P/C RATIO", "Put/Call Ratio", "Total P/C Ratio"]

_OUTPUT_COLUMNS = ["pc_total", "pc_equity", "pc_index"]


class CboeDownloader(BaseDataDownloader):
    """
    CBOE Put/Call Ratio Downloader.

    Downloads all six CBOE put/call CSV files (three current + three archives),
    merges them into a single wide DataFrame, and saves it as a single CSV.gz
    file at DATA_CACHE_DIR/cboe/cboe_putcall.csv.gz.  The file is fully
    replaced on every run.

    The result contains three columns (pc_total, pc_equity, pc_index) covering
    the full available history by combining archive and current files.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        request_timeout: int = 30,
    ):
        """
        Initialize the CBOE downloader.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       Output is stored at <cache_dir>/cboe/cboe_putcall.csv.gz.
            request_timeout: HTTP request timeout in seconds. Default: 30.
        """
        super().__init__()
        root = Path(cache_dir) if cache_dir else Path(DATA_CACHE_DIR)
        self._cboe_dir = root / "cboe"
        self._cboe_file = self._cboe_dir / "cboe_putcall.csv.gz"
        self._timeout = request_timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "e-trading-research cboe-downloader akossyrev@gmail.com",
        })

    # ------------------------------------------------------------------
    # BaseDataDownloader interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return "cboe"

    def get_supported_intervals(self) -> List[str]:
        """CBOE provides sentiment data — no OHLCV intervals."""
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
        _logger.warning("CBOE downloader does not provide OHLCV data. Use download() instead.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Main download
    # ------------------------------------------------------------------

    def download(self, force: bool = False) -> Optional[Path]:
        """
        Download all six CBOE put/call CSV files, merge, and save as a single CSV.gz.

        The output file is fully replaced on every call because CBOE updates
        its source files in-place daily.  ``force`` is accepted for API
        compatibility but has no effect — the download always runs.

        Args:
            force: Accepted for API compatibility; has no effect.

        Returns:
            Path to ``cboe_putcall.csv.gz``, or None if the download failed.
        """
        del force  # always re-downloads; CBOE files are replaced in-place daily
        frames: Dict[str, pd.Series] = {}
        errors = 0

        for source_key, source in _CBOE_SOURCES.items():
            series = self._fetch_pc_series(source["url"], source["col"], source_key)
            if series is not None:
                frames[source["col"]] = series
            else:
                errors += 1

        if not frames:
            _logger.error("All CBOE downloads failed — nothing to save")
            return None

        combined = pd.DataFrame(frames).sort_index()

        # Merge current files with archives: current takes priority,
        # archive fills historical gaps not covered by the current file.
        for metric in ["total", "equity", "index"]:
            current_col = f"pc_{metric}"
            archive_col = f"pc_{metric}_archive"
            if current_col in combined and archive_col in combined:
                combined[current_col] = combined[current_col].combine_first(combined[archive_col])  # type: ignore[arg-type]
            elif archive_col in combined and current_col not in combined:
                # Current file failed — fall back to archive only
                combined[current_col] = combined[archive_col]

        # Drop archive columns, keep only the three merged output series
        available_output = [c for c in _OUTPUT_COLUMNS if c in combined.columns]
        if not available_output:
            _logger.error("No output columns produced after merge")
            return None

        combined = combined[available_output].sort_index()
        combined.index.name = "date"

        self._cboe_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(self._cboe_file, compression="gzip")

        _logger.info(
            "Saved CBOE putcall: %d rows × %d cols, %s → %s → %s",
            len(combined), len(combined.columns),
            pd.Timestamp(combined.index.min()).date(),  # type: ignore[arg-type]
            pd.Timestamp(combined.index.max()).date(),  # type: ignore[arg-type]
            self._cboe_file,
        )
        if errors:
            _logger.warning("%d source file(s) failed to download", errors)

        return self._cboe_file

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load the cached CBOE put/call data, downloading if absent.

        Args:
            force_refresh: If True, re-download before loading.

        Returns:
            DataFrame with DatetimeIndex and columns pc_total, pc_equity,
            pc_index.  Returns an empty DataFrame on failure.
        """
        result = self.download(force=force_refresh)
        if result is None:
            return pd.DataFrame()
        return pd.read_csv(self._cboe_file, index_col=0, parse_dates=True, compression="gzip")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_pc_series(
        self,
        url: str,
        col_name: str,
        source_key: str,
    ) -> Optional[pd.Series]:
        """
        Fetch a single CBOE put/call CSV and return the P/C ratio as a Series.

        CBOE CSVs have 2 disclaimer rows before the header, which are skipped.
        The date and P/C ratio column names vary slightly between files, so
        multiple candidate names are tried.

        Args:
            url: Full URL of the CBOE CSV file.
            col_name: Name to assign to the returned Series.
            source_key: Human-readable key used in log messages.

        Returns:
            Series with DatetimeIndex, or None on error.
        """
        try:
            _logger.debug("GET %s", url)
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.HTTPError as exc:
            _logger.warning("HTTP error fetching %s (%s): %s", source_key, url, exc)
            return None
        except Exception:
            _logger.exception("Failed to fetch %s (%s)", source_key, url)
            return None

        try:
            df = pd.read_csv(
                io.StringIO(response.text),
                skiprows=2,
                na_values=["."],
            )

            # Normalise column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]

            # Find the date column
            date_col = next(
                (c for c in _DATE_COLUMN_CANDIDATES if c in df.columns), None
            )
            if date_col is None:
                _logger.warning(
                    "%s: could not find date column. Available: %s",
                    source_key, list(df.columns),
                )
                return None

            # Find the P/C ratio column
            pc_col = next(
                (c for c in _PC_COLUMN_CANDIDATES if c in df.columns), None
            )
            if pc_col is None:
                _logger.warning(
                    "%s: could not find P/C ratio column. Available: %s",
                    source_key, list(df.columns),
                )
                return None

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[pc_col] = pd.to_numeric(df[pc_col], errors="coerce")
            df = df[[date_col, pc_col]].dropna()
            df = df.set_index(date_col).sort_index()
            series: pd.Series = df[pc_col].copy()  # type: ignore[assignment]
            series.name = col_name
            series = pd.Series(series[~series.index.duplicated(keep="last")])

            _logger.debug("%s: %d rows, %s – %s", source_key, len(series),
                          str(series.index.min())[:10], str(series.index.max())[:10])
            return series

        except Exception:
            _logger.exception("Failed to parse CSV for %s", source_key)
            return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CBOE put/call ratio data to local cache.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="Download and cache CBOE put/call data")
    p_download.add_argument("--force", action="store_true", help="Re-download even if cache exists")

    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"Cache root directory (default: {DATA_CACHE_DIR})",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="HTTP request timeout in seconds (default: 30)",
    )

    args = parser.parse_args()
    dl = CboeDownloader(cache_dir=args.cache_dir, request_timeout=args.timeout)

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
