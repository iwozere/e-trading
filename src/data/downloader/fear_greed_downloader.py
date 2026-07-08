"""
CNN Fear & Greed Index Downloader

Downloads and caches the CNN Fear & Greed Index daily scores.

Two data sources are combined:
  - GitHub archive (2011-01-03 to present, updated every Friday) via a community CSV:
    https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv
  - CNN production API (recent window) via the official JSON endpoint, used to fill
    any gap between the archive's last entry and today.

Two update modes:
  - Full rebuild  (Fridays / on demand): re-download the GitHub archive (which is
    refreshed every Friday) + CNN API top-up, merge, deduplicate, and overwrite cache.
  - Incremental   (Mon–Thu): load existing cache, fetch only rows newer than the
    last cached date from the CNN API, append, deduplicate, and re-save.

Cache layout:
    DATA_CACHE_DIR/fear_greed/
        cnn_fear_greed.csv.gz   ← single file; appended and deduplicated in-place

Output columns:
    fear_greed_score  — float, 0–100
    label             — extreme_fear | fear | neutral | greed | extreme_greed

Classes:
- FearGreedDownloader: Main downloader class for the CNN Fear & Greed index
"""

import io
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_ARCHIVE_URL = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv"
_CNN_BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_CNN_ARCHIVE_START = "2021-02-01"


def _score_to_label(score: float) -> str:
    """Map a numeric Fear & Greed score to its category label."""
    if score < 25:
        return "extreme_fear"
    if score < 45:
        return "fear"
    if score < 55:
        return "neutral"
    if score < 75:
        return "greed"
    return "extreme_greed"


class FearGreedDownloader(BaseDataDownloader):
    """
    CNN Fear & Greed Index Downloader.

    Combines a long-running GitHub archive (2011–2021) with live data from the
    CNN production API to produce a single cached Parquet file covering the
    full available history.

    Intended run schedule:
      - Every Friday EOD  → ``download(full_rebuild=True)``  (re-downloads archive)
      - All other trading days → ``download()``  (incremental append only)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] | None = None,
        request_timeout: int = 30,
    ):
        """
        Initialize the Fear & Greed downloader.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       Output is stored under <cache_dir>/fear_greed/.
            request_timeout: HTTP request timeout in seconds. Default: 30.
        """
        super().__init__()
        root = Path(cache_dir) if cache_dir else Path(DATA_CACHE_DIR)
        self._fg_dir = root / "fear_greed"
        self._fg_file = self._fg_dir / "cnn_fear_greed.csv.gz"
        self._timeout = request_timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; e-trading-research; akossyrev@gmail.com)",
            }
        )

    # ------------------------------------------------------------------
    # BaseDataDownloader interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return "fear_greed"

    def get_supported_intervals(self) -> List[str]:
        """Fear & Greed is a daily sentiment index — no OHLCV intervals."""
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
        _logger.warning("FearGreedDownloader does not provide OHLCV data. Use download() instead.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def download(self, full_rebuild: bool = False) -> Path | None:
        """
        Download and cache the Fear & Greed index.

        Chooses between a full rebuild (re-downloads the archive + full CNN
        history) and an incremental append (fetches only rows newer than the
        last cached date).  The caller decides which mode to use; the scheduler
        passes ``full_rebuild=True`` on Fridays and the default on other days.

        Args:
            full_rebuild: If True, re-download the full archive and overwrite
                          the cache.  If False and a cache file exists, only
                          new rows are appended.

        Returns:
            Path to the saved Parquet file, or None if the download failed.
        """
        if full_rebuild or not self._fg_file.exists():
            return self._build_full_history()
        return self._append_recent()

    def load(self, full_rebuild: bool = False) -> pd.DataFrame:
        """
        Load the cached Fear & Greed data, downloading if absent.

        Args:
            full_rebuild: Passed through to ``download()``.

        Returns:
            DataFrame with DatetimeIndex and columns fear_greed_score, label.
            Returns an empty DataFrame on failure.
        """
        result = self.download(full_rebuild=full_rebuild)
        if result is None:
            return pd.DataFrame()
        return pd.read_csv(self._fg_file, index_col=0, parse_dates=True, compression="gzip")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_full_history(self) -> Path | None:
        """
        Full rebuild: download archive CSV + full CNN history, merge, and save.

        Returns:
            Path to saved Parquet, or None on failure.
        """
        archive = self._fetch_archive()
        fresh = self._fetch_cnn(start_date=_CNN_ARCHIVE_START)

        if archive is None and fresh is None:
            _logger.error("Both archive and CNN API failed — cannot build history")
            return None

        parts = [df for df in (archive, fresh) if df is not None]
        combined = pd.concat(parts, ignore_index=True)
        return self._normalise_and_save(combined, mode="full rebuild")

    def _append_recent(self) -> Path | None:
        """
        Incremental update: load cache and append rows newer than last date.

        Returns:
            Path to ``cnn_fear_greed.csv.gz``, or None on failure.
        """
        try:
            existing = pd.read_csv(self._fg_file, index_col=0, parse_dates=True, compression="gzip")
        except Exception:
            _logger.exception("Failed to read existing cache — falling back to full rebuild")
            return self._build_full_history()

        if existing.empty:
            return self._build_full_history()

        last_ts: pd.Timestamp = existing.index.max()  # type: ignore[assignment]
        fetch_start = (last_ts + timedelta(days=1)).strftime("%Y-%m-%d")

        _logger.info("Appending Fear & Greed data from %s", fetch_start)
        fresh = self._fetch_cnn(start_date=fetch_start)
        if fresh is None or fresh.empty:
            _logger.info("No new Fear & Greed data since %s", fetch_start)
            return self._fg_file

        existing_reset = existing.reset_index()
        combined = pd.concat([existing_reset, fresh], ignore_index=True)
        return self._normalise_and_save(combined, mode="incremental")

    def _normalise_and_save(self, df: pd.DataFrame, mode: str) -> Path | None:
        """
        Deduplicate, sort, ensure label column, set DatetimeIndex, and save.

        Args:
            df: Raw combined DataFrame with a 'date' column.
            mode: Human-readable mode label for log messages.

        Returns:
            Path to ``cnn_fear_greed.csv.gz``, or None on failure.
        """
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = pd.DataFrame(df[df["date"].notna() & df["fear_greed_score"].notna()])
            df["fear_greed_score"] = pd.to_numeric(df["fear_greed_score"], errors="coerce")
            df = pd.DataFrame(df[df["fear_greed_score"].notna()])

            # Compute label for any rows missing it
            mask = df["label"].isna() if "label" in df.columns else pd.Series(True, index=df.index)
            df.loc[mask, "label"] = df.loc[mask, "fear_greed_score"].apply(_score_to_label)

            df = (
                df[["date", "fear_greed_score", "label"]]
                .drop_duplicates(subset="date")  # type: ignore[call-overload]
                .sort_values("date")
                .set_index("date")
            )

            self._fg_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self._fg_file, compression="gzip")

            _logger.info(
                "Saved Fear & Greed (%s): %d rows, %s → %s",
                mode,
                len(df),
                pd.Timestamp(df.index.min()).date(),  # type: ignore[arg-type]
                pd.Timestamp(df.index.max()).date(),  # type: ignore[arg-type]
            )
            return self._fg_file

        except Exception:
            _logger.exception("Failed to normalise and save Fear & Greed data")
            return None

    def _fetch_cnn(self, start_date: str) -> pd.DataFrame | None:
        """
        Fetch Fear & Greed historical data from the CNN production API.

        Args:
            start_date: ISO date string (YYYY-MM-DD) for the start of the range.

        Returns:
            DataFrame with columns date, fear_greed_score, label; or None on error.
        """
        url = f"{_CNN_BASE_URL}/{start_date}"
        try:
            _logger.debug("GET %s", url)
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except requests.HTTPError as exc:
            _logger.warning("HTTP error fetching CNN Fear & Greed (%s): %s", url, exc)
            return None
        except Exception:
            _logger.exception("Failed to fetch CNN Fear & Greed (%s)", url)
            return None

        try:
            records = data["fear_and_greed_historical"]["data"]
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["x"] // 1000, unit="s").dt.normalize()  # type: ignore[union-attr]
            df = df.rename(columns={"y": "fear_greed_score"})
            df["fear_greed_score"] = pd.to_numeric(df["fear_greed_score"], errors="coerce")
            df["label"] = df["fear_greed_score"].apply(_score_to_label)
            result = df[["date", "fear_greed_score", "label"]].drop_duplicates(subset="date")  # type: ignore[call-overload]
            _logger.debug("CNN API returned %d rows from %s", len(result), start_date)
            return result
        except Exception:
            _logger.exception("Failed to parse CNN Fear & Greed response")
            return None

    def _fetch_archive(self) -> pd.DataFrame | None:
        """
        Download the community archive CSV (2011–2021) from GitHub.

        Returns:
            DataFrame with columns date, fear_greed_score, label; or None on error.
        """
        try:
            _logger.debug("GET %s", _ARCHIVE_URL)
            response = self._session.get(_ARCHIVE_URL, timeout=self._timeout)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
        except requests.HTTPError as exc:
            _logger.warning("HTTP error fetching Fear & Greed archive: %s", exc)
            return None
        except Exception:
            _logger.exception("Failed to fetch Fear & Greed archive")
            return None

        try:
            # Normalise column names: the archive may use 'score' or 'value' instead
            col_map: Dict[str, str] = {}
            for col in df.columns:
                low = col.lower().strip()
                if low in ("date", "datetime", "timestamp", "time"):
                    col_map[col] = "date"
                elif low in (
                    "score",
                    "value",
                    "fear_greed",
                    "fear_greed_score",
                    "fear_greed_value",
                    "fng_value",
                    "fear greed",
                ):
                    col_map[col] = "fear_greed_score"
                elif low in ("rating", "label", "category", "fear_greed_category", "fng_classification"):
                    col_map[col] = "label"
            if col_map:
                df = df.rename(columns=col_map)

            if "fear_greed_score" not in df.columns:
                _logger.warning("Archive CSV missing score column. Available: %s", list(df.columns))
                return None

            df["fear_greed_score"] = pd.to_numeric(df["fear_greed_score"], errors="coerce")
            if "label" not in df.columns:
                df["label"] = df["fear_greed_score"].apply(_score_to_label)

            sub = pd.DataFrame(df[["date", "fear_greed_score", "label"]])
            result = pd.DataFrame(sub[sub["date"].notna() & sub["fear_greed_score"].notna()])
            _logger.debug("Archive CSV: %d rows", len(result))
            return result
        except Exception:
            _logger.exception("Failed to parse Fear & Greed archive CSV")
            return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CNN Fear & Greed Index data to local cache.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=f"Cache root directory (default: {DATA_CACHE_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds (default: 30)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_dl = subparsers.add_parser("download", help="Download / update Fear & Greed data")
    p_dl.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Re-download archive + full CNN history and overwrite cache (use on Fridays)",
    )

    args = parser.parse_args()
    dl = FearGreedDownloader(cache_dir=args.cache_dir, request_timeout=args.timeout)

    if args.command == "download":
        directory = dl.download(full_rebuild=args.full_rebuild)
        rows = len(dl.load()) if directory is not None else 0
        result = {
            "success": directory is not None,
            "path": str(directory) if directory else None,
            "rows": rows,
            "full_rebuild": args.full_rebuild,
            "downloaded_at": datetime.now(UTC).isoformat(),
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
