"""
FRED (Federal Reserve Economic Data) Downloader

Downloads macroeconomic time-series from the St. Louis Fed FRED API and
caches them as per-year CSV.gz files.  Supports incremental updates
(only fetches new observations since the last cached date) and rebuilds
a single wide combined CSV.gz suitable for analysis.

Cache layout:
    DATA_CACHE_DIR/fred/
        FEDFUNDS.csv.gz             ← one flat file per FRED series ID
        T10Y2Y.csv.gz
        ...
        fred_combined.csv.gz        ← wide daily-reindexed forward-filled view
        fred_meta.json              ← last_updated / last_observation per series

Series covered (24 total, matching fred_api_example.py):
    Daily   : DFF, T10Y2Y, T10Y3M, DGS2, DGS10, T5YIE, T10YIE,
              BAMLH0A0HYM2, BAMLC0A0CM
    Weekly  : ICSA, WALCL, MORTGAGE30US
    Monthly : FEDFUNDS, CPIAUCSL, CPILFESL, PCEPILFE, PPIACO, UNRATE,
              PAYEMS, M2SL, UMCSENT, INDPRO, USREC
    Quarterly: DRTSCILM

CLI usage:
    python fred_downloader.py update-series --series ICSA WALCL [--rebuild] [--force]
    python fred_downloader.py update-freq --freq daily [--rebuild] [--force]
    python fred_downloader.py update-freq --freq monthly quarterly [--rebuild] [--force]
    python fred_downloader.py build-combined
    python fred_downloader.py update-all [--force]

Classes:
- FredDownloader: Main downloader class for FRED macro series
"""

import json
import time
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

_FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
_DEFAULT_START_DATE = "2010-01-01"
_REQUEST_DELAY = 0.15  # seconds — well within FRED's 120 req/min limit

# All 24 series from fred_api_example.py, with friendly names and frequencies
FRED_SERIES: Dict[str, Dict[str, str]] = {
    # Daily
    "DFF":          {"name": "fed_funds_daily",    "freq": "daily"},
    "T10Y2Y":       {"name": "yield_spread_10_2",  "freq": "daily"},
    "T10Y3M":       {"name": "yield_spread_10_3m", "freq": "daily"},
    "DGS2":         {"name": "yield_2y",           "freq": "daily"},
    "DGS10":        {"name": "yield_10y",          "freq": "daily"},
    "T5YIE":        {"name": "breakeven_5y",       "freq": "daily"},
    "T10YIE":       {"name": "breakeven_10y",      "freq": "daily"},
    "BAMLH0A0HYM2": {"name": "hy_spread",          "freq": "daily"},
    "BAMLC0A0CM":   {"name": "ig_spread",          "freq": "daily"},
    "USEPUINDXD":   {"name": "epu_daily",          "freq": "daily"},    # from 1985
    # Weekly (released Thursdays)
    "ICSA":         {"name": "jobless_claims",     "freq": "weekly"},
    "WALCL":        {"name": "fed_balance_sheet",  "freq": "weekly"},
    "MORTGAGE30US": {"name": "mortgage_rate_30y",  "freq": "weekly"},
    # Monthly
    "FEDFUNDS":     {"name": "fed_funds_rate",     "freq": "monthly"},
    "CPIAUCSL":     {"name": "cpi",                "freq": "monthly"},
    "CPILFESL":     {"name": "core_cpi",           "freq": "monthly"},
    "PCEPILFE":     {"name": "core_pce",           "freq": "monthly"},
    "PPIACO":       {"name": "ppi",                "freq": "monthly"},
    "UNRATE":       {"name": "unemployment",       "freq": "monthly"},
    "PAYEMS":       {"name": "nonfarm_payrolls",   "freq": "monthly"},
    "M2SL":         {"name": "m2",                 "freq": "monthly"},
    "UMCSENT":      {"name": "consumer_sentiment", "freq": "monthly"},
    "INDPRO":       {"name": "industrial_prod",    "freq": "monthly"},
    "USREC":        {"name": "recession_flag",     "freq": "monthly"},
    "USEPUINDXM":   {"name": "epu_monthly",        "freq": "monthly"},  # from 1985
    "EPUTRADE":     {"name": "epu_trade",          "freq": "monthly"},  # trade policy
    "EPUHEALTHCARE":{"name": "epu_healthcare",     "freq": "monthly"},
    "GEPUCURRENT":  {"name": "epu_global",         "freq": "monthly"},  # global EPU
    # Quarterly
    "DRTSCILM":     {"name": "loan_standards",     "freq": "quarterly"},
}


class FredDownloader(BaseDataDownloader):
    """
    FRED API Data Downloader.

    Fetches macroeconomic time-series from the St. Louis Fed FRED REST API,
    stores each series as an individual Parquet file, and optionally rebuilds
    a wide combined Parquet forward-filled to daily frequency.

    Incremental updates: if a cached Parquet already exists for a series,
    only observations after the last cached date are fetched.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        api_key: Optional[str] = None,
        start_date: str = _DEFAULT_START_DATE,
    ):
        """
        Initialize the FRED downloader.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       FRED files are stored under <cache_dir>/fred/.
            api_key: FRED API key. Falls back to FRED_API_KEY config/env if omitted.
            start_date: Earliest observation date for full downloads (default: 2010-01-01).
        """
        super().__init__()
        root = Path(cache_dir) if cache_dir else Path(DATA_CACHE_DIR)
        self._fred_dir = root / "fred"
        self._combined_file = self._fred_dir / "fred_combined.csv.gz"
        self._meta_path = self._fred_dir / "fred_meta.json"
        self._start_date = start_date
        self._session = requests.Session()
        self._last_request_time: float = 0.0

        resolved_key = api_key or self._get_config_value("FRED_API_KEY")
        if not resolved_key:
            raise ValueError(
                "FRED API key not found. Set FRED_API_KEY in environment or "
                "config/donotshare/donotshare.py, or pass api_key= explicitly."
            )
        self._api_key = resolved_key

    # ------------------------------------------------------------------
    # BaseDataDownloader interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return "fred"

    def get_supported_intervals(self) -> List[str]:
        """FRED provides macro series — no OHLCV intervals."""
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
        _logger.warning("FRED does not provide OHLCV data. Use update_series() instead.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    def fetch_series(self, series_id: str, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch observations for a single FRED series from the API.

        Args:
            series_id: FRED series identifier, e.g. ``"T10Y2Y"``.
            start_date: ISO date string for the first observation (inclusive).
                        Defaults to the downloader's configured start_date.

        Returns:
            DataFrame with a DatetimeIndex named ``date`` and a single column
            named after ``series_id``.  Missing FRED values (``"."``) are
            converted to NaN and dropped.
        """
        obs_start = start_date or self._start_date
        obs_end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self._throttle()
        params = {
            "series_id":         series_id,
            "api_key":           self._api_key,
            "file_type":         "json",
            "observation_start": obs_start,
            "observation_end":   obs_end,
        }
        _logger.debug("FRED API GET %s from %s", series_id, obs_start)
        response = self._session.get(_FRED_OBS_URL, params=params, timeout=30)
        response.raise_for_status()

        observations = response.json().get("observations", [])
        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[df["value"].notna()].rename(columns={"value": series_id})
        df = df.set_index("date").sort_index()
        return df

    # ------------------------------------------------------------------
    # Incremental update for a single series
    # ------------------------------------------------------------------

    def update_series(self, series_id: str, force_full: bool = False) -> pd.DataFrame:
        """
        Incrementally update (or fully download) a single FRED series.

        If a cached file exists and ``force_full`` is False, only observations
        after the last cached date are fetched and appended.

        Args:
            series_id: FRED series identifier.
            force_full: If True, re-download from ``start_date`` regardless
                        of what is already cached.

        Returns:
            The complete updated DataFrame for this series.
        """
        series_path = self._fred_dir / f"{series_id}.csv.gz"

        if series_path.exists() and not force_full:
            try:
                existing = pd.read_csv(
                    series_path, index_col=0, parse_dates=True, compression="gzip"
                )
                last_date = existing.index.max().date() if not existing.empty else None
            except Exception:
                _logger.warning("%s: failed to read cache — falling back to full download", series_id)
                existing = pd.DataFrame()
                last_date = None

            if last_date is not None:
                fetch_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                new_data = self.fetch_series(series_id, start_date=fetch_start)
                if new_data.empty:
                    _logger.info("%s: up to date (last observation: %s)", series_id, last_date)
                    return existing
                combined = pd.concat([existing, new_data]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                _logger.info(
                    "%s: +%d new rows (total %d, last: %s)",
                    series_id, len(new_data), len(combined), combined.index.max().date(),
                )
            else:
                combined = self.fetch_series(series_id)
                if combined.empty:
                    _logger.warning("%s: no data returned", series_id)
                    return pd.DataFrame()
        else:
            _logger.info("%s: full download from %s", series_id, self._start_date)
            combined = self.fetch_series(series_id)
            if combined.empty:
                _logger.warning("%s: no data returned", series_id)
                return pd.DataFrame()

        self._fred_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(series_path, compression="gzip")
        self._update_meta(series_id, combined)
        return combined

    # ------------------------------------------------------------------
    # Bulk update methods
    # ------------------------------------------------------------------

    def update_series_list(
        self,
        series_ids: List[str],
        force_full: bool = False,
        rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Update a specific list of FRED series.

        Args:
            series_ids: List of FRED series IDs to update.
            force_full: If True, force full re-download for all series.
            rebuild: If True, rebuild fred combined (combined/YYYY.csv.gz) after updating.

        Returns:
            Summary dict: ``total``, ``updated``, ``up_to_date``, ``errors``.
        """
        unknown = [s for s in series_ids if s not in FRED_SERIES]
        if unknown:
            _logger.warning("Unknown series IDs (not in FRED_SERIES): %s", unknown)

        return self._run_update_loop(series_ids, force_full=force_full, rebuild=rebuild)

    def update_by_freq(
        self,
        freqs: List[str],
        force_full: bool = False,
        rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Update all FRED series matching the given frequency (or frequencies).

        Args:
            freqs: List of frequency strings, e.g. ``["daily"]`` or
                   ``["monthly", "quarterly"]``.
            force_full: If True, force full re-download for all matched series.
            rebuild: If True, rebuild fred combined (combined/YYYY.csv.gz) after updating.

        Returns:
            Summary dict: ``total``, ``updated``, ``up_to_date``, ``errors``.
        """
        freq_set = {f.lower() for f in freqs}
        matched = [sid for sid, meta in FRED_SERIES.items() if meta["freq"] in freq_set]
        if not matched:
            _logger.warning("No series found for frequencies: %s", freqs)
            return {"total": 0, "updated": 0, "up_to_date": 0, "errors": 0}
        _logger.info("Updating %d series with freq in %s", len(matched), freq_set)
        return self._run_update_loop(matched, force_full=force_full, rebuild=rebuild)

    def update_all(self, force_full: bool = False) -> Dict[str, Any]:
        """
        Update all 24 FRED series and rebuild the combined Parquet.

        Args:
            force_full: If True, force full re-download from start_date.

        Returns:
            Summary dict: ``total``, ``updated``, ``up_to_date``, ``errors``,
            ``combined_rows``, ``combined_cols``.
        """
        return self._run_update_loop(list(FRED_SERIES.keys()), force_full=force_full, rebuild=True)

    # ------------------------------------------------------------------
    # Combined Parquet builder
    # ------------------------------------------------------------------

    def build_combined(self) -> pd.DataFrame:
        """
        Merge all cached raw Parquet files into one wide daily-frequency DataFrame.

        Columns are renamed to friendly names from FRED_SERIES.  The result is
        reindexed to a full daily calendar and forward-filled so monthly/weekly
        series have a value on every day.

        Returns:
            Wide DataFrame saved to DATA_CACHE_DIR/fred/fred_combined.csv.gz.
            Returns an empty DataFrame if no series files are cached.
        """
        frames = []
        for series_id, meta in FRED_SERIES.items():
            series_path = self._fred_dir / f"{series_id}.csv.gz"
            if not series_path.exists():
                _logger.debug("Skipping %s — not yet cached", series_id)
                continue
            try:
                df = pd.read_csv(
                    series_path, index_col=0, parse_dates=True, compression="gzip"
                )
            except Exception:
                _logger.warning("Could not read %s — skipping", series_path)
                continue
            if df.empty:
                _logger.debug("Skipping %s — empty cache", series_id)
                continue
            df.columns = [meta["name"]]
            frames.append(df)

        if not frames:
            _logger.warning("No cached series found — combined CSV not built")
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).sort_index()
        daily_index = pd.date_range(
            start=combined.index.min(),
            end=combined.index.max(),
            freq="D",
        )
        combined = combined.reindex(daily_index).ffill()
        combined.index.name = "date"

        self._fred_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(self._combined_file, compression="gzip")
        _logger.info(
            "Built combined: %d days × %d columns → %s",
            len(combined), len(combined.columns), self._combined_file,
        )
        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_update_loop(
        self,
        series_ids: List[str],
        force_full: bool,
        rebuild: bool,
    ) -> Dict[str, Any]:
        """Update a list of series and optionally rebuild combined."""
        total = len(series_ids)
        updated = up_to_date = errors = 0

        for series_id in series_ids:
            try:
                series_path = self._fred_dir / f"{series_id}.csv.gz"
                existed = series_path.exists()
                result = self.update_series(series_id, force_full=force_full)
                if result.empty:
                    errors += 1
                elif existed and not force_full:
                    # Distinguish "had new rows" vs "already current"
                    up_to_date += 1
                else:
                    updated += 1
            except Exception:
                _logger.exception("Failed to update series %s", series_id)
                errors += 1

        summary: Dict[str, Any] = {
            "total": total,
            "updated": updated,
            "up_to_date": up_to_date,
            "errors": errors,
            "success": errors == 0,
        }

        if rebuild:
            combined = self.build_combined()
            summary["combined_rows"] = len(combined)
            summary["combined_cols"] = len(combined.columns)

        _logger.info(
            "FRED update complete: updated=%d up_to_date=%d errors=%d / %d total",
            updated, up_to_date, errors, total,
        )
        return summary

    def _update_meta(self, series_id: str, df: pd.DataFrame) -> None:
        """Persist last_updated / last_observation / rows to fred_meta.json."""
        meta = self._load_meta()
        meta[series_id] = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "last_observation": df.index.max().isoformat(),
            "rows": len(df),
        }
        self._fred_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps(meta, indent=2))

    def _load_meta(self) -> Dict[str, Any]:
        """Load fred_meta.json, returning an empty dict if the file does not exist."""
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return {}

    def _throttle(self) -> None:
        """Sleep if necessary to respect the FRED rate limit."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download FRED macroeconomic series to local cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  update-freq  --freq daily --rebuild\n"
            "  update-series --series ICSA WALCL MORTGAGE30US --rebuild\n"
            "  update-freq  --freq monthly quarterly --rebuild\n"
            "  build-combined\n"
            "  update-all --force\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_upd_freq = subparsers.add_parser("update-freq", help="Update all series of given frequency")
    p_upd_freq.add_argument(
        "--freq", nargs="+", required=True,
        choices=["daily", "weekly", "monthly", "quarterly"],
        help="Frequency (or frequencies) to update",
    )
    p_upd_freq.add_argument("--rebuild", action="store_true", help="Rebuild fred_combined.csv.gz after update")
    p_upd_freq.add_argument("--force", action="store_true", help="Force full re-download (ignore cache)")

    p_upd_series = subparsers.add_parser("update-series", help="Update specific series by ID")
    p_upd_series.add_argument("--series", nargs="+", required=True, help="FRED series IDs, e.g. ICSA WALCL")
    p_upd_series.add_argument("--rebuild", action="store_true", help="Rebuild fred_combined.csv.gz after update")
    p_upd_series.add_argument("--force", action="store_true", help="Force full re-download (ignore cache)")

    subparsers.add_parser("build-combined", help="Rebuild fred_combined.csv.gz from cached series files")

    p_all = subparsers.add_parser("update-all", help="Update all 24 series and rebuild combined")
    p_all.add_argument("--force", action="store_true", help="Force full re-download for all series")

    parser.add_argument("--cache-dir", type=str, default=None, help=f"Cache root (default: {DATA_CACHE_DIR})")
    parser.add_argument("--api-key", type=str, default=None, help="FRED API key (overrides config)")

    args = parser.parse_args()
    dl = FredDownloader(cache_dir=args.cache_dir, api_key=args.api_key)

    if args.command == "update-freq":
        summary = dl.update_by_freq(args.freq, force_full=args.force, rebuild=args.rebuild)
        print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")

    elif args.command == "update-series":
        summary = dl.update_series_list(args.series, force_full=args.force, rebuild=args.rebuild)
        print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")

    elif args.command == "build-combined":
        combined = dl.build_combined()
        result = {"success": not combined.empty, "rows": len(combined), "cols": len(combined.columns)}
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

    elif args.command == "update-all":
        summary = dl.update_all(force_full=args.force)
        print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")
