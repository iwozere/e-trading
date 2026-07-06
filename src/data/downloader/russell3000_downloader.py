"""
Russell 3000 Index Constituent Downloader

Downloads and caches the Russell 3000 index constituent list.

Sources tried in order:
  1. Cache — DATA_CACHE_DIR/universe/russell3000.csv.gz (TTL 90 days)
  2. FMP /stable/russell-index-constituents (free tier — available if key present)
  3. Static fallback CSV — DATA_CACHE_DIR/universe/russell3000_static.csv
     (place a CSV with ticker,name,sector,industry,exchange columns there;
      update manually each quarter from Slickcharts or FTSE Russell)

Cache layout:
    DATA_CACHE_DIR/universe/
        russell3000.csv.gz       ← live cache; auto-managed, TTL 90 days
        russell3000_meta.json    ← last_updated, source_used, row_count
        russell3000_static.csv   ← static seed; user-managed
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import List, cast

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR, FMP_API_KEY
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"
    FMP_API_KEY = None

_CACHE_TTL_DAYS = 90
_FMP_STABLE_URL = "https://financialmodelingprep.com/stable/russell-index-constituents"
_REQUIRED_COLUMNS = ["ticker", "name", "sector", "industry", "exchange"]


class Russell3000Downloader(BaseDataDownloader):
    """
    Downloads and caches the Russell 3000 index constituents.

    Sources tried in order:
      1. FMP /stable/russell-index-constituents (free tier — available if key present)
      2. Bundled static CSV at src/data/downloader/data/russell3000_static.csv

    Cache: DATA_CACHE_DIR/universe/russell3000.csv.gz
    TTL:   90 days; force=True bypasses the TTL check.
    """

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: FMP API key. Defaults to FMP_API_KEY from donotshare config.
        """
        super().__init__()
        self._api_key = api_key or FMP_API_KEY or self._get_config_value("FMP_API_KEY")
        self._cache_dir = Path(DATA_CACHE_DIR) / "universe"
        self._cache_file = self._cache_dir / "russell3000.csv.gz"
        self._meta_file = self._cache_dir / "russell3000_meta.json"
        self._static_csv_path = self._cache_dir / "russell3000_static.csv"
        self.last_source_used: str | None = None

    # ------------------------------------------------------------------
    # BaseDataDownloader abstract method stubs (not applicable)
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        return "russell3000"

    def get_supported_intervals(self) -> List[str]:
        return []

    def get_ohlcv(self, symbol, interval, start_date, end_date, **kwargs):
        _logger.warning("Russell3000Downloader does not provide OHLCV data")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, force: bool = False) -> pd.DataFrame:
        """
        Return the Russell 3000 constituent list, refreshing from source if stale.

        Args:
            force: Bypass the 90-day TTL and force a fresh download.

        Returns:
            DataFrame with columns: ticker, name, sector, industry, exchange.
        """
        if not force and not self.is_stale():
            _logger.info("russell3000: cache is fresh — loading from %s", self._cache_file)
            df = pd.read_csv(self._cache_file, compression="gzip")
            self.last_source_used = self._read_meta().get("source_used", "cache")
            return df

        _logger.info("russell3000: cache is stale or missing — downloading")
        df = self._fetch_from_fmp()

        if df is None or df.empty:
            _logger.warning("russell3000: FMP fetch failed — falling back to static CSV")
            df = self._load_static_fallback()
            self.last_source_used = "static"
        else:
            self.last_source_used = "fmp"

        self._write_cache(df)
        return df

    def is_stale(self) -> bool:
        """Return True when cache is absent or older than 90 days."""
        if not self._cache_file.exists():
            return True
        meta = self._read_meta()
        last_updated_str = meta.get("last_updated")
        if not last_updated_str:
            return True
        try:
            last_updated = datetime.fromisoformat(last_updated_str)
            if last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - last_updated).days
            return age_days > _CACHE_TTL_DAYS
        except (ValueError, TypeError):
            return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_from_fmp(self) -> pd.DataFrame | None:
        """Call FMP stable API for Russell 3000 constituents; return None on any error."""
        if not self._api_key:
            _logger.warning("russell3000: no FMP API key — skipping FMP fetch")
            return None
        try:
            resp = requests.get(
                _FMP_STABLE_URL,
                params={"apikey": self._api_key},
                timeout=30,
            )
            if resp.status_code == 402:
                _logger.warning("russell3000: FMP endpoint requires paid plan (402) — falling back to static")
                return None
            if resp.status_code == 404:
                _logger.warning("russell3000: FMP endpoint not found (404) — falling back to static")
                return None
            resp.raise_for_status()
            data = resp.json()
            if not data:
                _logger.warning("russell3000: FMP returned empty response")
                return None
            df = pd.DataFrame(data)
            df = self._normalise_columns(df)
            if df is None or df.empty:
                return None
            _logger.info("russell3000: fetched %d constituents from FMP", len(df))
            return df
        except requests.exceptions.RequestException:
            _logger.exception("russell3000: HTTP error fetching from FMP")
            return None
        except Exception:
            _logger.exception("russell3000: unexpected error fetching from FMP")
            return None

    def _load_static_fallback(self) -> pd.DataFrame:
        """
        Load static fallback CSV from DATA_CACHE_DIR/universe/.

        Raises:
            FileNotFoundError: If the static CSV has not been seeded.
        """
        if not self._static_csv_path.exists():
            raise FileNotFoundError(
                f"Russell 3000 static fallback CSV not found at {self._static_csv_path}. "
                "Copy a CSV with columns ticker,name,sector,industry,exchange there "
                "(download from Slickcharts or FTSE Russell quarterly release)."
            )
        df = pd.read_csv(self._static_csv_path)
        df = self._normalise_columns(df)
        if df is None:
            raise ValueError("russell3000_static.csv is missing required columns")
        _logger.info("russell3000: loaded %d constituents from static CSV", len(df))
        return df

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Map provider column names to the canonical schema."""
        col_map = {
            "symbol": "ticker",
            "companyName": "name",
            "company": "name",
            "Sector": "sector",
            "Industry": "industry",
            "Exchange": "exchange",
            "Ticker": "ticker",
            "Name": "name",
        }
        df = df.rename(columns=col_map)
        if "ticker" not in df.columns:
            _logger.error("russell3000: no ticker column found in response (columns: %s)", list(df.columns))
            return None
        for col in _REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[_REQUIRED_COLUMNS].copy()
        df["ticker"] = df["ticker"].astype(str).str.strip()
        mask: pd.Series = df["ticker"].str.len() > 0  # type: ignore[assignment]
        df = df[mask]
        df = df.drop_duplicates(subset=["ticker"])
        return df.reset_index(drop=True)

    def _write_cache(self, df: pd.DataFrame) -> None:
        """Persist DataFrame to cache and write metadata."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self._cache_file, index=False, compression="gzip")
        meta = {
            "last_updated": datetime.now(UTC).isoformat(),
            "source_used": self.last_source_used,
            "row_count": len(df),
        }
        self._meta_file.write_text(json.dumps(meta, indent=2))
        _logger.info("russell3000: cache written — %d rows, source=%s", len(df), self.last_source_used)

    def _read_meta(self) -> dict:
        """Read metadata JSON; return empty dict on any error."""
        try:
            if self._meta_file.exists():
                return json.loads(self._meta_file.read_text())
        except Exception:
            pass
        return {}


if __name__ == "__main__":
    dl = Russell3000Downloader()
    result = dl.load()
    print(f"Rows: {len(result)}")
    print(f"Source: {dl.last_source_used}")
    print(result.head())
