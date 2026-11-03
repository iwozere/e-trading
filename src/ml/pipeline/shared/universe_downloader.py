"""
NASDAQ Universe Downloader

Downloads complete US stock universe from NASDAQ Trader (FTP, with HTTPS fallback).
Stores results in results/emps2/YYYY-MM-DD/ for date-based organization.
"""

import ftplib
import io
import json
import sys
import time
from datetime import datetime
from ftplib import FTP
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

from .config import UniverseConfig

_logger = setup_logger(__name__)

_FTP_HOST = "ftp.nasdaqtrader.com"
_HTTPS_BASE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir"
_SYMBOL_FILES = ("nasdaqlisted.txt", "otherlisted.txt")
_TIMEOUT_SECONDS = 30
_FETCH_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 20


class NasdaqUniverseDownloader:
    """
    Downloads and caches the complete US stock universe from NASDAQ Trader.

    Fetches from:
    - NASDAQ listed stocks
    - NYSE, AMEX, and other exchange listings
    """

    def __init__(
        self,
        config: UniverseConfig | None = None,
        results_dir: Path | None = None,
        target_date: str | None = None,
    ):
        """
        Initialize NASDAQ universe downloader.

        Args:
            config: Optional universe configuration (uses defaults if None)
            results_dir: Optional results directory.
            target_date: Target date for results folder (YYYY-MM-DD)
        """
        self.config = config or UniverseConfig()

        # Cache in results folder with target date
        if target_date is None:
            from datetime import timedelta

            target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        if results_dir:
            self._cache_dir = results_dir
        else:
            self._cache_dir = Path("results") / "shared" / target_date

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("NASDAQ Universe Downloader initialized (cache: %s)", self._cache_dir)

    def download_universe(self, force_refresh: bool = False) -> List[str]:
        """
        Download complete US stock universe.

        Args:
            force_refresh: If True, bypass cache and force fresh download

        Returns:
            List of ticker symbols
        """
        try:
            _logger.info("Downloading NASDAQ universe (force_refresh=%s)", force_refresh)

            # Check cache first (unless force refresh)
            if not force_refresh and self.config.cache_enabled:
                cached = self._load_from_cache()
                if cached:
                    _logger.info("Loaded %d tickers from cache", len(cached))
                    return cached

            # Download from NASDAQ Trader
            tickers = self._download_from_nasdaq_trader()

            # Save to cache
            if self.config.cache_enabled:
                self._save_to_cache(tickers)

            _logger.info("Successfully downloaded universe: %d tickers", len(tickers))
            return tickers

        except Exception:
            _logger.exception("Error downloading NASDAQ universe:")
            return []

    def _download_from_nasdaq_trader(self) -> List[str]:
        """
        Download universe from NASDAQ Trader FTP.

        Returns:
            List of ticker symbols
        """
        try:
            texts = self._fetch_symbol_files()
            df1 = pd.read_csv(io.StringIO(texts["nasdaqlisted.txt"]), sep="|")
            _logger.debug("Downloaded %d rows from nasdaqlisted.txt", len(df1))
            df2 = pd.read_csv(io.StringIO(texts["otherlisted.txt"]), sep="|")
            _logger.debug("Downloaded %d rows from otherlisted.txt", len(df2))
            # otherlisted.txt names its ticker column "ACT Symbol"; without this rename
            # every NYSE/AMEX row has a NaN "Symbol" after concat and is silently dropped.
            df2 = df2.rename(columns={"ACT Symbol": "Symbol"})

            # Combine
            df = pd.concat([df1, df2], ignore_index=True)
            _logger.info("Downloaded %d total symbols", len(df))
            _logger.debug("Columns: %s", df.columns.tolist())

            # Drop any completely empty rows (from trailing pipes in file)
            df = df.dropna(how="all")

            # Filter test issues
            if self.config.exclude_test_issues and "Test Issue" in df.columns:
                initial_count = len(df)
                df = df[df["Test Issue"] == "N"]
                _logger.info("Excluded %d test issues", initial_count - len(df))

            # Optionally exclude ETFs early to reduce unnecessary downstream
            # fundamentals lookups and related debug noise.
            if getattr(self.config, "exclude_etfs", False):
                etf_col = next((col for col in df.columns if col.strip().upper() == "ETF"), None)
                if etf_col:
                    initial_count = len(df)
                    etf_flags = df[etf_col].astype(str).str.strip().str.upper()
                    df = df[~etf_flags.isin({"Y", "YES", "TRUE", "1", "T"})]
                    _logger.info("Excluded %d ETFs (column: %s)", initial_count - len(df), etf_col)
                else:
                    _logger.debug("ETF exclusion requested but ETF column is missing")

            # Extract ticker symbols
            if "Symbol" in df.columns:
                tickers = df["Symbol"].dropna().tolist()
            else:
                _logger.error("Symbol column not found in data. Available columns: %s", df.columns.tolist())
                return []

            _logger.info("Extracted %d tickers before filtering", len(tickers))

            # Apply filters
            tickers = self._apply_filters(tickers)

            # Save full universe file
            self._save_universe_file(df, tickers)

            return tickers

        except Exception:
            _logger.exception("Error downloading from NASDAQ Trader:")
            _logger.error("Unable to download from NASDAQ Trader. Please check network/firewall settings.")
            _logger.info(
                "You can try manual download using: python src/ml/pipeline/p06_emps2/download_nasdaq_manual.py"
            )
            return []

    def _fetch_symbol_files(self) -> Dict[str, str]:
        """
        Fetch the raw NASDAQ Trader symbol directory files.

        Each attempt tries FTP first and falls back to the HTTPS mirror of the
        same files; a failed attempt is retried after a delay. Previously a
        single FTP connect timeout emptied the universe and aborted P06/P17.

        Returns:
            Mapping of file name (``nasdaqlisted.txt`` / ``otherlisted.txt``) to its text.

        Raises:
            RuntimeError: When every attempt over both transports failed.
        """
        fetchers = (("FTP", self._fetch_via_ftp), ("HTTPS", self._fetch_via_https))
        for attempt in range(1, _FETCH_ATTEMPTS + 1):
            for transport, fetch in fetchers:
                try:
                    _logger.info("Downloading NASDAQ Trader symbol files via %s (attempt %d/%d)", transport, attempt, _FETCH_ATTEMPTS)
                    return fetch()
                except (OSError, EOFError, ftplib.Error, requests.RequestException) as exc:
                    _logger.warning("NASDAQ Trader %s download failed (attempt %d/%d): %s", transport, attempt, _FETCH_ATTEMPTS, exc)
            if attempt < _FETCH_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)
        raise RuntimeError(f"NASDAQ Trader symbol files unavailable via FTP and HTTPS after {_FETCH_ATTEMPTS} attempts")

    @staticmethod
    def _fetch_via_ftp() -> Dict[str, str]:
        """
        Fetch the symbol files from the NASDAQ Trader FTP server.

        Returns:
            Mapping of file name to its text.
        """
        texts: Dict[str, str] = {}
        with FTP(_FTP_HOST, timeout=_TIMEOUT_SECONDS) as ftp:
            ftp.login()  # Anonymous login
            ftp.cwd("SymbolDirectory")
            for name in _SYMBOL_FILES:
                lines: List[str] = []
                ftp.retrlines(f"RETR {name}", lines.append)
                texts[name] = "\n".join(lines)
        return texts

    @staticmethod
    def _fetch_via_https() -> Dict[str, str]:
        """
        Fetch the symbol files from the NASDAQ Trader HTTPS mirror.

        Returns:
            Mapping of file name to its text.
        """
        texts: Dict[str, str] = {}
        for name in _SYMBOL_FILES:
            response = requests.get(f"{_HTTPS_BASE_URL}/{name}", timeout=_TIMEOUT_SECONDS)
            response.raise_for_status()
            texts[name] = response.text
        return texts

    def _apply_filters(self, tickers: List[str]) -> List[str]:
        """
        Apply filters to ticker list.

        Args:
            tickers: Raw ticker list

        Returns:
            Filtered ticker list
        """
        filtered = []

        for ticker in tickers:
            if not ticker or not isinstance(ticker, str):
                continue

            ticker = ticker.strip().upper()

            # Alphabetic only filter
            if self.config.alphabetic_only:
                if not ticker.isalpha():
                    continue

            # Length check (US tickers typically 1-5 chars)
            if len(ticker) > 5 or len(ticker) == 0:
                continue

            filtered.append(ticker)

        _logger.info("After filtering: %d tickers (removed %d)", len(filtered), len(tickers) - len(filtered))
        return filtered

    def _save_universe_file(self, df: pd.DataFrame, tickers: List[str]) -> None:
        """
        Save full universe DataFrame to CSV.

        Args:
            df: Full DataFrame from NASDAQ Trader
            tickers: Filtered ticker list
        """
        try:
            if not tickers:
                _logger.warning("No tickers to save to universe file")
                return

            output_path = self._cache_dir / "01_nasdaq_universe.csv"

            # Filter DataFrame to only include filtered tickers
            if "Symbol" in df.columns:
                df_filtered = df[df["Symbol"].isin(tickers)].copy()
            else:
                df_filtered = df.copy()

            if df_filtered.empty:
                _logger.warning("Filtered DataFrame is empty, saving ticker list only")
                # Create simple DataFrame with just tickers
                df_filtered = pd.DataFrame({"Symbol": tickers})

            df_filtered.to_csv(output_path, index=False)
            _logger.info("Saved %d tickers to: %s", len(df_filtered), output_path)

        except Exception:
            _logger.exception("Error saving universe file:")

    def _load_from_cache(self) -> List[str] | None:
        """
        Load universe from cache if valid.

        Returns:
            List of tickers from cache, or None if cache invalid/expired
        """
        try:
            cache_file = self._cache_dir / "nasdaq_universe_cache.json"

            if not cache_file.exists():
                _logger.debug("Cache file not found: %s", cache_file)
                return None

            # Check age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() > (self.config.cache_ttl_hours * 3600):
                _logger.info("Cache expired (age: %.1f hours)", cache_age.total_seconds() / 3600)
                return None

            # Load cache
            with open(cache_file) as f:
                cache_data = json.load(f)

            if not isinstance(cache_data, dict) or "tickers" not in cache_data:
                _logger.warning("Invalid cache structure")
                return None

            # Invalidate cache when filtering-related config changes.
            # This avoids reusing old universes generated before new filters
            # (e.g. ETF exclusion) were introduced or toggled.
            cached_config = cache_data.get("config") if isinstance(cache_data.get("config"), dict) else {}
            expected_config = {
                "exclude_test_issues": self.config.exclude_test_issues,
                "exclude_etfs": getattr(self.config, "exclude_etfs", False),
                "alphabetic_only": self.config.alphabetic_only,
            }
            if cached_config != expected_config:
                _logger.info(
                    "Universe cache invalidated due to config mismatch (cached=%s, current=%s)",
                    cached_config,
                    expected_config,
                )
                return None

            tickers = cache_data["tickers"]
            if not isinstance(tickers, list):
                _logger.warning("Invalid ticker list in cache")
                return None

            _logger.info(
                "Loaded %d tickers from cache (created: %s)", len(tickers), cache_data.get("created_at", "unknown")
            )
            return tickers

        except Exception:
            _logger.exception("Error loading from cache:")
            return None

    def _save_to_cache(self, tickers: List[str]) -> None:
        """
        Save universe to cache.

        Args:
            tickers: Ticker list to cache
        """
        try:
            cache_file = self._cache_dir / "nasdaq_universe_cache.json"

            cache_data = {
                "tickers": tickers,
                "created_at": datetime.now().isoformat(),
                "config": {
                    "exclude_test_issues": self.config.exclude_test_issues,
                    "exclude_etfs": getattr(self.config, "exclude_etfs", False),
                    "alphabetic_only": self.config.alphabetic_only,
                },
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            _logger.info("Saved %d tickers to cache", len(tickers))

        except Exception:
            _logger.exception("Error saving to cache:")

    def clear_cache(self) -> None:
        """Clear the universe cache."""
        try:
            cache_file = self._cache_dir / "nasdaq_universe_cache.json"
            if cache_file.exists():
                cache_file.unlink()
                _logger.info("Universe cache cleared")
        except Exception:
            _logger.exception("Error clearing cache:")


def create_universe_downloader(
    config: UniverseConfig | None = None, results_dir: Path | None = None, target_date: str | None = None
) -> NasdaqUniverseDownloader:
    """Factory function."""
    return NasdaqUniverseDownloader(config, results_dir, target_date)
