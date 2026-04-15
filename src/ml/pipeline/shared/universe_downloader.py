"""
NASDAQ Universe Downloader

Downloads complete US stock universe from NASDAQ Trader FTP service.
Stores results in results/emps2/YYYY-MM-DD/ for date-based organization.
"""

from pathlib import Path
import sys
from typing import List, Optional
from datetime import datetime
import io
import json
from ftplib import FTP

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from .config import UniverseConfig

_logger = setup_logger(__name__)


class NasdaqUniverseDownloader:
    """
    Downloads and caches the complete US stock universe from NASDAQ Trader.

    Fetches from:
    - NASDAQ listed stocks
    - NYSE, AMEX, and other exchange listings
    """

    def __init__(self, config: Optional[UniverseConfig] = None, results_dir: Optional[Path] = None, target_date: Optional[str] = None):
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
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

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
            _logger.info("Downloading from NASDAQ Trader FTP...")

            # Connect to FTP server
            ftp = FTP('ftp.nasdaqtrader.com', timeout=30)
            ftp.login()  # Anonymous login
            _logger.debug("Connected to FTP server: %s", ftp.getwelcome())

            # Change to SymbolDirectory
            ftp.cwd('SymbolDirectory')

            # Download NASDAQ listed
            _logger.debug("Fetching nasdaqlisted.txt")
            nasdaq_data = []
            ftp.retrlines('RETR nasdaqlisted.txt', nasdaq_data.append)
            nasdaq_text = '\n'.join(nasdaq_data)
            df1 = pd.read_csv(io.StringIO(nasdaq_text), sep="|")
            _logger.debug("Downloaded %d rows from nasdaqlisted.txt", len(df1))

            # Download other listed (NYSE, AMEX, etc.)
            _logger.debug("Fetching otherlisted.txt")
            other_data = []
            ftp.retrlines('RETR otherlisted.txt', other_data.append)
            other_text = '\n'.join(other_data)
            df2 = pd.read_csv(io.StringIO(other_text), sep="|")
            _logger.debug("Downloaded %d rows from otherlisted.txt", len(df2))

            ftp.quit()

            # Combine
            df = pd.concat([df1, df2], ignore_index=True)
            _logger.info("Downloaded %d total symbols", len(df))
            _logger.debug("Columns: %s", df.columns.tolist())

            # Drop any completely empty rows (from trailing pipes in file)
            df = df.dropna(how='all')

            # Filter test issues
            if self.config.exclude_test_issues and "Test Issue" in df.columns:
                initial_count = len(df)
                df = df[df["Test Issue"] == "N"]
                _logger.info("Excluded %d test issues", initial_count - len(df))

            # Optionally exclude ETFs early to reduce unnecessary downstream
            # fundamentals lookups and related debug noise.
            if getattr(self.config, "exclude_etfs", False):
                etf_col = next((col for col in df.columns if str(col).strip().upper() == "ETF"), None)
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
            _logger.exception("Error downloading from NASDAQ Trader FTP:")
            _logger.error("Unable to download from NASDAQ Trader. Please check network/firewall settings.")
            _logger.info("You can try manual download using: python src/ml/pipeline/p06_emps2/download_nasdaq_manual.py")
            return []

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
                df_filtered = pd.DataFrame({'Symbol': tickers})

            df_filtered.to_csv(output_path, index=False)
            _logger.info("Saved %d tickers to: %s", len(df_filtered), output_path)

        except Exception:
            _logger.exception("Error saving universe file:")

    def _load_from_cache(self) -> Optional[List[str]]:
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
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            if not isinstance(cache_data, dict) or 'tickers' not in cache_data:
                _logger.warning("Invalid cache structure")
                return None

            # Invalidate cache when filtering-related config changes.
            # This avoids reusing old universes generated before new filters
            # (e.g. ETF exclusion) were introduced or toggled.
            cached_config = cache_data.get('config') if isinstance(cache_data.get('config'), dict) else {}
            expected_config = {
                'exclude_test_issues': self.config.exclude_test_issues,
                'exclude_etfs': getattr(self.config, "exclude_etfs", False),
                'alphabetic_only': self.config.alphabetic_only,
            }
            if cached_config != expected_config:
                _logger.info(
                    "Universe cache invalidated due to config mismatch (cached=%s, current=%s)",
                    cached_config,
                    expected_config
                )
                return None

            tickers = cache_data['tickers']
            if not isinstance(tickers, list):
                _logger.warning("Invalid ticker list in cache")
                return None

            _logger.info("Loaded %d tickers from cache (created: %s)",
                        len(tickers), cache_data.get('created_at', 'unknown'))
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
                'tickers': tickers,
                'created_at': datetime.now().isoformat(),
                'config': {
                    'exclude_test_issues': self.config.exclude_test_issues,
                    'exclude_etfs': getattr(self.config, "exclude_etfs", False),
                    'alphabetic_only': self.config.alphabetic_only
                }
            }

            with open(cache_file, 'w') as f:
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


def create_universe_downloader(config: Optional[UniverseConfig] = None,
                               results_dir: Optional[Path] = None,
                               target_date: Optional[str] = None) -> NasdaqUniverseDownloader:
    """Factory function."""
    return NasdaqUniverseDownloader(config, results_dir, target_date)
