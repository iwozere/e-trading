"""
Fundamental Filter

Applies fundamental screening filters using Finnhub data.
Filters by market cap, float, and average volume.

Features:
- Persistent cache for Finnhub profile2 data (3-day TTL)
- Checkpoint/resume capability for crash recovery
- Progress tracking and intermediate saves
"""

from pathlib import Path
import sys
from typing import List, Optional, Set
from datetime import datetime
import time

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.ml.pipeline.p06_emps2.config import EMPS2FilterConfig
from src.ml.pipeline.p06_emps2.fundamental_cache import FundamentalCache

_logger = setup_logger(__name__)


class FundamentalFilter:
    """
    Applies fundamental filters to stock universe.

    Uses Finnhub API to fetch:
    - Market capitalization
    - Float shares
    - Average volume
    - Sector information

    Implements:
    - Persistent cache (3-day TTL) to reduce API calls
    - Checkpoint/resume capability for crash recovery
    - Progress tracking and intermediate saves
    """

    def __init__(self, downloader: FinnhubDataDownloader, config: EMPS2FilterConfig):
        """
        Initialize fundamental filter.

        Args:
            downloader: Finnhub data downloader instance
            config: Filter configuration
        """
        self.downloader = downloader
        self.config = config

        # Results directory (dated)
        today = datetime.now().strftime('%Y-%m-%d')
        self._results_dir = Path("results") / "emps2" / today
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache
        if config.fundamental_cache_enabled:
            self.cache = FundamentalCache(cache_ttl_days=config.fundamental_cache_ttl_days)
            cache_stats = self.cache.get_stats()
            _logger.info("Fundamental cache: %d valid, %d expired entries",
                        cache_stats['valid'], cache_stats['expired'])
        else:
            self.cache = None

        _logger.info("Fundamental Filter initialized: cap=$%dM-$%dB, volume=%dK, float<%dM",
                    config.min_market_cap // 1_000_000,
                    config.max_market_cap // 1_000_000_000,
                    config.min_avg_volume // 1_000,
                    config.max_float // 1_000_000)

    def apply_filters(self, tickers: List[str], force_refresh: bool = False) -> pd.DataFrame:
        """
        Apply fundamental filters to ticker list.

        Args:
            tickers: List of ticker symbols
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            DataFrame with filtered tickers and fundamental data
        """
        try:
            _logger.info("Applying fundamental filters to %d tickers", len(tickers))

            # Check for existing checkpoint
            checkpoint_data = self._load_checkpoint()
            if checkpoint_data is not None and not force_refresh:
                _logger.info("Resuming from checkpoint with %d existing records", len(checkpoint_data))
                processed_tickers = set(checkpoint_data['ticker'].str.upper())
                remaining_tickers = [t for t in tickers if t.upper() not in processed_tickers]
                _logger.info("Already processed: %d, Remaining: %d",
                            len(processed_tickers), len(remaining_tickers))
            else:
                checkpoint_data = pd.DataFrame()
                remaining_tickers = tickers

            # Fetch fundamentals for remaining tickers
            if remaining_tickers:
                new_fundamentals = self._fetch_fundamentals_batch(remaining_tickers, force_refresh)
                new_df = pd.DataFrame(new_fundamentals)

                # Combine with checkpoint data
                if not checkpoint_data.empty and not new_df.empty:
                    df = pd.concat([checkpoint_data, new_df], ignore_index=True)
                elif not new_df.empty:
                    df = new_df
                else:
                    df = checkpoint_data
            else:
                df = checkpoint_data

            if df.empty:
                _logger.warning("No fundamental data retrieved")
                return df

            _logger.info("Total fundamentals retrieved: %d tickers", len(df))

            # Save raw data BEFORE filtering for inspection
            raw_output_path = self._results_dir / "fundamental_raw_data.csv"
            df.to_csv(raw_output_path, index=False)
            _logger.info("Saved raw fundamental data to: %s", raw_output_path)

            # Log sample data for debugging
            if len(df) > 0:
                sample = df.head(5)
                _logger.info("Sample raw data (first 5 tickers):")
                for idx, row in sample.iterrows():
                    _logger.info("  %s: market_cap=$%.2fM, volume=%.0f, float=%.2fM",
                                row['ticker'],
                                row['market_cap'] / 1_000_000 if pd.notna(row['market_cap']) else 0,
                                row['avg_volume'] if pd.notna(row['avg_volume']) else 0,
                                row['float'] / 1_000_000 if pd.notna(row['float']) else 0)

            # Apply filters
            df_filtered = self._apply_fundamental_filters(df)

            # Save results
            self._save_results(df_filtered)

            # Clear checkpoint after successful completion
            self._clear_checkpoint()

            _logger.info("After fundamental filtering: %d tickers (%.1f%%)",
                        len(df_filtered),
                        100.0 * len(df_filtered) / len(tickers) if tickers else 0)

            return df_filtered

        except Exception:
            _logger.exception("Error applying fundamental filters:")
            return pd.DataFrame()

    def _fetch_fundamentals_batch(self, tickers: List[str], force_refresh: bool = False) -> List[dict]:
        """
        Fetch fundamentals for multiple tickers with caching, checkpointing, and rate limiting.

        Args:
            tickers: List of ticker symbols
            force_refresh: If True, bypass cache

        Returns:
            List of fundamental data dictionaries
        """
        fundamentals = []
        total_tickers = len(tickers)
        processed = 0
        failed = 0
        cache_hits = 0

        _logger.info("Fetching fundamentals for %d tickers (cache: %s, checkpoints: every %d)",
                    total_tickers,
                    "enabled" if self.config.fundamental_cache_enabled else "disabled",
                    self.config.checkpoint_interval)

        for ticker in tickers:
            try:
                # Try cache first (unless force_refresh)
                profile_data = None
                if self.cache and not force_refresh:
                    profile_data = self.cache.get(ticker)
                    if profile_data:
                        cache_hits += 1

                # Fetch from API if not cached
                if not profile_data:
                    profile_data = self._fetch_ticker_profile(ticker)

                    # Save to cache
                    if profile_data and self.cache:
                        self.cache.set(ticker, profile_data)

                    # Rate limiting (only when fetching from API)
                    time.sleep(1.1)

                if profile_data:
                    fundamentals.append(profile_data)
                else:
                    failed += 1

                processed += 1

                # Progress logging
                if processed % 100 == 0:
                    _logger.info("Progress: %d/%d (%.1f%%), successful: %d, failed: %d, cache hits: %d",
                                processed, total_tickers,
                                100.0 * processed / total_tickers,
                                len(fundamentals), failed, cache_hits)

                # Checkpoint save
                if self.config.checkpoint_enabled and processed % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(fundamentals)

            except KeyboardInterrupt:
                _logger.warning("Interrupted by user. Saving checkpoint...")
                self._save_checkpoint(fundamentals)
                raise

            except Exception:
                _logger.exception("Error fetching fundamentals for %s:", ticker)
                failed += 1
                continue

        # Final checkpoint
        if self.config.checkpoint_enabled and fundamentals:
            self._save_checkpoint(fundamentals)

        _logger.info("Fetched fundamentals: %d successful, %d failed, %d cache hits (%.1f%% cache hit rate)",
                    len(fundamentals), failed, cache_hits,
                    100.0 * cache_hits / total_tickers if total_tickers > 0 else 0)

        return fundamentals

    def _fetch_ticker_profile(self, ticker: str) -> Optional[dict]:
        """
        Fetch profile and quote data for a single ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with fundamental data or None if failed
        """
        try:
            # Use get_fundamentals method from Finnhub downloader
            fundamentals = self.downloader.get_fundamentals(ticker)

            if not fundamentals or fundamentals.market_cap == 0:
                return None

            # Extract relevant fields
            # NOTE: market_cap and shares_outstanding are already in MILLIONS from Finnhub
            profile_data = {
                'ticker': ticker.upper(),
                'market_cap': fundamentals.market_cap * 1_000_000 if fundamentals.market_cap else None,  # Already in millions, convert to dollars
                'float': fundamentals.shares_outstanding * 1_000_000 if fundamentals.shares_outstanding else None,  # Already in millions, convert to shares
                'sector': fundamentals.sector,
                'current_price': fundamentals.current_price,
            }

            # Get average volume from metrics API
            # Finnhub's /quote endpoint returns None for volume
            # Use /metrics endpoint which has 10DayAverageTradingVolume in millions
            import requests
            from config.donotshare.donotshare import FINNHUB_KEY

            metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_KEY}"
            metrics_response = requests.get(metrics_url, timeout=10)

            if metrics_response.status_code == 200:
                metrics_data = metrics_response.json()
                # Get 10-day average volume (in millions), convert to shares
                avg_vol_millions = metrics_data.get('metric', {}).get('10DayAverageTradingVolume', 0)
                profile_data['avg_volume'] = avg_vol_millions * 1_000_000 if avg_vol_millions else 0
            else:
                profile_data['avg_volume'] = 0

            return profile_data

        except Exception:
            _logger.debug("Failed to fetch profile for %s", ticker)
            return None

    def _apply_fundamental_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fundamental filter criteria.

        Args:
            df: DataFrame with fundamental data

        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)

        # Drop rows with missing critical data
        df = df.dropna(subset=['market_cap', 'avg_volume'])
        _logger.info("After dropping missing data: %d tickers", len(df))

        # Market cap filter
        df = df[
            (df['market_cap'] >= self.config.min_market_cap) &
            (df['market_cap'] <= self.config.max_market_cap)
        ]
        _logger.info("After market cap filter ($%dM-$%dB): %d tickers",
                    self.config.min_market_cap // 1_000_000,
                    self.config.max_market_cap // 1_000_000_000,
                    len(df))

        # Volume filter
        df = df[df['avg_volume'] >= self.config.min_avg_volume]
        _logger.info("After volume filter (>%dK): %d tickers",
                    self.config.min_avg_volume // 1_000,
                    len(df))

        # Float filter (if available)
        if 'float' in df.columns:
            df_with_float = df[df['float'].notna()]
            df_with_float = df_with_float[df_with_float['float'] <= self.config.max_float]

            # Keep tickers without float data (conservative approach)
            df_without_float = df[df['float'].isna()]
            df = pd.concat([df_with_float, df_without_float], ignore_index=True)

            _logger.info("After float filter (<%dM): %d tickers (kept %d without float data)",
                        self.config.max_float // 1_000_000,
                        len(df),
                        len(df_without_float))

        _logger.info("Fundamental filtering removed %d tickers (%.1f%%)",
                    initial_count - len(df),
                    100.0 * (initial_count - len(df)) / initial_count if initial_count > 0 else 0)

        return df

    def _save_results(self, df: pd.DataFrame) -> None:
        """
        Save filtered results to CSV.

        Args:
            df: Filtered DataFrame
        """
        try:
            output_path = self._results_dir / "fundamental_filtered.csv"
            df.to_csv(output_path, index=False)
            _logger.info("Saved fundamental filter results to: %s", output_path)

        except Exception:
            _logger.exception("Error saving fundamental filter results:")

    def _save_checkpoint(self, fundamentals: List[dict]) -> None:
        """
        Save checkpoint for resume capability.

        Args:
            fundamentals: List of fundamental data dicts
        """
        if not self.config.checkpoint_enabled:
            return

        try:
            checkpoint_path = self._results_dir / "fundamental_checkpoint.csv"
            df = pd.DataFrame(fundamentals)

            if not df.empty:
                df.to_csv(checkpoint_path, index=False)
                _logger.info("Checkpoint saved: %d records", len(df))

        except Exception:
            _logger.exception("Error saving checkpoint:")

    def _load_checkpoint(self) -> Optional[pd.DataFrame]:
        """
        Load existing checkpoint if available.

        Returns:
            DataFrame from checkpoint or None
        """
        if not self.config.checkpoint_enabled:
            return None

        try:
            checkpoint_path = self._results_dir / "fundamental_checkpoint.csv"

            if not checkpoint_path.exists():
                return None

            df = pd.read_csv(checkpoint_path)

            if df.empty:
                return None

            _logger.info("Loaded checkpoint: %d records", len(df))
            return df

        except Exception:
            _logger.debug("Error loading checkpoint (will start fresh)")
            return None

    def _clear_checkpoint(self) -> None:
        """Clear checkpoint file after successful completion."""
        if not self.config.checkpoint_enabled:
            return

        try:
            checkpoint_path = self._results_dir / "fundamental_checkpoint.csv"

            if checkpoint_path.exists():
                checkpoint_path.unlink()
                _logger.info("Checkpoint cleared")

        except Exception:
            _logger.debug("Error clearing checkpoint")


def create_fundamental_filter(
    downloader: FinnhubDataDownloader,
    config: EMPS2FilterConfig
) -> FundamentalFilter:
    """
    Factory function to create fundamental filter.

    Args:
        downloader: Finnhub data downloader instance
        config: Filter configuration

    Returns:
        FundamentalFilter instance
    """
    return FundamentalFilter(downloader, config)
