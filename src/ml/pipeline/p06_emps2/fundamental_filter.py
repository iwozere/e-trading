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

    def __init__(self,
                 downloader: FinnhubDataDownloader,
                 config: EMPS2FilterConfig,
                 target_date: Optional[str] = None,
                 cache_enabled: bool = True,
                 cache_ttl_days: int = 3,
                 checkpoint_enabled: bool = True,
                 checkpoint_interval: int = 100):
        """
        Initialize fundamental filter.

        Args:
            downloader: Finnhub data downloader instance
            config: Filter configuration
            target_date: Target trading date (YYYY-MM-DD). Defaults to today.
            cache_enabled: Whether to enable caching
            cache_ttl_days: Cache TTL in days
            checkpoint_enabled: Whether to enable checkpoints
            checkpoint_interval: Checkpoint save interval
        """
        self.downloader = downloader
        self.config = config
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint_interval = checkpoint_interval

        # Results directory (dated)
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        self._results_dir = Path("results") / "emps2" / target_date
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache (with negative caching support)
        # Initialize cache (with negative caching support)
        if cache_enabled:
            self.cache = FundamentalCache(
                cache_ttl_days=cache_ttl_days,
                negative_cache_ttl_days=2  # 2 days for failed/empty responses
            )
            cache_stats = self.cache.get_stats()
            _logger.info("Fundamental cache: %d valid (%d positive, %d negative), %d expired",
                        cache_stats['valid'], cache_stats['positive'],
                        cache_stats['negative'], cache_stats['expired'])
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
            raw_output_path = self._results_dir / "02_fundamental_raw_data.csv"
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
        cache_hits = 0  # Re-adding for logging compatibility if needed
        negative_cache_skipped = 0

        _logger.info("Fetching fundamentals for %d tickers (cache enabled, tiered optimization active)", total_tickers)

        for ticker in tickers:
            try:
                # Try cache first (unless force_refresh)
                profile_data = self._fetch_ticker_profile(ticker, force_refresh)

                if profile_data:
                    fundamentals.append(profile_data)

                    # Save checkpoint after successful fetch
                    if self.checkpoint_enabled:
                        self._save_checkpoint(fundamentals)
                else:
                    failed += 1

                processed += 1

                # Progress logging
                if processed % 100 == 0:
                    _logger.info("Progress: %d/%d (%.1f%%), successful: %d, failed/skipped: %d",
                                processed, total_tickers,
                                100.0 * processed / total_tickers,
                                len(fundamentals), failed)

            except KeyboardInterrupt:
                _logger.warning("Interrupted by user. Saving checkpoint...")
                self._save_checkpoint(fundamentals)
                raise

            except Exception:
                _logger.exception("Error fetching fundamentals for %s:", ticker)
                failed += 1
                continue

        # Final checkpoint
        if self.checkpoint_enabled and fundamentals:
            self._save_checkpoint(fundamentals)

        _logger.info("Fetched fundamentals: %d successful, %d failed/skipped",
                    len(fundamentals), failed)

        return fundamentals

    def _fetch_ticker_profile(self, ticker: str, force_refresh: bool = False) -> Optional[dict]:
        """
        Fetch profile and quote data for a single ticker with pre-filtering.
        """
        try:
            # 1. Background Check (Check cache for all components)
            profile_data = None
            metrics_data = None
            quote_data = None

            if self.cache and not force_refresh:
                profile_data = self.cache.get_data(ticker, 'profile')
                metrics_data = self.cache.get_data(ticker, 'metrics')
                quote_data = self.cache.get_data(ticker, 'quote')

                # Check negative cache
                if profile_data and profile_data.get('is_negative_cache'):
                    _logger.debug("Skipping %s (negative cache hit)", ticker)
                    return None

            # 2. Fetch missing Profile/Metrics if necessary
            if not profile_data:
                profile_data = self.downloader.get_company_profile(ticker)
                if not profile_data:
                    if self.cache: self.cache.set_data(ticker, 'profile', None)
                    return None
                if self.cache: self.cache.set_data(ticker, 'profile', profile_data)

            if not metrics_data:
                metrics_data = self.downloader.get_basic_financials(ticker)
                if not metrics_data:
                    # Non-critical metrics failure? No, we need avg volume.
                    return None
                if self.cache: self.cache.set_data(ticker, 'metrics', metrics_data)

            # 3. Pre-Filter (Step 2 of optimization plan)
            # Use Profile/Metrics to estimate if ticker is worth a Quote call
            shares = profile_data.get('shareOutstanding', 0) * 1_000_000
            metrics = metrics_data.get('metric', {})
            avg_vol = metrics.get('10DayAverageTradingVolume', 0) * 1_000_000

            # Estimate market cap using cached price if available, otherwise use profile marketCap
            last_price = quote_data.get('c', 0) if quote_data else 0
            market_cap_est = shares * last_price if last_price > 0 else profile_data.get('marketCapitalization', 0) * 1_000_000

            # Pre-filter checks (with 20% buffer to avoid accidental drops)
            if avg_vol < self.config.min_avg_volume * 0.8:
                _logger.debug("Pre-filter skip %s: Volume %d < %d", ticker, avg_vol, self.config.min_avg_volume)
                return None

            if market_cap_est < self.config.min_market_cap * 0.8:
                _logger.debug("Pre-filter skip %s: Market Cap Est $%dM < $%dM",
                            ticker, market_cap_est // 1_000_000, self.config.min_market_cap // 1_000_000)
                return None

            # 4. Intentional Call (Step 3 of optimization plan)
            # Fetch fresh quote ONLY for survivors
            if not quote_data:
                quote_data = self.downloader.get_quote(ticker)
                if not quote_data:
                    return None
                if self.cache: self.cache.set_data(ticker, 'quote', quote_data)

            # 5. Build final record
            return {
                'ticker': ticker.upper(),
                'market_cap': shares * quote_data.get('c', 0) if quote_data.get('c') else profile_data.get('marketCapitalization', 0) * 1_000_000,
                'float': shares, # Finnhub doesn't have float, using shares as proxy
                'sector': profile_data.get('finnhubIndustry'),
                'current_price': quote_data.get('c', 0),
                'avg_volume': avg_vol,
            }

        except Exception:
            _logger.debug("Failed to process ticker %s", ticker)
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
            output_path = self._results_dir / "03_fundamental_filtered.csv"
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
        if not self.checkpoint_enabled:
            return

        try:
            checkpoint_path = self._results_dir / "fundamental_checkpoint.csv"
            df = pd.DataFrame(fundamentals)

            if not df.empty:
                df.to_csv(checkpoint_path, index=False)
                # Only log every 10 saves to reduce noise
                if len(df) % 10 == 0:
                    _logger.info("Checkpoint saved: %d records", len(df))

        except Exception:
            _logger.exception("Error saving checkpoint:")

    def _load_checkpoint(self) -> Optional[pd.DataFrame]:
        """
        Load existing checkpoint if available.

        Returns:
            DataFrame from checkpoint or None
        """
        if not self.checkpoint_enabled:
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
        if not self.checkpoint_enabled:
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
