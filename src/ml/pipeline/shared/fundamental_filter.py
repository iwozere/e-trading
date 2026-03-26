from pathlib import Path
import sys
from typing import List, Optional, Set, Dict, Any
from datetime import datetime
import time

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.data_manager import DataManager
from .config import FundamentalFilterConfig

_logger = setup_logger(__name__)


class FundamentalFilter:
    """
    Applies fundamental filters to stock universe.

    Uses DataManager to fetch:
    - Market capitalization
    - Float shares
    - Average volume
    - Sector information

    Implements:
    - DataManager-based centralized caching (7-day TTL)
    - Checkpoint/resume capability for crash recovery
    - Progress tracking and intermediate saves
    """

    def __init__(self,
                 data_manager: DataManager,
                 config: FundamentalFilterConfig,
                 results_dir: Path,
                 checkpoint_enabled: bool = True,
                 checkpoint_interval: int = 100):
        """
        Initialize fundamental filter.

        Args:
            data_manager: DataManager instance for data retrieval and caching
            config: Filter configuration
            results_dir: Directory to save results and checkpoints
            checkpoint_enabled: Whether to enable checkpoints
            checkpoint_interval: Checkpoint save interval
        """
        self.data_manager = data_manager
        self.config = config
        self._results_dir = results_dir
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint_interval = checkpoint_interval
        self.ttl_days = getattr(config, 'fundamental_cache_ttl_days', 14)

        self._results_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Fundamental Filter initialized: cap=$%dM-$%dB, volume=%dK, float<%dM",
                    config.min_market_cap // 1_000_000,
                    config.max_market_cap // 1_000_000_000,
                    config.min_avg_volume // 1_000,
                    config.max_float // 1_000_000)
        _logger.info("Results directory: %s", self._results_dir)

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
        Fetch fundamentals for multiple tickers with caching and checkpointing.
        """
        fundamentals = []
        total_tickers = len(tickers)
        processed = 0
        failed = 0

        _logger.info("Fetching fundamentals for %d tickers via DataManager", total_tickers)

        for ticker in tickers:
            try:
                profile_data = self._fetch_ticker_profile(ticker, force_refresh)

                if profile_data:
                    fundamentals.append(profile_data)

                    # Save checkpoint
                    if self.checkpoint_enabled and len(fundamentals) % self.checkpoint_interval == 0:
                        self._save_checkpoint(fundamentals)
                else:
                    failed += 1

                processed += 1

                if processed % 100 == 0:
                    _logger.info("Progress: %d/%d (%.1f%%), successful: %d, failed: %d",
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

        return fundamentals

    def _fetch_ticker_profile(self, ticker: str, force_refresh: bool = False) -> Optional[dict]:
        """
        Fetch fundamentals for a single ticker using DataManager.
        """
        try:
            # Use DataManager to get combined fundamentals (includes profile, metrics, quote)
            # This uses the centralized cache in c:\data-cache
            data = self.data_manager.get_fundamentals(
                ticker, 
                force_refresh=force_refresh,
                max_age_days=self.ttl_days
            )
            
            if not data:
                return None

            # The combined data may contain a nested 'profile' dict from FMP
            # with additional fields not present at the top level
            profile = data.get('profile', {}) or {}

            # Map DataManager fields to FundamentalFilter expected format
            # with fallbacks from nested profile dict
            # Use explicit None/zero checks — 0.0 from a bad provider should fall back
            market_cap = data.get('market_cap')
            if not market_cap:  # None or 0.0
                market_cap = profile.get('marketCap')

            shares = data.get('shares_outstanding')
            if not shares:
                shares = profile.get('sharesOutstanding')

            avg_vol = data.get('avg_volume')
            if not avg_vol:
                avg_vol = profile.get('averageVolume') or profile.get('volume')

            current_price = data.get('current_price')
            if not current_price:
                current_price = profile.get('price')

            sector = data.get('sector')
            if not sector:
                sector = profile.get('sector')
            
            # Fallbacks and calculations if some fields are missing but derivable
            if not market_cap and shares and current_price:
                market_cap = shares * current_price
            
            if market_cap is None or avg_vol is None:
                _logger.debug("Ticker %s missing critical fundamentals: cap=%s, vol=%s", 
                            ticker, market_cap, avg_vol)
                return None

            return {
                'ticker': ticker.upper(),
                'market_cap': market_cap,
                'float': shares, # Using shares as proxy for float as before
                'sector': sector,
                'current_price': current_price or 0,
                'avg_volume': avg_vol,
            }

        except Exception:
            _logger.debug("Failed to process ticker %s", ticker)
            return None

    def _apply_fundamental_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fundamental filter criteria."""
        initial_count = len(df)
        df = df.dropna(subset=['market_cap', 'avg_volume'])
        
        df = df[
            (df['market_cap'] >= self.config.min_market_cap) &
            (df['market_cap'] <= self.config.max_market_cap)
        ]
        
        df = df[df['avg_volume'] >= self.config.min_avg_volume]

        if 'float' in df.columns:
            df_with_float = df[df['float'].notna()]
            df_with_float = df_with_float[df_with_float['float'] <= self.config.max_float]
            df_without_float = df[df['float'].isna()]
            df = pd.concat([df_with_float, df_without_float], ignore_index=True)

        _logger.info("Fundamental filtering: %d -> %d tickers", initial_count, len(df))
        return df

    def _save_results(self, df: pd.DataFrame) -> None:
        """Save filtered results to CSV."""
        try:
            output_path = self._results_dir / "03_fundamental_filtered.csv"
            df.to_csv(output_path, index=False)
            _logger.info("Saved fundamental filter results to: %s", output_path)
        except Exception:
            _logger.exception("Error saving results:")

    def _save_checkpoint(self, fundamentals: List[dict]) -> None:
        """Save checkpoint for resume capability."""
        if not self.checkpoint_enabled or not fundamentals:
            return
        try:
            checkpoint_path = self._results_dir / "fundamental_checkpoint.csv"
            pd.DataFrame(fundamentals).to_csv(checkpoint_path, index=False)
        except Exception:
            _logger.exception("Error saving checkpoint:")

    def _load_checkpoint(self) -> Optional[pd.DataFrame]:
        """Load existing checkpoint if available."""
        if not self.checkpoint_enabled:
            return None
        try:
            checkpoint_path = self._results_dir / "fundamental_checkpoint.csv"
            if checkpoint_path.exists():
                df = pd.read_csv(checkpoint_path)
                return df if not df.empty else None
        except Exception:
            return None
        return None

    def _clear_checkpoint(self) -> None:
        """Clear checkpoint file after successful completion."""
        if not self.checkpoint_enabled:
            return
        try:
            checkpoint_path = self._results_dir / "fundamental_checkpoint.csv"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        except Exception:
            pass


def create_fundamental_filter(
    data_manager: DataManager,
    config: FundamentalFilterConfig,
    results_dir: Path
) -> FundamentalFilter:
    """Factory function to create fundamental filter."""
    return FundamentalFilter(data_manager, config, results_dir)
