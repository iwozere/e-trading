"""
Universe Loader for Short Squeeze Detection Pipeline

This module provides functionality to load and filter the universe of stocks
for short squeeze analysis using FMP data provider.
"""

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.ml.pipeline.p04_short_squeeze.config.data_classes import UniverseConfig

_logger = setup_logger(__name__)


class UniverseLoader:
    """
    Universe loader for short squeeze detection pipeline.

    Loads and filters stocks from FMP based on market cap, volume, and exchange criteria.
    Optionally filters using FINRA short interest data from database (no API calls).
    Includes caching functionality for performance optimization.
    """

    def __init__(self, fmp_downloader: FMPDataDownloader, config: UniverseConfig):
        """
        Initialize Universe Loader.

        Args:
            fmp_downloader: FMP data downloader instance
            config: Universe configuration with filtering criteria
        """
        self.fmp_downloader = fmp_downloader
        self.config = config
        self._cache_dir = Path("data/cache/universe")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_hours = 24  # Cache universe for 24 hours

        _logger.info("Universe Loader initialized with config: min_cap=%s, max_cap=%s, min_volume=%s",
                    self.config.min_market_cap, self.config.max_market_cap, self.config.min_avg_volume)

    def load_universe(self) -> List[str]:
        """
        Load the complete universe of stocks for short squeeze analysis.

        Returns:
            List of ticker symbols that meet all filtering criteria
        """
        try:
            _logger.info("Loading universe with filtering criteria")

            # Check cache first
            cached_universe = self._load_from_cache()
            if cached_universe:
                _logger.info("Loaded %d tickers from cache", len(cached_universe))
                return cached_universe

            # Load from FMP screener
            universe = self._load_from_screener()

            # Apply additional filters
            filtered_universe = self._apply_filters(universe)

            # Cache the results
            self._save_to_cache(filtered_universe)

            _logger.info("Successfully loaded universe of %d tickers", len(filtered_universe))
            return filtered_universe

        except Exception:
            _logger.exception("Error loading universe:")
            return []

    def _load_from_screener(self) -> List[str]:
        """
        Load universe from FMP stock screener with optimized criteria for short squeeze detection.

        Returns:
            List of ticker symbols from screener
        """
        try:
            # Use multiple screening strategies to get better candidates
            all_tickers = set()

            # Calculate mid-cap range (between min and max from config)
            mid_cap_min = max(self.config.min_market_cap, 500_000_000)  # At least $500M for mid-cap
            mid_cap_max = min(self.config.max_market_cap, 5_000_000_000)  # Max $5B for mid-cap

            # Calculate strategy limits from config
            max_universe = getattr(self.config, 'max_universe_size', 1000)
            strategy_1_limit = min(500, max_universe // 2)  # Up to half for mid-cap
            strategy_2_limit = min(300, max_universe // 3)  # Up to third for small-cap

            # Strategy 1: High volume, mid-cap stocks (most likely to have short interest data)
            criteria_1 = {
                'marketCapMoreThan': mid_cap_min,
                'marketCapLowerThan': mid_cap_max,
                'volumeMoreThan': max(self.config.min_avg_volume * 2, 1_000_000),  # 2x min volume or 1M
                'exchange': ','.join(self.config.exchanges),
                'limit': strategy_1_limit
            }

            _logger.info("Loading high-volume mid-cap universe: %s", criteria_1)
            tickers_1 = self.fmp_downloader.load_universe_from_screener(criteria_1)
            all_tickers.update(tickers_1)
            _logger.info("Strategy 1: %d tickers", len(tickers_1))

            # Strategy 2: Small-cap high volume (potential squeeze candidates)
            small_cap_max = min(mid_cap_min, 1_000_000_000)  # Max $1B for small-cap
            criteria_2 = {
                'marketCapMoreThan': self.config.min_market_cap,
                'marketCapLowerThan': small_cap_max,
                'volumeMoreThan': max(self.config.min_avg_volume, 500_000),  # Config volume or 500K
                'exchange': ','.join(self.config.exchanges),
                'limit': strategy_2_limit
            }

            _logger.info("Loading small-cap high-volume universe: %s", criteria_2)
            tickers_2 = self.fmp_downloader.load_universe_from_screener(criteria_2)
            all_tickers.update(tickers_2)
            _logger.info("Strategy 2: %d additional tickers", len(tickers_2) - len(all_tickers & set(tickers_2)))

            # Strategy 3: Add known high short interest sectors/stocks
            known_candidates = self._get_known_short_interest_candidates()
            all_tickers.update(known_candidates)
            _logger.info("Strategy 3: Added %d known candidates", len(known_candidates))

            final_tickers = list(all_tickers)

            # Apply max universe size limit from config
            if len(final_tickers) > max_universe:
                _logger.info("Limiting universe from %d to %d tickers per config", len(final_tickers), max_universe)
                final_tickers = final_tickers[:max_universe]

            _logger.info("Combined universe: %d unique tickers", len(final_tickers))
            return final_tickers

        except Exception:
            _logger.exception("Error loading from screener:")
            # Fallback to original method
            return self._load_from_screener_fallback()

    def _load_from_screener_fallback(self) -> List[str]:
        """Fallback to original screener method."""
        try:
            criteria = {
                'marketCapMoreThan': self.config.min_market_cap,
                'marketCapLowerThan': self.config.max_market_cap,
                'volumeMoreThan': self.config.min_avg_volume,
                'exchange': ','.join(self.config.exchanges),
                'limit': 1000
            }

            tickers = self.fmp_downloader.load_universe_from_screener(criteria)
            _logger.info("Fallback screener returned %d tickers", len(tickers))
            return tickers

        except Exception:
            _logger.exception("Error in fallback screener:")
            return []

    def _get_known_short_interest_candidates(self) -> List[str]:
        """
        Get a curated list of stocks known to frequently have short interest data.

        Returns:
            List of ticker symbols with historically high short interest
        """
        # These are stocks that typically have active short interest and are good for testing
        known_candidates = [
            # Meme stocks / frequently shorted
            'GME', 'AMC', 'BBBY', 'CLOV', 'WISH', 'PLTR', 'SPCE',

            # High-profile tech stocks often shorted
            'TSLA', 'NKLA', 'RIVN', 'LCID', 'HOOD', 'COIN',

            # Biotech (often heavily shorted)
            'MRNA', 'BNTX', 'NVAX', 'SAVA', 'BIIB', 'GILD',

            # Energy/EV (volatile sector)
            'PLUG', 'FCEL', 'BLNK', 'CHPT', 'QS', 'HYLN',

            # SPACs and growth stocks
            'DKNG', 'OPEN', 'SKLZ', 'SOFI', 'UPST', 'AFRM',

            # Traditional high short interest sectors
            'NFLX', 'SHOP', 'ZM', 'PTON', 'ROKU', 'SQ'
        ]

        _logger.info("Added %d known short interest candidates", len(known_candidates))
        return known_candidates

    def _apply_filters(self, tickers: List[str]) -> List[str]:
        """
        Apply additional filters to the ticker list, including optional short interest availability check.

        Args:
            tickers: List of ticker symbols to filter

        Returns:
            Filtered list of ticker symbols
        """
        try:
            if not tickers:
                return []

            _logger.info("Applying additional filters to %d tickers", len(tickers))

            # Step 1: Basic ticker validation
            valid_tickers = []
            for ticker in tickers:
                if self._is_valid_ticker(ticker):
                    valid_tickers.append(ticker)

            _logger.info("After basic validation: %d tickers", len(valid_tickers))

            # Step 2: Check if FINRA filtering is enabled in config
            use_finra_filtering = getattr(self.config, 'use_finra_filtering', False)

            if use_finra_filtering and len(valid_tickers) > 100:
                _logger.info("FINRA filtering enabled - pre-filtering using database data (no API calls)")
                filtered_tickers = self._pre_filter_short_interest_availability(valid_tickers)
            else:
                if not use_finra_filtering:
                    _logger.info("FINRA filtering disabled in config - using known candidates approach")
                else:
                    _logger.info("Small universe (%d tickers) - using known candidates approach", len(valid_tickers))
                filtered_tickers = self._fallback_to_known_candidates(valid_tickers)

            _logger.info("Final filtered universe: %d tickers", len(filtered_tickers))
            return filtered_tickers

        except Exception:
            _logger.exception("Error applying filters:")
            return tickers  # Return original list on error

    def _pre_filter_short_interest_availability(self, tickers: List[str]) -> List[str]:
        """
        Pre-filter tickers using FINRA data from database to check short interest availability.

        Args:
            tickers: List of ticker symbols to check

        Returns:
            Filtered list prioritizing tickers with FINRA short interest data
        """
        try:
            _logger.info("Pre-filtering %d tickers using FINRA data from database", len(tickers))

            # Get bulk FINRA data from database (no API calls)
            finra_data = self._get_finra_data_from_db(tickers)

            if not finra_data:
                _logger.warning("No FINRA data available in database, using known candidates approach")
                return self._fallback_to_known_candidates(tickers)

            # Separate tickers with and without FINRA data
            tickers_with_finra = []
            tickers_without_finra = []

            for ticker in tickers:
                ticker_upper = ticker.upper()
                if ticker_upper in finra_data:
                    # Check if the ticker has meaningful short interest data
                    ticker_data = finra_data[ticker_upper]
                    short_interest_pct = ticker_data.get('short_interest_pct', 0)
                    days_to_cover = ticker_data.get('days_to_cover', 0)

                    # Only include if there's actual short interest activity
                    if short_interest_pct > 0 and days_to_cover > 0:
                        tickers_with_finra.append(ticker)
                    else:
                        tickers_without_finra.append(ticker)
                else:
                    tickers_without_finra.append(ticker)

            _logger.info("FINRA data availability: %d with data, %d without data",
                        len(tickers_with_finra), len(tickers_without_finra))

            # Prioritize tickers with FINRA data, add known candidates from remaining
            known_candidates = self._get_known_short_interest_candidates()
            known_without_finra = [t for t in tickers_without_finra if t.upper() in [k.upper() for k in known_candidates]]

            # Combine: FINRA tickers first, then known candidates, then limit others
            final_list = tickers_with_finra + known_without_finra

            # Add some others if we don't have enough
            if len(final_list) < 200:
                other_tickers = [t for t in tickers_without_finra if t not in known_without_finra]
                remaining_slots = 200 - len(final_list)
                final_list.extend(other_tickers[:remaining_slots])

            _logger.info("Final universe: %d with FINRA data + %d known candidates + %d others = %d total",
                        len(tickers_with_finra), len(known_without_finra),
                        len(final_list) - len(tickers_with_finra) - len(known_without_finra), len(final_list))

            return final_list

        except Exception as e:
            _logger.warning("Error in FINRA pre-filtering: %s", e)
            return self._fallback_to_known_candidates(tickers)

    def _get_finra_data_from_db(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get FINRA short interest data from database for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to FINRA data
        """
        try:
            _logger.info("Querying FINRA data from database for %d tickers", len(tickers))

            # Service manages sessions internally via UoW pattern
            service = ShortSqueezeService()
            finra_data = service.get_bulk_finra_short_interest(tickers)

            _logger.info("Retrieved FINRA data for %d tickers from database", len(finra_data))
            return finra_data

        except Exception as e:
            _logger.warning("Error getting FINRA data from database: %s", e)
            return {}

    def _fallback_to_known_candidates(self, tickers: List[str]) -> List[str]:
        """
        Fallback method when FINRA data is not available.

        Args:
            tickers: Original ticker list

        Returns:
            Filtered list prioritizing known candidates
        """
        _logger.info("Using fallback approach with known candidates")

        known_candidates = self._get_known_short_interest_candidates()
        known_in_universe = [t for t in tickers if t.upper() in [k.upper() for k in known_candidates]]
        other_tickers = [t for t in tickers if t.upper() not in [k.upper() for k in known_candidates]]

        # Limit to avoid too many potential API failures later
        limited_others = other_tickers[:min(150, len(other_tickers))]

        final_list = known_in_universe + limited_others
        _logger.info("Fallback universe: %d known + %d others = %d total",
                    len(known_in_universe), len(limited_others), len(final_list))

        return final_list

    def _is_valid_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker is suitable for short squeeze analysis.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            True if ticker is valid, False otherwise
        """
        try:
            # Basic validation
            if not ticker or len(ticker.strip()) == 0:
                return False

            ticker = ticker.strip().upper()

            # Length check (typical US tickers are 1-5 characters)
            if len(ticker) > 5:
                return False

            # Exclude certain ticker patterns that are typically not suitable
            # for short squeeze analysis (ETFs, preferred shares, etc.)
            exclude_patterns = [
                '.', '-', '/', ' ',  # Special characters
                'WARR', 'WS', 'WT',  # Warrants
                'PR',  # Preferred shares (though this might be too broad)
            ]

            for pattern in exclude_patterns:
                if pattern in ticker:
                    return False

            return True

        except Exception as e:
            _logger.warning("Error validating ticker %s: %s", ticker, e)
            return False

    # NOTE: Market cap and volume filtering methods removed to avoid inefficient per-ticker API calls.
    # The FMP screener already handles these filters efficiently in bulk.
    # Individual filtering would require hundreds of API calls vs. single screener call.

    def get_finra_coverage_stats(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Get statistics about FINRA data coverage for the universe from database.

        Args:
            tickers: List of ticker symbols to check

        Returns:
            Dictionary with coverage statistics
        """
        try:
            _logger.info("Checking FINRA coverage for %d tickers from database", len(tickers))

            finra_data = self._get_finra_data_from_db(tickers)

            total_tickers = len(tickers)
            covered_tickers = len(finra_data)
            coverage_rate = covered_tickers / total_tickers if total_tickers > 0 else 0

            # Analyze short interest levels
            high_activity = 0
            medium_activity = 0
            low_activity = 0

            for ticker_data in finra_data.values():
                short_interest_pct = ticker_data.get('short_interest_pct', 0)

                if short_interest_pct > 20.0:  # > 20% short interest
                    high_activity += 1
                elif short_interest_pct > 10.0:  # > 10% short interest
                    medium_activity += 1
                else:
                    low_activity += 1

            stats = {
                'total_tickers': total_tickers,
                'finra_covered': covered_tickers,
                'coverage_rate': coverage_rate,
                'high_short_activity': high_activity,
                'medium_short_activity': medium_activity,
                'low_short_activity': low_activity,
                'report_date': datetime.now().date(),
                'data_source': 'database'
            }

            _logger.info("FINRA coverage: %.1f%% (%d/%d), High activity: %d, Medium: %d, Low: %d",
                        coverage_rate * 100, covered_tickers, total_tickers,
                        high_activity, medium_activity, low_activity)

            return stats

        except Exception:
            _logger.exception("Error getting FINRA coverage stats:")
            return {}

    def _load_from_cache(self) -> Optional[List[str]]:
        """
        Load universe from cache if available and not expired.

        Returns:
            Cached ticker list or None if cache is invalid/expired
        """
        try:
            cache_file = self._cache_dir / "universe_cache.json"

            if not cache_file.exists():
                return None

            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() > (self._cache_ttl_hours * 3600):
                _logger.info("Universe cache expired (age: %s hours)", cache_age.total_seconds() / 3600)
                return None

            # Load cache
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            # Validate cache structure
            if not isinstance(cache_data, dict) or 'tickers' not in cache_data:
                _logger.warning("Invalid cache structure, ignoring cache")
                return None

            tickers = cache_data['tickers']
            if not isinstance(tickers, list):
                _logger.warning("Invalid ticker list in cache, ignoring cache")
                return None

            _logger.info("Loaded %d tickers from cache (created: %s)",
                        len(tickers), cache_data.get('created_at', 'unknown'))
            return tickers

        except Exception as e:
            _logger.warning("Error loading from cache: %s", e)
            return None

    def _save_to_cache(self, tickers: List[str]) -> None:
        """
        Save universe to cache.

        Args:
            tickers: List of ticker symbols to cache
        """
        try:
            cache_file = self._cache_dir / "universe_cache.json"

            cache_data = {
                'tickers': tickers,
                'created_at': datetime.now().isoformat(),
                'config': {
                    'min_market_cap': self.config.min_market_cap,
                    'max_market_cap': self.config.max_market_cap,
                    'min_avg_volume': self.config.min_avg_volume,
                    'exchanges': self.config.exchanges
                }
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            _logger.info("Saved %d tickers to cache", len(tickers))

        except Exception as e:
            _logger.warning("Error saving to cache: %s", e)

    def clear_cache(self) -> None:
        """Clear the universe cache."""
        try:
            cache_file = self._cache_dir / "universe_cache.json"
            if cache_file.exists():
                cache_file.unlink()
                _logger.info("Universe cache cleared")
        except Exception as e:
            _logger.warning("Error clearing cache: %s", e)

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache.

        Returns:
            Dictionary with cache information
        """
        try:
            cache_file = self._cache_dir / "universe_cache.json"

            if not cache_file.exists():
                return {'exists': False}

            stat = cache_file.stat()
            cache_age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)

            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            return {
                'exists': True,
                'size': len(cache_data.get('tickers', [])),
                'created_at': cache_data.get('created_at'),
                'age_hours': cache_age.total_seconds() / 3600,
                'expired': cache_age.total_seconds() > (self._cache_ttl_hours * 3600),
                'config': cache_data.get('config', {})
            }

        except Exception as e:
            _logger.warning("Error getting cache info: %s", e)
            return {'exists': False, 'error': str(e)}


def create_universe_loader(fmp_downloader: FMPDataDownloader,
                          config: UniverseConfig) -> UniverseLoader:
    """
    Factory function to create Universe Loader.

    Args:
        fmp_downloader: FMP data downloader instance
        config: Universe configuration

    Returns:
        Configured Universe Loader instance
    """
    return UniverseLoader(fmp_downloader, config)


# Example usage
if __name__ == "__main__":
    from src.ml.pipeline.p04_short_squeeze.config.data_classes import UniverseConfig

    # Create FMP downloader
    fmp_downloader = FMPDataDownloader()

    # Create universe config
    universe_config = UniverseConfig(
        min_market_cap=100_000_000,  # $100M
        max_market_cap=10_000_000_000,  # $10B
        min_avg_volume=200_000,
        exchanges=['NYSE', 'NASDAQ']
    )

    # Create universe loader
    loader = create_universe_loader(fmp_downloader, universe_config)

    # Test connection
    if fmp_downloader.test_connection():
        print("✅ FMP API connection successful")

        # Load universe
        universe = loader.load_universe()
        if universe:
            print(f"✅ Loaded universe of {len(universe)} stocks")
            print(f"Sample tickers: {universe[:10]}")

            # Show cache info
            cache_info = loader.get_cache_info()
            print(f"Cache info: {cache_info}")
        else:
            print("❌ Failed to load universe")
    else:
        print("❌ FMP API connection failed")