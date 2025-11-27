"""
EMPS Universe Loader

Provides universe selection functionality for EMPS pipeline.
Standalone implementation - NO dependencies on P04 pipeline.
"""

from pathlib import Path
import sys
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader

logger = setup_logger(__name__)


@dataclass
class EMPSUniverseConfig:
    """
    Configuration for EMPS universe loading.

    Standalone config - not dependent on P04.
    """
    min_market_cap: int = 100_000_000  # $100M
    max_market_cap: int = 10_000_000_000  # $10B
    min_avg_volume: int = 500_000  # 500K shares/day (optimized for EMPS)
    exchanges: List[str] = field(default_factory=lambda: ['NYSE', 'NASDAQ'])
    max_universe_size: int = 1000


class EMPSUniverseLoader:
    """
    Universe loader for EMPS pipeline.

    Loads and filters stocks from FMP screener based on criteria suitable
    for explosive move detection.

    Standalone implementation - NO P04 dependencies.
    """

    def __init__(self, downloader: FMPDataDownloader, config: Optional[EMPSUniverseConfig] = None):
        """
        Initialize EMPS Universe Loader.

        Args:
            downloader: FMP data downloader instance
            config: Optional universe configuration (uses defaults if None)
        """
        self.downloader = downloader
        self.config = config or EMPSUniverseConfig()

        # Store universe files in results/emps/YYYY-MM-DD/ instead of data/cache/
        today = datetime.now().strftime('%Y-%m-%d')
        self._cache_dir = Path("results") / "emps" / today
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_hours = 24

        logger.info("EMPS Universe Loader initialized: cap=$%dM-$%dB, volume=%dK",
                    self.config.min_market_cap // 1_000_000,
                    self.config.max_market_cap // 1_000_000_000,
                    self.config.min_avg_volume // 1_000)

    def load_universe(self, force_refresh: bool = False) -> List[str]:
        """
        Load universe of stocks suitable for EMPS analysis.

        Args:
            force_refresh: If True, bypass cache and force fresh fetch from screener

        Returns:
            List of ticker symbols
        """
        try:
            logger.info("Loading EMPS universe... (force_refresh=%s)", force_refresh)

            # Clear cache if force refresh requested
            if force_refresh:
                logger.info("Force refresh requested - clearing cache")
                self.clear_cache()

            # Check cache first (unless force refresh)
            if not force_refresh:
                cached = self._load_from_cache()
                if cached:
                    logger.info("Loaded %d tickers from cache", len(cached))
                    return cached

            # Load from FMP screener
            universe = self._load_from_screener()

            # Apply filters
            filtered = self._apply_filters(universe)

            # Cache results
            self._save_to_cache(filtered)

            logger.info("Successfully loaded EMPS universe: %d tickers", len(filtered))
            return filtered

        except Exception:
            logger.exception("Error loading EMPS universe:")
            return []

    def _load_from_screener(self) -> List[str]:
        """
        Load universe from FMP screener.

        Uses multiple strategies optimized for explosive move detection.
        """
        try:
            all_tickers = set()

            # Strategy 1: Mid-cap with high volume (sweet spot for explosions)
            mid_cap_min = max(self.config.min_market_cap, 500_000_000)  # $500M+
            mid_cap_max = min(self.config.max_market_cap, 5_000_000_000)  # < $5B

            criteria_1 = {
                'marketCapMoreThan': mid_cap_min,
                'marketCapLowerThan': mid_cap_max,
                'volumeMoreThan': max(self.config.min_avg_volume * 2, 1_000_000),
                'exchange': ','.join(self.config.exchanges),
                'limit': min(500, self.config.max_universe_size // 2)
            }

            logger.info("Strategy 1: Mid-cap high-volume screening")
            tickers_1 = self.downloader.load_universe_from_screener(criteria_1)
            all_tickers.update(tickers_1)
            logger.info("Strategy 1: Found %d tickers", len(tickers_1))

            # Strategy 2: Small-cap with good volume (explosive potential)
            small_cap_max = min(mid_cap_min, 1_000_000_000)  # < $1B
            criteria_2 = {
                'marketCapMoreThan': self.config.min_market_cap,
                'marketCapLowerThan': small_cap_max,
                'volumeMoreThan': max(self.config.min_avg_volume, 500_000),
                'exchange': ','.join(self.config.exchanges),
                'limit': min(300, self.config.max_universe_size // 3)
            }

            logger.info("Strategy 2: Small-cap screening")
            tickers_2 = self.downloader.load_universe_from_screener(criteria_2)
            all_tickers.update(tickers_2)
            logger.info("Strategy 2: Found %d new tickers", len(tickers_2) - len(all_tickers & set(tickers_2)))

            # Strategy 3: Known explosive tickers (prioritize these)
            known_tickers = self._get_known_explosive_tickers()
            logger.info("Strategy 3: Adding %d known explosive tickers", len(known_tickers))

            # Prioritize known explosive tickers first, then add screener results
            # This ensures high-quality explosive candidates are never excluded
            final_tickers = []

            # 1. Add known explosive tickers that aren't already included
            for ticker in known_tickers:
                if ticker not in all_tickers:
                    final_tickers.append(ticker)

            # 2. Add screener results (maintain their order - usually sorted by screener's relevance)
            screener_tickers = list(all_tickers)
            for ticker in screener_tickers:
                if ticker not in final_tickers:
                    final_tickers.append(ticker)

            # 3. Add remaining known tickers that were in screener results (move to front)
            known_in_screener = [t for t in known_tickers if t in all_tickers]
            final_tickers = known_in_screener + [t for t in final_tickers if t not in known_in_screener]

            # Limit to max universe size (after prioritization)
            if len(final_tickers) > self.config.max_universe_size:
                logger.info("Limiting universe from %d to %d tickers (prioritized by explosion potential)",
                           len(final_tickers), self.config.max_universe_size)
                final_tickers = final_tickers[:self.config.max_universe_size]

            logger.info("Combined universe: %d unique tickers (known explosive: %d)",
                       len(final_tickers), len([t for t in final_tickers if t in known_tickers]))
            return final_tickers

        except Exception:
            logger.exception("Error loading from screener:")
            return self._fallback_screener()

    def _fallback_screener(self) -> List[str]:
        """Fallback to simple screener if multi-strategy fails."""
        try:
            criteria = {
                'marketCapMoreThan': self.config.min_market_cap,
                'marketCapLowerThan': self.config.max_market_cap,
                'volumeMoreThan': self.config.min_avg_volume,
                'exchange': ','.join(self.config.exchanges),
                'limit': self.config.max_universe_size
            }

            logger.info("Using fallback screener")
            tickers = self.downloader.load_universe_from_screener(criteria)
            logger.info("Fallback screener returned %d tickers", len(tickers))
            return tickers

        except Exception:
            logger.exception("Fallback screener failed:")
            return []

    def _get_known_explosive_tickers(self) -> List[str]:
        """
        Get curated list of tickers known for explosive moves.

        These are stocks that historically show explosive price movements
        and are good candidates for EMPS detection.
        """
        known_explosive = [
            # Meme stocks / Frequent squeeze candidates
            'GME', 'AMC', 'BBBY', 'CLOV', 'WISH', 'PLTR', 'SPCE',

            # High-profile volatile stocks
            'TSLA', 'NKLA', 'RIVN', 'LCID', 'HOOD', 'COIN',

            # Biotech (volatile sector)
            'MRNA', 'BNTX', 'NVAX', 'SAVA', 'BIIB', 'GILD',

            # Energy/EV (explosive moves common)
            'PLUG', 'FCEL', 'BLNK', 'CHPT', 'QS', 'HYLN',

            # Growth/Fintech
            'DKNG', 'OPEN', 'SKLZ', 'SOFI', 'UPST', 'AFRM',

            # Tech with high volatility
            'NFLX', 'SHOP', 'ZM', 'PTON', 'ROKU', 'SQ'
        ]

        logger.info("Added %d known explosive tickers", len(known_explosive))
        return known_explosive

    def _apply_filters(self, tickers: List[str]) -> List[str]:
        """
        Apply validation filters to ticker list.

        Args:
            tickers: Raw ticker list

        Returns:
            Filtered ticker list
        """
        if not tickers:
            return []

        logger.info("Applying filters to %d tickers", len(tickers))

        valid_tickers = []
        for ticker in tickers:
            if self._is_valid_ticker(ticker):
                valid_tickers.append(ticker)

        logger.info("After validation: %d tickers", len(valid_tickers))
        return valid_tickers

    def _is_valid_ticker(self, ticker: str) -> bool:
        """
        Validate ticker symbol.

        Args:
            ticker: Ticker symbol

        Returns:
            True if valid
        """
        try:
            if not ticker or len(ticker.strip()) == 0:
                return False

            ticker = ticker.strip().upper()

            # Length check (US tickers typically 1-5 chars)
            if len(ticker) > 5:
                return False

            # Exclude patterns unsuitable for EMPS
            exclude_patterns = [
                '.', '-', '/', ' ',  # Special characters
                'WARR', 'WS', 'WT',  # Warrants
            ]

            for pattern in exclude_patterns:
                if pattern in ticker:
                    return False

            return True

        except Exception as e:
            logger.warning("Error validating ticker %s: %s", ticker, e)
            return False

    def _load_from_cache(self) -> Optional[List[str]]:
        """Load universe from cache if valid."""
        try:
            cache_file = self._cache_dir / "emps_universe_cache.json"

            if not cache_file.exists():
                return None

            # Check age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() > (self._cache_ttl_hours * 3600):
                logger.info("Cache expired (age: %.1f hours)", cache_age.total_seconds() / 3600)
                return None

            # Load cache
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            if not isinstance(cache_data, dict) or 'tickers' not in cache_data:
                logger.warning("Invalid cache structure")
                return None

            tickers = cache_data['tickers']
            if not isinstance(tickers, list):
                logger.warning("Invalid ticker list in cache")
                return None

            logger.info("Loaded %d tickers from cache (created: %s)",
                       len(tickers), cache_data.get('created_at', 'unknown'))
            return tickers

        except Exception as e:
            logger.warning("Error loading from cache: %s", e)
            return None

    def _save_to_cache(self, tickers: List[str]) -> None:
        """Save universe to cache."""
        try:
            cache_file = self._cache_dir / "emps_universe_cache.json"

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

            logger.info("Saved %d tickers to cache", len(tickers))

        except Exception as e:
            logger.warning("Error saving to cache: %s", e)

    def clear_cache(self) -> None:
        """Clear the universe cache."""
        try:
            cache_file = self._cache_dir / "emps_universe_cache.json"
            if cache_file.exists():
                cache_file.unlink()
                logger.info("EMPS universe cache cleared")
        except Exception as e:
            logger.warning("Error clearing cache: %s", e)


# Factory function
def create_universe_loader(
    downloader: FMPDataDownloader,
    config: Optional[EMPSUniverseConfig] = None
) -> EMPSUniverseLoader:
    """
    Factory function to create EMPS universe loader.

    Args:
        downloader: FMP data downloader instance
        config: Optional universe configuration

    Returns:
        Configured EMPS universe loader
    """
    return EMPSUniverseLoader(downloader, config)


# Example usage
if __name__ == "__main__":
    from src.data.downloader.fmp_data_downloader import FMPDataDownloader

    print("EMPS Universe Loader Test")
    print("=" * 60)

    # Initialize
    fmp = FMPDataDownloader()

    if not fmp.test_connection():
        print("FMP connection failed")
        sys.exit(1)

    print("FMP connection successful\n")

    # Create loader with default config
    loader = create_universe_loader(fmp)

    # Load universe
    universe = loader.load_universe()

    if universe:
        print(f"Loaded universe of {len(universe)} stocks")
        print(f"Sample tickers: {universe[:20]}")
    else:
        print("Failed to load universe")

    print("=" * 60)
