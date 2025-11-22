"""
EMPS Integration Module

Provides universe scanning with EMPS (Explosive Move Probability Score).
Standalone P05 implementation with optional P04 short squeeze integration.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from datetime import datetime

import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.ml.pipeline.p05_emps.emps import compute_emps_from_intraday, DEFAULTS as EMPS_DEFAULTS
from src.ml.pipeline.p05_emps.emps_data_adapter import EMPSDataAdapter
from src.ml.pipeline.p05_emps.universe_loader import EMPSUniverseLoader, EMPSUniverseConfig

logger = setup_logger(__name__)

# Optional P04 integration (for short squeeze metrics)
try:
    from src.data.db.services.short_squeeze_service import ShortSqueezeService
    P04_AVAILABLE = True
except ImportError as e:
    logger.warning("P04 short squeeze service not available: %s. P04 integration will be disabled.", e)
    ShortSqueezeService = None
    P04_AVAILABLE = False


class EMPSUniverseScanner:
    """
    Scanner that applies EMPS scoring to a universe of stocks.

    Standalone P05 implementation using EMPSUniverseLoader.
    """

    def __init__(
        self,
        downloader: FMPDataDownloader,
        universe_loader: EMPSUniverseLoader,
        emps_params: Optional[Dict[str, Any]] = None,
        fetch_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize EMPS universe scanner.

        Args:
            downloader: Data downloader instance
            universe_loader: EMPS universe loader (standalone P05 component)
            emps_params: Optional EMPS parameters (overrides defaults)
            fetch_params: Optional data fetch parameters (interval, days_back)
        """
        self.downloader = downloader
        self.universe_loader = universe_loader
        self.emps_params = {**EMPS_DEFAULTS, **(emps_params or {})}
        self.fetch_params = fetch_params or {'interval': '15m', 'days_back': 2}
        self.adapter = EMPSDataAdapter(downloader)

        logger.info("EMPS Universe Scanner initialized (interval=%s, days=%d)",
                    self.fetch_params.get('interval'), self.fetch_params.get('days_back'))

    def scan_universe(
        self,
        limit: Optional[int] = None,
        min_emps_score: float = 0.5
    ) -> pd.DataFrame:
        """
        Scan universe and compute EMPS scores for all tickers.

        Args:
            limit: Optional limit on number of tickers to scan
            min_emps_score: Minimum EMPS score to include in results

        Returns:
            DataFrame with ticker, EMPS scores, and key metrics
        """
        logger.info("Starting EMPS universe scan...")

        # Load universe
        universe = self.universe_loader.load_universe()
        if not universe:
            logger.warning("No tickers in universe")
            return pd.DataFrame()

        if limit:
            universe = universe[:limit]

        logger.info("Scanning %d tickers from universe", len(universe))

        results = []
        for idx, ticker in enumerate(universe, 1):
            try:
                logger.info("[%d/%d] Scanning %s...", idx, len(universe), ticker)

                # Fetch data and compute EMPS
                result = self.scan_single_ticker(ticker)

                if result and result.get('emps_score', 0) >= min_emps_score:
                    results.append(result)
                    logger.info("  [OK] %s: EMPS=%.3f, explosion=%s",
                                ticker, result['emps_score'], result['explosion_flag'])
                else:
                    logger.info("  [SKIP] %s: Below threshold", ticker)

            except Exception as e:
                logger.warning("  [ERROR] %s: Error - %s", ticker, e)
                continue

        if not results:
            logger.warning("No tickers met EMPS threshold")
            return pd.DataFrame()

        # Create results DataFrame
        df_results = pd.DataFrame(results)

        # Sort by EMPS score descending
        df_results = df_results.sort_values('emps_score', ascending=False)

        logger.info("Scan complete: %d/%d tickers above threshold %.2f",
                    len(df_results), len(universe), min_emps_score)

        return df_results

    def scan_single_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Scan single ticker and compute EMPS score.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with EMPS metrics or None if failed
        """
        try:
            # Fetch metadata
            metadata = self.adapter.fetch_ticker_metadata(ticker)

            # Fetch intraday data using configured parameters
            df_intraday = self.adapter.fetch_intraday_for_emps(
                ticker,
                self.fetch_params
            )

            if df_intraday.empty:
                logger.warning("No intraday data for %s", ticker)
                return None

            # Compute EMPS
            df_emps = compute_emps_from_intraday(
                df_intraday,
                market_cap=metadata.get('market_cap'),
                float_shares=metadata.get('float_shares'),
                avg_volume=metadata.get('avg_volume'),
                ticker=ticker,
                params=self.emps_params
            )

            # Extract latest row
            latest = df_emps.iloc[-1]

            return {
                'ticker': ticker,
                'timestamp': latest['emps_timestamp'],
                'emps_score': float(latest['emps_score']),
                'explosion_flag': bool(latest['explosion_flag']),
                'hard_flag': bool(latest['hard_flag']),
                'vol_zscore': float(latest['vol_zscore']),
                'vwap_dev': float(latest['vwap_dev']),
                'rv_ratio': float(latest['rv_ratio']),
                'liquidity_score': float(latest['liquidity_score']),
                'market_cap': metadata.get('market_cap'),
                'avg_volume': metadata.get('avg_volume'),
                'sector': metadata.get('sector'),
            }

        except Exception as e:
            logger.error("Error scanning %s: %s", ticker, e)
            return None

    def scan_with_p04_integration(
        self,
        limit: Optional[int] = None,
        combine_scores: bool = True
    ) -> pd.DataFrame:
        """
        Scan universe with P04 short squeeze integration.

        Combines EMPS scores with P04 short squeeze metrics for enhanced detection.

        Args:
            limit: Optional limit on number of tickers to scan
            combine_scores: If True, fetches P04 metrics and combines with EMPS

        Returns:
            DataFrame with combined EMPS and P04 metrics
        """
        logger.info("Starting combined EMPS + P04 scan...")

        # Get EMPS scores
        df_emps = self.scan_universe(limit=limit, min_emps_score=0.3)

        if df_emps.empty:
            logger.warning("No EMPS results to combine")
            return df_emps

        if not combine_scores:
            return df_emps

        # Check if P04 integration is available
        if not P04_AVAILABLE:
            logger.warning("P04 short squeeze service not available - skipping P04 integration")
            return df_emps

        # Try to fetch P04 short squeeze data
        try:
            tickers = df_emps['ticker'].tolist()

            # Service manages sessions internally via UoW pattern
            service = ShortSqueezeService()
            p04_data = service.get_bulk_finra_short_interest(tickers)

            if p04_data:
                logger.info("Found P04 data for %d tickers", len(p04_data))

                # Add P04 columns to results
                df_emps['short_interest_pct'] = df_emps['ticker'].apply(
                    lambda t: p04_data.get(t.upper(), {}).get('short_interest_pct')
                )
                df_emps['days_to_cover'] = df_emps['ticker'].apply(
                    lambda t: p04_data.get(t.upper(), {}).get('days_to_cover')
                )

                # Compute combined score (weighted average)
                # EMPS: 60%, Short Interest: 40%
                df_emps['combined_score'] = (
                    0.6 * df_emps['emps_score']
                    + 0.4 * (df_emps['short_interest_pct'].fillna(0) / 100.0)
                )

                df_emps = df_emps.sort_values('combined_score', ascending=False)

                logger.info("Combined scoring complete")

        except Exception as e:
            logger.warning("Could not integrate P04 data: %s", e)

        return df_emps


class EMPSEnhancedCandidate:
    """
    Enhanced candidate model combining EMPS and P04 short squeeze metrics.
    """

    def __init__(
        self,
        ticker: str,
        emps_score: float,
        emps_metrics: Dict[str, Any],
        short_squeeze_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced candidate.

        Args:
            ticker: Stock ticker symbol
            emps_score: EMPS score [0, 1]
            emps_metrics: Dictionary of EMPS component metrics
            short_squeeze_metrics: Optional P04 short squeeze metrics
        """
        self.ticker = ticker
        self.emps_score = emps_score
        self.emps_metrics = emps_metrics
        self.short_squeeze_metrics = short_squeeze_metrics or {}
        self.timestamp = datetime.now()

    def compute_combined_score(self) -> float:
        """
        Compute combined score from EMPS and short squeeze metrics.

        Returns:
            Combined score [0, 1]
        """
        # Base score from EMPS
        score = self.emps_score * 0.6

        # Add short interest component (if available)
        short_pct = self.short_squeeze_metrics.get('short_interest_pct', 0)
        if short_pct > 0:
            # Normalize short interest: 20% = 0.5, 40%+ = 1.0
            short_score = min(1.0, short_pct / 40.0)
            score += short_score * 0.4

        return min(1.0, score)

    def get_alert_level(self) -> str:
        """
        Determine alert level based on scores.

        Returns:
            Alert level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
        """
        combined = self.compute_combined_score()

        if combined >= 0.8:
            return 'CRITICAL'
        elif combined >= 0.65:
            return 'HIGH'
        elif combined >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'emps_score': self.emps_score,
            'combined_score': self.compute_combined_score(),
            'alert_level': self.get_alert_level(),
            **self.emps_metrics,
            **{f'ss_{k}': v for k, v in self.short_squeeze_metrics.items()}
        }


def create_emps_scanner(
    downloader: FMPDataDownloader,
    universe_config: Optional[EMPSUniverseConfig] = None,
    emps_params: Optional[Dict[str, Any]] = None,
    fetch_params: Optional[Dict[str, Any]] = None
) -> EMPSUniverseScanner:
    """
    Factory function to create EMPS universe scanner.

    Standalone P05 implementation - no P04 dependencies required.

    Args:
        downloader: Data downloader instance
        universe_config: Optional EMPS universe configuration (uses defaults if None)
        emps_params: Optional EMPS parameters
        fetch_params: Optional data fetch parameters (interval, days_back)

    Returns:
        Configured EMPS universe scanner
    """
    if universe_config is None:
        universe_config = EMPSUniverseConfig(
            min_market_cap=100_000_000,  # $100M
            max_market_cap=10_000_000_000,  # $10B
            min_avg_volume=500_000,
            exchanges=['NYSE', 'NASDAQ'],
            max_universe_size=1000
        )

    universe_loader = EMPSUniverseLoader(downloader, universe_config)

    return EMPSUniverseScanner(downloader, universe_loader, emps_params, fetch_params)


# Example usage
if __name__ == "__main__":
    from src.data.downloader.fmp_data_downloader import FMPDataDownloader

    print(f"\n{'='*60}")
    print("EMPS + P04 Integration Test")
    print(f"{'='*60}\n")

    # Initialize components
    fmp = FMPDataDownloader()

    # Test connection
    if not fmp.test_connection():
        print("❌ FMP connection failed")
        sys.exit(1)

    print("✅ FMP connection successful\n")

    # Create scanner
    scanner = create_emps_scanner(fmp)

    # Scan a small sample
    print("Scanning top 10 tickers from universe...\n")
    results = scanner.scan_universe(limit=10, min_emps_score=0.3)

    if not results.empty:
        print(f"\n✅ Found {len(results)} candidates:\n")
        print(results[['ticker', 'emps_score', 'explosion_flag', 'vol_zscore', 'vwap_dev']].to_string())
    else:
        print("\n⚠️  No candidates found above threshold\n")

    print(f"\n{'='*60}\n")
