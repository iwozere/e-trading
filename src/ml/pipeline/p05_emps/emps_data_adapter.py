"""
EMPS Data Adapter

Adapter module to integrate EMPS scoring with existing data downloaders (FMP, Polygon, etc.).
Provides data fetching functions compatible with EMPS module expectations.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader

logger = setup_logger(__name__)


class EMPSDataAdapter:
    """
    Adapter to fetch and format data for EMPS computation.

    Integrates with existing data downloaders (FMP, Polygon, etc.) and provides
    standardized data retrieval for EMPS scoring.
    """

    def __init__(self, downloader: FMPDataDownloader):
        """
        Initialize EMPS data adapter.

        Args:
            downloader: Data downloader instance (FMP, Polygon, etc.)
        """
        self.downloader = downloader
        logger.info("EMPS Data Adapter initialized with %s", type(downloader).__name__)

    def fetch_intraday_for_emps(
        self,
        ticker: str,
        fetch_kwargs: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fetch intraday data for EMPS computation.

        Args:
            ticker: Stock ticker symbol
            fetch_kwargs: Optional parameters (period, interval, days_back)

        Returns:
            DataFrame with OHLCV columns and datetime index

        Raises:
            RuntimeError: If data fetch fails
        """
        if fetch_kwargs is None:
            fetch_kwargs = {}

        # Extract parameters with defaults optimized for EMPS
        interval = fetch_kwargs.get('interval', '5m')
        days_back = fetch_kwargs.get('days_back', 7)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info("Fetching %s intraday data: ticker=%s, interval=%s, days=%d",
                    type(self.downloader).__name__, ticker, interval, days_back)

        try:
            df = self.downloader.get_ohlcv(
                symbol=ticker,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                raise RuntimeError(f"No data returned for {ticker}")

            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise RuntimeError(f"Missing required columns: {missing_cols}")

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                else:
                    logger.warning("No datetime column found, using default index")

            # Sort by time (ascending)
            df = df.sort_index()

            logger.info("Successfully fetched %d bars for %s", len(df), ticker)
            return df

        except Exception as e:
            logger.error("Error fetching data for %s: %s", ticker, e)
            raise RuntimeError(f"Failed to fetch data for {ticker}: {e}")

    def fetch_ticker_metadata(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch ticker metadata required for EMPS scoring.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with: market_cap, float_shares, avg_volume, sector, etc.
        """
        logger.info("Fetching metadata for %s", ticker)

        try:
            # Try to get company profile (FMP-specific, adapt for other providers)
            if hasattr(self.downloader, 'get_company_profile'):
                profile = self.downloader.get_company_profile(ticker)

                if profile:
                    metadata = {
                        'market_cap': profile.get('mktCap') or profile.get('marketCap'),
                        'float_shares': profile.get('floatShares'),
                        'avg_volume': profile.get('volAvg') or profile.get('avgVolume'),
                        'sector': profile.get('sector'),
                        'industry': profile.get('industry'),
                        'exchange': profile.get('exchange'),
                    }

                    logger.info("Metadata for %s: cap=%s, float=%s, vol=%s",
                                ticker, metadata.get('market_cap'),
                                metadata.get('float_shares'), metadata.get('avg_volume'))

                    return metadata

            # Fallback: try to estimate from recent data
            logger.warning("Company profile not available, using fallback metadata for %s", ticker)
            return self._estimate_metadata_from_data(ticker)

        except Exception as e:
            logger.error("Error fetching metadata for %s: %s", ticker, e)
            return self._get_default_metadata()

    def _estimate_metadata_from_data(self, ticker: str) -> Dict[str, Any]:
        """Estimate metadata from recent trading data (fallback)."""
        try:
            # Get recent daily data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            df = self.downloader.get_ohlcv(
                symbol=ticker,
                interval='1d',
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
                avg_volume = int(df['Volume'].mean())
                recent_close = float(df['Close'].iloc[-1])
                # Rough market cap estimate (assuming ~100M shares)
                estimated_market_cap = int(recent_close * 100_000_000)

                return {
                    'market_cap': estimated_market_cap,
                    'float_shares': None,
                    'avg_volume': avg_volume,
                    'sector': None,
                    'industry': None,
                    'exchange': None,
                    'estimated': True
                }

        except Exception as e:
            logger.warning("Could not estimate metadata for %s: %s", ticker, e)

        return self._get_default_metadata()

    def _get_default_metadata(self) -> Dict[str, Any]:
        """Return default/empty metadata."""
        return {
            'market_cap': None,
            'float_shares': None,
            'avg_volume': None,
            'sector': None,
            'industry': None,
            'exchange': None,
        }

    def fetch_daily_for_breakout(
        self,
        ticker: str,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch daily data for breakout detection.

        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to fetch

        Returns:
            DataFrame with daily OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        logger.info("Fetching daily data for breakout: ticker=%s, days=%d", ticker, lookback_days)

        try:
            df = self.downloader.get_ohlcv(
                symbol=ticker,
                interval='1d',
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                logger.warning("No daily data for %s", ticker)
                return pd.DataFrame()

            return df.sort_index()

        except Exception as e:
            logger.error("Error fetching daily data for %s: %s", ticker, e)
            return pd.DataFrame()


def create_emps_adapter(downloader: FMPDataDownloader) -> EMPSDataAdapter:
    """
    Factory function to create EMPS data adapter.

    Args:
        downloader: Data downloader instance

    Returns:
        Configured EMPS data adapter
    """
    return EMPSDataAdapter(downloader)


# Example usage
if __name__ == "__main__":
    from src.data.downloader.fmp_data_downloader import FMPDataDownloader

    # Initialize downloader and adapter
    fmp = FMPDataDownloader()
    adapter = create_emps_adapter(fmp)

    # Test data fetch
    test_ticker = "AAPL"
    print(f"\n{'='*60}")
    print(f"Testing EMPS Data Adapter with {test_ticker}")
    print(f"{'='*60}\n")

    # Fetch intraday data
    try:
        df_intraday = adapter.fetch_intraday_for_emps(test_ticker, {'interval': '5m', 'days_back': 5})
        print(f"✅ Intraday data: {len(df_intraday)} bars")
        print(f"   Date range: {df_intraday.index.min()} to {df_intraday.index.max()}")
        print(f"   Columns: {list(df_intraday.columns)}")
    except Exception as e:
        print(f"❌ Intraday data fetch failed: {e}")

    # Fetch metadata
    try:
        metadata = adapter.fetch_ticker_metadata(test_ticker)
        print(f"\n✅ Metadata:")
        for key, value in metadata.items():
            if value:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Metadata fetch failed: {e}")

    # Fetch daily for breakout
    try:
        df_daily = adapter.fetch_daily_for_breakout(test_ticker, lookback_days=30)
        print(f"\n✅ Daily data: {len(df_daily)} days")
    except Exception as e:
        print(f"❌ Daily data fetch failed: {e}")

    print(f"\n{'='*60}\n")
