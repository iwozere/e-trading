
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import asyncio
import logging

try:
    import san
    SANPY_AVAILABLE = True
except ImportError:
    SANPY_AVAILABLE = False

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.model.schemas import SentimentData
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class SantimentDataDownloader(BaseDataDownloader):
    """
    Downloader for Santiment.net data using sanpy library.
    Focuses on social metrics and sentiment data for crypto and stocks.
    """

    def __init__(self):
        super().__init__()
        if not SANPY_AVAILABLE:
            _logger.warning("sanpy library not installed. Santiment data will be unavailable. Run 'pip install sanpy'")
            return

        # Configure API key
        self.api_key = self._get_config_value("SANTIMENT_API_KEY", "SANTIMENT_API_KEY")
        if self.api_key:
            san.ApiConfig.api_key = self.api_key
        else:
            _logger.warning("SANTIMENT_API_KEY not found. Some functionality may be limited.")

    def get_supported_intervals(self) -> List[str]:
        """Return supported intervals."""
        return ["1d", "1h"]

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Download OHLCV data.
        Note: Santiment is primarily for crypto, coverage for stocks varies.
        """
        if not SANPY_AVAILABLE:
            _logger.error("sanpy not installed")
            return pd.DataFrame()

        try:
            # Map interval to Santiment format if needed
            # For now, simplistic mapping
            slug = self._symbol_to_slug(symbol)

            df = san.get(
                "ohlcv",
                slug=slug,
                from_date=start_date,
                to_date=end_date,
                interval=interval
            )

            if df.empty:
                return pd.DataFrame()

            # Rename columns to standard format
            df.index.name = "date"
            df = df.rename(columns={
                "openPriceUsd": "open",
                "highPriceUsd": "high",
                "lowPriceUsd": "low",
                "closePriceUsd": "close",
                "volume": "volume"
            })

            return df[["open", "high", "low", "close", "volume"]]

        except Exception as e:
            _logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    async def get_social_volume(self, symbol: str, days_back: int = 7) -> SentimentData:
        """
        Get social volume metrics (mentions).

        Args:
            symbol: Ticker symbol
            days_back: Number of days to look back

        Returns:
            SentimentData object
        """
        if not SANPY_AVAILABLE:
            return SentimentData(
                symbol=symbol,
                provider="santiment",
                timestamp=datetime.now().isoformat(),
                raw_data={"error": "sanpy not installed"}
            )

        try:
            slug = self._symbol_to_slug(symbol)
            start_date = datetime.now() - timedelta(days=days_back)
            end_date = datetime.now()

            # Run blocking sanpy call in executor
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: san.get(
                    "social_volume_total",
                    slug=slug,
                    from_date=start_date,
                    to_date=end_date,
                    interval="1d"
                )
            )

            if df is None or df.empty:
                 return SentimentData(
                    symbol=symbol,
                    provider="santiment",
                    timestamp=datetime.now().isoformat()
                )

            # Calculate total mentions
            total_mentions = int(df["value"].sum()) if "value" in df.columns else 0

            # Simple buzz calculation (today vs avg)
            buzz_ratio = None
            if len(df) > 1 and "value" in df.columns:
                last_val = df["value"].iloc[-1]
                avg_val = df["value"].iloc[:-1].mean()
                if avg_val > 0:
                    buzz_ratio = float(last_val / avg_val)

            return SentimentData(
                symbol=symbol,
                provider="santiment",
                timestamp=datetime.now().isoformat(),
                mention_count=total_mentions,
                buzz_ratio=buzz_ratio,
                raw_data={"social_volume": df.to_dict()}
            )

        except Exception as e:
            _logger.error(f"Error fetching social volume for {symbol}: {e}")
            return SentimentData(
                symbol=symbol,
                provider="santiment",
                timestamp=datetime.now().isoformat(),
                raw_data={"error": str(e)}
            )

    async def get_sentiment_metrics(self, symbol: str, days_back: int = 7) -> SentimentData:
        """
        Get sentiment balance metrics.
        """
        if not SANPY_AVAILABLE:
             return SentimentData(
                symbol=symbol,
                provider="santiment",
                timestamp=datetime.now().isoformat(),
                raw_data={"error": "sanpy not installed"}
            )

        try:
            slug = self._symbol_to_slug(symbol)
            start_date = datetime.now() - timedelta(days=days_back)
            end_date = datetime.now()

            # Fetch sentiment positive/negative totals
            # Note: Tickers logic might vary for non-crypto
            loop = asyncio.get_event_loop()

            # Helper to fetch multiple metrics
            def fetch_metrics():
                pos = san.get("sentiment_positive_total", slug=slug, from_date=start_date, to_date=end_date)
                neg = san.get("sentiment_negative_total", slug=slug, from_date=start_date, to_date=end_date)
                return pos, neg

            pos_df, neg_df = await loop.run_in_executor(None, fetch_metrics)

            total_pos = pos_df["value"].sum() if pos_df is not None and not pos_df.empty else 0
            total_neg = neg_df["value"].sum() if neg_df is not None and not neg_df.empty else 0
            total = total_pos + total_neg

            sentiment_score = 0.0
            if total > 0:
                sentiment_score = (total_pos - total_neg) / total  # -1 to 1 range

            return SentimentData(
                symbol=symbol,
                provider="santiment",
                timestamp=datetime.now().isoformat(),
                sentiment_score=sentiment_score,
                bullish_score=total_pos,
                bearish_score=total_neg,
                raw_data={
                    "positive_total": total_pos,
                    "negative_total": total_neg
                }
            )

        except Exception as e:
            _logger.error(f"Error fetching sentiment metrics for {symbol}: {e}")
            return SentimentData(
                symbol=symbol,
                provider="santiment",
                timestamp=datetime.now().isoformat(),
                raw_data={"error": str(e)}
            )

    def _symbol_to_slug(self, symbol: str) -> str:
        """
        Map ticker symbol to Santiment slug.
        Since Santiment is crypto-first, stocks often need specific slugs or project IDs.
        For now, we'll try lowercase symbol name or simple mapping.
        """
        # TODO: Implement proper mapping for stocks if needed
        # Commonly, crypto uses full names (bitcoin), stocks might use tickers
        return symbol.lower()

