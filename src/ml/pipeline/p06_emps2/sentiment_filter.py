"""
Sentiment Filter

Applies sentiment-based filters using social media data.
Integrates with src/common/sentiments module for async sentiment collection.

Filters stocks based on:
- Social media mentions (volume)
- Sentiment score (positive/negative/neutral)
- Virality index (growth in mentions)
- Bot activity percentage
- Unique author count (organic discussion)
"""

from pathlib import Path
import sys
from typing import List, Optional
from datetime import datetime
import asyncio

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p06_emps2.config import SentimentFilterConfig

_logger = setup_logger(__name__)


class SentimentFilter:
    """
    Applies sentiment filters to stock universe.

    Uses async sentiment collection from src/common/sentiments to:
    - Fetch social media data (StockTwits, Reddit)
    - Calculate sentiment scores
    - Detect viral stocks and bot activity

    Filters for stocks with genuine social momentum and positive sentiment.
    """

    def __init__(self, config: SentimentFilterConfig):
        """
        Initialize sentiment filter.

        Args:
            config: Sentiment filter configuration
        """
        self.config = config

        # Results directory (dated)
        today = datetime.now().strftime('%Y-%m-%d')
        self._results_dir = Path("results") / "emps2" / today
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Try to import sentiment collection (graceful degradation)
        self.sentiment_available = False
        try:
            from src.common.sentiments.collect_sentiment_async import collect_sentiment_batch
            self.collect_sentiment_batch = collect_sentiment_batch
            self.sentiment_available = True
            _logger.info("Sentiment Filter initialized: mentions>=%d, sentiment>=%.2f, bot<%.0f%%, virality>=%.2f",
                        config.min_mentions_24h,
                        config.min_sentiment_score,
                        config.max_bot_pct * 100,
                        config.min_virality_index)
        except ImportError:
            _logger.warning("Sentiment module not available - sentiment filtering will be skipped")
            _logger.info("To enable sentiment filtering, ensure src/common/sentiments is configured")

    async def apply_filters_async(self, tickers: List[str]) -> pd.DataFrame:
        """
        Apply sentiment filters to ticker list (async).

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with sentiment-filtered tickers and metrics
        """
        try:
            if not self.config.enabled:
                _logger.info("Sentiment filtering disabled by configuration")
                return pd.DataFrame({'ticker': tickers})

            if not self.sentiment_available:
                _logger.warning("Sentiment module not available - returning all tickers")
                return pd.DataFrame({'ticker': tickers})

            _logger.info("Applying sentiment filters to %d tickers", len(tickers))

            # Collect sentiment data (concurrent API calls)
            sentiment_data = await self.collect_sentiment_batch(
                tickers,
                lookback_hours=24
            )

            # Apply filters
            results = []
            for ticker, sentiment in sentiment_data.items():
                if sentiment is None:
                    _logger.debug("No sentiment data for %s - excluding", ticker)
                    continue

                # Check filters
                if self._passes_filters(ticker, sentiment):
                    results.append({
                        'ticker': ticker,
                        'mentions_24h': sentiment.mentions_24h,
                        'sentiment_score': sentiment.sentiment_normalized,
                        'sentiment_raw': sentiment.sentiment_score_24h,
                        'virality_index': sentiment.virality_index,
                        'bot_pct': sentiment.bot_pct,
                        'unique_authors': sentiment.unique_authors_24h,
                        'positive_ratio': sentiment.positive_ratio_24h
                    })

            df = pd.DataFrame(results)

            _logger.info("After sentiment filtering: %d tickers (%.1f%%)",
                        len(df),
                        100.0 * len(df) / len(tickers) if tickers else 0)

            # Save results
            self._save_results(df)

            return df

        except Exception:
            _logger.exception("Error applying sentiment filters:")
            return pd.DataFrame({'ticker': tickers})

    def apply_filters(self, tickers: List[str]) -> pd.DataFrame:
        """
        Apply sentiment filters to ticker list (sync wrapper).

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with sentiment-filtered tickers and metrics
        """
        return asyncio.run(self.apply_filters_async(tickers))

    def _passes_filters(self, ticker: str, sentiment) -> bool:
        """
        Check if ticker passes sentiment filters.

        Args:
            ticker: Ticker symbol
            sentiment: SentimentFeatures object

        Returns:
            True if passes all filters
        """
        try:
            # Mentions filter
            if sentiment.mentions_24h < self.config.min_mentions_24h:
                _logger.debug("%s failed: mentions=%d (min=%d)",
                             ticker, sentiment.mentions_24h, self.config.min_mentions_24h)
                return False

            # Sentiment score filter
            if sentiment.sentiment_normalized < self.config.min_sentiment_score:
                _logger.debug("%s failed: sentiment=%.2f (min=%.2f)",
                             ticker, sentiment.sentiment_normalized, self.config.min_sentiment_score)
                return False

            # Bot activity filter
            if sentiment.bot_pct > self.config.max_bot_pct:
                _logger.debug("%s failed: bot_pct=%.2f (max=%.2f)",
                             ticker, sentiment.bot_pct, self.config.max_bot_pct)
                return False

            # Virality filter
            if sentiment.virality_index < self.config.min_virality_index:
                _logger.debug("%s failed: virality=%.2f (min=%.2f)",
                             ticker, sentiment.virality_index, self.config.min_virality_index)
                return False

            # Unique authors filter
            if sentiment.unique_authors_24h < self.config.min_unique_authors:
                _logger.debug("%s failed: authors=%d (min=%d)",
                             ticker, sentiment.unique_authors_24h, self.config.min_unique_authors)
                return False

            # Passed all filters
            _logger.debug("%s passed: mentions=%d, sentiment=%.2f, virality=%.2f, bot_pct=%.2f",
                         ticker, sentiment.mentions_24h, sentiment.sentiment_normalized,
                         sentiment.virality_index, sentiment.bot_pct)

            return True

        except Exception:
            _logger.exception("Error checking sentiment filters for %s:", ticker)
            return False

    def _save_results(self, df: pd.DataFrame) -> None:
        """
        Save sentiment filter results to CSV.

        Args:
            df: Filtered DataFrame
        """
        try:
            if df.empty:
                _logger.warning("No sentiment results to save")
                return

            # Sort by sentiment score (highest first)
            df = df.sort_values('sentiment_score', ascending=False)

            output_path = self._results_dir / "06_sentiment_filtered.csv"
            df.to_csv(output_path, index=False)

            _logger.info("Saved sentiment filter results to: %s", output_path)

            # Log top candidates
            _logger.info("Top 10 by sentiment score:")
            for _, row in df.head(10).iterrows():
                _logger.info("  %s: sentiment=%.2f, mentions=%d, virality=%.2f, bot_pct=%.1f%%",
                            row['ticker'], row['sentiment_score'], row['mentions_24h'],
                            row['virality_index'], row['bot_pct'] * 100)

        except Exception:
            _logger.exception("Error saving sentiment filter results:")


def create_sentiment_filter(config: SentimentFilterConfig) -> SentimentFilter:
    """
    Factory function to create sentiment filter.

    Args:
        config: Sentiment filter configuration

    Returns:
        SentimentFilter instance
    """
    return SentimentFilter(config)
