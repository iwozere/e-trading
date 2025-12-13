"""
Standalone script to collect sentiment data for tickers from the last 14 days of Phase 2 alerts.
Saves results to yesterday's results folder.

Usage:
    python -m src.ml.pipeline.p06_emps2.collect_sentiments

This script:

Collects all unique tickers from the last 14 days of Phase 2 alerts
Processes them in batches of 20 to respect rate limits
Saves the results to yesterday's results folder as 10_phase2_sentiments.csv
Includes comprehensive logging and error handling

The script will:
Skip any days without Phase 2 alerts
Process all available tickers regardless of sentiment thresholds
Save all available sentiment metrics
Work independently of the main pipeline
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from typing import List, Set
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p06_emps2.sentiment_filter import SentimentFilter, SentimentFilterConfig
from src.ml.pipeline.p06_emps2.config import SentimentFilterConfig

_logger = setup_logger(__name__)

def get_phase2_tickers_from_date(date: datetime) -> Set[str]:
    """
    Get unique tickers from a specific date's Phase 2 alerts.

    Args:
        date: Date to check for Phase 2 alerts

    Returns:
        Set of ticker symbols
    """
    date_str = date.strftime('%Y-%m-%d')
    alerts_file = PROJECT_ROOT / 'results' / 'emps2' / date_str / '08_phase2_alerts.csv'

    if not alerts_file.exists():
        _logger.debug(f"No Phase 2 alerts found for {date_str}")
        return set()

    try:
        df = pd.read_csv(alerts_file)
        if 'ticker' not in df.columns:
            _logger.warning(f"No 'ticker' column in {alerts_file}")
            return set()
        return set(df['ticker'].dropna().unique())
    except Exception as e:
        _logger.error(f"Error reading {alerts_file}: {e}")
        return set()

def collect_sentiments():
    """Main function to collect sentiments for tickers from the last 14 days."""
    try:
        _logger.info("Starting sentiment collection for Phase 2 tickers (last 14 days)")

        # Get dates for the last 14 days
        end_date = datetime.now(timezone.utc).date()
        dates = [end_date - timedelta(days=i) for i in range(14)]

        # Collect all unique tickers
        all_tickers = set()
        for date in tqdm(dates, desc="Collecting tickers from past 14 days"):
            all_tickers.update(get_phase2_tickers_from_date(date))

        if not all_tickers:
            _logger.warning("No tickers found in the last 14 days")
            return

        ticker_list = sorted(list(all_tickers))
        _logger.info(f"Found {len(ticker_list)} unique tickers from the last 14 days")

        # Initialize sentiment filter
        config = SentimentFilterConfig(
            min_mentions_24h=0,  # Get all tickers regardless of mentions
            min_sentiment_score=-1.0,  # Include all sentiment scores
            max_bot_pct=1.0,  # Include all bot percentages
            min_virality_index=0.0,  # No virality requirement
            min_unique_authors=0,  # No minimum authors
            enabled=True
        )

        sentiment_filter = SentimentFilter(config)

        # Process in batches of 20
        batch_size = 20
        all_results = []

        for i in tqdm(range(0, len(ticker_list), batch_size), desc="Processing sentiment batches"):
            batch = ticker_list[i:i + batch_size]
            try:
                batch_results = sentiment_filter.apply_filters(batch)
                if not batch_results.empty:
                    batch_results['batch_timestamp'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                    all_results.append(batch_results)
            except Exception as e:
                _logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

        if not all_results:
            _logger.warning("No sentiment data collected")
            return

        # Combine all results
        results_df = pd.concat(all_results, ignore_index=True)

        # Save to yesterday's results folder
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        output_dir = PROJECT_ROOT / 'results' / 'emps2' / yesterday
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / '10_sentiments.csv'
        results_df.to_csv(output_file, index=False)

        _logger.info(f"Saved sentiment data for {len(results_df)} tickers to {output_file}")

        # Log summary
        if not results_df.empty:
            _logger.info("\nSentiment Summary:")
            _logger.info(f"  Total tickers with data: {len(results_df)}")
            _logger.info(f"  Avg. sentiment score: {results_df['sentiment_score'].mean():.2f}")
            _logger.info(f"  Avg. mentions (24h): {results_df['mentions_24h'].mean():.1f}")
            _logger.info(f"  Avg. virality index: {results_df['virality_index'].mean():.2f}")
            _logger.info(f"  Avg. bot percentage: {results_df['bot_pct'].mean()*100:.1f}%")

    except Exception as e:
        _logger.exception(f"Error in collect_sentiments: {e}")
        raise

if __name__ == "__main__":
    collect_sentiments()