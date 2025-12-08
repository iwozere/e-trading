#!/usr/bin/env python3
"""
Unit test for sentiment filter - simple test with single ticker.

Run with: python -m pytest src/ml/pipeline/p06_emps2/tests/test_sentiment_filter.py -v -s
Or directly: python src/ml/pipeline/p06_emps2/tests/test_sentiment_filter.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p06_emps2.sentiment_filter import SentimentFilter, SentimentFilterConfig


def test_sentiment_single_ticker():
    """Test sentiment filter with single ticker and print results."""
    print("\n" + "="*70)
    print("Testing Sentiment Filter with Single Ticker")
    print("="*70)

    # Create config with relaxed filters for testing
    config = SentimentFilterConfig(
        min_mentions_24h=1,  # Very low threshold for testing
        min_sentiment_score=0.0,  # Accept any sentiment
        max_bot_pct=1.0,  # Accept high bot activity
        min_virality_index=0.0,  # No virality requirement
        min_unique_authors=1,  # Minimum unique authors
        enabled=True
    )

    # Create sentiment filter
    sentiment_filter = SentimentFilter(config)

    # Test with a popular ticker
    test_ticker = "NVDA"
    print(f"\nAnalyzing sentiment for: {test_ticker}")
    print("-" * 70)

    try:
        # Apply filter
        result_df = sentiment_filter.apply_filters([test_ticker])

        if result_df.empty:
            print(f"❌ No sentiment data available for {test_ticker}")
            print("\nPossible reasons:")
            print("  - Sentiment module not configured")
            print("  - API credentials missing")
            print("  - No social media mentions found")
        else:
            print(f"✅ Successfully retrieved sentiment data for {test_ticker}\n")

            # Print all columns and values
            print("Sentiment Metrics:")
            print("-" * 70)
            for col in result_df.columns:
                value = result_df.iloc[0][col]
                print(f"  {col:25s}: {value}")

            print("\n" + "="*70)
            print("Test Complete")
            print("="*70)

    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        print(f"\nException type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    # Run test directly
    test_sentiment_single_ticker()
