#!/usr/bin/env python3
"""
Data Validation and Gap Filling Script

This script:
1. Validates cached data for quality and gaps
2. Attempts to fill gaps using alternative providers
3. Reports data quality metrics

Usage:
    python src/data/cache/validate_and_fill_gaps.py --help
    python src/data/cache/validate_and_fill_gaps.py --validate --symbols BTCUSDT,ETHUSDT --intervals 5m,15m,1h
    python src/data/cache/validate_and_fill_gaps.py --validate-all
    python src/data/cache/validate_and_fill_gaps.py --fill-gaps --symbols BTCUSDT --intervals 5m
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.cache.unified_cache import configure_unified_cache, get_unified_cache
from src.data.utils.validation import validate_ohlcv_data, get_data_quality_score
from src.notification.logger import setup_logger

# Import data downloaders
from src.data.binance_data_downloader import BinanceDataDownloader
from src.data.yahoo_data_downloader import YahooDataDownloader
from src.data.alpha_vantage_data_downloader import AlphaVantageDataDownloader

# Import ticker classifier for automatic provider selection
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
from src.common.ticker_classifier import TickerClassifier, DataProvider

# Initialize logger
_logger = setup_logger(__name__)


def validate_cached_data(
    symbols: List[str],
    intervals: List[str],
    cache_dir: str = "d:/data-cache"
) -> Dict[str, Any]:
    """
    Validate cached data for quality and gaps.

    Args:
        symbols: List of symbols to validate
        intervals: List of intervals to validate
        cache_dir: Cache directory path

    Returns:
        Dictionary with validation results
    """
    print(f"🔍 Validating cached data at: {cache_dir}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Intervals: {', '.join(intervals)}")
    print()

    _logger.info("Starting data validation for %d symbols, %d intervals", len(symbols), len(intervals))

    # Configure cache
    cache = configure_unified_cache(cache_dir=cache_dir)

    results = {
        'valid': [],
        'invalid': [],
        'missing': [],
        'quality_scores': {},
        'gap_analysis': {}
    }

    for symbol in symbols:
        for interval in intervals:
            print(f"🔍 Validating {symbol} {interval}...")

            try:
                # Get cached data
                df = cache.get(symbol, interval)

                if df is None or df.empty:
                    results['missing'].append(f"{symbol}_{interval}")
                    print(f"  ❌ No cached data found")
                    _logger.warning("No cached data found for %s_%s", symbol, interval)
                    continue

                # Validate data
                is_valid, errors = validate_ohlcv_data(df, symbol=symbol, interval=interval)
                quality_score = get_data_quality_score(df)

                # Analyze gaps
                gap_analysis = analyze_data_gaps(df, interval)

                results['quality_scores'][f"{symbol}_{interval}"] = quality_score
                results['gap_analysis'][f"{symbol}_{interval}"] = gap_analysis

                if is_valid:
                    results['valid'].append(f"{symbol}_{interval}")
                    print(f"  ✅ Valid data: {len(df)} rows, quality: {quality_score['quality_score']:.2f}")
                    _logger.info("VALID: %s_%s - %d rows, quality: %.2f", symbol, interval, len(df), quality_score['quality_score'])
                else:
                    results['invalid'].append(f"{symbol}_{interval}")
                    print(f"  ⚠️  Invalid data: {errors}")
                    _logger.warning("INVALID: %s_%s - %s", symbol, interval, errors)

                # Report gaps
                if gap_analysis['total_gaps'] > 0:
                    print(f"  📊 Gaps found: {gap_analysis['total_gaps']} gaps, largest: {gap_analysis['largest_gap_hours']:.1f}h")
                    _logger.info("GAPS: %s_%s - %d gaps, largest: %.1fh", symbol, interval, gap_analysis['total_gaps'], gap_analysis['largest_gap_hours'])

            except Exception as e:
                results['invalid'].append(f"{symbol}_{interval}")
                print(f"  ❌ Validation error: {str(e)}")
                _logger.exception("VALIDATION ERROR: %s_%s - %s", symbol, interval, str(e))

    return results


def analyze_data_gaps(df: pd.DataFrame, interval: str) -> Dict[str, Any]:
    """
    Analyze gaps in the data.

    Args:
        df: DataFrame with DatetimeIndex
        interval: Time interval

    Returns:
        Dictionary with gap analysis
    """
    if len(df) < 2:
        return {'total_gaps': 0, 'largest_gap_hours': 0, 'gap_details': []}

    # Calculate gaps
    gaps = df.index.to_series().diff().dropna()
    gap_hours = gaps / pd.Timedelta(hours=1)

    # Find large gaps (more than expected interval)
    expected_interval_minutes = parse_interval_to_minutes(interval)
    if expected_interval_minutes:
        expected_gap_hours = expected_interval_minutes / 60
        large_gaps = gaps[gap_hours > expected_gap_hours * 1.5]  # 50% tolerance
    else:
        large_gaps = gaps[gap_hours > 24]  # Default: gaps > 24 hours

    gap_details = []
    for gap_start, gap_duration in large_gaps.items():
        gap_details.append({
            'start': gap_start,
            'duration_hours': gap_duration / pd.Timedelta(hours=1)
        })

    return {
        'total_gaps': len(large_gaps),
        'largest_gap_hours': gap_hours.max() if not gap_hours.empty else 0,
        'gap_details': gap_details
    }


def fill_data_gaps(
    symbols: List[str],
    intervals: List[str],
    cache_dir: str = "d:/data-cache",
    max_gap_hours: float = 24.0
) -> Dict[str, Any]:
    """
    Attempt to fill gaps in cached data using alternative providers.

    Args:
        symbols: List of symbols to process
        intervals: List of intervals to process
        cache_dir: Cache directory path
        max_gap_hours: Maximum gap size to attempt filling

    Returns:
        Dictionary with gap filling results
    """
    print(f"🔧 Attempting to fill gaps in cached data...")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Intervals: {', '.join(intervals)}")
    print(f"⏰ Max gap size: {max_gap_hours} hours")
    print()

    _logger.info("Starting gap filling for %d symbols, %d intervals", len(symbols), len(intervals))

    # Configure cache
    cache = configure_unified_cache(cache_dir=cache_dir)

    # Initialize alternative downloaders
    downloaders = {}
    try:
        downloaders['binance'] = BinanceDataDownloader()
        print("  ✅ Binance downloader initialized")
    except Exception as e:
        print(f"  ⚠️  Binance downloader failed: {str(e)}")

    try:
        downloaders['yahoo'] = YahooDataDownloader()
        print("  ✅ Yahoo downloader initialized")
    except Exception as e:
        print(f"  ⚠️  Yahoo downloader failed: {str(e)}")

    try:
        # Check if Alpha Vantage API key is available
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_vantage_key:
            downloaders['alpha_vantage'] = AlphaVantageDataDownloader(api_key=alpha_vantage_key)
            print("  ✅ Alpha Vantage downloader initialized")
        else:
            print("  ⚠️  Alpha Vantage downloader skipped: No API key found (set ALPHA_VANTAGE_API_KEY env var)")
    except Exception as e:
        print(f"  ⚠️  Alpha Vantage downloader failed: {str(e)}")

    print()

    results = {
        'gaps_filled': [],
        'gaps_failed': [],
        'no_gaps': []
    }

    for symbol in symbols:
        for interval in intervals:
            print(f"🔧 Processing {symbol} {interval}...")

            try:
                # Get cached data
                df = cache.get(symbol, interval)

                if df is None or df.empty:
                    print(f"  ❌ No cached data found")
                    continue

                # Analyze gaps
                gap_analysis = analyze_data_gaps(df, interval)

                if gap_analysis['total_gaps'] == 0:
                    results['no_gaps'].append(f"{symbol}_{interval}")
                    print(f"  ✅ No gaps found")
                    continue

                # Check if gaps are small enough to fill
                if gap_analysis['largest_gap_hours'] > max_gap_hours:
                    results['gaps_failed'].append(f"{symbol}_{interval}")
                    print(f"  ⚠️  Gaps too large to fill: {gap_analysis['largest_gap_hours']:.1f}h > {max_gap_hours}h")
                    continue

                # TODO: Implement gap filling logic
                # This would involve:
                # 1. Identifying specific gap periods
                # 2. Using alternative providers to fetch data for those periods
                # 3. Merging the new data with existing data
                # 4. Updating the cache

                print(f"  🔧 Gap filling not yet implemented")
                print(f"  📊 Found {gap_analysis['total_gaps']} gaps, largest: {gap_analysis['largest_gap_hours']:.1f}h")

                results['gaps_failed'].append(f"{symbol}_{interval}")

            except Exception as e:
                results['gaps_failed'].append(f"{symbol}_{interval}")
                print(f"  ❌ Gap filling error: {str(e)}")
                _logger.exception("GAP FILLING ERROR: %s_%s - %s", symbol, interval, str(e))

    return results


def parse_interval_to_minutes(interval: str) -> Optional[int]:
    """
    Parse interval string to minutes.

    Args:
        interval: Interval string (e.g., '1m', '1h', '1d')

    Returns:
        Number of minutes or None if invalid
    """
    interval = interval.lower()

    if interval.endswith('m'):
        try:
            return int(interval[:-1])
        except ValueError:
            return None
    elif interval.endswith('h'):
        try:
            return int(interval[:-1]) * 60
        except ValueError:
            return None
    elif interval.endswith('d'):
        try:
            return int(interval[:-1]) * 24 * 60
        except ValueError:
            return None
    else:
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate and fill gaps in cached data")

    # Action arguments
    parser.add_argument('--validate', action='store_true', help='Validate cached data')
    parser.add_argument('--fill-gaps', action='store_true', help='Attempt to fill gaps in cached data')
    parser.add_argument('--validate-all', action='store_true', help='Validate all cached data')

    # Data selection arguments
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)')
    parser.add_argument('--intervals', type=str, help='Comma-separated list of intervals (e.g., 5m,15m,1h)')

    # Configuration arguments
    parser.add_argument('--cache-dir', type=str, default='d:/data-cache', help='Cache directory path')
    parser.add_argument('--max-gap-hours', type=float, default=24.0, help='Maximum gap size to attempt filling (hours)')

    args = parser.parse_args()

    # Determine symbols and intervals
    if args.validate_all:
        # TODO: Implement logic to discover all cached symbols and intervals
        symbols = ['BTCUSDT', 'ETHUSDT', 'AAPL', 'MSFT']  # Default for now
        intervals = ['5m', '15m', '1h', '4h', '1d']
    else:
        if not args.symbols or not args.intervals:
            print("❌ Error: --symbols and --intervals are required unless using --validate-all")
            return

        symbols = [s.strip() for s in args.symbols.split(',')]
        intervals = [i.strip() for i in args.intervals.split(',')]

    # Execute requested actions
    if args.validate or args.validate_all:
        print("🔍 Starting data validation...")
        validation_results = validate_cached_data(symbols, intervals, args.cache_dir)

        print("\n📊 Validation Summary:")
        print(f"  ✅ Valid: {len(validation_results['valid'])}")
        print(f"  ❌ Invalid: {len(validation_results['invalid'])}")
        print(f"  ❓ Missing: {len(validation_results['missing'])}")

        if validation_results['invalid']:
            print(f"\n❌ Invalid data:")
            for item in validation_results['invalid']:
                print(f"  - {item}")

        if validation_results['missing']:
            print(f"\n❓ Missing data:")
            for item in validation_results['missing']:
                print(f"  - {item}")

    if args.fill_gaps:
        print("\n🔧 Starting gap filling...")
        gap_results = fill_data_gaps(symbols, intervals, args.cache_dir, args.max_gap_hours)

        print("\n📊 Gap Filling Summary:")
        print(f"  ✅ No gaps: {len(gap_results['no_gaps'])}")
        print(f"  🔧 Gaps filled: {len(gap_results['gaps_filled'])}")
        print(f"  ❌ Gaps failed: {len(gap_results['gaps_failed'])}")

        if gap_results['gaps_failed']:
            print(f"\n❌ Failed to fill gaps:")
            for item in gap_results['gaps_failed']:
                print(f"  - {item}")


if __name__ == "__main__":
    main()
