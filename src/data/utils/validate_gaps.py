#!/usr/bin/env python3
"""
Gap Validation Script

This script analyzes cached data for gaps and generates a comprehensive validation report.
It focuses solely on gap detection and analysis, creating detailed JSON metadata.

Usage:
    python validate_gaps.py                                    # Validate all cached data
    python validate_gaps.py --symbols BTCUSDT,ETHUSDT         # Validate specific symbols
    python validate_gaps.py --intervals 1h,4h,1d              # Validate specific intervals
    python validate_gaps.py --cache-dir /path/to/cache        # Use custom cache directory
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.cache.unified_cache import configure_unified_cache
from src.data.utils.validation import validate_ohlcv_data, get_data_quality_score
from src.notification.logger import setup_logger

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"  # Fallback if import fails

_logger = setup_logger(__name__)


def validate_cached_data(
    symbols: List[str],
    intervals: List[str],
    cache_dir: str = DATA_CACHE_DIR
) -> Dict[str, Any]:
    """
    Validate cached data for quality and gaps by examining each year's data file.

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
        'gap_analysis': {},
        'yearly_analysis': {},
        'validation_metadata': {}
    }

    for symbol in symbols:
        for interval in intervals:
            print(f"🔍 Validating {symbol} {interval}...")

            try:
                # Get available years for this symbol/interval
                available_years = cache.list_years(symbol, interval)

                if not available_years:
                    results['missing'].append(f"{symbol}_{interval}")
                    print(f"  ❌ No cached data found")
                    _logger.warning("No cached data found for %s_%s", symbol, interval)
                    continue

                print(f"  📅 Found data for years: {sorted(available_years)}")

                # Validate each year's data file individually
                yearly_results = {}
                all_data_valid = True
                total_rows = 0
                all_gaps = []

                for year in sorted(available_years):
                    print(f"    🔍 Validating {year} data...")

                    try:
                        # Load year-specific data
                        year_data = cache.get(symbol, interval, start_date=datetime(year, 1, 1), end_date=datetime(year, 12, 31))

                        if year_data is None or year_data.empty:
                            print(f"      ❌ No data for {year}")
                            yearly_results[year] = {'status': 'missing', 'rows': 0, 'gaps': []}
                            all_data_valid = False
                            continue

                        # Validate year data
                        is_valid, errors = validate_ohlcv_data(year_data, symbol=symbol, interval=interval)
                        quality_score = get_data_quality_score(year_data)
                        gaps = analyze_data_gaps(year_data, interval)

                        yearly_results[year] = {
                            'status': 'valid' if is_valid else 'invalid',
                            'rows': len(year_data),
                            'quality_score': quality_score['quality_score'],
                            'errors': errors if not is_valid else [],
                            'gaps': gaps,
                            'date_range': {
                                'start': year_data.index.min().isoformat(),
                                'end': year_data.index.max().isoformat()
                            }
                        }

                        total_rows += len(year_data)
                        all_gaps.extend(gaps)

                        if is_valid:
                            print(f"      ✅ {year}: {len(year_data)} rows, quality: {quality_score['quality_score']:.2f}")
                            if gaps:
                                print(f"      📊 Gaps found: {len(gaps)} gaps, largest: {max(gap['duration_hours'] for gap in gaps):.1f}h")
                        else:
                            print(f"      ❌ {year}: Invalid data - {errors}")
                            all_data_valid = False

                    except Exception as e:
                        print(f"      ❌ Error validating {year}: {e}")
                        yearly_results[year] = {'status': 'error', 'error': str(e), 'rows': 0, 'gaps': []}
                        all_data_valid = False
                        _logger.exception("Error validating %s_%s_%d: %s", symbol, interval, year, e)

                # Store results
                if all_data_valid:
                    results['valid'].append(f"{symbol}_{interval}")
                    print(f"  ✅ All years valid: {total_rows} total rows")
                else:
                    results['invalid'].append(f"{symbol}_{interval}")
                    print(f"  ❌ Some years have issues")

                # Store detailed analysis
                results['yearly_analysis'] = results.get('yearly_analysis', {})
                results['yearly_analysis'][f"{symbol}_{interval}"] = yearly_results
                results['gap_analysis'][f"{symbol}_{interval}"] = all_gaps

                # Create comprehensive validation metadata
                results['validation_metadata'] = results.get('validation_metadata', {})
                results['validation_metadata'][f"{symbol}_{interval}"] = {
                    'symbol': symbol,
                    'interval': interval,
                    'available_years': sorted(available_years),
                    'total_rows': total_rows,
                    'overall_status': 'valid' if all_data_valid else 'invalid',
                    'yearly_breakdown': yearly_results,
                    'gap_summary': {
                        'total_gaps': len(all_gaps),
                        'largest_gap_hours': max(gap['duration_hours'] for gap in all_gaps) if all_gaps else 0,
                        'gaps_by_year': {year: len(yearly_results[year].get('gaps', [])) for year in available_years}
                    },
                    'validation_timestamp': datetime.now().isoformat()
                }

                _logger.info("VALIDATION_COMPLETE: %s_%s - %d years, %d total rows, %d gaps",
                           symbol, interval, len(available_years), total_rows, len(all_gaps))

            except Exception as e:
                results['invalid'].append(f"{symbol}_{interval}")
                print(f"  ❌ Error validating {symbol} {interval}: {e}")
                _logger.exception("Error validating %s_%s: %s", symbol, interval, e)

    return results


def analyze_data_gaps(df: pd.DataFrame, interval: str) -> List[Dict[str, Any]]:
    """
    Analyze gaps in the data.

    Args:
        df: DataFrame with DatetimeIndex
        interval: Time interval

    Returns:
        List of gap dictionaries with start time and duration
    """
    if len(df) < 2:
        return []

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
            'start': gap_start.isoformat(),
            'duration_hours': gap_duration / pd.Timedelta(hours=1)
        })

    return gap_details


def save_validation_metadata(results: Dict[str, Any], cache_dir: str) -> None:
    """
    Save comprehensive validation metadata to validate-metadata.json in the cache directory.

    Args:
        results: Validation results dictionary
        cache_dir: Cache directory path
    """
    import json
    from pathlib import Path

    metadata_file = Path(cache_dir) / "validate-metadata.json"

    # Create comprehensive metadata
    validation_metadata = {
        'validation_summary': {
            'total_symbols_intervals': len(results.get('validation_metadata', {})),
            'valid': len(results.get('valid', [])),
            'invalid': len(results.get('invalid', [])),
            'missing': len(results.get('missing', [])),
            'validation_timestamp': datetime.now().isoformat()
        },
        'detailed_results': results.get('validation_metadata', {}),
        'gap_analysis': results.get('gap_analysis', {}),
        'yearly_analysis': results.get('yearly_analysis', {})
    }

    try:
        with open(metadata_file, 'w') as f:
            json.dump(validation_metadata, f, indent=2, default=str)
        print(f"📄 Validation metadata saved to: {metadata_file}")
        _logger.info("Validation metadata saved to: %s", metadata_file)
    except Exception as e:
        print(f"❌ Error saving validation metadata: {e}")
        _logger.exception("Error saving validation metadata:")


def parse_interval_to_minutes(interval: str) -> Optional[int]:
    """
    Parse interval string to minutes.

    Args:
        interval: Interval string (e.g., '1m', '1h', '1d')

    Returns:
        Minutes or None if invalid
    """
    interval_map = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    return interval_map.get(interval)


def main():
    """Main function for gap validation."""
    parser = argparse.ArgumentParser(description='Validate cached data for gaps and quality issues')

    # Data selection arguments
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)')
    parser.add_argument('--intervals', type=str, help='Comma-separated list of intervals (e.g., 5m,15m,1h)')
    parser.add_argument('--validate-all', action='store_true', help='Validate all cached data')

    # Configuration arguments
    parser.add_argument('--cache-dir', type=str, default=DATA_CACHE_DIR, help='Cache directory path')

    args = parser.parse_args()

    # Default to validate-all if no action is specified
    if not any([args.symbols, args.intervals, args.validate_all]):
        args.validate_all = True
        print("🔍 No arguments specified, defaulting to --validate-all")

    # Determine symbols and intervals
    if args.validate_all:
        # Discover all cached symbols and intervals
        try:
            from src.data.cache.unified_cache import configure_unified_cache
            cache = configure_unified_cache(cache_dir=args.cache_dir)
            symbols = cache.list_symbols()
            if not symbols:
                print("❌ No cached symbols found")
                return

            # Get all unique intervals across all symbols
            all_intervals = set()
            for symbol in symbols:
                symbol_intervals = cache.list_timeframes(symbol)
                all_intervals.update(symbol_intervals)
            intervals = sorted(list(all_intervals))

            print(f"🔍 Discovered {len(symbols)} symbols and {len(intervals)} intervals in cache")
            print(f"   Symbols: {', '.join(symbols)}")
            print(f"   Intervals: {', '.join(intervals)}")
        except Exception as e:
            print(f"❌ Error discovering cached data: {e}")
            print("   Using default symbols and intervals")
            symbols = ['BTCUSDT', 'ETHUSDT', 'AAPL', 'MSFT']  # Fallback
            intervals = ['5m', '15m', '1h', '4h', '1d']
    else:
        if not args.symbols or not args.intervals:
            print("❌ Error: --symbols and --intervals are required unless using --validate-all")
            return

        symbols = [s.strip() for s in args.symbols.split(',')]
        intervals = [i.strip() for i in args.intervals.split(',')]

    # Execute validation
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

    # Save comprehensive validation metadata
    save_validation_metadata(validation_results, args.cache_dir)


if __name__ == "__main__":
    main()
