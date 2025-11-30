#!/usr/bin/env python3
"""
Fundamentals Cache Refresh Script

This script can be used to periodically refresh fundamentals cache data.
It supports:
- Refreshing specific symbols or all cached symbols
- Loading symbols from text files
- Force refresh (bypass TTL)
- Cleanup expired data
- Batch processing with rate limiting

Default Behavior:
- Scans DATA_CACHE_DIR/fundamentals directory for all tickers with cached data
- Refreshes fundamentals for all found tickers

Usage:
    python src/data/utils/refresh_fundamentals_cache.py                    # Scan and refresh all cached tickers
    python src/data/utils/refresh_fundamentals_cache.py --symbols AAPL,GOOGL,MSFT
    python src/data/utils/refresh_fundamentals_cache.py --all-symbols      # Same as default
    python src/data/utils/refresh_fundamentals_cache.py --cleanup-only
    python src/data/utils/refresh_fundamentals_cache.py --symbols-file my_symbols.txt
    python src/data/utils/refresh_fundamentals_cache.py --force-refresh --symbols AAPL
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.data.cache.fundamentals_cache import get_fundamentals_cache
from src.data.cache.fundamentals_combiner import get_fundamentals_combiner
from src.notification.logger import setup_logger

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "data-cache"

_logger = setup_logger(__name__)

def get_cached_symbols(cache_dir: str) -> List[str]:
    """Get list of symbols that have cached fundamentals data by scanning the fundamentals directory."""
    try:
        fundamentals_dir = Path(cache_dir) / "fundamentals"

        if not fundamentals_dir.exists():
            _logger.warning("Fundamentals directory does not exist: %s", fundamentals_dir)
            return []

        symbols = []
        total_dirs = 0

        # Scan all subdirectories in the fundamentals directory
        for item in fundamentals_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                total_dirs += 1
                # Each subdirectory represents a ticker symbol
                symbol = item.name.upper()

                # Check if the directory contains any JSON files (cached data)
                json_files = list(item.glob("*.json"))
                if json_files:
                    symbols.append(symbol)
                    _logger.debug("Found cached fundamentals for %s (%d files)", symbol, len(json_files))
                else:
                    _logger.debug("Skipping %s (no JSON files found)", symbol)

        symbols.sort()  # Sort alphabetically for consistent output
        _logger.info("Scanned %d directories, found %d symbols with cached fundamentals data",
                    total_dirs, len(symbols))
        return symbols

    except Exception:
        _logger.exception("Error scanning fundamentals directory:")
        return []

def load_symbols_from_file(file_path: str) -> List[str]:
    """Load symbols from a text file (one symbol per line)."""
    try:
        symbols = []
        with open(file_path, 'r') as f:
            for line in f:
                symbol = line.strip()
                if symbol and not symbol.startswith('#'):  # Skip empty lines and comments
                    symbols.append(symbol.upper())

        _logger.info("Loaded %d symbols from %s", len(symbols), file_path)
        return symbols
    except Exception as e:
        _logger.error("Error loading symbols from %s: %s", file_path, e)
        return []

def refresh_symbol_fundamentals(dm: DataManager, symbol: str, data_types: List[str],
                               force_refresh: bool = False) -> Dict[str, Any]:
    """Refresh fundamentals for a specific symbol."""
    results = {
        'symbol': symbol,
        'data_types': {},
        'success': True,
        'errors': []
    }

    for data_type in data_types:
        try:
            _logger.info("Refreshing %s fundamentals for %s (force_refresh=%s)",
                        data_type, symbol, force_refresh)

            fundamentals = dm.get_fundamentals(
                symbol=symbol,
                data_type=data_type,
                force_refresh=force_refresh,
                combination_strategy='priority_based'
            )

            if fundamentals:
                results['data_types'][data_type] = {
                    'success': True,
                    'fields_count': len(fundamentals),
                    'has_metadata': '_metadata' in fundamentals
                }
                _logger.info("Successfully refreshed %s for %s: %d fields",
                           data_type, symbol, len(fundamentals))
            else:
                results['data_types'][data_type] = {
                    'success': False,
                    'error': 'No data returned'
                }
                results['errors'].append(f"No data for {data_type}")

        except Exception as e:
            error_msg = f"Error refreshing {data_type} for {symbol}: {e}"
            _logger.error(error_msg)
            results['data_types'][data_type] = {
                'success': False,
                'error': str(e)
            }
            results['errors'].append(error_msg)
            results['success'] = False

    return results

def cleanup_expired_cache(cache_dir: str, data_types: List[str]) -> Dict[str, Any]:
    """Clean up expired cache data."""
    cleanup_results = {
        'data_types': {},
        'total_removed_files': 0,
        'total_removed_symbols': 0
    }

    try:
        combiner = get_fundamentals_combiner()
        cache = get_fundamentals_cache(cache_dir, combiner)

        for data_type in data_types:
            _logger.info("Cleaning up expired %s cache data", data_type)

            stats = cache.cleanup_expired_data(data_type=data_type)
            cleanup_results['data_types'][data_type] = stats
            cleanup_results['total_removed_files'] += stats.get('removed_files', 0)
            cleanup_results['total_removed_symbols'] += stats.get('removed_symbols', 0)

            _logger.info("Cleaned up %s: %d files, %d symbols removed",
                        data_type, stats.get('removed_files', 0), stats.get('removed_symbols', 0))

    except Exception as e:
        _logger.exception("Error during cache cleanup:")
        cleanup_results['error'] = str(e)

    return cleanup_results

def main():
    """Main function to run the fundamentals cache refresh script."""
    parser = argparse.ArgumentParser(
        description='Refresh fundamentals cache data. By default, scans DATA_CACHE_DIR/fundamentals and refreshes all cached tickers.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refresh all cached tickers (default behavior)
  python src/data/utils/refresh_fundamentals_cache.py

  # Refresh specific tickers
  python src/data/utils/refresh_fundamentals_cache.py --symbols AAPL,MSFT,GOOGL

  # Refresh from file
  python src/data/utils/refresh_fundamentals_cache.py --symbols-file my_tickers.txt

  # Force refresh (bypass TTL)
  python src/data/utils/refresh_fundamentals_cache.py --force-refresh

  # Dry run to see what would be refreshed
  python src/data/utils/refresh_fundamentals_cache.py --dry-run

  # Cleanup only
  python src/data/utils/refresh_fundamentals_cache.py --cleanup-only
        """
    )

    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group(required=False)
    symbol_group.add_argument('--symbols', type=str,
                             help='Comma-separated list of symbols to refresh (e.g., AAPL,GOOGL,MSFT)')
    symbol_group.add_argument('--all-symbols', action='store_true',
                             help='Refresh all cached symbols (same as default behavior)')
    symbol_group.add_argument('--cleanup-only', action='store_true',
                             help='Only cleanup expired cache data, no refresh')
    symbol_group.add_argument('--symbols-file', type=str,
                             help='Path to text file containing symbols (one per line)')

    # Data types
    parser.add_argument('--data-types', type=str, default='ratios,profile,statements',
                       help='Comma-separated list of data types to refresh (default: ratios,profile,statements)')

    # Options
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh even if cache is valid')
    parser.add_argument('--cache-dir', type=str, default=DATA_CACHE_DIR,
                       help=f'Cache directory path (default: {DATA_CACHE_DIR})')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between symbol refreshes in seconds (default: 1.0)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually doing it')

    args = parser.parse_args()

    # Parse data types
    data_types = [dt.strip() for dt in args.data_types.split(',')]

    _logger.info("Starting fundamentals cache refresh")
    _logger.info("Cache directory: %s", args.cache_dir)
    _logger.info("Data types: %s", data_types)
    _logger.info("Force refresh: %s", args.force_refresh)
    _logger.info("Dry run: %s", args.dry_run)

    if args.dry_run:
        _logger.info("DRY RUN MODE - No actual changes will be made")

    try:
        # Initialize DataManager
        dm = DataManager(args.cache_dir)

        if args.cleanup_only:
            # Only cleanup expired data
            _logger.info("Running cache cleanup only")
            cleanup_results = cleanup_expired_cache(args.cache_dir, data_types)

            _logger.info("Cleanup completed:")
            _logger.info("  Total files removed: %d", cleanup_results['total_removed_files'])
            _logger.info("  Total symbols removed: %d", cleanup_results['total_removed_symbols'])

            return

        # Determine symbols to refresh
        if args.all_symbols:
            symbols = get_cached_symbols(args.cache_dir)
            _logger.info("Found %d cached symbols to refresh", len(symbols))
        elif args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            _logger.info("Refreshing %d specified symbols", len(symbols))
        elif args.symbols_file:
            symbols = load_symbols_from_file(args.symbols_file)
            _logger.info("Refreshing %d symbols from file", len(symbols))
        else:
            # Default: scan all tickers in DATA_CACHE_DIR/fundamentals
            symbols = get_cached_symbols(args.cache_dir)
            if symbols:
                _logger.info("No symbols specified, scanning fundamentals directory")
                _logger.info("Found %d symbols with cached fundamentals data to refresh", len(symbols))
            else:
                # Fallback to example_tickers.txt if no cached symbols found
                default_file = os.path.join(os.path.dirname(__file__), 'example_tickers.txt')
                if os.path.exists(default_file):
                    symbols = load_symbols_from_file(default_file)
                    _logger.info("No cached symbols found, using default file: %s", default_file)
                    _logger.info("Refreshing %d symbols from default file", len(symbols))
                else:
                    _logger.warning("No cached symbols found and no default file available")
                    symbols = []

        if not symbols:
            _logger.warning("No symbols to refresh")
            return

        # Refresh fundamentals for each symbol
        results = []
        for i, symbol in enumerate(symbols):
            _logger.info("Processing symbol %d/%d: %s", i+1, len(symbols), symbol)

            if not args.dry_run:
                result = refresh_symbol_fundamentals(dm, symbol, data_types, args.force_refresh)
                results.append(result)

                # Add delay between symbols to respect rate limits
                if i < len(symbols) - 1 and args.delay > 0:
                    time.sleep(args.delay)
            else:
                _logger.info("DRY RUN: Would refresh %s for data types: %s", symbol, data_types)

        # Summary
        if not args.dry_run:
            successful_symbols = [r for r in results if r['success']]
            failed_symbols = [r for r in results if not r['success']]

            _logger.info("Refresh completed:")
            _logger.info("  Successful symbols: %d", len(successful_symbols))
            _logger.info("  Failed symbols: %d", len(failed_symbols))

            if failed_symbols:
                _logger.warning("Failed symbols:")
                for result in failed_symbols:
                    _logger.warning("  %s: %s", result['symbol'], ', '.join(result['errors']))

        # Optional cleanup after refresh
        if not args.dry_run and not args.cleanup_only:
            _logger.info("Running post-refresh cleanup")
            cleanup_results = cleanup_expired_cache(args.cache_dir, data_types)
            _logger.info("Post-refresh cleanup: %d files removed",
                        cleanup_results['total_removed_files'])

        # Output result for scheduler
        if not args.dry_run:
            import json
            result = {
                "success": True,
                "total_symbols": len(symbols),
                "successful_symbols": len(successful_symbols),
                "failed_symbols": len(failed_symbols),
                "cleanup_files_removed": cleanup_results['total_removed_files'] if not args.cleanup_only else 0
            }
            print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

    except Exception as e:
        _logger.exception("Error during cache refresh:")

        # Output error result for scheduler
        import json
        result = {
            "success": False,
            "error": str(e)
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

        sys.exit(1)

if __name__ == "__main__":
    main()
