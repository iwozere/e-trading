#!/usr/bin/env python3
"""
Periodic Fundamentals Cache Refresh

This script can be run periodically (e.g., via cron) to refresh fundamentals cache.
It's designed to be lightweight and safe for automated execution.

Usage:
    # Use default example_tickers.txt (recommended for daily cron)
    python src/data/utils/periodic_fundamentals_refresh.py

    # Refresh all cached symbols
    python src/data/utils/periodic_fundamentals_refresh.py --all-symbols

    # Refresh specific symbols (for testing)
    python src/data/utils/periodic_fundamentals_refresh.py --symbols AAPL,GOOGL,MSFT

    # Cleanup only (for weekly maintenance)
    python src/data/utils/periodic_fundamentals_refresh.py --cleanup-only
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.utils.refresh_fundamentals_cache import (
    get_cached_symbols,
    refresh_symbol_fundamentals,
    cleanup_expired_cache,
    load_symbols_from_file
)
from src.data.data_manager import DataManager

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "data-cache"

# Setup simple logging for cron jobs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fundamentals_refresh.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function for periodic refresh."""
    parser = argparse.ArgumentParser(description='Periodic fundamentals cache refresh')

    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group(required=False)
    symbol_group.add_argument('--symbols', type=str,
                             help='Comma-separated list of symbols to refresh')
    symbol_group.add_argument('--all-symbols', action='store_true',
                             help='Refresh all cached symbols')
    symbol_group.add_argument('--cleanup-only', action='store_true',
                             help='Only cleanup expired cache data')

    # Options
    parser.add_argument('--cache-dir', type=str, default=DATA_CACHE_DIR,
                       help=f'Cache directory path (default: {DATA_CACHE_DIR})')
    parser.add_argument('--data-types', type=str, default='ratios,profile',
                       help='Data types to refresh (default: ratios,profile)')
    parser.add_argument('--max-symbols', type=int, default=50,
                       help='Maximum number of symbols to process (default: 50)')

    args = parser.parse_args()

    start_time = datetime.now()
    logger.info("Starting periodic fundamentals refresh at %s", start_time)

    try:
        # Initialize DataManager
        dm = DataManager(args.cache_dir)

        if args.cleanup_only:
            # Cleanup only
            logger.info("Running cache cleanup")
            data_types = [dt.strip() for dt in args.data_types.split(',')]
            cleanup_results = cleanup_expired_cache(args.cache_dir, data_types)

            logger.info("Cleanup completed: %d files, %d symbols removed",
                       cleanup_results['total_removed_files'],
                       cleanup_results['total_removed_symbols'])
            return

        # Determine symbols to refresh
        if args.all_symbols:
            all_symbols = get_cached_symbols(args.cache_dir)
            symbols = all_symbols[:args.max_symbols]  # Limit for safety
            logger.info("Found %d cached symbols, processing first %d",
                       len(all_symbols), len(symbols))
        elif args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            logger.info("Processing %d specified symbols", len(symbols))
        else:
            # Default: use example_tickers.txt
            default_file = os.path.join(os.path.dirname(__file__), 'example_tickers.txt')
            symbols = load_symbols_from_file(default_file)
            symbols = symbols[:args.max_symbols]  # Limit for safety
            logger.info("No symbols specified, using default file: %s", default_file)
            logger.info("Processing first %d symbols from default file", len(symbols))

        if not symbols:
            logger.warning("No symbols to process")
            return

        # Parse data types
        data_types = [dt.strip() for dt in args.data_types.split(',')]

        # Process symbols
        successful = 0
        failed = 0

        for i, symbol in enumerate(symbols):
            try:
                logger.info("Processing %d/%d: %s", i+1, len(symbols), symbol)

                result = refresh_symbol_fundamentals(dm, symbol, data_types, force_refresh=False)

                if result['success']:
                    successful += 1
                    logger.info("Successfully refreshed %s", symbol)
                else:
                    failed += 1
                    logger.warning("Failed to refresh %s: %s", symbol, ', '.join(result['errors']))

                # Small delay to be respectful to APIs
                if i < len(symbols) - 1:
                    import time
                    time.sleep(0.5)

            except Exception as e:
                failed += 1
                logger.exception("Error processing %s: %s", symbol, e)

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("Refresh completed in %s", duration)
        logger.info("Results: %d successful, %d failed", successful, failed)

        # Exit with error code if too many failures
        if failed > successful:
            logger.exception("Too many failures (%d > %d), exiting with error", failed, successful)
            sys.exit(1)

    except Exception as e:
        logger.exception("Fatal error during refresh:")
        sys.exit(1)

if __name__ == "__main__":
    main()
