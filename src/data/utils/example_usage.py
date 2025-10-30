#!/usr/bin/env python3
"""
Example Usage of Fundamentals Downloader
========================================

This script demonstrates how to use the fundamentals downloader programmatically.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.utils.download_fundamentals import FundamentalsDownloader, load_tickers_from_file


def example_basic_usage():
    """Example of basic usage with a few tickers."""
    print("🔍 Example 1: Basic Usage")
    print("=" * 40)

    # Initialize downloader
    downloader = FundamentalsDownloader()

    # Download fundamentals for a few tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    stats = downloader.download_fundamentals(
        tickers=tickers,
        provider="yf",  # Yahoo Finance (no API key needed)
        show_progress=True
    )

    print(f"Download completed: {stats['successful_downloads']} successful, {stats['failed_downloads']} failed")


def example_batch_processing():
    """Example of batch processing from file."""
    print("\n📁 Example 2: Batch Processing from File")
    print("=" * 40)

    # Load tickers from file
    ticker_file = Path(__file__).parent / "example_tickers.txt"
    tickers = load_tickers_from_file(str(ticker_file))

    if not tickers:
        print("No tickers loaded from file")
        return

    # Initialize downloader with more workers for batch processing
    downloader = FundamentalsDownloader(max_workers=3)

    # Download fundamentals
    stats = downloader.download_fundamentals(
        tickers=tickers,
        provider="yf",
        show_progress=True
    )

    print(f"Batch download completed: {stats['successful_downloads']} successful, {stats['failed_downloads']} failed")


def example_cache_management():
    """Example of cache management operations."""
    print("\n💾 Example 3: Cache Management")
    print("=" * 40)

    downloader = FundamentalsDownloader()

    # Show cache information
    cache_info = downloader.get_cache_info()
    print("Cache Statistics:")
    print(f"  Total symbols: {cache_info.get('total_symbols', 0)}")
    print(f"  Total files: {cache_info.get('total_files', 0)}")
    print(f"  Total size: {cache_info.get('total_size', 0) / 1024:.1f} KB")

    # Show info for specific ticker
    aapl_info = downloader.get_cache_info("AAPL")
    print(f"\nAAPL Cache Info:")
    print(f"  Files: {aapl_info.get('files', 0)}")
    print(f"  Size: {aapl_info.get('total_size', 0)} bytes")
    print(f"  Providers: {aapl_info.get('providers', [])}")


def example_force_refresh():
    """Example of force refresh functionality."""
    print("\n🔄 Example 4: Force Refresh")
    print("=" * 40)

    downloader = FundamentalsDownloader()

    # Force refresh even if cache is valid
    stats = downloader.download_fundamentals(
        tickers=["AAPL"],
        provider="yf",
        force_refresh=True,
        show_progress=True
    )

    print(f"Force refresh completed: {stats['successful_downloads']} successful")


def example_multiple_providers():
    """Example of using multiple providers."""
    print("\n🌐 Example 5: Multiple Providers")
    print("=" * 40)

    downloader = FundamentalsDownloader()

    # Try different providers for the same ticker
    ticker = "AAPL"
    providers = ["yf", "fmp"]  # Yahoo Finance and Financial Modeling Prep

    for provider in providers:
        print(f"\nDownloading {ticker} from {provider}...")
        try:
            stats = downloader.download_fundamentals(
                tickers=[ticker],
                provider=provider,
                show_progress=True
            )
            print(f"Provider {provider}: {stats['successful_downloads']} successful")
        except Exception as e:
            print(f"Provider {provider} failed: {e}")


def main():
    """Run all examples."""
    print("📊 Fundamentals Downloader Examples")
    print("=" * 50)

    try:
        example_basic_usage()
        example_batch_processing()
        example_cache_management()
        example_force_refresh()
        example_multiple_providers()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
