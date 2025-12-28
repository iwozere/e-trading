#!/usr/bin/env python3
"""
Data Downloader Factory Example
-------------------------------

This example demonstrates how to use the DataDownloaderFactory to create
data downloaders using short provider codes like "yf", "av", "bnc", etc.

Features demonstrated:
- Creating downloaders with short provider codes
- Environment variable support for API keys
- Error handling for missing API keys
- Provider information and listing
- Fundamental data retrieval
"""

import os
from datetime import datetime, timedelta
from src.data.data_downloader_factory import DataDownloaderFactory
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def demonstrate_provider_listing():
    """Demonstrate listing all available providers."""
    print("Available Data Providers:")
    print("=" * 80)
    DataDownloaderFactory.list_providers()
    print()


def demonstrate_yahoo_finance():
    """Demonstrate Yahoo Finance downloader (no API key required)."""
    print("Yahoo Finance Example (no API key required):")
    print("-" * 50)

    try:
        # Create Yahoo Finance downloader using short code
        downloader = DataDownloaderFactory.create_downloader("yf")

        if downloader:
            print("✓ Successfully created Yahoo Finance downloader")

            # Get fundamental data
            fundamentals = downloader.get_fundamentals("AAPL")
            print("✓ Retrieved fundamentals for AAPL:")
            print(f"  Company: {fundamentals.company_name}")
            print(f"  Current Price: ${fundamentals.current_price:.2f}")
            print(f"  PE Ratio: {fundamentals.pe_ratio:.2f}")
            print(f"  Market Cap: ${fundamentals.market_cap:,.0f}")
            print(f"  Data Source: {fundamentals.data_source}")
        else:
            print("✗ Failed to create Yahoo Finance downloader")

    except Exception as e:
        print(f"✗ Error with Yahoo Finance: {e}")

    print()


def demonstrate_alpha_vantage():
    """Demonstrate Alpha Vantage downloader (API key required)."""
    print("Alpha Vantage Example (API key required):")
    print("-" * 50)

    # Check if API key is available
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("⚠️  ALPHA_VANTAGE_API_KEY environment variable not set")
        print("   Skipping Alpha Vantage example")
        print()
        return

    try:
        # Create Alpha Vantage downloader using short code
        downloader = DataDownloaderFactory.create_downloader("av")

        if downloader:
            print("✓ Successfully created Alpha Vantage downloader")

            # Get fundamental data
            fundamentals = downloader.get_fundamentals("AAPL")
            print("✓ Retrieved fundamentals for AAPL:")
            print(f"  Company: {fundamentals.company_name}")
            print(f"  Current Price: ${fundamentals.current_price:.2f}")
            print(f"  PE Ratio: {fundamentals.pe_ratio:.2f}")
            print(f"  Market Cap: ${fundamentals.market_cap:,.0f}")
            print(f"  Data Source: {fundamentals.data_source}")
        else:
            print("✗ Failed to create Alpha Vantage downloader")

    except Exception as e:
        print(f"✗ Error with Alpha Vantage: {e}")

    print()


def demonstrate_binance():
    """Demonstrate Binance downloader (API key and secret required)."""
    print("Binance Example (API key and secret required):")
    print("-" * 50)

    # Check if API credentials are available
    api_key = os.getenv("BINANCE_KEY")
    secret_key = os.getenv("BINANCE_SECRET")

    if not api_key or not secret_key:
        print("⚠️  BINANCE_KEY or BINANCE_SECRET environment variables not set")
        print("   Skipping Binance example")
        print()
        return

    try:
        # Create Binance downloader using short code
        downloader = DataDownloaderFactory.create_downloader("bnc")

        if downloader:
            print("✓ Successfully created Binance downloader")

            # Get historical data (fundamentals not available for crypto)
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            df = downloader.get_ohlcv("BTCUSDT", "1d", start_date, end_date)
            print("✓ Retrieved historical data for BTCUSDT:")
            print(f"  Data points: {len(df)}")
            print(f"  Date range: {start_date} to {end_date}")
            print(f"  Latest close: ${df.iloc[-1]['close']:.2f}")

            # Try to get fundamentals (should raise NotImplementedError)
            try:
                fundamentals = downloader.get_fundamentals("BTCUSDT")
            except NotImplementedError:
                print("✓ Correctly raised NotImplementedError for crypto fundamentals")

        else:
            print("✗ Failed to create Binance downloader")

    except Exception as e:
        print(f"✗ Error with Binance: {e}")

    print()


def demonstrate_provider_codes():
    """Demonstrate different provider codes for the same provider."""
    print("Provider Code Examples:")
    print("-" * 50)

    # Test different codes for Yahoo Finance
    yahoo_codes = ["yf", "yahoo", "yf_finance"]

    for code in yahoo_codes:
        normalized = DataDownloaderFactory.get_provider_by_code(code)
        print(f"Code '{code}' -> Provider: {normalized}")

    # Test different codes for Alpha Vantage
    av_codes = ["av", "alphavantage", "alpha_vantage"]

    for code in av_codes:
        normalized = DataDownloaderFactory.get_provider_by_code(code)
        print(f"Code '{code}' -> Provider: {normalized}")

    # Test different codes for Binance
    binance_codes = ["bnc", "binance"]

    for code in binance_codes:
        normalized = DataDownloaderFactory.get_provider_by_code(code)
        print(f"Code '{code}' -> Provider: {normalized}")

    print()


def demonstrate_error_handling():
    """Demonstrate error handling for invalid providers and missing API keys."""
    print("Error Handling Examples:")
    print("-" * 50)

    # Test invalid provider code
    print("Testing invalid provider code 'invalid':")
    downloader = DataDownloaderFactory.create_downloader("invalid")
    if downloader is None:
        print("✓ Correctly returned None for invalid provider")
    else:
        print("✗ Should have returned None for invalid provider")

    # Test Alpha Vantage without API key
    print("\nTesting Alpha Vantage without API key:")
    # Temporarily remove API key
    original_key = os.environ.pop("ALPHA_VANTAGE_API_KEY", None)

    try:
        downloader = DataDownloaderFactory.create_downloader("av")
        print("✗ Should have raised ValueError for missing API key")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    finally:
        # Restore API key if it existed
        if original_key:
            os.environ["ALPHA_VANTAGE_API_KEY"] = original_key

    print()


def main():
    """Main function to demonstrate the DataDownloaderFactory."""
    print("Data Downloader Factory Example")
    print("=" * 80)
    print("This example demonstrates how to use the DataDownloaderFactory")
    print("to create data downloaders using short provider codes.")
    print()

    # List all available providers
    demonstrate_provider_listing()

    # Demonstrate provider codes
    demonstrate_provider_codes()

    # Demonstrate Yahoo Finance (no API key required)
    demonstrate_yahoo_finance()

    # Demonstrate Alpha Vantage (API key required)
    demonstrate_alpha_vantage()

    # Demonstrate Binance (API key and secret required)
    demonstrate_binance()

    # Demonstrate error handling
    demonstrate_error_handling()

    print("Example completed!")
    print("\nTo use this factory in your code:")
    print("1. Set environment variables for API keys (if required)")
    print("2. Use short provider codes like 'yf', 'av', 'bnc'")
    print("3. Handle errors appropriately")
    print("\nExample:")
    print("  downloader = DataDownloaderFactory.create_downloader('yf')")
    print("  fundamentals = downloader.get_fundamentals('AAPL')")


if __name__ == "__main__":
    main()