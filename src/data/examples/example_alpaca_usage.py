#!/usr/bin/env python3
"""
Example usage of Alpaca Data Downloader

This script demonstrates how to use the AlpacaDataDownloader to fetch
historical market data and fundamental information.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from datetime import datetime, timedelta
from src.data.downloader.alpaca_data_downloader import AlpacaDataDownloader
from src.data.downloader.data_downloader_factory import DataDownloaderFactory
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def example_direct_usage():
    """Example of using AlpacaDataDownloader directly."""
    print("Example 1: Direct Usage")
    print("-" * 40)

    try:
        # Initialize with credentials from environment or config
        downloader = AlpacaDataDownloader()

        # Define date range (last 30 days)
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)

        # Download daily data for Apple
        print(f"Downloading AAPL daily data from {start_date.date()} to {end_date.date()}")
        df = downloader.get_ohlcv("AAPL", "1d", start_date, end_date)

        if not df.empty:
            print(f"Downloaded {len(df)} rows")
            print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            print(f"Average volume: {df['volume'].mean():,.0f}")

            # Show recent data
            print("\nLast 5 days:")
            print(df.tail()[['open', 'high', 'low', 'close', 'volume']])
        else:
            print("No data received")

    except Exception as e:
        print(f"Error: {e}")


def example_factory_usage():
    """Example of using AlpacaDataDownloader via factory."""
    print("\nExample 2: Factory Usage")
    print("-" * 40)

    try:
        # Create downloader via factory
        downloader = DataDownloaderFactory.create_downloader("alpaca")

        if downloader:
            # Download 1-hour data for Microsoft
            end_date = datetime.now() - timedelta(hours=1)
            start_date = end_date - timedelta(days=3)

            print(f"Downloading MSFT hourly data from {start_date.date()} to {end_date.date()}")
            df = downloader.get_ohlcv("MSFT", "1h", start_date, end_date)

            if not df.empty:
                print(f"Downloaded {len(df)} hourly bars")
                print(f"Latest close: ${df['close'].iloc[-1]:.2f}")

                # Calculate simple statistics
                daily_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                print(f"3-day return: {daily_return:.2f}%")
            else:
                print("No data received")
        else:
            print("Failed to create downloader")

    except Exception as e:
        print(f"Error: {e}")


def example_multiple_symbols():
    """Example of downloading data for multiple symbols."""
    print("\nExample 3: Multiple Symbols")
    print("-" * 40)

    try:
        downloader = AlpacaDataDownloader()

        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)

        results = {}

        for symbol in symbols:
            print(f"Downloading {symbol}...")
            try:
                df = downloader.get_ohlcv(symbol, "1d", start_date, end_date)
                if not df.empty:
                    # Calculate weekly return
                    weekly_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                    results[symbol] = {
                        'rows': len(df),
                        'latest_price': df['close'].iloc[-1],
                        'weekly_return': weekly_return
                    }
                else:
                    results[symbol] = {'error': 'No data'}
            except Exception as e:
                results[symbol] = {'error': str(e)}

        # Display results
        print("\nResults Summary:")
        print(f"{'Symbol':<8} {'Price':<10} {'Weekly %':<10} {'Rows':<6}")
        print("-" * 40)

        for symbol, data in results.items():
            if 'error' in data:
                print(f"{symbol:<8} {'ERROR':<10} {data['error']:<10}")
            else:
                price = f"${data['latest_price']:.2f}"
                return_pct = f"{data['weekly_return']:+.2f}%"
                rows = str(data['rows'])
                print(f"{symbol:<8} {price:<10} {return_pct:<10} {rows:<6}")

    except Exception as e:
        print(f"Error: {e}")


def example_fundamentals():
    """Example of getting fundamental data."""
    print("\nExample 4: Fundamental Data")
    print("-" * 40)

    try:
        downloader = AlpacaDataDownloader()

        symbols = ["AAPL", "MSFT"]

        for symbol in symbols:
            print(f"\nFundamentals for {symbol}:")
            fundamentals = downloader.get_fundamentals(symbol)

            if fundamentals:
                print(f"  Company: {fundamentals.company_name}")
                print(f"  Sector: {fundamentals.sector}")
                print(f"  Industry: {fundamentals.industry}")
                if fundamentals.pe_ratio:
                    print(f"  P/E Ratio: {fundamentals.pe_ratio}")
                if fundamentals.market_cap:
                    print(f"  Market Cap: ${fundamentals.market_cap:,.0f}")
            else:
                print(f"  No fundamental data available")

    except Exception as e:
        print(f"Error: {e}")


def example_different_intervals():
    """Example of using different time intervals."""
    print("\nExample 5: Different Intervals")
    print("-" * 40)

    try:
        downloader = AlpacaDataDownloader()

        # Show supported intervals
        intervals = downloader.get_supported_intervals()
        print(f"Supported intervals: {intervals}")

        symbol = "SPY"  # S&P 500 ETF
        end_date = datetime.now() - timedelta(hours=1)

        for interval in ["1m", "5m", "1h", "1d"]:
            if interval in intervals:
                # Adjust start date based on interval
                if interval == "1m":
                    start_date = end_date - timedelta(hours=2)
                elif interval == "5m":
                    start_date = end_date - timedelta(hours=6)
                elif interval == "1h":
                    start_date = end_date - timedelta(days=2)
                else:  # 1d
                    start_date = end_date - timedelta(days=10)

                try:
                    df = downloader.get_ohlcv(symbol, interval, start_date, end_date)
                    if not df.empty:
                        print(f"  {interval:<4}: {len(df):>3} bars, latest: ${df['close'].iloc[-1]:.2f}")
                    else:
                        print(f"  {interval:<4}: No data")
                except Exception as e:
                    print(f"  {interval:<4}: Error - {e}")

    except Exception as e:
        print(f"Error: {e}")


def example_bar_limits():
    """Example demonstrating the 10,000 bar limit."""
    print("\nExample 6: Bar Limits (10,000 max)")
    print("-" * 40)

    try:
        downloader = AlpacaDataDownloader()

        symbol = "AAPL"
        end_date = datetime.now() - timedelta(days=1)

        # Test with a very long date range (should be limited to 10,000 bars)
        start_date = end_date - timedelta(days=365*5)  # 5 years ago

        print(f"Requesting 5 years of daily data for {symbol}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")

        df = downloader.get_ohlcv(symbol, "1d", start_date, end_date)

        if not df.empty:
            print(f"✅ Received {len(df)} bars (limited to 10,000 max)")
            print(f"Actual date range: {df.index.min().date()} to {df.index.max().date()}")

            # Test with custom limit
            print(f"\nTesting with custom limit of 100 bars...")
            df_limited = downloader.get_ohlcv(symbol, "1d", start_date, end_date, limit=100)
            print(f"✅ Received {len(df_limited)} bars (custom limit: 100)")
        else:
            print("⚠️ No data received")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples."""
    print("Alpaca Data Downloader Examples")
    print("=" * 50)

    # Run examples
    example_direct_usage()
    example_factory_usage()
    example_multiple_symbols()
    example_fundamentals()
    example_different_intervals()
    example_bar_limits()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Make sure you have ALPACA_API_KEY and ALPACA_SECRET_KEY")
    print("set in your environment variables or config/donotshare/donotshare.py")


if __name__ == "__main__":
    main()