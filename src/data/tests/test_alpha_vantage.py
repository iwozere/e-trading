#!/usr/bin/env python3
"""
Test script for Alpha Vantage integration.

This script demonstrates how to use Alpha Vantage for intraday data
that's not limited by the 60-day restriction of yfinance.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader.alpha_vantage_data_downloader import AlphaVantageDataDownloader

# Import API key from donotshare configuration
from config.donotshare.donotshare import ALPHA_VANTAGE_KEY


def test_alpha_vantage_intraday():
    """Test Alpha Vantage intraday data download."""
    print("🧪 Testing Alpha Vantage Intraday Data Download")
    print("=" * 60)

    # Check if API key is available
    api_key = ALPHA_VANTAGE_KEY
    if not api_key:
        print("❌ ALPHA_VANTAGE_KEY not found in donotshare.py!")
        print("Please get a free API key from: https://www.alphavantage.co/support/#api-key")
        print("Then add it to config/donotshare/donotshare.py")
        return False

    try:
        # Initialize downloader
        print("🔑 API Key: Found")
        downloader = AlphaVantageDataDownloader(api_key=api_key)
        print("✅ Alpha Vantage downloader initialized")

        # Test parameters
        symbol = "AAPL"  # Apple stock
        interval = "5m"  # 5-minute data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 90 days of data

        print(f"\n📊 Test Parameters:")
        print(f"  Symbol: {symbol}")
        print(f"  Interval: {interval}")
        print(f"  Date Range: {start_date.date()} to {end_date.date()}")
        print(f"  Expected: Full historical data (no 60-day limit like yfinance)")

        # Download data
        print(f"\n📥 Downloading data...")
        df = downloader.get_ohlcv(symbol, interval, start_date, end_date)

        if df is not None and not df.empty:
            print(f"✅ Success! Downloaded {len(df)} rows")
            print(f"📊 Data shape: {df.shape}")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"📈 Columns: {list(df.columns)}")

            # Show sample data
            print(f"\n📋 Sample data (first 5 rows):")
            print(df.head())

            return True
        else:
            print("❌ No data returned")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_vantage_crypto():
    """Test Alpha Vantage crypto data download."""
    print("\n🧪 Testing Alpha Vantage Crypto Data Download")
    print("=" * 60)

    api_key = ALPHA_VANTAGE_KEY
    if not api_key:
        print("❌ ALPHA_VANTAGE_KEY not found in donotshare.py")
        return False

    try:
        downloader = AlphaVantageDataDownloader(api_key=api_key)

        # Test crypto
        symbol = "BTCUSD"
        interval = "15m"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        print(f"📊 Test Parameters:")
        print(f"  Symbol: {symbol}")
        print(f"  Interval: {interval}")
        print(f"  Date Range: {start_date.date()} to {end_date.date()}")

        print(f"\n📥 Downloading crypto data...")
        df = downloader.get_ohlcv(symbol, interval, start_date, end_date)

        if df is not None and not df.empty:
            print(f"✅ Success! Downloaded {len(df)} rows")
            print(f"📊 Data shape: {df.shape}")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return True
        else:
            print("❌ No crypto data returned")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Main test function."""
    print("🚀 Alpha Vantage Integration Test")
    print("=" * 60)
    print()

    # Test stock intraday data
    stock_success = test_alpha_vantage_intraday()

    # Test crypto data
    crypto_success = test_alpha_vantage_crypto()

    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"📈 Stock Intraday: {'✅ PASS' if stock_success else '❌ FAIL'}")
    print(f"🪙 Crypto Data: {'✅ PASS' if crypto_success else '❌ FAIL'}")

    if stock_success and crypto_success:
        print(f"\n🎉 All tests passed! Alpha Vantage is working correctly.")
        print(f"💡 You can now use this for full historical intraday data.")
    else:
        print(f"\n⚠️  Some tests failed. Check the error messages above.")

    print(f"\n🔑 To use Alpha Vantage in populate_cache.py:")
    print(f"   1. Add your API key to config/donotshare/donotshare.py")
    print(f"   2. Run: python src/data/cache/populate_cache.py --populate --intervals 5m,15m")


if __name__ == "__main__":
    main()
