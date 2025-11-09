#!/usr/bin/env python3
"""
Demo script for FMP Data Downloader functionality.
This script demonstrates how to use the FMP API for stock screening.
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.notification.logger import setup_logger

# Import API key from donotshare configuration
from config.donotshare.donotshare import FMP_API_KEY

logger = setup_logger(__name__)


def demo_fmp_screener():
    """Demonstrate FMP stock screener functionality."""
    print("🚀 FMP Stock Screener Demo")
    print("=" * 50)

    # Check if API key is available
    api_key = FMP_API_KEY
    if not api_key:
        print("❌ FMP_API_KEY not found in donotshare.py")
        print("Please add your FMP API key to config/donotshare/donotshare.py")
        return

    try:
        # Initialize FMP downloader
        downloader = FMPDataDownloader(api_key=api_key)
        print("✅ FMP Data Downloader initialized")

        # Load predefined screener criteria
        criteria_file = PROJECT_ROOT / "config" / "screener" / "fmp_screener_criteria.json"
        if criteria_file.exists():
            with open(criteria_file, 'r') as f:
                screener_configs = json.load(f)
            print(f"✅ Loaded {len(screener_configs)} predefined screener configurations")
        else:
            print("⚠️  Screener criteria file not found, using default criteria")
            screener_configs = {}

        # Demo different screener strategies
        demo_strategies = [
            "conservative_value",
            "growth_at_reasonable_price",
            "dividend_aristocrats"
        ]

        for strategy in demo_strategies:
            if strategy in screener_configs:
                print(f"\n📊 Testing {strategy} screener...")
                print(f"Description: {screener_configs[strategy]['description']}")

                criteria = screener_configs[strategy]['criteria']
                print(f"Criteria: {json.dumps(criteria, indent=2)}")

                try:
                    # Run screener
                    results = downloader.get_stock_screener(criteria)
                    print(f"✅ Found {len(results)} stocks matching criteria")

                    # Display top results
                    if results:
                        print("\n🏆 Top Results:")
                        for i, stock in enumerate(results[:5], 1):
                            print(f"  {i}. {stock['symbol']}: {stock['companyName']}")
                            print(f"     Price: ${stock.get('price', 0):.2f}")
                            print(f"     Market Cap: ${stock.get('marketCap', 0):,.0f}")
                            print(f"     PE Ratio: {stock.get('peRatio', 0):.1f}")
                            print(f"     ROE: {stock.get('returnOnEquity', 0):.1%}")
                            print(f"     Sector: {stock.get('sector', 'Unknown')}")
                            print()

                except Exception as e:
                    print(f"❌ Error running {strategy} screener: {e}")

        # Demo custom criteria
        print("\n🔧 Testing Custom Criteria...")
        custom_criteria = {
            "marketCapMoreThan": 1000000000,  # $1B+ market cap
            "peRatioLessThan": 15,
            "returnOnEquityMoreThan": 0.12,
            "limit": 10
        }

        try:
            results = downloader.get_stock_screener(custom_criteria)
            print(f"✅ Custom screener found {len(results)} stocks")

            if results:
                print("\n📈 Custom Screener Results:")
                for i, stock in enumerate(results[:3], 1):
                    print(f"  {i}. {stock['symbol']}: {stock['companyName']}")
                    print(f"     Price: ${stock.get('price', 0):.2f}")
                    print(f"     PE: {stock.get('peRatio', 0):.1f}")
                    print(f"     ROE: {stock.get('returnOnEquity', 0):.1%}")
                    print()

        except Exception as e:
            print(f"❌ Error running custom screener: {e}")

    except Exception as e:
        print(f"❌ Error initializing FMP downloader: {e}")


def demo_fundamentals_retrieval():
    """Demonstrate FMP fundamentals retrieval."""
    print("\n💰 FMP Fundamentals Retrieval Demo")
    print("=" * 50)

    api_key = FMP_API_KEY
    if not api_key:
        print("❌ FMP_API_KEY not found in donotshare.py")
        return

    try:
        downloader = FMPDataDownloader(api_key=api_key)

        # Test symbols
        test_symbols = ["AAPL", "MSFT", "GOOGL"]

        for symbol in test_symbols:
            print(f"\n📊 Getting fundamentals for {symbol}...")

            try:
                fundamentals = downloader.get_fundamentals(symbol)
                print(f"✅ {fundamentals.company_name}")
                print(f"   Price: ${fundamentals.current_price:.2f}")
                print(f"   Market Cap: ${fundamentals.market_cap:,.0f}")
                print(f"   PE Ratio: {fundamentals.pe_ratio:.1f}")
                print(f"   ROE: {fundamentals.return_on_equity:.1%}")
                print(f"   Sector: {fundamentals.sector}")
                print(f"   Data Source: {fundamentals.data_source}")

            except Exception as e:
                print(f"❌ Error getting fundamentals for {symbol}: {e}")

    except Exception as e:
        print(f"❌ Error in fundamentals demo: {e}")


def demo_company_profile():
    """Demonstrate FMP company profile retrieval."""
    print("\n🏢 FMP Company Profile Demo")
    print("=" * 50)

    api_key = FMP_API_KEY
    if not api_key:
        print("❌ FMP_API_KEY not found in donotshare.py")
        return

    try:
        downloader = FMPDataDownloader(api_key=api_key)

        symbol = "AAPL"
        print(f"📊 Getting company profile for {symbol}...")

        try:
            profile = downloader.get_company_profile(symbol)
            if profile:
                print(f"✅ Company: {profile.get('companyName', 'Unknown')}")
                print(f"   Exchange: {profile.get('exchange', 'Unknown')}")
                print(f"   Sector: {profile.get('sector', 'Unknown')}")
                print(f"   Industry: {profile.get('industry', 'Unknown')}")
                print(f"   Country: {profile.get('country', 'Unknown')}")
                print(f"   Market Cap: ${profile.get('mktCap', 0):,.0f}")
                print(f"   Revenue: ${profile.get('revenue', 0):,.0f}")
                print(f"   Employees: {profile.get('fullTimeEmployees', 'Unknown')}")
            else:
                print("❌ No profile data found")

        except Exception as e:
            print(f"❌ Error getting company profile: {e}")

    except Exception as e:
        print(f"❌ Error in company profile demo: {e}")


if __name__ == "__main__":
    print("🚀 Starting FMP Data Downloader Demo")
    print("=" * 70)

    # Demo stock screener
    demo_fmp_screener()

    # Demo fundamentals retrieval
    demo_fundamentals_retrieval()

    # Demo company profile
    demo_company_profile()

    print("\n🎉 FMP Data Downloader demo completed!")
    print("\n💡 To use FMP in your screener:")
    print("1. Get an API key from https://financialmodelingprep.com/")
    print("2. Add your API key to config/donotshare/donotshare.py")
    print("3. Use FMPDataDownloader in your screener logic")
