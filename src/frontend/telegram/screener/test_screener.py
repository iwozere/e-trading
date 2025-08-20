"""
Test script for the fundamental screener functionality.
This script can be run independently to test the screener without the full bot.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.frontend.telegram.screener.fundamental_screener import screener
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def test_screener():
    """Test the fundamental screener with a small list."""
    print("🧪 Testing Fundamental Screener")
    print("=" * 50)

    # Test with a small list first
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")

    try:
        # Test fundamental data collection
        print("\n📊 Collecting fundamental data...")
        fundamentals_data = screener.collect_fundamentals(test_tickers)

        print(f"✅ Collected data for {len(fundamentals_data)} tickers")

        if fundamentals_data:
            # Test screening
            print("\n🔍 Applying screening criteria...")
            results = screener.apply_screening_criteria(fundamentals_data)

            print(f"✅ Found {len(results)} undervalued stocks")

            # Test report generation
            print("\n📋 Generating report...")
            report = screener.generate_report("test_list", results, len(test_tickers))

            # Format and display report
            print("\n" + "=" * 50)
            print("📊 SCREENER REPORT")
            print("=" * 50)

            telegram_message = screener.format_telegram_message(report)
            print(telegram_message)

            print("\n✅ Screener test completed successfully!")

        else:
            print("❌ No fundamental data collected")

    except Exception as e:
        print(f"❌ Error during screener test: {e}")
        logger.exception("Screener test failed")


def test_ticker_lists():
    """Test ticker list loading."""
    print("\n📋 Testing Ticker Lists")
    print("=" * 30)

    list_types = ['us_small_cap', 'us_medium_cap', 'us_large_cap', 'swiss_shares']

    for list_type in list_types:
        try:
            tickers = screener.load_ticker_list(list_type)
            print(f"✅ {list_type}: {len(tickers)} tickers loaded")
            if tickers:
                print(f"   Sample: {', '.join(tickers[:5])}")
        except Exception as e:
            print(f"❌ {list_type}: Error - {e}")


def test_dcf_calculation():
    """Test DCF calculation with sample data."""
    print("\n💰 Testing DCF Calculation")
    print("=" * 30)

    from src.model.telegram_bot import Fundamentals

    # Create sample fundamentals data
    sample_fundamentals = Fundamentals(
        ticker="TEST",
        company_name="Test Company",
        current_price=100.0,
        free_cash_flow=1000000.0,
        revenue_growth=10.0,
        net_income_growth=8.0,
        beta=1.2,
        shares_outstanding=10000000.0
    )

    try:
        dcf_result = screener._calculate_dcf_valuation(sample_fundamentals)

        if dcf_result.error:
            print(f"❌ DCF Error: {dcf_result.error}")
        else:
            print(f"✅ DCF Calculation successful:")
            print(f"   Fair Value: ${dcf_result.fair_value:.2f}")
            print(f"   Growth Rate: {dcf_result.growth_rate:.2%}")
            print(f"   Discount Rate: {dcf_result.discount_rate:.2%}")
            print(f"   Confidence: {dcf_result.confidence_level}")

    except Exception as e:
        print(f"❌ DCF calculation error: {e}")


if __name__ == "__main__":
    print("🚀 Starting Fundamental Screener Tests")
    print("=" * 60)

    # Test ticker lists
    test_ticker_lists()

    # Test DCF calculation
    test_dcf_calculation()

    # Test full screener (comment out if you want to skip this)
    # test_screener()

    print("\n🎉 All tests completed!")
