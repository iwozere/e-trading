#!/usr/bin/env python3
"""
Test script for FMP Data Downloader functionality.
This script tests the Financial Modeling Prep API integration.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def test_fmp_initialization():
    """Test FMP data downloader initialization."""
    print("🧪 Testing FMP Data Downloader Initialization")
    print("=" * 50)

    # Test with API key parameter
    try:
        api_key = "test_api_key"
        downloader = FMPDataDownloader(api_key=api_key)
        print(f"✅ FMP downloader initialized with API key: {api_key[:10]}...")
        assert downloader.api_key == api_key
        assert downloader.base_url == "https://financialmodelingprep.com/api/v3"
        print("✅ Base URL and API key correctly set")
    except Exception as e:
        print(f"❌ Error initializing with API key: {e}")

    # Test with environment variable
    try:
        os.environ['FMP_API_KEY'] = "env_test_key"
        downloader = FMPDataDownloader()
        print(f"✅ FMP downloader initialized with env API key: {downloader.api_key[:10]}...")
        assert downloader.api_key == "env_test_key"
        print("✅ Environment variable API key correctly loaded")
    except Exception as e:
        print(f"❌ Error initializing with environment variable: {e}")

    # Test without API key
    try:
        if 'FMP_API_KEY' in os.environ:
            del os.environ['FMP_API_KEY']
        downloader = FMPDataDownloader()
        print("❌ Should have raised error for missing API key")
    except ValueError as e:
        print(f"✅ Correctly raised error for missing API key: {e}")


def test_screener_criteria_validation():
    """Test screener criteria validation."""
    print("\n🔍 Testing Screener Criteria Validation")
    print("=" * 50)

    api_key = "test_api_key"
    downloader = FMPDataDownloader(api_key=api_key)

    # Test valid criteria
    valid_criteria = {
        "marketCapMoreThan": 1000000000,
        "peRatioLessThan": 15,
        "priceToBookRatioLessThan": 1.5,
        "debtToEquityLessThan": 0.5,
        "returnOnEquityMoreThan": 0.12,
        "limit": 50
    }

    try:
        downloader._validate_screener_criteria(valid_criteria)
        print("✅ Valid criteria validation passed")
    except Exception as e:
        print(f"❌ Valid criteria validation failed: {e}")

    # Test invalid criteria
    invalid_criteria = {
        "marketCapMoreThan": 1000000000,
        "invalidCriterion": "value",
        "anotherInvalid": 123
    }

    try:
        downloader._validate_screener_criteria(invalid_criteria)
        print("✅ Invalid criteria validation handled gracefully")
    except Exception as e:
        print(f"❌ Invalid criteria validation failed: {e}")


def test_stock_screener_mock():
    """Test stock screener with mocked API response."""
    print("\n📊 Testing Stock Screener (Mocked)")
    print("=" * 50)

    api_key = "test_api_key"
    downloader = FMPDataDownloader(api_key=api_key)

    # Mock API response
    mock_response = [
        {
            "symbol": "AAPL",
            "companyName": "Apple Inc.",
            "price": 150.0,
            "marketCap": 2500000000000,
            "peRatio": 25.0,
            "priceToBookRatio": 15.0,
            "debtToEquity": 0.3,
            "returnOnEquity": 0.15,
            "returnOnAssets": 0.10,
            "currentRatio": 1.5,
            "quickRatio": 1.2,
            "beta": 1.2,
            "dividendYield": 0.5,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchange": "NASDAQ"
        },
        {
            "symbol": "MSFT",
            "companyName": "Microsoft Corporation",
            "price": 300.0,
            "marketCap": 2200000000000,
            "peRatio": 30.0,
            "priceToBookRatio": 12.0,
            "debtToEquity": 0.4,
            "returnOnEquity": 0.18,
            "returnOnAssets": 0.12,
            "currentRatio": 1.8,
            "quickRatio": 1.5,
            "beta": 1.1,
            "dividendYield": 0.8,
            "sector": "Technology",
            "industry": "Software",
            "exchange": "NASDAQ"
        }
    ]

    criteria = {
        "marketCapMoreThan": 1000000000,
        "peRatioLessThan": 35,
        "returnOnEquityMoreThan": 0.10,
        "limit": 10
    }

    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status.return_value = None
        try:
            results = downloader.get_stock_screener(criteria)
            print(f"✅ Stock screener returned {len(results)} results")

            for result in results:
                print(f"  📈 {result['symbol']}: {result['companyName']}")
                print(f"     Price: ${result['price']:.2f}")
                print(f"     Market Cap: ${result['marketCap']:,.0f}")
                print(f"     PE Ratio: {result['peRatio']:.1f}")
                print(f"     ROE: {result['returnOnEquity']:.1%}")
                print()

        except Exception as e:
            print(f"❌ Stock screener test failed: {e}")


def test_fundamentals_retrieval_mock():
    """Test fundamentals retrieval with mocked API response."""
    print("\n💰 Testing Fundamentals Retrieval (Mocked)")
    print("=" * 50)

    api_key = "test_api_key"
    downloader = FMPDataDownloader(api_key=api_key)

    # Mock profile response
    mock_profile = [{
        "companyName": "Apple Inc.",
        "price": 150.0,
        "mktCap": 2500000000000,
        "revenue": 400000000000,
        "netIncome": 100000000000,
        "freeCashFlow": 80000000000,
        "sharesOutstanding": 16000000000,
        "sharesFloat": 15000000000,
        "sharesShort": 100000000,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "US",
        "exchange": "NASDAQ",
        "currency": "USD",
        "enterpriseValue": 2600000000000,
        "lastDiv": 0.75,
        "beta": 1.2
    }]

    # Mock ratios response
    mock_ratios = [{
        "peRatio": 25.0,
        "forwardPeRatio": 24.0,
        "eps": 6.0,
        "priceToBookRatio": 15.0,
        "returnOnEquity": 0.15,
        "returnOnAssets": 0.10,
        "debtEquityRatio": 0.3,
        "currentRatio": 1.5,
        "quickRatio": 1.2,
        "operatingMargin": 0.25,
        "netProfitMargin": 0.20,
        "dividendPayoutRatio": 0.25,
        "pegRatio": 1.5,
        "priceToSalesRatio": 5.0,
        "enterpriseValueMultiple": 20.0
    }]

    # Mock metrics response
    mock_metrics = [{
        "revenueGrowth": 0.08,
        "netIncomeGrowth": 0.12
    }]

    with patch('requests.get') as mock_get:
        # Set up the mock to return different responses for different calls
        mock_responses = [mock_profile, mock_metrics, mock_ratios]
        mock_get.return_value.raise_for_status.return_value = None

        def mock_json():
            return mock_responses.pop(0) if mock_responses else []

        mock_get.return_value.json = mock_json

        try:
            fundamentals = downloader.get_fundamentals("AAPL")
            print(f"✅ Fundamentals retrieved for {fundamentals.ticker}")
            print(f"   Company: {fundamentals.company_name}")
            print(f"   Price: ${fundamentals.current_price:.2f}")
            print(f"   Market Cap: ${fundamentals.market_cap:,.0f}")
            print(f"   PE Ratio: {fundamentals.pe_ratio:.1f}")
            print(f"   ROE: {fundamentals.return_on_equity:.1%}")
            print(f"   Sector: {fundamentals.sector}")
            print(f"   Data Source: {fundamentals.data_source}")

        except Exception as e:
            print(f"❌ Fundamentals retrieval test failed: {e}")


def test_ohlcv_retrieval_mock():
    """Test OHLCV data retrieval with mocked API response."""
    print("\n📈 Testing OHLCV Data Retrieval (Mocked)")
    print("=" * 50)

    api_key = "test_api_key"
    downloader = FMPDataDownloader(api_key=api_key)

    # Mock OHLCV response
    mock_ohlcv = {
        "symbol": "AAPL",
        "historical": [
            {
                "date": "2024-01-15",
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.0,
                "adjClose": 151.0,
                "volume": 50000000
            },
            {
                "date": "2024-01-16",
                "open": 151.0,
                "high": 153.0,
                "low": 150.0,
                "close": 152.0,
                "adjClose": 152.0,
                "volume": 48000000
            }
        ]
    }

    start_date = datetime(2024, 1, 15)
    end_date = datetime(2024, 1, 16)

    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_ohlcv
        mock_get.return_value.raise_for_status.return_value = None
        try:
            df = downloader.get_ohlcv("AAPL", "1d", start_date, end_date)
            print(f"✅ OHLCV data retrieved for AAPL")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Sample data:")
            print(df.head())

        except Exception as e:
            print(f"❌ OHLCV retrieval test failed: {e}")


def test_interval_conversion():
    """Test interval format conversion."""
    print("\n⏰ Testing Interval Format Conversion")
    print("=" * 50)

    api_key = "test_api_key"
    downloader = FMPDataDownloader(api_key=api_key)

    test_cases = [
        ("1m", "1min"),
        ("5m", "5min"),
        ("15m", "15min"),
        ("30m", "30min"),
        ("1h", "1hour"),
        ("4h", "4hour"),
        ("1d", "1day"),
        ("invalid", "1day")  # Default fallback
    ]

    for input_interval, expected_output in test_cases:
        result = downloader._convert_interval(input_interval)
        status = "✅" if result == expected_output else "❌"
        print(f"{status} {input_interval} -> {result} (expected: {expected_output})")


def test_period_interval_validation():
    """Test period and interval validation."""
    print("\n✅ Testing Period and Interval Validation")
    print("=" * 50)

    api_key = "test_api_key"
    downloader = FMPDataDownloader(api_key=api_key)

    # Test valid combinations
    valid_combinations = [
        ("1d", "1d"),
        ("1mo", "1d"),
        ("6mo", "1d"),
        ("1y", "1d")
    ]

    for period, interval in valid_combinations:
        is_valid = downloader.is_valid_period_interval(period, interval)
        status = "✅" if is_valid else "❌"
        print(f"{status} {period} + {interval} = {is_valid}")

    # Test invalid combinations
    invalid_combinations = [
        ("invalid", "1d"),
        ("1d", "invalid"),
        ("invalid", "invalid")
    ]

    for period, interval in invalid_combinations:
        is_valid = downloader.is_valid_period_interval(period, interval)
        status = "✅" if not is_valid else "❌"
        print(f"{status} {period} + {interval} = {is_valid} (should be False)")


def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\n⏱️ Testing Rate Limiting")
    print("=" * 50)

    api_key = "test_api_key"
    downloader = FMPDataDownloader(api_key=api_key)

    # Test that rate_limit_delay is set correctly
    assert downloader.rate_limit_delay == 0.1, f"Rate limit delay should be 0.1s, got {downloader.rate_limit_delay}s"
    print("✅ Rate limiting delay configured correctly")


if __name__ == "__main__":
    print("🚀 Starting FMP Data Downloader Tests")
    print("=" * 70)

    # Test initialization
    test_fmp_initialization()

    # Test screener criteria validation
    test_screener_criteria_validation()

    # Test stock screener (mocked)
    test_stock_screener_mock()

    # Test fundamentals retrieval (mocked)
    test_fundamentals_retrieval_mock()

    # Test OHLCV retrieval (mocked)
    test_ohlcv_retrieval_mock()

    # Test interval conversion
    test_interval_conversion()

    # Test period/interval validation
    test_period_interval_validation()

    # Test rate limiting
    test_rate_limiting()

    print("\n🎉 All FMP Data Downloader tests completed!")
