#!/usr/bin/env python3
"""
Test script for DataDownloaderFactory
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.downloader.data_downloader_factory import DataDownloaderFactory
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

def test_provider_codes():
    """Test provider code mapping."""
    print("Testing provider code mapping:")

    # Test Yahoo Finance codes
    assert DataDownloaderFactory.get_provider_by_code("yf") == "yahoo"
    assert DataDownloaderFactory.get_provider_by_code("yahoo") == "yahoo"
    assert DataDownloaderFactory.get_provider_by_code("yf_finance") == "yahoo"

    # Test Alpha Vantage codes
    assert DataDownloaderFactory.get_provider_by_code("av") == "alphavantage"
    assert DataDownloaderFactory.get_provider_by_code("alphavantage") == "alphavantage"
    assert DataDownloaderFactory.get_provider_by_code("alpha_vantage") == "alphavantage"

    # Test Binance codes
    assert DataDownloaderFactory.get_provider_by_code("bnc") == "binance"
    assert DataDownloaderFactory.get_provider_by_code("binance") == "binance"

    # Test invalid code
    assert DataDownloaderFactory.get_provider_by_code("invalid") is None

    print("✓ All provider code tests passed!")

def test_yahoo_downloader():
    """Test Yahoo Finance downloader creation."""
    print("\nTesting Yahoo Finance downloader creation:")

    downloader = DataDownloaderFactory.create_downloader("yf")
    assert downloader is not None
    assert hasattr(downloader, 'get_fundamentals')
    assert hasattr(downloader, 'get_ohlcv')

    print("✓ Yahoo Finance downloader created successfully!")

def test_invalid_provider():
    """Test invalid provider handling."""
    print("\nTesting invalid provider handling:")

    downloader = DataDownloaderFactory.create_downloader("invalid")
    assert downloader is None

    print("✓ Invalid provider correctly handled!")

def test_provider_listing():
    """Test provider listing functionality."""
    print("\nTesting provider listing:")

    providers = DataDownloaderFactory.get_supported_providers()
    assert len(providers) > 0
    assert "yf" in providers
    assert "av" in providers
    assert "bnc" in providers

    print(f"✓ Found {len(providers)} supported providers")
    print(f"  Sample providers: {providers[:5]}")

def test_provider_info():
    """Test provider information retrieval."""
    print("\nTesting provider information:")

    info = DataDownloaderFactory.get_provider_info()
    assert "yahoo" in info
    assert "alphavantage" in info
    assert "binance" in info

    yahoo_info = info["yahoo"]
    assert "codes" in yahoo_info
    assert "name" in yahoo_info
    assert "requires_api_key" in yahoo_info

    print("✓ Provider information retrieved successfully!")

def main():
    """Run all tests."""
    print("DataDownloaderFactory Test Suite")
    print("=" * 50)

    try:
        test_provider_codes()
        test_yahoo_downloader()
        test_invalid_provider()
        test_provider_listing()
        test_provider_info()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("\nDataDownloaderFactory is working correctly!")

    except Exception as e:
        logger.exception("Test failed: %s", str(e))
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
