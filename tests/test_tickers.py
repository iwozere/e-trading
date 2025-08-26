#!/usr/bin/env python3
"""
Test script for the refactored tickers_list.py
"""

from src.util.tickers_list import (
    get_all_us_tickers,
    get_us_delisted_tickers,
    get_us_small_cap_tickers,
    get_us_medium_cap_tickers,
    get_us_large_cap_tickers
)

def test_ticker_loading():
    """Test that all ticker functions load data correctly"""

    print("Testing ticker loading from CSV files...")

    # Test all functions
    functions = [
        ("All US Tickers", get_all_us_tickers),
        ("US Delisted Tickers", get_us_delisted_tickers),
        ("US Small Cap Tickers", get_us_small_cap_tickers),
        ("US Medium Cap Tickers", get_us_medium_cap_tickers),
        ("US Large Cap Tickers", get_us_large_cap_tickers),
    ]

    for name, func in functions:
        try:
            tickers = func()
            print(f"✓ {name}: {len(tickers)} tickers loaded")
            if tickers:
                print(f"  Sample tickers: {tickers[:5]}")
        except Exception as e:
            print(f"✗ {name}: Error - {str(e)}")

    print("\nAll tests completed!")

if __name__ == "__main__":
    test_ticker_loading()