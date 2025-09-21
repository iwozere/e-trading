#!/usr/bin/env python3
"""
Test script for YFinance batch download functionality.
This script demonstrates how to use batch operations for better performance.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def test_batch_ohlcv_download():
    """Test batch OHLCV download functionality."""
    print("📊 Testing Batch OHLCV Download")
    print("=" * 50)

    # Test tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days

    downloader = YahooDataDownloader()

    print(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Test individual downloads (for comparison)
    print("\n🔄 Testing Individual Downloads...")
    start_time = time.time()

    individual_results = {}
    for ticker in test_tickers:
        try:
            df = downloader.get_ohlcv(ticker, "1d", start_date, end_date)
            individual_results[ticker] = df
            print(f"  ✅ {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")

    individual_time = time.time() - start_time
    print(f"Individual downloads completed in {individual_time:.2f} seconds")

    # Test batch download
    print("\n🚀 Testing Batch Download...")
    start_time = time.time()

    try:
        batch_results = downloader.get_ohlcv_batch(test_tickers, "1d", start_date, end_date)

        batch_time = time.time() - start_time
        print(f"Batch download completed in {batch_time:.2f} seconds")

        # Compare results
        print(f"\n📈 Performance Comparison:")
        print(f"  Individual: {individual_time:.2f} seconds")
        print(f"  Batch:      {batch_time:.2f} seconds")
        print(f"  Speedup:    {individual_time/batch_time:.1f}x faster")

        # Verify data consistency
        print(f"\n🔍 Data Verification:")
        for ticker in test_tickers:
            if ticker in individual_results and ticker in batch_results:
                ind_df = individual_results[ticker]
                batch_df = batch_results[ticker]

                if len(ind_df) == len(batch_df):
                    print(f"  ✅ {ticker}: {len(ind_df)} rows (consistent)")
                else:
                    print(f"  ⚠️  {ticker}: Individual={len(ind_df)}, Batch={len(batch_df)} rows")
            else:
                print(f"  ❌ {ticker}: Missing data")

    except Exception as e:
        print(f"❌ Batch download failed: {e}")


def test_batch_fundamentals_download():
    """Test batch fundamentals download functionality."""
    print("\n💰 Testing Batch Fundamentals Download")
    print("=" * 50)

    # Test tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]

    downloader = YahooDataDownloader()

    print(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")

    # Test individual downloads (for comparison)
    print("\n🔄 Testing Individual Downloads...")
    start_time = time.time()

    individual_results = {}
    for ticker in test_tickers:
        try:
            fundamentals = downloader.get_fundamentals(ticker)
            individual_results[ticker] = fundamentals
            print(f"  ✅ {ticker}: {fundamentals.company_name}")
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")

    individual_time = time.time() - start_time
    print(f"Individual downloads completed in {individual_time:.2f} seconds")

    # Test batch download
    print("\n🚀 Testing Batch Download...")
    start_time = time.time()

    try:
        batch_results = downloader.get_fundamentals_batch(test_tickers)

        batch_time = time.time() - start_time
        print(f"Batch download completed in {batch_time:.2f} seconds")

        # Compare results
        print(f"\n📈 Performance Comparison:")
        print(f"  Individual: {individual_time:.2f} seconds")
        print(f"  Batch:      {batch_time:.2f} seconds")
        print(f"  Speedup:    {individual_time/batch_time:.1f}x faster")

        # Verify data consistency
        print(f"\n🔍 Data Verification:")
        for ticker in test_tickers:
            if ticker in individual_results and ticker in batch_results:
                ind_fund = individual_results[ticker]
                batch_fund = batch_results[ticker]

                if ind_fund.company_name == batch_fund.company_name:
                    print(f"  ✅ {ticker}: {ind_fund.company_name} (consistent)")
                else:
                    print(f"  ⚠️  {ticker}: Individual='{ind_fund.company_name}', Batch='{batch_fund.company_name}'")
            else:
                print(f"  ❌ {ticker}: Missing data")

    except Exception as e:
        print(f"❌ Batch fundamentals download failed: {e}")


def test_enhanced_screener_batch():
    """Test enhanced screener with batch operations."""
    print("\n🎯 Testing Enhanced Screener with Batch Operations")
    print("=" * 60)

    from src.telegram.screener.enhanced_screener import enhanced_screener
    from src.telegram.screener.screener_config_parser import parse_screener_config

    # Test configuration
    test_config = {
        "screener_type": "hybrid",
        "list_type": "us_medium_cap",
        "fundamental_criteria": [
            {
                "indicator": "PE",
                "operator": "max",
                "value": 25,
                "weight": 1.0,
                "required": True
            }
        ],
        "technical_criteria": [
            {
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "range", "min": 20, "max": 80},
                "weight": 0.6,
                "required": False
            }
        ],
        "period": "3mo",
        "interval": "1d",
        "max_results": 5,
        "min_score": 5.0,
        "email": False
    }

    try:
        # Parse configuration
        config_json = '{"screener_type":"hybrid","list_type":"us_medium_cap","fundamental_criteria":[{"indicator":"PE","operator":"max","value":25,"weight":1.0,"required":true}],"technical_criteria":[{"indicator":"RSI","parameters":{"period":14},"condition":{"operator":"range","min":20,"max":80},"weight":0.6,"required":false}],"period":"3mo","interval":"1d","max_results":5,"min_score":5.0,"email":false}'
        screener_config = parse_screener_config(config_json)

        print(f"Configuration: {screener_config.screener_type} screener for {screener_config.list_type}")
        print(f"Period: {screener_config.period}, Interval: {screener_config.interval}")
        print(f"Max Results: {screener_config.max_results}, Min Score: {screener_config.min_score}")

        # Run enhanced screener
        print("\n🔄 Running Enhanced Screener...")
        start_time = time.time()

        report = enhanced_screener.run_enhanced_screener(screener_config)

        total_time = time.time() - start_time
        print(f"Enhanced screener completed in {total_time:.2f} seconds")

        if report.error:
            print(f"❌ Screener error: {report.error}")
        else:
            print(f"✅ Screener completed successfully!")
            print(f"   Processed: {report.total_tickers_processed} tickers")
            print(f"   Found: {len(report.top_results)} matching stocks")

            if report.top_results:
                print("\n🏆 Top Results:")
                for i, result in enumerate(report.top_results, 1):
                    print(f"   {i}. {result.ticker}")
                    print(f"      Score: {result.composite_score:.1f}/10")
                    print(f"      Fundamental: {result.fundamental_score:.1f}/10")
                    print(f"      Technical: {result.technical_score:.1f}/10")
                    print(f"      Recommendation: {result.recommendation}")
                    if result.current_price:
                        print(f"      Price: ${result.current_price:.2f}")
                    print()

            # Display formatted report
            telegram_message = enhanced_screener.format_enhanced_telegram_message(report, screener_config)
            print("📊 ENHANCED SCREENER REPORT")
            print("=" * 50)
            print(telegram_message[:1000] + "..." if len(telegram_message) > 1000 else telegram_message)

    except Exception as e:
        print(f"❌ Enhanced screener test failed: {e}")


def test_large_batch():
    """Test with a larger number of tickers to demonstrate scalability."""
    print("\n🚀 Testing Large Batch Download")
    print("=" * 40)

    # Larger test set
    large_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "JNJ",
        "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "ADBE", "CRM",
        "NKE", "CMCSA", "XOM", "VZ", "ABT", "KO", "PEP", "TMO", "COST", "AVGO"
    ]

    downloader = YahooDataDownloader()

    print(f"Testing with {len(large_tickers)} tickers")

    # Test batch OHLCV
    print("\n📊 Batch OHLCV Download:")
    start_time = time.time()

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days

        ohlcv_results = downloader.get_ohlcv_batch(large_tickers, "1d", start_date, end_date)

        ohlcv_time = time.time() - start_time
        successful_ohlcv = len([df for df in ohlcv_results.values() if not df.empty])
        print(f"  Completed in {ohlcv_time:.2f} seconds")
        print(f"  Success rate: {successful_ohlcv}/{len(large_tickers)} ({successful_ohlcv/len(large_tickers)*100:.1f}%)")

    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Test batch fundamentals
    print("\n💰 Batch Fundamentals Download:")
    start_time = time.time()

    try:
        fundamentals_results = downloader.get_fundamentals_batch(large_tickers)

        fundamentals_time = time.time() - start_time
        successful_fundamentals = len([f for f in fundamentals_results.values() if f.company_name != "Unknown"])
        print(f"  Completed in {fundamentals_time:.2f} seconds")
        print(f"  Success rate: {successful_fundamentals}/{len(large_tickers)} ({successful_fundamentals/len(large_tickers)*100:.1f}%)")

    except Exception as e:
        print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting YFinance Batch Download Tests")
    print("=" * 70)

    # Test batch OHLCV download
    test_batch_ohlcv_download()

    # Test batch fundamentals download
    test_batch_fundamentals_download()

    # Test enhanced screener with batch operations
    test_enhanced_screener_batch()

    # Test large batch
    test_large_batch()

    print("\n🎉 All batch download tests completed!")
