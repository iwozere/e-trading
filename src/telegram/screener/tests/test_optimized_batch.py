#!/usr/bin/env python3
"""
Test script for optimized YFinance batch download functionality.
This script demonstrates the difference between individual, regular batch, and optimized batch operations.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def test_individual_vs_batch_vs_optimized():
    """Test individual vs regular batch vs optimized batch download performance."""
    print("⚡ Testing Individual vs Batch vs Optimized Batch Performance")
    print("=" * 70)

    # Test tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]

    downloader = YahooDataDownloader()

    print(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")

    # Test 1: Individual downloads
    print("\n🔄 Testing Individual Downloads...")
    start_time = time.time()

    individual_fundamentals = {}
    for ticker in test_tickers:
        try:
            fundamentals = downloader.get_fundamentals(ticker)
            individual_fundamentals[ticker] = fundamentals
            print(f"  ✅ {ticker}: {fundamentals.company_name}")
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")

    individual_time = time.time() - start_time
    print(f"Individual downloads completed in {individual_time:.2f} seconds")

    # Test 2: Regular batch downloads
    print("\n🚀 Testing Regular Batch Downloads...")
    start_time = time.time()

    try:
        batch_fundamentals = downloader.get_fundamentals_batch(test_tickers)

        batch_time = time.time() - start_time
        print(f"Regular batch downloads completed in {batch_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Regular batch download failed: {e}")
        batch_fundamentals = {}
        batch_time = individual_time  # Use individual time as fallback

    # Test 3: Optimized batch downloads
    print("\n🚀 Testing Optimized Batch Downloads...")
    start_time = time.time()

    try:
        optimized_fundamentals = downloader.get_fundamentals_batch_optimized(test_tickers, include_financials=False)

        optimized_time = time.time() - start_time
        print(f"Optimized batch downloads completed in {optimized_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Optimized batch download failed: {e}")
        optimized_fundamentals = {}
        optimized_time = individual_time  # Use individual time as fallback

    # Compare performance
    print("\n📈 Performance Comparison:")
    print(f"  Individual:  {individual_time:.2f} seconds")
    print(f"  Regular Batch: {batch_time:.2f} seconds")
    print(f"  Optimized Batch: {optimized_time:.2f} seconds")

    if batch_time < individual_time:
        batch_speedup = individual_time / batch_time
        print(f"  Regular Batch Speedup: {batch_speedup:.1f}x faster")

    if optimized_time < individual_time:
        optimized_speedup = individual_time / optimized_time
        print(f"  Optimized Batch Speedup: {optimized_speedup:.1f}x faster")

    if optimized_time < batch_time:
        optimization_improvement = batch_time / optimized_time
        print(f"  Optimization Improvement: {optimization_improvement:.1f}x faster than regular batch")

    # Verify data consistency
    print("\n🔍 Data Verification:")
    for ticker in test_tickers:
        if ticker in individual_fundamentals and ticker in batch_fundamentals and ticker in optimized_fundamentals:
            ind_fund = individual_fundamentals[ticker]
            batch_fund = batch_fundamentals[ticker]
            opt_fund = optimized_fundamentals[ticker]

            if (ind_fund.company_name == batch_fund.company_name == opt_fund.company_name):
                print(f"  ✅ {ticker}: {ind_fund.company_name} (all consistent)")
            else:
                print(f"  ⚠️  {ticker}: Individual='{ind_fund.company_name}', Batch='{batch_fund.company_name}', Optimized='{opt_fund.company_name}'")
        else:
            print(f"  ❌ {ticker}: Missing data in one or more methods")


def test_large_scale_optimized():
    """Test optimized batch with a larger number of tickers."""
    print("\n🚀 Testing Large Scale Optimized Batch Download")
    print("=" * 60)

    # Larger test set
    large_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "JNJ",
        "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "ADBE", "CRM",
        "NKE", "CMCSA", "XOM", "VZ", "ABT", "KO", "PEP", "TMO", "COST", "AVGO"
    ]

    downloader = YahooDataDownloader()

    print(f"Testing with {len(large_tickers)} tickers")

    # Test optimized batch fundamentals
    print("\n💰 Optimized Batch Fundamentals Download:")
    start_time = time.time()

    try:
        fundamentals_results = downloader.get_fundamentals_batch_optimized(large_tickers, include_financials=False)

        fundamentals_time = time.time() - start_time
        successful_fundamentals = len([f for f in fundamentals_results.values() if f.company_name != "Unknown"])
        print(f"  Completed in {fundamentals_time:.2f} seconds")
        print(f"  Success rate: {successful_fundamentals}/{len(large_tickers)} ({successful_fundamentals/len(large_tickers)*100:.1f}%)")

        # Show some sample results
        print("\n  Sample Results:")
        for i, (ticker, fundamentals) in enumerate(list(fundamentals_results.items())[:5], 1):
            print(f"    {i}. {ticker}: {fundamentals.company_name} - PE: {fundamentals.pe_ratio:.1f}")

    except Exception as e:
        print(f"  ❌ Failed: {e}")


def test_enhanced_screener_optimized():
    """Test enhanced screener with optimized batch operations."""
    print("\n🎯 Testing Enhanced Screener with Optimized Batch Operations")
    print("=" * 70)

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

        # Run enhanced screener with optimized batch operations
        print("\n🔄 Running Enhanced Screener with Optimized Batch Operations...")
        start_time = time.time()

        report = enhanced_screener.run_enhanced_screener(screener_config)

        total_time = time.time() - start_time
        print(f"Enhanced screener completed in {total_time:.2f} seconds")

        if report.error:
            print(f"❌ Screener error: {report.error}")
        else:
            print("✅ Screener completed successfully!")
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


def test_api_call_reduction():
    """Test to demonstrate the reduction in individual API calls."""
    print("\n🔍 Testing API Call Reduction")
    print("=" * 40)

    # Small test set for demonstration
    test_tickers = ["AAPL", "MSFT", "GOOGL"]

    downloader = YahooDataDownloader()

    print(f"Testing with {len(test_tickers)} tickers")
    print("This test demonstrates the reduction in individual API calls")

    # Test optimized batch
    print("\n🚀 Testing Optimized Batch (Minimal API Calls)...")
    start_time = time.time()

    try:
        fundamentals_results = downloader.get_fundamentals_batch_optimized(test_tickers, include_financials=False)

        optimized_time = time.time() - start_time
        print(f"  Completed in {optimized_time:.2f} seconds")
        print(f"  Retrieved data for {len(fundamentals_results)} tickers")

        # Show what data was retrieved
        print("\n  Retrieved Data:")
        for ticker, fundamentals in fundamentals_results.items():
            print(f"    {ticker}: {fundamentals.company_name}")
            print(f"      PE: {fundamentals.pe_ratio:.1f}")
            print(f"      Market Cap: ${fundamentals.market_cap:,.0f}")
            print(f"      Sector: {fundamentals.sector}")
            print()

    except Exception as e:
        print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Optimized YFinance Batch Download Tests")
    print("=" * 70)

    # Test individual vs batch vs optimized performance
    test_individual_vs_batch_vs_optimized()

    # Test large scale optimized batch
    test_large_scale_optimized()

    # Test enhanced screener with optimized batch operations
    test_enhanced_screener_optimized()

    # Test API call reduction
    test_api_call_reduction()

    print("\n🎉 All optimized batch download tests completed!")
