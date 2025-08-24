#!/usr/bin/env python3
"""
Test script for the unified indicator system.

This script tests the unified indicator service that uses TA-Lib directly
for technical indicator calculations with simple memory caching.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from src.common.indicator_service import get_indicator_service
from src.models.indicators import (
    IndicatorCalculationRequest, BatchIndicatorRequest,
    TECHNICAL_INDICATORS, FUNDAMENTAL_INDICATORS
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


async def test_single_ticker():
    """Test indicator calculation for a single ticker."""
    print("=== Testing Single Ticker ===")

    service = get_indicator_service()

    # Test request
    request = IndicatorCalculationRequest(
        ticker="AAPL",
        indicators=["RSI", "MACD", "PE_RATIO", "ROE", "DIVIDEND_YIELD"],
        timeframe="1d",
        period="1y",
        include_recommendations=True
    )

    print("Calculating indicators for AAPL...")
    result = await service.get_indicators(request)

    print(f"Results for AAPL:")
    print(f"  Technical indicators: {len([r for r in result.get_all_indicators().values() if r.category.value == 'technical'])}")
    print(f"  Fundamental indicators: {len([r for r in result.get_all_indicators().values() if r.category.value == 'fundamental'])}")
    print(f"  Composite score: {result.composite_score:.2f}")
    print(f"  Overall recommendation: {result.overall_recommendation.recommendation.value if result.overall_recommendation else 'N/A'}")
    print(f"  Confidence: {result.overall_recommendation.confidence if result.overall_recommendation else 'N/A'}")
    print(f"  Reasoning: {result.overall_recommendation.reasoning if result.overall_recommendation else 'N/A'}")

    print("\n  Technical Indicators:")
    for name, indicator in result.get_all_indicators().items():
        if indicator.category.value == 'technical':
            print(f"    {name}: {indicator.value:.4f} ({indicator.recommendation.recommendation.value})")

    print("\n  Fundamental Indicators:")
    for name, indicator in result.get_all_indicators().items():
        if indicator.category.value == 'fundamental':
            print(f"    {name}: {indicator.value:.4f} ({indicator.recommendation.recommendation.value})")


async def test_batch_tickers():
    """Test batch indicator calculation."""
    print("\n=== Testing Batch Tickers ===")

    service = get_indicator_service()

    # Test batch request
    request = BatchIndicatorRequest(
        tickers=["AAPL", "MSFT", "GOOGL"],
        indicators=["RSI", "MACD"],
        timeframe="1d",
        period="1y",
        max_concurrent=3,
        include_recommendations=True
    )

    print("Calculating indicators for 3 tickers...")
    results = await service.get_batch_indicators(request)

    print("Results for 3 tickers:")
    for ticker, result in results.items():
        indicators = result.get_all_indicators()
        score = result.composite_score if result.composite_score is not None else 0.0
        print(f"  {ticker}: {len(indicators)} indicators, score: {score:.2f}")


async def test_cache_functionality():
    """Test cache functionality."""
    print("\n=== Testing Cache Functionality ===")

    service = get_indicator_service()

    # First request (should miss cache)
    print("First request (should miss cache)...")
    request1 = IndicatorCalculationRequest(
        ticker="TSLA",
        indicators=["RSI", "MACD"],
        timeframe="1d",
        period="1y",
        include_recommendations=True
    )

    result1 = await service.get_indicators(request1)

    # Second request (should hit cache)
    print("Second request (should hit cache)...")
    result2 = await service.get_indicators(request1)

    # Check cache stats
    stats = service.get_cache_stats()
    print(f"Cache stats: {stats}")


async def test_service_information():
    """Test service information."""
    print("\n=== Testing Service Information ===")

    service = get_indicator_service()

    # Get available indicators
    available = service.get_available_indicators()
    print("Available indicators:")
    print(f"  Technical: {len(available['technical'])}")
    print(f"  Fundamental: {len(available['fundamental'])}")
    print(f"  Total: {len(available['all'])}")

    # Get service info
    info = service.get_service_info()
    print(f"Service info: {info['service']} v{info['version']}")


async def test_error_handling():
    """Test error handling with invalid ticker."""
    print("\n=== Testing Error Handling ===")

    service = get_indicator_service()

    # Test with invalid ticker
    print("Testing with invalid ticker...")
    request = IndicatorCalculationRequest(
        ticker="INVALID_TICKER_12345",
        indicators=["RSI", "MACD"],
        timeframe="1d",
        period="1y",
        include_recommendations=True
    )

    result = await service.get_indicators(request)
    indicators = result.get_all_indicators()
    print(f"Result for invalid ticker: {len(indicators)} indicators")


async def main():
    """Run all tests."""
    print("🧪 Testing Unified Indicator System")
    print("=" * 50)
    print()

    try:
        await test_single_ticker()
        await test_batch_tickers()
        await test_cache_functionality()
        await test_service_information()
        await test_error_handling()

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        _logger.exception("Test failed")


if __name__ == "__main__":
    asyncio.run(main())
