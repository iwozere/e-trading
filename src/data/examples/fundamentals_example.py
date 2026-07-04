"""
Fundamentals Cache System Example

This example demonstrates how to use the new fundamentals cache system
with the DataManager to retrieve and cache fundamentals data.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.data_manager import DataManager


def main():
    """Demonstrate fundamentals cache usage."""
    print("📊 Fundamentals Cache System Example\n")

    # Initialize DataManager
    print("1. Initializing DataManager...")
    dm = DataManager("data-cache")
    print("   ✅ DataManager initialized\n")

    # Example 1: Get fundamentals for a stock
    print("2. Getting fundamentals for AAPL...")
    try:
        fundamentals = dm.get_fundamentals("AAPL")

        if fundamentals:
            print(f"   ✅ Retrieved fundamentals with {len(fundamentals)} fields")

            # Show some key fields
            key_fields = ["market_cap", "pe_ratio", "pb_ratio", "dividend_yield", "revenue"]
            print("   Key fundamentals:")
            for field in key_fields:
                if field in fundamentals:
                    value = fundamentals[field]
                    if isinstance(value, (int, float)) and value > 1000000000:
                        # Format large numbers
                        formatted_value = f"{value / 1000000000:.2f}B"
                    else:
                        formatted_value = value
                    print(f"     {field}: {formatted_value}")

            # Show metadata if available
            if "_metadata" in fundamentals:
                metadata = fundamentals["_metadata"]
                print(f"   Data sources: {metadata.get('providers_used', [])}")
                print(f"   Combination strategy: {metadata.get('combination_strategy', 'unknown')}")
        else:
            print("   ⚠️  No fundamentals data available (API keys may be required)")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Example 2: Get fundamentals with specific providers
    print("3. Getting fundamentals with specific providers...")
    try:
        fundamentals = dm.get_fundamentals("GOOGL", providers=["yfinance", "fmp"])

        if fundamentals:
            print(f"   ✅ Retrieved fundamentals with {len(fundamentals)} fields")
            if "_metadata" in fundamentals:
                print(f"   Providers used: {fundamentals['_metadata']['providers_used']}")
        else:
            print("   ⚠️  No fundamentals data available")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Example 3: Force refresh (bypass cache)
    print("4. Force refresh fundamentals...")
    try:
        fundamentals = dm.get_fundamentals("MSFT", force_refresh=True)

        if fundamentals:
            print(f"   ✅ Force refreshed fundamentals with {len(fundamentals)} fields")
        else:
            print("   ⚠️  No fundamentals data available")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Example 4: Different combination strategies
    print("5. Testing different combination strategies...")
    strategies = ["priority_based", "quality_based", "consensus"]

    for strategy in strategies:
        try:
            print(f"   Testing {strategy} strategy...")
            fundamentals = dm.get_fundamentals("TSLA", combination_strategy=strategy)

            if fundamentals and "_metadata" in fundamentals:
                print(f"     ✅ {strategy}: {fundamentals['_metadata']['combination_strategy']}")
            else:
                print(f"     ⚠️  {strategy}: No data available")

        except Exception as e:
            print(f"     ❌ {strategy}: Error - {e}")

    print()

    # Example 5: Cache statistics
    print("6. Cache statistics...")
    try:
        stats = dm.get_cache_stats()
        print(f"   Total cache size: {stats.get('total_size_gb', 0):.2f} GB")
        print(f"   Total files: {stats.get('files_count', 0)}")
        print(f"   Last updated: {stats.get('last_updated', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Error getting cache stats: {e}")

    print()
    print("🎉 Fundamentals cache example completed!")
    print("\nKey Features Demonstrated:")
    print("• 7-day cache-first rule")
    print("• Multi-provider data combination")
    print("• Automatic stale data cleanup")
    print("• Multiple combination strategies")
    print("• Provider priority system")


if __name__ == "__main__":
    main()
