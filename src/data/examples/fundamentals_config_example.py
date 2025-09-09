"""
Fundamentals Configuration Example

This example demonstrates the new configuration-based fundamentals system with:
- Provider sequences for different data types
- Field-specific provider priorities
- Configurable TTL settings
- Configuration validation
"""

import logging
from src.data.data_manager import DataManager
from src.data.cache.fundamentals_combiner import get_fundamentals_combiner
from src.data.cache.fundamentals_cache import get_fundamentals_cache
from src.data.config.fundamentals_config_validator import validate_fundamentals_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_configuration_validation():
    """Demonstrate configuration validation."""
    print("=== Configuration Validation ===")

    # Validate the fundamentals configuration
    is_valid, errors = validate_fundamentals_config()

    if is_valid:
        print("✅ Fundamentals configuration is valid")
    else:
        print("❌ Fundamentals configuration has errors:")
        for error in errors:
            print(f"  - {error}")

def demonstrate_provider_sequences():
    """Demonstrate provider sequences for different data types."""
    print("\n=== Provider Sequences ===")

    combiner = get_fundamentals_combiner()

    data_types = ['statements', 'ratios', 'profile', 'calendar', 'dividends']

    for data_type in data_types:
        sequence = combiner.get_provider_sequence(data_type)
        print(f"{data_type:12}: {sequence}")

def demonstrate_ttl_settings():
    """Demonstrate TTL settings for different data types."""
    print("\n=== TTL Settings ===")

    combiner = get_fundamentals_combiner()

    data_types = ['profiles', 'ratios', 'statements', 'calendar', 'dividends']

    for data_type in data_types:
        ttl_days = combiner.get_ttl_for_data_type(data_type)
        print(f"{data_type:12}: {ttl_days} days")

def demonstrate_field_priorities():
    """Demonstrate field-specific provider priorities."""
    print("\n=== Field-Specific Provider Priorities ===")

    combiner = get_fundamentals_combiner()

    # Test different field paths
    field_paths = [
        'ttm_metrics.pe_ratio',
        'ttm_metrics.pb_ratio',
        'company_profile.sector',
        'company_profile.industry',
        'share_data.shares_outstanding',
        'calendar_events.earnings_date'
    ]

    for field_path in field_paths:
        priority = combiner.get_field_provider_priority(field_path)
        print(f"{field_path:30}: {priority}")

def demonstrate_data_manager_usage():
    """Demonstrate DataManager usage with new configuration system."""
    print("\n=== DataManager Usage ===")

    try:
        dm = DataManager("data-cache")

        # Test different data types
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        data_types = ['ratios', 'profile', 'statements']

        for symbol in symbols:
            print(f"\n--- {symbol} ---")

            for data_type in data_types:
                try:
                    print(f"Fetching {data_type} for {symbol}...")
                    fundamentals = dm.get_fundamentals(
                        symbol,
                        data_type=data_type,
                        combination_strategy='priority_based'
                    )

                    if fundamentals:
                        # Show some key fields
                        key_fields = ['market_cap', 'pe_ratio', 'sector', 'name']
                        available_fields = [field for field in key_fields if field in fundamentals]

                        if available_fields:
                            print(f"  {data_type}: Found {len(fundamentals)} fields, including: {available_fields[:3]}")
                        else:
                            print(f"  {data_type}: Found {len(fundamentals)} fields")
                    else:
                        print(f"  {data_type}: No data available")

                except Exception as e:
                    print(f"  {data_type}: Error - {e}")

    except Exception as e:
        print(f"Error initializing DataManager: {e}")

def demonstrate_cache_operations():
    """Demonstrate cache operations with configuration."""
    print("\n=== Cache Operations ===")

    try:
        # Initialize with configuration
        combiner = get_fundamentals_combiner()
        cache = get_fundamentals_cache("data-cache", combiner)

        # Get cache statistics
        stats = cache.get_cache_stats()
        print(f"Cache statistics: {stats}")

        # Test cache validity with different data types
        symbol = 'AAPL'
        data_types = ['ratios', 'profile', 'statements']

        for data_type in data_types:
            cached_data = cache.find_latest_json(symbol, data_type=data_type)
            if cached_data:
                is_valid = cache.is_cache_valid(cached_data.timestamp, data_type=data_type)
                ttl_days = combiner.get_ttl_for_data_type(data_type)
                print(f"{symbol} {data_type}: {'Valid' if is_valid else 'Expired'} (TTL: {ttl_days}d)")
            else:
                print(f"{symbol} {data_type}: No cached data")

    except Exception as e:
        print(f"Error with cache operations: {e}")

def main():
    """Run all demonstrations."""
    print("Fundamentals Configuration System Demo")
    print("=" * 50)

    # Run demonstrations
    demonstrate_configuration_validation()
    demonstrate_provider_sequences()
    demonstrate_ttl_settings()
    demonstrate_field_priorities()
    demonstrate_cache_operations()
    demonstrate_data_manager_usage()

    print("\n" + "=" * 50)
    print("Demo completed!")

if __name__ == "__main__":
    main()
