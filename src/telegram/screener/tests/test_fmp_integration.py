#!/usr/bin/env python3
"""
Test script for FMP Integration with Enhanced Screener.
This script tests the FMP-based screening functionality.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.telegram.screener.fmp_integration import get_fmp_integration, run_fmp_screening
from src.telegram.screener.screener_config_parser import parse_screener_config, validate_screener_config
from src.telegram.screener.enhanced_screener import enhanced_screener
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def test_fmp_integration():
    """Test FMP integration functionality."""
    print("🧪 Testing FMP Integration")
    print("=" * 50)

    # Test 1: FMP Integration Initialization
    try:
        fmp_integration = get_fmp_integration()
        print("✅ FMP Integration initialized successfully")

        # Test available strategies
        strategies = fmp_integration.get_available_fmp_strategies()
        print(f"✅ Available FMP strategies: {len(strategies)}")

        # Test default configs
        configs = fmp_integration.get_default_fmp_configs()
        print(f"✅ Available default configs: {len(configs)}")

    except Exception as e:
        print(f"❌ FMP Integration initialization failed: {e}")
        return

    # Test 2: FMP Criteria Validation
    print("\n🔍 Testing FMP Criteria Validation")

    valid_fmp_criteria = {
        "marketCapMoreThan": 1000000000,
        "peRatioLessThan": 15,
        "returnOnEquityMoreThan": 0.12,
        "limit": 50
    }

    try:
        is_valid, errors = fmp_integration.validate_fmp_criteria(valid_fmp_criteria)
        if is_valid:
            print("✅ Valid FMP criteria validation passed")
        else:
            print(f"❌ Valid FMP criteria validation failed: {errors}")
    except Exception as e:
        print(f"❌ FMP criteria validation error: {e}")

    # Test 3: Screener Config with FMP
    print("\n📊 Testing Screener Config with FMP")

    # Test config with FMP criteria
    fmp_config_json = json.dumps({
        "screener_type": "hybrid",
        "list_type": "us_medium_cap",
        "fmp_criteria": {
            "marketCapMoreThan": 2000000000,
            "peRatioLessThan": 20,
            "returnOnEquityMoreThan": 0.12,
            "limit": 30
        },
        "fundamental_criteria": [
            {
                "indicator": "PE",
                "operator": "max",
                "value": 15,
                "weight": 1.0,
                "required": True
            }
        ],
        "technical_criteria": [
            {
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "<", "value": 70},
                "weight": 0.6,
                "required": False
            }
        ],
        "max_results": 10,
        "min_score": 7.0
    })

    try:
        # Validate config
        is_valid, errors = validate_screener_config(fmp_config_json)
        if is_valid:
            print("✅ FMP screener config validation passed")
        else:
            print(f"❌ FMP screener config validation failed: {errors}")
            return

        # Parse config
        config = parse_screener_config(fmp_config_json)
        print(f"✅ FMP screener config parsed successfully")
        print(f"   Screener type: {config.screener_type}")
        print(f"   List type: {config.list_type}")
        print(f"   FMP criteria: {len(config.fmp_criteria) if config.fmp_criteria else 0} criteria")
        print(f"   Fundamental criteria: {len(config.fundamental_criteria) if config.fundamental_criteria else 0}")
        print(f"   Technical criteria: {len(config.technical_criteria) if config.technical_criteria else 0}")

    except Exception as e:
        print(f"❌ FMP screener config test failed: {e}")
        return

    # Test 4: FMP Screening (Mock/Test)
    print("\n🚀 Testing FMP Screening")

    try:
        # Convert config to dictionary for FMP screening
        screener_config = {
            "screener_type": config.screener_type,
            "list_type": config.list_type,
            "fmp_criteria": config.fmp_criteria,
            "fmp_strategy": config.fmp_strategy
        }

        # Run FMP screening (this will fail if no API key, but we can test the logic)
        ticker_list, fmp_results = run_fmp_screening(screener_config)

        if ticker_list:
            print(f"✅ FMP screening returned {len(ticker_list)} tickers")
            print(f"   Sample tickers: {ticker_list[:5]}")
        else:
            print("⚠️  FMP screening returned no tickers (likely no API key)")
            print("   This is expected if FMP_API_KEY is not set")

    except Exception as e:
        print(f"❌ FMP screening test failed: {e}")

    # Test 5: Enhanced Screener with FMP
    print("\n🎯 Testing Enhanced Screener with FMP")

    try:
        # Run enhanced screener with FMP config
        report = enhanced_screener.run_enhanced_screener(config)

        if report.error:
            print(f"⚠️  Enhanced screener with FMP returned error: {report.error}")
        else:
            print(f"✅ Enhanced screener with FMP completed successfully")
            print(f"   Total processed: {report.total_tickers_processed}")
            print(f"   Results found: {len(report.top_results)}")

            # Check if FMP results are included
            if hasattr(report, 'fmp_results') and report.fmp_results:
                print(f"   FMP results included: Yes")
                fmp_criteria = report.fmp_results.get('fmp_criteria', {})
                print(f"   FMP criteria used: {list(fmp_criteria.keys())}")
            else:
                print(f"   FMP results included: No (fallback to traditional screening)")

    except Exception as e:
        print(f"❌ Enhanced screener with FMP test failed: {e}")


def test_fmp_strategies():
    """Test predefined FMP strategies."""
    print("\n📋 Testing FMP Strategies")
    print("=" * 50)

    try:
        fmp_integration = get_fmp_integration()
        strategies = fmp_integration.get_available_fmp_strategies()

        print(f"Available strategies ({len(strategies)}):")
        for name, description in strategies.items():
            print(f"  • {name}: {description}")

        # Test a specific strategy
        if strategies:
            strategy_name = list(strategies.keys())[0]
            print(f"\nTesting strategy: {strategy_name}")

            config_json = json.dumps({
                "screener_type": "hybrid",
                "list_type": "us_medium_cap",
                "fmp_strategy": strategy_name,
                "fundamental_criteria": [
                    {
                        "indicator": "PE",
                        "operator": "max",
                        "value": 15,
                        "weight": 1.0,
                        "required": True
                    }
                ],
                "max_results": 10,
                "min_score": 7.0
            })

            is_valid, errors = validate_screener_config(config_json)
            if is_valid:
                print(f"✅ Strategy '{strategy_name}' config validation passed")
            else:
                print(f"❌ Strategy '{strategy_name}' config validation failed: {errors}")

    except Exception as e:
        print(f"❌ FMP strategies test failed: {e}")


def test_fmp_config_examples():
    """Test FMP configuration examples."""
    print("\n📝 Testing FMP Configuration Examples")
    print("=" * 50)

    examples = [
        {
            "name": "Basic FMP Criteria",
            "config": {
                "screener_type": "hybrid",
                "list_type": "us_medium_cap",
                "fmp_criteria": {
                    "marketCapMoreThan": 1000000000,
                    "peRatioLessThan": 15,
                    "limit": 50
                },
                "fundamental_criteria": [
                    {
                        "indicator": "PE",
                        "operator": "max",
                        "value": 12,
                        "weight": 1.0,
                        "required": True
                    }
                ],
                "max_results": 10,
                "min_score": 7.0
            }
        },
        {
            "name": "FMP Strategy",
            "config": {
                "screener_type": "hybrid",
                "list_type": "us_large_cap",
                "fmp_strategy": "conservative_value",
                "fundamental_criteria": [
                    {
                        "indicator": "ROE",
                        "operator": "min",
                        "value": 15,
                        "weight": 1.0,
                        "required": True
                    }
                ],
                "max_results": 15,
                "min_score": 7.5
            }
        },
        {
            "name": "Advanced FMP + Technical",
            "config": {
                "screener_type": "hybrid",
                "list_type": "us_small_cap",
                "fmp_criteria": {
                    "marketCapMoreThan": 500000000,
                    "marketCapLowerThan": 2000000000,
                    "peRatioLessThan": 20,
                    "returnOnEquityMoreThan": 0.10,
                    "limit": 40
                },
                "fundamental_criteria": [
                    {
                        "indicator": "PE",
                        "operator": "max",
                        "value": 15,
                        "weight": 1.0,
                        "required": True
                    }
                ],
                "technical_criteria": [
                    {
                        "indicator": "RSI",
                        "parameters": {"period": 14},
                        "condition": {"operator": "<", "value": 70},
                        "weight": 0.6,
                        "required": False
                    }
                ],
                "max_results": 20,
                "min_score": 6.5
            }
        }
    ]

    for example in examples:
        print(f"\n🔍 Testing: {example['name']}")

        try:
            config_json = json.dumps(example['config'])

            # Validate config
            is_valid, errors = validate_screener_config(config_json)
            if is_valid:
                print(f"✅ Validation passed")

                # Parse config
                config = parse_screener_config(config_json)
                print(f"✅ Parsing successful")
                print(f"   FMP criteria: {len(config.fmp_criteria) if config.fmp_criteria else 0}")
                print(f"   FMP strategy: {config.fmp_strategy or 'None'}")

            else:
                print(f"❌ Validation failed: {errors}")

        except Exception as e:
            print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting FMP Integration Tests")
    print("=" * 70)

    # Test FMP integration
    test_fmp_integration()

    # Test FMP strategies
    test_fmp_strategies()

    # Test FMP configuration examples
    test_fmp_config_examples()

    print("\n🎉 FMP Integration tests completed!")
    print("\n💡 To use FMP screening:")
    print("1. Set FMP_API_KEY environment variable")
    print("2. Use fmp_criteria or fmp_strategy in your screener config")
    print("3. Run enhanced screener with FMP integration")
