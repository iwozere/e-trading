#!/usr/bin/env python3
"""
Test script for Alert Config Parser
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from src.frontend.telegram.screener.alert_config_parser import AlertConfigParser

def test_alert_parser():
    """Test the alert configuration parser."""
    print("Testing Alert Configuration Parser...")

    parser = AlertConfigParser()
    samples = parser.create_sample_configs()

    print(f"✅ Parser loaded successfully!")
    print(f"📊 Found {len(samples)} sample configurations")

    # Test each sample
    for name, config_json in samples.items():
        print(f"\n🔍 Testing {name}:")
        print(f"   Config: {config_json[:100]}...")

        # Validate the config
        is_valid, errors = parser.validate_config(config_json)
        if is_valid:
            print(f"   ✅ Valid configuration")
        else:
            print(f"   ❌ Invalid configuration: {errors}")

        # Get required data points
        data_points = parser.get_required_data_points(config_json)
        print(f"   📊 Required data points: {data_points}")

        # Parse the config
        try:
            config = parser.parse_config(config_json)
            print(f"   🎯 Parsed successfully: {config.alert_type} alert")
            print(f"   📋 Conditions: {len(config.conditions)}")
            print(f"   ⏰ Timeframe: {config.timeframe}")
            print(f"   🎬 Action: {config.alert_action}")
        except Exception as e:
            print(f"   ❌ Parse error: {e}")

    print(f"\n🎉 All tests completed!")

if __name__ == "__main__":
    test_alert_parser()
