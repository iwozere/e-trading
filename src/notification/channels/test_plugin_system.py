#!/usr/bin/env python3
"""
Test script for the channel plugin system.

Tests plugin loading, registration, configuration validation, and basic functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.channels import (
    load_all_channels, channel_registry, MessageContent,
    ConfigValidationError
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


async def test_plugin_loading():
    """Test plugin discovery and loading."""
    print("Testing plugin loading...")

    try:
        # Load all available plugins
        loaded_plugins = load_all_channels()
        print(f"✓ Loaded {len(loaded_plugins)} plugins: {list(loaded_plugins.keys())}")

        # Check if test plugin was loaded
        if 'test_plugin' in loaded_plugins:
            print("✓ Test plugin loaded successfully")
        else:
            print("✗ Test plugin not found")
            return False

        # List all registered channels
        available_channels = channel_registry.list_channels()
        print(f"✓ Available channels: {available_channels}")

        return True

    except Exception as e:
        print(f"✗ Plugin loading failed: {e}")
        return False


async def test_channel_configuration():
    """Test channel configuration validation."""
    print("\nTesting channel configuration...")

    try:
        # Test valid configuration
        valid_config = {
            "simulate_delay_ms": 50,
            "failure_rate": 0.1,
            "max_message_length": 500,
            "rate_limit_per_minute": 50
        }

        channel = channel_registry.get_channel("test_plugin", valid_config)
        print("✓ Valid configuration accepted")

        # Test invalid configuration
        try:
            invalid_config = {
                "simulate_delay_ms": -100,  # Invalid negative value
                "failure_rate": 2.0,        # Invalid rate > 1.0
            }

            channel_registry.get_channel("test_plugin", invalid_config)
            print("✗ Invalid configuration should have been rejected")
            return False

        except (ConfigValidationError, ValueError):
            print("✓ Invalid configuration properly rejected")

        return True

    except Exception as e:
        print(f"✗ Configuration testing failed: {e}")
        return False


async def test_message_delivery():
    """Test message delivery functionality."""
    print("\nTesting message delivery...")

    try:
        # Create channel instance
        config = {
            "simulate_delay_ms": 10,
            "failure_rate": 0.0,  # No failures for this test
            "max_message_length": 100
        }

        channel = channel_registry.get_channel("test_plugin", config)

        # Test simple message
        content = MessageContent(
            text="Hello, this is a test message!",
            subject="Test Subject"
        )

        result = await channel.send_message("test_recipient", content, "msg_001", "HIGH")

        if result.is_successful:
            print(f"✓ Message delivered successfully: {result.external_id}")
            print(f"  Response time: {result.response_time_ms}ms")
            print(f"  Status: {result.status}")
        else:
            print(f"✗ Message delivery failed: {result.error_message}")
            return False

        # Test message splitting
        long_content = MessageContent(
            text="This is a very long message that should be split into multiple parts because it exceeds the maximum length configured for the test channel. " * 3
        )

        parts = channel.split_long_message(long_content)
        print(f"✓ Long message split into {len(parts)} parts")

        # Test message formatting
        formatted = channel.format_message(content)
        if formatted.text.startswith("[TEST]"):
            print("✓ Message formatting works correctly")
        else:
            print("✗ Message formatting failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Message delivery testing failed: {e}")
        return False


async def test_health_monitoring():
    """Test channel health monitoring."""
    print("\nTesting health monitoring...")

    try:
        # Create channel instance
        config = {"simulate_delay_ms": 20}
        channel = channel_registry.get_channel("test_plugin", config)

        # Check individual channel health
        health = await channel.check_health()

        if health.is_healthy:
            print(f"✓ Channel health check passed: {health.status}")
            print(f"  Response time: {health.response_time_ms}ms")
        else:
            print(f"✗ Channel health check failed: {health.status}")
            return False

        # Check all channels health
        all_health = await channel_registry.check_all_health()
        print(f"✓ All channels health checked: {len(all_health)} channels")

        return True

    except Exception as e:
        print(f"✗ Health monitoring testing failed: {e}")
        return False


async def test_feature_support():
    """Test feature support checking."""
    print("\nTesting feature support...")

    try:
        config = {}
        channel = channel_registry.get_channel("test_plugin", config)

        # Test supported features
        supported_features = ["html", "attachments", "formatting"]
        unsupported_features = ["replies", "voice_messages"]

        for feature in supported_features:
            if channel.supports_feature(feature):
                print(f"✓ Feature '{feature}' is supported")
            else:
                print(f"✗ Feature '{feature}' should be supported")
                return False

        for feature in unsupported_features:
            if not channel.supports_feature(feature):
                print(f"✓ Feature '{feature}' is correctly not supported")
            else:
                print(f"✗ Feature '{feature}' should not be supported")
                return False

        # Test rate limit
        rate_limit = channel.get_rate_limit()
        print(f"✓ Rate limit: {rate_limit} messages/minute")

        # Test max message length
        max_length = channel.get_max_message_length()
        print(f"✓ Max message length: {max_length} characters")

        return True

    except Exception as e:
        print(f"✗ Feature support testing failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=== Channel Plugin System Tests ===\n")

    tests = [
        ("Plugin Loading", test_plugin_loading),
        ("Configuration Validation", test_channel_configuration),
        ("Message Delivery", test_message_delivery),
        ("Health Monitoring", test_health_monitoring),
        ("Feature Support", test_feature_support),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

        print()

    # Summary
    print("=== Test Results ===")
    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)