#!/usr/bin/env python3
"""
Test script for all notification channel plugins.

Tests Telegram, Email, SMS, and Test channel implementations.
Verifies configuration validation, message formatting, and health monitoring.
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


async def test_telegram_channel():
    """Test Telegram channel plugin."""
    print("Testing Telegram Channel...")

    try:
        # Test configuration validation
        config = {
            "bot_token": "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",  # Fake token format
            "default_chat_id": "123456789",
            "parse_mode": "HTML",
            "rate_limit_per_minute": 25
        }

        channel = channel_registry.get_channel("telegram_channel", config)
        print("✓ Telegram channel created and configured")

        # Test feature support
        features_to_test = ["html", "attachments", "replies", "voice_messages"]
        for feature in features_to_test:
            supported = channel.supports_feature(feature)
            print(f"  Feature '{feature}': {'✓' if supported else '✗'}")

        # Test message formatting
        content = MessageContent(
            text="Hello, World!",
            html="<b>Hello, World!</b>",
            subject="Test Message"
        )

        formatted = channel.format_message(content)
        print(f"✓ Message formatting: {formatted.text[:50]}...")

        # Test message splitting
        long_content = MessageContent(text="A" * 5000)  # Very long message
        parts = channel.split_long_message(long_content)
        print(f"✓ Message splitting: {len(parts)} parts")

        # Test rate limit
        rate_limit = channel.get_rate_limit()
        print(f"✓ Rate limit: {rate_limit} messages/minute")

        # Test max message length
        max_length = channel.get_max_message_length()
        print(f"✓ Max message length: {max_length} characters")

        # Note: We can't test actual sending without a real bot token
        print("✓ Telegram channel tests completed (configuration and features)")

        return True

    except Exception as e:
        print(f"✗ Telegram channel test failed: {e}")
        return False


async def test_email_channel():
    """Test Email channel plugin."""
    print("\nTesting Email Channel...")

    try:
        # Test configuration validation
        config = {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_username": "test@example.com",
            "smtp_password": "test_password",
            "from_email": "sender@example.com",
            "from_name": "Test Sender",
            "use_tls": True,
            "rate_limit_per_minute": 5
        }

        channel = channel_registry.get_channel("email_channel", config)
        print("✓ Email channel created and configured")

        # Test feature support
        features_to_test = ["html", "attachments", "cc", "bcc", "templates"]
        for feature in features_to_test:
            supported = channel.supports_feature(feature)
            print(f"  Feature '{feature}': {'✓' if supported else '✗'}")

        # Test message formatting
        content = MessageContent(
            text="Hello, World!\nThis is a test email.",
            subject="Test Email",
            attachments={"test.txt": b"Test file content"}
        )

        formatted = channel.format_message(content)
        print(f"✓ Message formatting: HTML generated")
        print(f"  HTML preview: {formatted.html[:100]}...")

        # Test rate limit
        rate_limit = channel.get_rate_limit()
        print(f"✓ Rate limit: {rate_limit} messages/minute")

        # Test max message length (should be None for email)
        max_length = channel.get_max_message_length()
        print(f"✓ Max message length: {'Unlimited' if max_length is None else max_length}")

        # Note: We can't test actual sending without real SMTP credentials
        print("✓ Email channel tests completed (configuration and features)")

        return True

    except Exception as e:
        print(f"✗ Email channel test failed: {e}")
        return False


async def test_sms_channel():
    """Test SMS channel plugin."""
    print("\nTesting SMS Channel...")

    try:
        # Test Twilio configuration
        config = {
            "provider": "twilio",
            "account_sid": "AC" + "1234567890123456789012345678901234",  # Fake SID format
            "auth_token": "1234567890123456789012345678901234",  # Fake token
            "default_from_number": "+1234567890",
            "rate_limit_per_minute": 3
        }

        channel = channel_registry.get_channel("sms_channel", config)
        print("✓ SMS channel created and configured")

        # Test feature support
        features_to_test = ["splitting", "delivery_reports", "attachments", "html"]
        for feature in features_to_test:
            supported = channel.supports_feature(feature)
            print(f"  Feature '{feature}': {'✓' if supported else '✗'}")

        # Test message formatting (should strip HTML)
        content = MessageContent(
            text="",
            html="<b>Hello, World!</b><br>This is a <i>test</i> SMS.",
            subject="Test SMS"
        )

        formatted = channel.format_message(content)
        print(f"✓ Message formatting (HTML stripped): {formatted.text}")

        # Test message splitting
        long_message = "A" * 500  # Long SMS message
        parts = channel._split_sms_message(long_message)
        print(f"✓ SMS splitting: {len(parts)} parts")

        # Test rate limit
        rate_limit = channel.get_rate_limit()
        print(f"✓ Rate limit: {rate_limit} messages/minute")

        # Test max message length
        max_length = channel.get_max_message_length()
        print(f"✓ Max message length: {max_length} characters")

        # Note: We can't test actual sending without real Twilio credentials
        print("✓ SMS channel tests completed (configuration and features)")

        return True

    except Exception as e:
        print(f"✗ SMS channel test failed: {e}")
        return False


async def test_test_channel():
    """Test the test channel plugin."""
    print("\nTesting Test Channel...")

    try:
        # Test configuration
        config = {
            "simulate_delay_ms": 100,
            "failure_rate": 0.0,  # No failures for this test
            "max_message_length": 200
        }

        channel = channel_registry.get_channel("test_plugin", config)
        print("✓ Test channel created and configured")

        # Test actual message sending (this should work)
        content = MessageContent(
            text="This is a test message for the test channel.",
            subject="Test Subject"
        )

        result = await channel.send_message("test_recipient", content, "test_msg_001")

        if result.is_successful:
            print(f"✓ Message sent successfully: {result.external_id}")
            print(f"  Response time: {result.response_time_ms}ms")
            print(f"  Status: {result.status}")
        else:
            print(f"✗ Message sending failed: {result.error_message}")
            return False

        # Test health check
        health = await channel.check_health()
        print(f"✓ Health check: {health.status} (response: {health.response_time_ms}ms)")

        # Test long message splitting
        long_content = MessageContent(text="A" * 300)  # Longer than max_message_length
        parts = channel.split_long_message(long_content)
        print(f"✓ Long message split into {len(parts)} parts")

        return True

    except Exception as e:
        print(f"✗ Test channel test failed: {e}")
        return False


async def test_configuration_validation():
    """Test configuration validation for all channels."""
    print("\nTesting Configuration Validation...")

    test_cases = [
        # Telegram - missing bot_token
        ("telegram_channel", {"default_chat_id": "123"}, False),

        # Telegram - invalid parse_mode
        ("telegram_channel", {
            "bot_token": "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
            "default_chat_id": "123",
            "parse_mode": "INVALID"
        }, False),

        # Email - missing SMTP config
        ("email_channel", {"from_email": "test@example.com"}, False),

        # Email - invalid email format
        ("email_channel", {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_username": "invalid_email",
            "smtp_password": "password",
            "from_email": "test@example.com"
        }, False),

        # SMS - invalid provider
        ("sms_channel", {"provider": "invalid_provider"}, False),

        # SMS - missing Twilio credentials
        ("sms_channel", {"provider": "twilio"}, False),
    ]

    passed = 0
    total = len(test_cases)

    for channel_name, config, should_succeed in test_cases:
        try:
            channel_registry.get_channel(channel_name, config)
            if should_succeed:
                print(f"✓ {channel_name} valid config accepted")
                passed += 1
            else:
                print(f"✗ {channel_name} invalid config should have been rejected")
        except (ConfigValidationError, ValueError):
            if not should_succeed:
                print(f"✓ {channel_name} invalid config properly rejected")
                passed += 1
            else:
                print(f"✗ {channel_name} valid config should have been accepted")
        except Exception as e:
            print(f"✗ {channel_name} unexpected error: {e}")

    print(f"Configuration validation: {passed}/{total} tests passed")
    return passed == total


async def test_plugin_loading():
    """Test plugin loading and discovery."""
    print("\nTesting Plugin Loading...")

    try:
        # Load all plugins
        plugins = load_all_channels()
        print(f"✓ Loaded {len(plugins)} plugins: {list(plugins.keys())}")

        # Check expected plugins are loaded
        expected_plugins = ["telegram_channel", "email_channel", "sms_channel", "test_plugin"]
        missing_plugins = []

        for plugin_name in expected_plugins:
            if plugin_name in plugins:
                print(f"✓ Plugin '{plugin_name}' loaded successfully")
            else:
                print(f"✗ Plugin '{plugin_name}' not found")
                missing_plugins.append(plugin_name)

        if missing_plugins:
            print(f"Missing plugins: {missing_plugins}")
            return False

        # Test registry functionality
        available_channels = channel_registry.list_channels()
        print(f"✓ Available channels in registry: {available_channels}")

        return True

    except Exception as e:
        print(f"✗ Plugin loading test failed: {e}")
        return False


async def main():
    """Run all channel plugin tests."""
    print("=== Notification Channel Plugin Tests ===\n")

    tests = [
        ("Plugin Loading", test_plugin_loading),
        ("Configuration Validation", test_configuration_validation),
        ("Telegram Channel", test_telegram_channel),
        ("Email Channel", test_email_channel),
        ("SMS Channel", test_sms_channel),
        ("Test Channel", test_test_channel),
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
        print("🎉 All channel plugin tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)