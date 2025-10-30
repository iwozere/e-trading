#!/usr/bin/env python3
"""
Test Runner for Web UI Backend Tests
-----------------------------------

Comprehensive test runner that executes all backend tests with coverage reporting.
Provides options for running specific test modules or all tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))


def run_all_tests():
    """Run all backend tests with coverage reporting."""
    test_dir = Path(__file__).parent

    # Test arguments for comprehensive testing
    args = [
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable pytest warnings for cleaner output
        "-x",  # Stop on first failure (remove for full test run)
    ]

    # Add coverage if available
    try:
        import coverage
        args.extend([
            "--cov=src.api",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=80"  # Require 80% coverage
        ])
        print("Running tests with coverage reporting...")
    except ImportError:
        print("Coverage not available. Install with: pip install pytest-cov")
        print("Running tests without coverage...")

    return pytest.main(args)


def run_specific_tests(test_pattern):
    """Run specific tests matching the pattern."""
    test_dir = Path(__file__).parent

    args = [
        str(test_dir),
        "-v",
        "-k", test_pattern,  # Run tests matching pattern
        "--tb=short"
    ]

    return pytest.main(args)


def run_auth_tests():
    """Run only authentication-related tests."""
    return run_specific_tests("auth")


def run_api_tests():
    """Run only API endpoint tests."""
    return run_specific_tests("api or main")


def run_service_tests():
    """Run only service layer tests."""
    return run_specific_tests("service")


def run_telegram_tests():
    """Run only Telegram-related tests."""
    return run_specific_tests("telegram")


def main():
    """Main test runner with command line options."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "auth":
            print("Running authentication tests...")
            return run_auth_tests()
        elif command == "api":
            print("Running API tests...")
            return run_api_tests()
        elif command == "services":
            print("Running service tests...")
            return run_service_tests()
        elif command == "telegram":
            print("Running Telegram tests...")
            return run_telegram_tests()
        elif command == "help":
            print("Available commands:")
            print("  auth      - Run authentication tests")
            print("  api       - Run API endpoint tests")
            print("  services  - Run service layer tests")
            print("  telegram  - Run Telegram-related tests")
            print("  all       - Run all tests (default)")
            print("  help      - Show this help message")
            return 0
        elif command == "all":
            print("Running all tests...")
            return run_all_tests()
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' to see available commands")
            return 1
    else:
        print("Running all tests...")
        return run_all_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)