#!/usr/bin/env python3
"""
Test Runner for DataManager Architecture

This script runs all tests for the new DataManager architecture,
including integration tests and performance tests.

Usage:
    python src/data/tests/run_tests.py [--integration] [--performance] [--all]
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run_integration_tests():
    """Run integration tests."""
    print("Running Integration Tests...")
    print("=" * 50)

    test_file = Path(__file__).parent / "integration" / "test_data_manager_integration.py"

    pytest.main([
        str(test_file),
        "-v",
        "--tb=short",
        "--color=yes",
        "--durations=10"
    ])


def run_performance_tests():
    """Run performance tests."""
    print("Running Performance Tests...")
    print("=" * 50)

    test_file = Path(__file__).parent / "performance" / "test_cache_performance.py"

    pytest.main([
        str(test_file),
        "-v",
        "--tb=short",
        "--color=yes",
        "--durations=10"
    ])


def run_all_tests():
    """Run all tests."""
    print("Running All DataManager Tests...")
    print("=" * 50)

    test_dir = Path(__file__).parent

    pytest.main([
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes",
        "--durations=10"
    ])


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run DataManager architecture tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run integration tests only
  python src/data/tests/run_tests.py --integration

  # Run performance tests only
  python src/data/tests/run_tests.py --performance

  # Run all tests
  python src/data/tests/run_tests.py --all
        """
    )

    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run integration tests only'
    )

    parser.add_argument(
        '--performance',
        action='store_true',
        help='Run performance tests only'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests'
    )

    args = parser.parse_args()

    if args.integration:
        run_integration_tests()
    elif args.performance:
        run_performance_tests()
    elif args.all:
        run_all_tests()
    else:
        # Default: run all tests
        run_all_tests()


if __name__ == "__main__":
    main()
