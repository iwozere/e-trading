#!/usr/bin/env python3
"""
Test Runner for Data Downloaders and Live Feeds

This script runs comprehensive tests for all data downloaders and live feeds
with detailed reporting and coverage analysis.

Usage:
    python run_data_tests.py [--verbose] [--coverage] [--downloaders] [--live-feeds]

Examples:
    python run_data_tests.py                    # Run all tests
    python run_data_tests.py --downloaders      # Run only data downloader tests
    python run_data_tests.py --live-feeds       # Run only live feed tests
    python run_data_tests.py --verbose          # Run with verbose output
    python run_data_tests.py --coverage         # Run with coverage report
"""

import sys
import os
import time
import argparse
import subprocess
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_tests(test_pattern, verbose=False, coverage=False):
    """
    Run tests with the specified pattern.

    Args:
        test_pattern: Test pattern to run
        verbose: Whether to run with verbose output
        coverage: Whether to run with coverage

    Returns:
        bool: True if all tests passed, False otherwise
    """
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=src/data", "--cov-report=html", "--cov-report=term-missing"])

    cmd.append(test_pattern)

    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=not verbose)
    end_time = time.time()

    duration = end_time - start_time

    if result.returncode == 0:
        print(f"✓ Tests passed in {duration:.2f} seconds")
        return True
    else:
        print(f"✗ Tests failed in {duration:.2f} seconds")
        if not verbose and result.stdout:
            print("STDOUT:")
            print(result.stdout.decode())
        if result.stderr:
            print("STDERR:")
            print(result.stderr.decode())
        return False


def run_specific_test(test_file, test_class=None, test_method=None, verbose=False):
    """
    Run a specific test or test class.

    Args:
        test_file: Test file to run
        test_class: Specific test class (optional)
        test_method: Specific test method (optional)
        verbose: Whether to run with verbose output

    Returns:
        bool: True if test passed, False otherwise
    """
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if test_class:
        if test_method:
            cmd.append(f"{test_file}::{test_class}::{test_method}")
        else:
            cmd.append(f"{test_file}::{test_class}")
    else:
        cmd.append(test_file)

    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=not verbose)
    end_time = time.time()

    duration = end_time - start_time

    if result.returncode == 0:
        print(f"✓ Test passed in {duration:.2f} seconds")
        return True
    else:
        print(f"✗ Test failed in {duration:.2f} seconds")
        if not verbose and result.stdout:
            print("STDOUT:")
            print(result.stdout.decode())
        if result.stderr:
            print("STDERR:")
            print(result.stderr.decode())
        return False


def run_data_downloader_tests(verbose=False, coverage=False):
    """Run all data downloader tests."""
    print("Running Data Downloader Tests")
    print("=" * 80)

    tests = [
        "tests/test_data_downloaders.py",
        "tests/test_base_data_downloader.py"
    ]

    all_passed = True
    for test in tests:
        if os.path.exists(test):
            print(f"\nRunning {test}...")
            if not run_tests(test, verbose, coverage):
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test}")

    return all_passed


def run_live_feed_tests(verbose=False, coverage=False):
    """Run all live feed tests."""
    print("Running Live Feed Tests")
    print("=" * 80)

    tests = [
        "tests/test_live_feeds.py",
        "tests/test_live_data_feeds.py"
    ]

    all_passed = True
    for test in tests:
        if os.path.exists(test):
            print(f"\nRunning {test}...")
            if not run_tests(test, verbose, coverage):
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test}")

    return all_passed


def run_factory_tests(verbose=False, coverage=False):
    """Run factory tests."""
    print("Running Factory Tests")
    print("=" * 80)

    tests = [
        "test_data_downloader_factory.py"
    ]

    all_passed = True
    for test in tests:
        if os.path.exists(test):
            print(f"\nRunning {test}...")
            if not run_tests(test, verbose, coverage):
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test}")

    return all_passed


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    print("Running All Data Tests")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run data downloader tests
    downloader_passed = run_data_downloader_tests(verbose, coverage)

    # Run live feed tests
    live_feed_passed = run_live_feed_tests(verbose, coverage)

    # Run factory tests
    factory_passed = run_factory_tests(verbose, coverage)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Data Downloaders: {'✓ PASSED' if downloader_passed else '✗ FAILED'}")
    print(f"Live Feeds:       {'✓ PASSED' if live_feed_passed else '✗ FAILED'}")
    print(f"Factories:        {'✓ PASSED' if factory_passed else '✗ FAILED'}")
    print()

    all_passed = downloader_passed and live_feed_passed and factory_passed
    print(f"Overall Result:   {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"Completed at:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run data downloader and live feed tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Run with verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage report")
    parser.add_argument("--downloaders", "-d", action="store_true", help="Run only data downloader tests")
    parser.add_argument("--live-feeds", "-l", action="store_true", help="Run only live feed tests")
    parser.add_argument("--factories", "-f", action="store_true", help="Run only factory tests")
    parser.add_argument("--test-file", help="Run specific test file")
    parser.add_argument("--test-class", help="Run specific test class")
    parser.add_argument("--test-method", help="Run specific test method")

    args = parser.parse_args()

    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed. Please install it with:")
        print("pip install pytest pytest-cov")
        return 1

    # Run specific test if requested
    if args.test_file:
        success = run_specific_test(args.test_file, args.test_class, args.test_method, args.verbose)
        return 0 if success else 1

    # Run specific test categories
    if args.downloaders:
        success = run_data_downloader_tests(args.verbose, args.coverage)
    elif args.live_feeds:
        success = run_live_feed_tests(args.verbose, args.coverage)
    elif args.factories:
        success = run_factory_tests(args.verbose, args.coverage)
    else:
        # Run all tests
        success = run_all_tests(args.verbose, args.coverage)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
