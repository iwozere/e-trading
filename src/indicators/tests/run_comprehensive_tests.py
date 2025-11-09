#!/usr/bin/env python3
"""
Comprehensive test runner for indicator service consolidation.

This script runs all test suites and generates a comprehensive report
of test coverage, performance benchmarks, and integration results.
"""

import sys
import subprocess
import time
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))


def run_test_suite(test_file, description, verbose=False):
    """Run a specific test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print(f"{'='*60}")

    start_time = time.time()

    cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
    if verbose:
        cmd.append("-s")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        end_time = time.time()

        print(f"Exit code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.2f} seconds")

        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        return {
            'name': description,
            'file': test_file,
            'exit_code': result.returncode,
            'duration': end_time - start_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }

    except Exception as e:
        print(f"Error running test: {e}")
        return {
            'name': description,
            'file': test_file,
            'exit_code': -1,
            'duration': 0,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }


def main():
    """Run comprehensive test suite."""
    parser = argparse.ArgumentParser(description='Run comprehensive indicator service tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', '-q', action='store_true', help='Skip performance benchmarks')
    parser.add_argument('--suite', '-s', choices=['unit', 'integration', 'migration', 'performance', 'backtrader'],
                       help='Run specific test suite only')

    args = parser.parse_args()

    # Define test suites
    test_suites = [
        {
            'file': 'src/indicators/tests/test_core_functionality.py',
            'description': 'Core Functionality Unit Tests',
            'category': 'unit'
        },
        {
            'file': 'src/indicators/tests/test_config_validation.py',
            'description': 'Configuration Validation Tests',
            'category': 'unit'
        },
        {
            'file': 'src/indicators/tests/test_batch_processing.py',
            'description': 'Batch Processing Tests',
            'category': 'unit'
        },
        {
            'file': 'src/indicators/tests/test_adapters.py',
            'description': 'Adapter Unit Tests',
            'category': 'unit'
        },
        {
            'file': 'src/indicators/tests/test_adapter_integration.py',
            'description': 'Adapter Integration Tests',
            'category': 'integration'
        },
        {
            'file': 'src/indicators/tests/test_error_handling_fallbacks.py',
            'description': 'Error Handling and Fallback Tests',
            'category': 'integration'
        },
        {
            'file': 'src/indicators/tests/test_migration_compatibility.py',
            'description': 'Migration Compatibility Tests',
            'category': 'migration'
        },
        {
            'file': 'src/indicators/tests/test_performance_benchmarks.py',
            'description': 'Performance Benchmarks',
            'category': 'performance'
        },
        {
            'file': 'src/indicators/adapters/tests/test_backtrader_integration.py',
            'description': 'Backtrader Integration Tests',
            'category': 'backtrader'
        }
    ]

    # Filter test suites based on arguments
    if args.suite:
        test_suites = [suite for suite in test_suites if suite['category'] == args.suite]

    if args.quick:
        test_suites = [suite for suite in test_suites if suite['category'] != 'performance']

    print("Comprehensive Indicator Service Test Suite")
    print("=" * 50)
    print(f"Running {len(test_suites)} test suite(s)")

    results = []
    total_start_time = time.time()

    for suite in test_suites:
        result = run_test_suite(suite['file'], suite['description'], args.verbose)
        result['category'] = suite['category']
        results.append(result)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Generate summary report
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")

    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]

    print(f"Total test suites: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Total duration: {total_duration:.2f} seconds")

    # Category breakdown
    categories = {}
    for result in results:
        category = result['category']
        if category not in categories:
            categories[category] = {'total': 0, 'passed': 0, 'failed': 0}

        categories[category]['total'] += 1
        if result['success']:
            categories[category]['passed'] += 1
        else:
            categories[category]['failed'] += 1

    print("\nResults by category:")
    for category, stats in categories.items():
        print(f"  {category:12s}: {stats['passed']}/{stats['total']} passed")

    # Detailed results
    print("\nDetailed Results:")
    print(f"{'Test Suite':<40} {'Status':<10} {'Duration':<10}")
    print("-" * 60)

    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        duration = f"{result['duration']:.2f}s"
        name = result['name'][:37] + "..." if len(result['name']) > 40 else result['name']
        print(f"{name:<40} {status:<10} {duration:<10}")

    # Failed test details
    if failed_tests:
        print("\nFailed Test Details:")
        print("-" * 60)
        for result in failed_tests:
            print(f"\nTest: {result['name']}")
            print(f"File: {result['file']}")
            print(f"Exit code: {result['exit_code']}")
            if result['stderr']:
                print(f"Error: {result['stderr'][:200]}...")

    # Performance summary (if performance tests were run)
    perf_results = [r for r in results if r['category'] == 'performance' and r['success']]
    if perf_results:
        print("\nPerformance Test Summary:")
        print("-" * 30)
        for result in perf_results:
            print(f"Performance benchmarks completed in {result['duration']:.2f}s")

    # Exit with appropriate code
    exit_code = 0 if len(failed_tests) == 0 else 1

    print(f"\nTest suite completed with exit code: {exit_code}")
    return exit_code


if __name__ == '__main__':
    sys.exit(main())