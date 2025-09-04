"""
Phase 4 Test Runner.

Comprehensive test runner for Phase 4 that executes all unit tests,
integration tests, and performance benchmarks.
"""

import sys
import time
import argparse
from pathlib import Path
import unittest
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def run_unit_tests():
    """Run all unit tests."""
    print("Running Unit Tests...")
    print("=" * 50)

    # Discover and run unit tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "unit"
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\nUnit Tests Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result


def run_integration_tests():
    """Run all integration tests."""
    print("\nRunning Integration Tests...")
    print("=" * 50)

    # Discover and run integration tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "integration"
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\nIntegration Tests Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result


def run_performance_tests():
    """Run all performance tests."""
    print("\nRunning Performance Tests...")
    print("=" * 50)

    # Discover and run performance tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "performance"
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\nPerformance Tests Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result


def run_specific_test(test_path):
    """Run a specific test file."""
    print(f"Running specific test: {test_path}")
    print("=" * 50)

    # Import and run the specific test
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_module", test_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    # Find and run the test function if it exists
    if hasattr(test_module, 'run_phase4_integration_tests'):
        return test_module.run_phase4_integration_tests()
    elif hasattr(test_module, 'run_performance_benchmarks'):
        return test_module.run_performance_benchmarks()
    else:
        # Run as regular unittest
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return len(result.failures) == 0 and len(result.errors) == 0


def generate_test_report(unit_result, integration_result, performance_result):
    """Generate a comprehensive test report."""
    print("\n" + "=" * 80)
    print("PHASE 4 COMPREHENSIVE TEST REPORT")
    print("=" * 80)

    # Calculate totals
    total_tests = unit_result.testsRun + integration_result.testsRun + performance_result.testsRun
    total_failures = len(unit_result.failures) + len(integration_result.failures) + len(performance_result.failures)
    total_errors = len(unit_result.errors) + len(integration_result.errors) + len(performance_result.errors)
    total_success = total_tests - total_failures - total_errors

    # Overall success rate
    success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0

    print(f"Overall Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Successful: {total_success}")
    print(f"  Failures: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Success Rate: {success_rate:.1f}%")

    print(f"\nBreakdown by Test Type:")
    print(f"  Unit Tests: {unit_result.testsRun} run, {len(unit_result.failures)} failures, {len(unit_result.errors)} errors")
    print(f"  Integration Tests: {integration_result.testsRun} run, {len(integration_result.failures)} failures, {len(integration_result.errors)} errors")
    print(f"  Performance Tests: {performance_result.testsRun} run, {len(performance_result.failures)} failures, {len(performance_result.errors)} errors")

    # Detailed failure/error reporting
    if total_failures > 0 or total_errors > 0:
        print(f"\nDetailed Issues:")

        if unit_result.failures:
            print(f"  Unit Test Failures:")
            for test, traceback in unit_result.failures:
                print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")

        if unit_result.errors:
            print(f"  Unit Test Errors:")
            for test, traceback in unit_result.errors:
                print(f"    - {test}: {traceback.split('Exception:')[-1].strip()}")

        if integration_result.failures:
            print(f"  Integration Test Failures:")
            for test, traceback in integration_result.failures:
                print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")

        if integration_result.errors:
            print(f"  Integration Test Errors:")
            for test, traceback in integration_result.errors:
                print(f"    - {test}: {traceback.split('Exception:')[-1].strip()}")

        if performance_result.failures:
            print(f"  Performance Test Failures:")
            for test, traceback in performance_result.failures:
                print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")

        if performance_result.errors:
            print(f"  Performance Test Errors:")
            for test, traceback in performance_result.errors:
                print(f"    - {test}: {traceback.split('Exception:')[-1].strip()}")

    # Phase 4 specific checks
    print(f"\nPhase 4 Specific Checks:")

    # Check if file-based cache is working
    try:
        from src.data import FileBasedCache, get_file_cache
        print(f"  ✓ File-based cache system: Available")
    except ImportError as e:
        print(f"  ✗ File-based cache system: Import error - {e}")

    # Check if Redis dependency is removed
    try:
        from src.data import RedisCache
        print(f"  ✗ Redis dependency: Still present (should be removed)")
    except ImportError:
        print(f"  ✓ Redis dependency: Successfully removed")

    # Test enhanced CSV cache functionality
    print(f"\nEnhanced CSV Cache Functionality:")
    try:
        from src.data.utils.file_based_cache import (
            CSVFormatConventions, SafeCSVAppender, SmartDataAppender, CacheMetadata
        )
        print(f"  ✓ CSV Format Conventions: Available")
        print(f"  ✓ Safe CSV Appender: Available")
        print(f"  ✓ Smart Data Appender: Available")
        print(f"  ✓ Enhanced Cache Metadata: Available")

        # Test basic functionality
        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='h'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        })

        # Test CSV validation
        is_valid = CSVFormatConventions.validate_dataframe(test_df)
        print(f"  ✓ CSV Validation: {'Working' if is_valid else 'Failed'}")

        # Test standardization
        standardized_df = CSVFormatConventions.standardize_dataframe(test_df, 'test_provider')
        has_provider_ts = 'provider_download_ts' in standardized_df.columns
        print(f"  ✓ CSV Standardization: {'Working' if has_provider_ts else 'Failed'}")

    except ImportError as e:
        print(f"  ✗ Enhanced CSV Cache: Import failed - {e}")
    except Exception as e:
        print(f"  ✗ Enhanced CSV Cache: Test failed - {e}")

    # Check test organization
    test_dirs = ["unit", "integration", "performance"]
    for test_dir in test_dirs:
        test_path = Path(__file__).parent / test_dir
        if test_path.exists():
            test_files = list(test_path.glob("test_*.py"))
            print(f"  ✓ {test_dir.capitalize()} tests: {len(test_files)} files found")

            # Check for enhanced CSV cache tests
            if test_dir == "unit":
                enhanced_csv_tests = [f for f in test_files if "enhanced_csv" in f.name]
                if enhanced_csv_tests:
                    print(f"    - Enhanced CSV Cache tests: {len(enhanced_csv_tests)} files found")
                else:
                    print(f"    - Enhanced CSV Cache tests: Not found")
        else:
            print(f"  ✗ {test_dir.capitalize()} tests: Directory not found")

    # Final verdict
    print(f"\nFinal Verdict:")
    if success_rate >= 95:
        print(f"  🟢 EXCELLENT: Phase 4 implementation is ready for production")
    elif success_rate >= 80:
        print(f"  🟡 GOOD: Phase 4 implementation needs minor fixes")
    elif success_rate >= 60:
        print(f"  🟠 FAIR: Phase 4 implementation needs significant work")
    else:
        print(f"  🔴 POOR: Phase 4 implementation needs major fixes")

    return total_failures == 0 and total_errors == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Phase 4 Test Runner")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--specific-test", type=str, help="Run a specific test file")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")

    args = parser.parse_args()

    start_time = time.time()

    try:
        if args.specific_test:
            # Run specific test
            test_path = Path(args.specific_test)
            if not test_path.exists():
                print(f"Error: Test file {test_path} not found")
                return 1

            success = run_specific_test(test_path)
        elif args.unit_only:
            # Run only unit tests
            unit_result = run_unit_tests()
            success = len(unit_result.failures) == 0 and len(unit_result.errors) == 0
        elif args.integration_only:
            # Run only integration tests
            integration_result = run_integration_tests()
            success = len(integration_result.failures) == 0 and len(integration_result.errors) == 0
        elif args.performance_only:
            # Run only performance tests
            performance_result = run_performance_tests()
            success = len(performance_result.failures) == 0 and len(performance_result.errors) == 0
        else:
            # Run all tests
            unit_result = run_unit_tests()
            integration_result = run_integration_tests()

            if args.skip_performance:
                performance_result = type('MockResult', (), {
                    'testsRun': 0,
                    'failures': [],
                    'errors': []
                })()
            else:
                performance_result = run_performance_tests()

            success = generate_test_report(unit_result, integration_result, performance_result)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\nTotal execution time: {total_time:.2f} seconds")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error during test execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
