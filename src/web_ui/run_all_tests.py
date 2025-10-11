#!/usr/bin/env python3
"""
Comprehensive Test Runner for Web UI Module
------------------------------------------

This script runs all unit tests for both backend and frontend components
of the web UI module, providing comprehensive coverage reporting and
test result summaries.

Features:
- Backend Python tests with pytest
- Frontend TypeScript tests with Vitest
- Coverage reporting for both environments
- Parallel test execution
- Detailed reporting and summaries
- CI/CD integration support
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestResult:
    """Represents the result of a test run."""

    def __init__(self, name: str, success: bool, duration: float,
                 output: str = "", error: str = "", coverage: Optional[float] = None):
        self.name = name
        self.success = success
        self.duration = duration
        self.output = output
        self.error = error
        self.coverage = coverage


class WebUITestRunner:
    """Comprehensive test runner for the Web UI module."""

    def __init__(self, verbose: bool = False, coverage: bool = True, parallel: bool = True):
        """Initialize the test runner."""
        self.verbose = verbose
        self.coverage = coverage
        self.parallel = parallel
        self.backend_dir = Path(__file__).parent / "backend"
        self.frontend_dir = Path(__file__).parent / "frontend"
        self.results: List[TestResult] = []

    def run_backend_tests(self) -> TestResult:
        """Run backend Python tests with pytest."""
        _logger.info("Starting backend tests...")
        start_time = time.time()

        try:
            # Change to backend directory
            os.chdir(self.backend_dir)

            # Build pytest command
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "-v" if self.verbose else "-q",
                "--tb=short",
                "--strict-markers",
                "--disable-warnings"
            ]

            if self.coverage:
                cmd.extend([
                    "--cov=src.web_ui.backend",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov",
                    "--cov-report=json:coverage.json",
                    "--cov-fail-under=80"
                ])

            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            duration = time.time() - start_time

            # Parse coverage if available
            coverage_percent = None
            if self.coverage and result.returncode == 0:
                coverage_percent = self._parse_backend_coverage()

            return TestResult(
                name="Backend Tests",
                success=result.returncode == 0,
                duration=duration,
                output=result.stdout,
                error=result.stderr,
                coverage=coverage_percent
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                name="Backend Tests",
                success=False,
                duration=duration,
                error="Tests timed out after 5 minutes"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Backend Tests",
                success=False,
                duration=duration,
                error=f"Failed to run backend tests: {e}"
            )

    def run_frontend_tests(self) -> TestResult:
        """Run frontend TypeScript tests with Vitest."""
        _logger.info("Starting frontend tests...")
        start_time = time.time()

        try:
            # Change to frontend directory
            os.chdir(self.frontend_dir)

            # Check if node_modules exists
            if not (self.frontend_dir / "node_modules").exists():
                _logger.warning("Frontend dependencies not installed. Installing...")
                install_result = subprocess.run(
                    ["npm", "install"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if install_result.returncode != 0:
                    raise Exception(f"Failed to install dependencies: {install_result.stderr}")

            # Build test command
            cmd = ["npm", "run", "test"]
            if not self.verbose:
                cmd.append("--reporter=basic")

            if self.coverage:
                cmd.extend(["--coverage", "--coverage.reporter=text", "--coverage.reporter=json"])

            # Set environment variables
            env = os.environ.copy()
            env["CI"] = "true"  # Prevent interactive mode

            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )

            duration = time.time() - start_time

            # Parse coverage if available
            coverage_percent = None
            if self.coverage and result.returncode == 0:
                coverage_percent = self._parse_frontend_coverage()

            return TestResult(
                name="Frontend Tests",
                success=result.returncode == 0,
                duration=duration,
                output=result.stdout,
                error=result.stderr,
                coverage=coverage_percent
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                name="Frontend Tests",
                success=False,
                duration=duration,
                error="Tests timed out after 5 minutes"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Frontend Tests",
                success=False,
                duration=duration,
                error=f"Failed to run frontend tests: {e}"
            )

    def _parse_backend_coverage(self) -> Optional[float]:
        """Parse backend coverage from JSON report."""
        try:
            coverage_file = self.backend_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    return coverage_data.get('totals', {}).get('percent_covered')
        except Exception as e:
            _logger.warning(f"Failed to parse backend coverage: {e}")
        return None

    def _parse_frontend_coverage(self) -> Optional[float]:
        """Parse frontend coverage from JSON report."""
        try:
            coverage_file = self.frontend_dir / "coverage" / "coverage-summary.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    return coverage_data.get('total', {}).get('lines', {}).get('pct')
        except Exception as e:
            _logger.warning(f"Failed to parse frontend coverage: {e}")
        return None

    def run_all_tests(self) -> List[TestResult]:
        """Run all tests (backend and frontend)."""
        _logger.info("Starting comprehensive test run...")

        if self.parallel:
            # Run tests in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(self.run_backend_tests): "backend",
                    executor.submit(self.run_frontend_tests): "frontend"
                }

                results = []
                for future in as_completed(futures):
                    test_type = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        _logger.info(f"{test_type.capitalize()} tests completed: {'✓' if result.success else '✗'}")
                    except Exception as e:
                        _logger.error(f"{test_type.capitalize()} tests failed: {e}")
                        results.append(TestResult(
                            name=f"{test_type.capitalize()} Tests",
                            success=False,
                            duration=0,
                            error=str(e)
                        ))
        else:
            # Run tests sequentially
            results = [
                self.run_backend_tests(),
                self.run_frontend_tests()
            ]

        self.results = results
        return results

    def print_summary(self):
        """Print a comprehensive test summary."""
        print("\n" + "="*80)
        print("WEB UI TEST SUMMARY")
        print("="*80)

        total_duration = sum(r.duration for r in self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        total_tests = len(self.results)

        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Tests Passed: {successful_tests}/{total_tests}")
        print(f"Overall Status: {'✓ PASSED' if successful_tests == total_tests else '✗ FAILED'}")

        print("\nDetailed Results:")
        print("-" * 80)

        for result in self.results:
            status = "✓ PASSED" if result.success else "✗ FAILED"
            coverage_info = f" (Coverage: {result.coverage:.1f}%)" if result.coverage else ""

            print(f"{result.name}: {status} ({result.duration:.2f}s){coverage_info}")

            if not result.success and result.error:
                print(f"  Error: {result.error}")

            if self.verbose and result.output:
                print(f"  Output: {result.output[:200]}...")

        # Coverage summary
        if self.coverage:
            print("\nCoverage Summary:")
            print("-" * 40)
            for result in self.results:
                if result.coverage is not None:
                    status = "✓" if result.coverage >= 80 else "⚠"
                    print(f"{result.name}: {status} {result.coverage:.1f}%")

        print("\n" + "="*80)

    def generate_junit_report(self, output_file: str = "test-results.xml"):
        """Generate JUnit XML report for CI/CD integration."""
        try:
            from xml.etree.ElementTree import Element, SubElement, tostring
            from xml.dom import minidom

            testsuites = Element("testsuites")
            testsuites.set("tests", str(len(self.results)))
            testsuites.set("failures", str(sum(1 for r in self.results if not r.success)))
            testsuites.set("time", str(sum(r.duration for r in self.results)))

            for result in self.results:
                testsuite = SubElement(testsuites, "testsuite")
                testsuite.set("name", result.name)
                testsuite.set("tests", "1")
                testsuite.set("failures", "0" if result.success else "1")
                testsuite.set("time", str(result.duration))

                testcase = SubElement(testsuite, "testcase")
                testcase.set("name", result.name)
                testcase.set("time", str(result.duration))

                if not result.success:
                    failure = SubElement(testcase, "failure")
                    failure.set("message", "Test failed")
                    failure.text = result.error

            # Pretty print XML
            rough_string = tostring(testsuites, 'utf-8')
            reparsed = minidom.parseString(rough_string)

            with open(output_file, 'w') as f:
                f.write(reparsed.toprettyxml(indent="  "))

            _logger.info(f"JUnit report generated: {output_file}")

        except ImportError:
            _logger.warning("xml.etree.ElementTree not available, skipping JUnit report")
        except Exception as e:
            _logger.error(f"Failed to generate JUnit report: {e}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Web UI Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--sequential", action="store_true", help="Run tests sequentially")
    parser.add_argument("--backend-only", action="store_true", help="Run only backend tests")
    parser.add_argument("--frontend-only", action="store_true", help="Run only frontend tests")
    parser.add_argument("--junit", help="Generate JUnit XML report")

    args = parser.parse_args()

    # Initialize test runner
    runner = WebUITestRunner(
        verbose=args.verbose,
        coverage=not args.no_coverage,
        parallel=not args.sequential
    )

    try:
        # Run specific test suites
        if args.backend_only:
            results = [runner.run_backend_tests()]
        elif args.frontend_only:
            results = [runner.run_frontend_tests()]
        else:
            results = runner.run_all_tests()

        runner.results = results

        # Print summary
        runner.print_summary()

        # Generate JUnit report if requested
        if args.junit:
            runner.generate_junit_report(args.junit)

        # Exit with appropriate code
        success = all(r.success for r in results)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        _logger.info("Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        _logger.error(f"Test run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()