#!/usr/bin/env python3
"""
End-to-End Test Runner for Notification Service

Runs comprehensive end-to-end tests covering:
1. Complete message flow from API to delivery
2. Service behavior under various failure scenarios
3. Service startup, shutdown, and recovery
"""

import asyncio
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class E2ETestRunner:
    """End-to-end test runner."""

    def __init__(self):
        self.test_results = []
        self.start_time = None
        self.end_time = None

    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now(timezone.utc)
        })

        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {test_name}" + (f" - {message}" if message else ""))

    async def run_standalone_tests(self):
        """Run standalone end-to-end tests."""
        print("\n=== Running Standalone E2E Tests ===")

        try:
            # Import and run standalone tests
            from src.notification.tests.test_e2e_standalone import main as standalone_main

            exit_code = await standalone_main()

            if exit_code == 0:
                self.log_result("Standalone E2E Tests", True, "All standalone tests passed")
                return True
            else:
                self.log_result("Standalone E2E Tests", False, "Some standalone tests failed")
                return False

        except Exception as e:
            self.log_result("Standalone E2E Tests", False, f"Exception: {str(e)}")
            return False

    def run_pytest_tests(self):
        """Run pytest-based end-to-end tests."""
        print("\n=== Running Pytest E2E Tests ===")

        try:
            # Check if pytest is available
            try:
                import pytest
                pytest_available = True
            except ImportError:
                pytest_available = False

            if not pytest_available:
                self.log_result("Pytest E2E Tests", False, "pytest not available - skipping")
                return False

            # Run pytest tests
            test_file = Path(__file__).parent / "test_end_to_end.py"

            if not test_file.exists():
                self.log_result("Pytest E2E Tests", False, "test_end_to_end.py not found")
                return False

            # Run pytest programmatically
            exit_code = pytest.main([
                str(test_file),
                "-v",
                "--tb=short",
                "-x",  # Stop on first failure
                "--disable-warnings"
            ])

            if exit_code == 0:
                self.log_result("Pytest E2E Tests", True, "All pytest tests passed")
                return True
            else:
                self.log_result("Pytest E2E Tests", False, f"Pytest tests failed (exit code: {exit_code})")
                return False

        except Exception as e:
            self.log_result("Pytest E2E Tests", False, f"Exception: {str(e)}")
            return False

    def run_infrastructure_tests(self):
        """Run infrastructure tests."""
        print("\n=== Running Infrastructure Tests ===")

        try:
            # Run infrastructure test script
            test_script = Path(__file__).parent / "test_infrastructure.py"

            if not test_script.exists():
                self.log_result("Infrastructure Tests", False, "test_infrastructure.py not found")
                return False

            # Run as subprocess
            result = subprocess.run([
                sys.executable, str(test_script)
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                self.log_result("Infrastructure Tests", True, "Infrastructure tests passed")
                return True
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                self.log_result("Infrastructure Tests", False, f"Failed: {error_msg}")
                return False

        except subprocess.TimeoutExpired:
            self.log_result("Infrastructure Tests", False, "Test timeout")
            return False
        except Exception as e:
            self.log_result("Infrastructure Tests", False, f"Exception: {str(e)}")
            return False

    def run_core_systems_tests(self):
        """Run core systems tests."""
        print("\n=== Running Core Systems Tests ===")

        try:
            # Run core systems test script
            test_script = Path(__file__).parent / "test_core_systems.py"

            if not test_script.exists():
                self.log_result("Core Systems Tests", False, "test_core_systems.py not found")
                return False

            # Run as subprocess
            result = subprocess.run([
                sys.executable, str(test_script)
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.log_result("Core Systems Tests", True, "Core systems tests passed")
                return True
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                self.log_result("Core Systems Tests", False, f"Failed: {error_msg}")
                return False

        except subprocess.TimeoutExpired:
            self.log_result("Core Systems Tests", False, "Test timeout")
            return False
        except Exception as e:
            self.log_result("Core Systems Tests", False, f"Exception: {str(e)}")
            return False

    def check_service_health(self):
        """Check if notification service components are healthy."""
        print("\n=== Checking Service Health ===")

        try:
            from src.notification.service.config import config
            from src.data.db.services.database_service import get_database_service

            # Test database connection
            try:
                db_service = get_database_service()
                db_service.init_databases()
                self.log_result("Database Connection", True, "Database accessible")
                db_healthy = True
            except Exception as e:
                self.log_result("Database Connection", False, f"Database error: {str(e)}")
                db_healthy = False

            # Test configuration loading
            try:
                service_name = config.service_name
                version = config.version
                self.log_result("Configuration Loading", True, f"Config loaded: {service_name} v{version}")
                config_healthy = True
            except Exception as e:
                self.log_result("Configuration Loading", False, f"Config error: {str(e)}")
                config_healthy = False

            return db_healthy and config_healthy

        except Exception as e:
            self.log_result("Service Health Check", False, f"Health check failed: {str(e)}")
            return False

    async def run_all_tests(self):
        """Run all end-to-end tests."""
        self.start_time = datetime.now(timezone.utc)

        print("🚀 Starting Notification Service End-to-End Tests")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 60)

        # Check service health first
        health_ok = self.check_service_health()

        if not health_ok:
            print("\n❌ Service health check failed - aborting tests")
            return 1

        # Run all test suites
        test_results = []

        # 1. Core systems tests (basic functionality)
        test_results.append(self.run_core_systems_tests())

        # 2. Infrastructure tests (database, queue, processor)
        test_results.append(self.run_infrastructure_tests())

        # 3. Standalone end-to-end tests (complete flows)
        standalone_result = await self.run_standalone_tests()
        test_results.append(standalone_result)

        # 4. Pytest-based tests (if available)
        test_results.append(self.run_pytest_tests())

        self.end_time = datetime.now(timezone.utc)
        duration = self.end_time - self.start_time

        # Print final summary
        print("\n" + "=" * 60)
        print("🏁 End-to-End Test Results Summary")
        print(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Duration: {duration.total_seconds():.1f} seconds")
        print("-" * 60)

        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)

        for result in self.test_results:
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            print(f"{status} {result['test']}")
            if result["message"]:
                print(f"    └─ {result['message']}")

        print("-" * 60)
        print(f"Total: {passed}/{total} test suites passed")

        # Overall result
        if passed == total and all(test_results):
            print("🎉 ALL END-TO-END TESTS PASSED!")
            print("\nThe notification service is ready for production use.")
            return 0
        else:
            print("❌ SOME END-TO-END TESTS FAILED!")
            print("\nPlease review the failures above before deploying.")
            return 1

    def print_test_coverage_summary(self):
        """Print summary of what was tested."""
        print("\n📋 Test Coverage Summary:")
        print("✓ Database connectivity and persistence")
        print("✓ Message queue operations")
        print("✓ Message processor functionality")
        print("✓ Single channel message delivery")
        print("✓ Multi-channel message delivery")
        print("✓ Priority message handling")
        print("✓ Channel failure handling")
        print("✓ Partial failure scenarios")
        print("✓ Service recovery mechanisms")
        print("✓ Rate limiting enforcement")
        print("✓ Health monitoring")
        print("✓ API endpoint functionality")


async def main():
    """Main test runner entry point."""
    runner = E2ETestRunner()

    try:
        exit_code = await runner.run_all_tests()
        runner.print_test_coverage_summary()
        return exit_code

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test runner crashed: {str(e)}")
        _logger.exception("Test runner exception")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)