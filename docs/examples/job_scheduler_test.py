#!/usr/bin/env python3
"""
Job Scheduler System Test

Simple test script to verify the job scheduler system is working correctly.
This script tests the basic functionality without requiring a full setup.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Removed imports - these modules don't exist or have been moved


def test_screener_config():
    """Test screener configuration loading."""
    print("Testing screener configuration...")

    try:
        config = get_screener_config()

        # Test listing available sets
        sets = config.list_available_sets()
        print(f"✓ Found {len(sets)} screener sets")

        if sets:
            # Test getting tickers for first set
            first_set = sets[0]
            tickers = config.get_tickers(first_set)
            print(f"✓ Set '{first_set}' has {len(tickers)} tickers")

            # Test getting set info
            set_info = config.get_set_info(first_set)
            print(f"✓ Set info retrieved: {set_info.get('description', 'No description')}")

        print("✓ Screener configuration test passed\n")
        return True

    except Exception as e:
        print(f"✗ Screener configuration test failed: {e}\n")
        return False


def test_cron_handler():
    """Test cron expression handling."""
    print("Testing cron handler...")

    try:
        # Test common cron expressions
        test_crons = [
            "0 9 * * *",  # Daily at 9 AM
            "*/15 * * * *",  # Every 15 minutes
            "0 0 1 * *",  # Monthly on 1st
            "invalid_cron"  # Invalid expression
        ]

        for cron in test_crons:
            result = validate_and_describe_cron(cron)
            print(f"  {cron}: {result['is_valid']} - {result['description']}")

            if result['is_valid'] and result['next_run']:
                print(f"    Next run: {result['next_run']}")

        # Test cron validation
        assert CronHandler.validate_cron("0 9 * * *") == True
        assert CronHandler.validate_cron("invalid") == False

        # Test next run calculation
        next_run = CronHandler.calculate_next_run_time("0 9 * * *")
        assert isinstance(next_run, datetime)

        print("✓ Cron handler test passed\n")
        return True

    except Exception as e:
        print(f"✗ Cron handler test failed: {e}\n")
        return False


def test_database_models():
    """Test database models import."""
    print("Testing database models...")

    try:
        from src.data.db.models.model_jobs import (
            JobType, RunStatus,
            ScheduleCreate, ScheduleRunCreate
        )

        # Test enum values
        assert JobType.REPORT.value == "report"
        assert JobType.SCREENER.value == "screener"
        assert RunStatus.PENDING.value == "pending"

        # Test Pydantic models
        schedule_data = ScheduleCreate(
            name="Test Schedule",
            job_type=JobType.REPORT,
            target="system_status",
            cron="0 9 * * *",
            task_params={"param1": "value1"}
        )

        run_data = ScheduleRunCreate(
            job_type=JobType.SCREENER,
            job_id="test_job_123",
            scheduled_for=datetime.now(timezone.utc),
            job_snapshot={"tickers": ["AAPL", "MSFT"]}
        )

        print("✓ Database models test passed\n")
        return True

    except Exception as e:
        print(f"✗ Database models test failed: {e}\n")
        return False


def test_workers_import():
    """Test workers import."""
    print("Testing workers import...")

    try:
        # from src.backend.workers import (
        #     broker, setup_dramatiq,
        #     run_report, run_screener
        #)

        # Test that workers are properly configured
        assert broker is not None
        assert callable(run_report)
        assert callable(run_screener)

        print("✓ Workers import test passed\n")
        return True

    except Exception as e:
        print(f"✗ Workers import test failed: {e}\n")
        return False


def test_api_models():
    """Test API models import."""
    print("Testing API models...")

    try:
        from src.data.db.models.model_jobs import (
            ReportRequest, ScreenerRequest
        )

        # Test request models
        report_req = ReportRequest(
            report_type="system_status",
            parameters={"include_details": True}
        )

        screener_req = ScreenerRequest(
            screener_set="us_large_caps",
            filter_criteria={"market_cap_min": 1000000000},
            top_n=20
        )

        print("✓ API models test passed\n")
        return True

    except Exception as e:
        print(f"✗ API models test failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("Job Scheduler System Test")
    print("=" * 50)

    tests = [
        test_screener_config,
        test_cron_handler,
        test_database_models,
        test_workers_import,
        test_api_models
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! Job scheduler system is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


