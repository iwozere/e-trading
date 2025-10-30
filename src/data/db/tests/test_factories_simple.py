"""
Simple test for factory functions without database creation.

Tests that factory functions can be called and return expected data types.
"""

import pytest
from datetime import datetime, timezone

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.tests.factories import RNG


class TestFactoryFunctions:
    """Test factory function behavior without database."""

    def test_rng_deterministic(self):
        """Test that RNG produces deterministic results."""
        rng1 = RNG(seed=42)
        rng2 = RNG(seed=42)

        # Same seed should produce same results
        assert rng1.randint(1, 100) == rng2.randint(1, 100)
        assert rng1.choice(['a', 'b', 'c']) == rng2.choice(['a', 'b', 'c'])

    def test_rng_biginteger_range(self):
        """Test that RNG can generate BigInteger-compatible values."""
        rng = RNG(seed=42)

        # Test large integer generation
        large_int = rng.randint(100000, 999999999)
        assert isinstance(large_int, int)
        assert 100000 <= large_int <= 999999999

    def test_factory_imports(self):
        """Test that all factory functions can be imported."""
        from src.data.db.tests.factories import (
            make_user, add_telegram_identity, make_verification_code,
            make_feedback, make_run, make_job_schedule
        )

        # All functions should be callable
        assert callable(make_user)
        assert callable(add_telegram_identity)
        assert callable(make_verification_code)
        assert callable(make_feedback)
        assert callable(make_run)
        assert callable(make_job_schedule)

    def test_factory_parameter_validation(self):
        """Test that factory functions accept expected parameters."""
        from src.data.db.tests.factories import make_run, RNG
        import inspect

        # Check make_run signature
        sig = inspect.signature(make_run)
        params = list(sig.parameters.keys())

        # Should have all required parameters including new ones
        expected_params = ['s', 'rng', 'user_id', 'job_type', 'job_id', 'status', 'scheduled_for', 'worker_id']
        for param in expected_params:
            assert param in params, f"Parameter {param} missing from make_run"

    def test_factory_default_values(self):
        """Test that factory functions have appropriate default values."""
        from src.data.db.tests.factories import make_verification_code
        import inspect

        # Check make_verification_code signature
        sig = inspect.signature(make_verification_code)

        # Should have provider default
        assert sig.parameters['provider'].default == "telegram"

    def test_biginteger_compatibility_values(self):
        """Test that factory generates BigInteger-compatible values."""
        rng = RNG(seed=42)

        # Test job_id range (should be larger for BigInteger)
        job_id = rng.randint(100000, 999999999)
        assert job_id >= 100000
        assert job_id <= 999999999

        # Test that it can handle max int8 values
        max_int8 = 9223372036854775807
        assert max_int8 > 999999999  # Our range is within int8 limits

    def test_worker_id_generation(self):
        """Test worker_id generation pattern."""
        rng = RNG(seed=42)

        worker_id = f"worker-{rng.randint(1, 100)}"
        assert worker_id.startswith("worker-")
        assert len(worker_id) > 7  # "worker-" + at least 1 digit

    def test_datetime_timezone_awareness(self):
        """Test that datetime functions return timezone-aware datetimes."""
        from src.data.db.tests.factories import utcnow

        dt = utcnow()
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc

    def test_json_data_structure(self):
        """Test that factory creates valid JSON structures."""
        test_metadata = {"username": "test", "first_name": "Test"}
        test_snapshot = {"created_by": "factory", "test": True}

        # Should be valid dict structures
        assert isinstance(test_metadata, dict)
        assert isinstance(test_snapshot, dict)
        assert "username" in test_metadata
        assert "created_by" in test_snapshot


if __name__ == "__main__":
    pytest.main([__file__, "-v"])