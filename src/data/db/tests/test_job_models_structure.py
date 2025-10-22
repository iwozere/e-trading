"""
Job Models Structure Tests

Tests the corrected job models structure to ensure they match
the PostgreSQL database schema exactly, without requiring database operations.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime, timezone
from sqlalchemy import inspect, BigInteger, Text, String
from sqlalchemy.dialects.postgresql import JSONB

from src.data.db.models.model_jobs import Schedule, ScheduleRun


class TestJobModelStructure:
    """Test the structure of job models to ensure they match PostgreSQL schema."""

    def test_schedule_table_name(self):
        """Test that Schedule model has correct table name."""
        assert Schedule.__tablename__ == "job_schedules"

    def test_schedule_column_types(self):
        """Test that Schedule model has correct column types."""
        columns = Schedule.__table__.columns

        # Check key column types
        assert columns['id'].type.__class__.__name__ == 'Integer'
        assert columns['user_id'].type.__class__.__name__ == 'Integer'
        assert columns['name'].type.__class__.__name__ == 'String'
        assert columns['job_type'].type.__class__.__name__ == 'String'
        assert columns['target'].type.__class__.__name__ == 'String'
        # Check JSONB type for PostgreSQL
        assert columns['task_params'].type.__class__.__name__ == 'JSONB'
        assert columns['cron'].type.__class__.__name__ == 'String'
        assert columns['enabled'].type.__class__.__name__ == 'Boolean'

    def test_run_table_name(self):
        """Test that Run model has correct table name."""
        assert Run.__tablename__ == "job_runs"

    def test_run_column_types_corrected(self):
        """Test that Run model has corrected column types matching PostgreSQL."""
        columns = Run.__table__.columns

        # Check corrected column types
        assert columns['job_type'].type.__class__.__name__ == 'Text'  # Changed from String(50)
        assert columns['job_id'].type.__class__.__name__ == 'BigInteger'  # Changed from String(255)
        assert columns['user_id'].type.__class__.__name__ == 'BigInteger'  # Changed from Integer
        assert columns['status'].type.__class__.__name__ == 'Text'  # Changed from String(20)

        # Check other important columns
        assert columns['run_id'].type.__class__.__name__ == 'UUID'
        # Check JSONB types for PostgreSQL
        assert columns['job_snapshot'].type.__class__.__name__ == 'JSONB'
        assert columns['result'].type.__class__.__name__ == 'JSONB'
        assert columns['error'].type.__class__.__name__ == 'Text'
        assert columns['worker_id'].type.__class__.__name__ == 'String'

    def test_run_nullable_fields(self):
        """Test that Run model has correct nullable settings."""
        columns = Run.__table__.columns

        # These should be nullable according to PostgreSQL schema
        assert columns['job_id'].nullable is True
        assert columns['user_id'].nullable is True
        assert columns['status'].nullable is True
        assert columns['scheduled_for'].nullable is True
        assert columns['job_snapshot'].nullable is True
        assert columns['result'].nullable is True
        assert columns['error'].nullable is True
        assert columns['worker_id'].nullable is True

    def test_run_unique_constraint_name(self):
        """Test that Run model has correct unique constraint name."""
        constraints = Run.__table__.constraints
        unique_constraints = [c for c in constraints if hasattr(c, 'columns') and len(c.columns) == 3]

        # Find the unique constraint on job_type, job_id, scheduled_for
        job_constraint = None
        for constraint in unique_constraints:
            col_names = [col.name for col in constraint.columns]
            if set(col_names) == {"job_type", "job_id", "scheduled_for"}:
                job_constraint = constraint
                break

        assert job_constraint is not None
        assert job_constraint.name == "ux_runs_job_scheduled_for"

    def test_schedule_has_required_columns(self):
        """Test that Schedule model has all required columns."""
        columns = Schedule.__table__.columns
        required_columns = {
            'id', 'user_id', 'name', 'job_type', 'target',
            'task_params', 'cron', 'enabled', 'next_run_at',
            'created_at', 'updated_at'
        }

        actual_columns = set(columns.keys())
        assert required_columns.issubset(actual_columns)

    def test_run_has_required_columns(self):
        """Test that Run model has all required columns."""
        columns = Run.__table__.columns
        required_columns = {
            'run_id', 'job_type', 'job_id', 'user_id', 'status',
            'scheduled_for', 'enqueued_at', 'started_at', 'finished_at',
            'job_snapshot', 'result', 'error', 'worker_id'
        }

        actual_columns = set(columns.keys())
        assert required_columns.issubset(actual_columns)

    def test_model_instantiation(self):
        """Test that models can be instantiated with correct types."""
        # Test Schedule instantiation
        schedule = Schedule(
            user_id=1,
            name="Test Schedule",
            job_type="report",
            target="portfolio_summary",
            task_params={"format": "pdf"},
            cron="0 9 * * *",
            enabled=True
        )

        assert schedule.user_id == 1
        assert schedule.name == "Test Schedule"
        assert schedule.job_type == "report"
        assert schedule.task_params == {"format": "pdf"}

        # Test ScheduleRun instantiation with corrected types
        run = ScheduleRun(
            job_type="screener",
            job_id=12345,  # BigInteger
            user_id=67890,  # BigInteger
            status="pending",  # Text
            scheduled_for=datetime.now(timezone.utc),
            job_snapshot={"test": "data"},
            worker_id="worker-123"
        )

        assert run.job_type == "screener"
        assert run.job_id == 12345
        assert run.user_id == 67890
        assert run.status == "pending"
        assert run.worker_id == "worker-123"

    def test_run_nullable_instantiation(self):
        """Test that ScheduleRun can be instantiated with nullable fields as None."""
        run = ScheduleRun(
            job_type="alert",
            job_id=None,
            user_id=None,
            status=None,
            scheduled_for=None,
            job_snapshot=None,
            worker_id=None
        )

        assert run.job_type == "alert"
        assert run.job_id is None
        assert run.user_id is None
        assert run.status is None
        assert run.scheduled_for is None
        assert run.job_snapshot is None
        assert run.worker_id is None


class TestJobModelConstraints:
    """Test constraint definitions in job models."""

    def test_schedule_unique_constraint_exists(self):
        """Test that Schedule has unique constraint on user_id + name."""
        constraints = Schedule.__table__.constraints
        unique_constraints = [c for c in constraints if hasattr(c, 'columns')]

        # Find unique constraint on user_id and name
        user_name_constraint = None
        for constraint in unique_constraints:
            col_names = [col.name for col in constraint.columns]
            if set(col_names) == {"user_id", "name"}:
                user_name_constraint = constraint
                break

        assert user_name_constraint is not None
        assert user_name_constraint.name == "unique_user_schedule_name"

    def test_run_primary_key(self):
        """Test that Run has correct primary key."""
        primary_key_cols = [col.name for col in Run.__table__.primary_key.columns]
        assert primary_key_cols == ["run_id"]

    def test_schedule_primary_key(self):
        """Test that Schedule has correct primary key."""
        primary_key_cols = [col.name for col in Schedule.__table__.primary_key.columns]
        assert primary_key_cols == ["id"]


class TestDataTypeCompatibility:
    """Test that data types are compatible with PostgreSQL."""

    def test_bigint_compatibility(self):
        """Test that BigInteger fields can handle large values."""
        run = ScheduleRun(
            job_type="test",
            job_id=9223372036854775807,  # Max int64 value
            user_id=9223372036854775806,
        )

        assert run.job_id == 9223372036854775807
        assert run.user_id == 9223372036854775806

    def test_text_field_compatibility(self):
        """Test that Text fields can handle long strings."""
        long_text = "x" * 10000

        run = ScheduleRun(
            job_type=long_text,
            status=long_text,
            error=long_text
        )

        assert run.job_type == long_text
        assert run.status == long_text
        assert run.error == long_text

    def test_jsonb_field_compatibility(self):
        """Test that JSONB fields can handle complex data structures."""
        complex_data = {
            "nested": {
                "array": [1, 2, 3],
                "object": {"key": "value"},
                "boolean": True,
                "null": None
            },
            "numbers": [1.5, 2.7, 3.14159],
            "strings": ["hello", "world"]
        }

        schedule = Schedule(
            user_id=1,
            name="Complex Data Test",
            job_type="test",
            target="test",
            task_params=complex_data,
            cron="0 * * * *"
        )

        run = ScheduleRun(
            job_type="test",
            job_snapshot=complex_data,
            result=complex_data
        )

        assert schedule.task_params == complex_data
        assert run.job_snapshot == complex_data
        assert run.result == complex_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])