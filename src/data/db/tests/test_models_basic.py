"""
Basic Model Tests

Tests that validate the model definitions and table structure
without requiring a database connection.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from sqlalchemy import create_engine, MetaData
from sqlalchemy.schema import CreateTable

# Import all models to register them
from src.data.db.core.base import Base
import src.data.db.models.model_users
import src.data.db.models.model_telegram
import src.data.db.models.model_trading
import src.data.db.models.model_webui
import src.data.db.models.model_jobs


def test_models_can_be_imported():
    """Test that all model modules can be imported without errors."""
    # If we get here, all imports succeeded
    assert True


def test_base_metadata_has_tables():
    """Test that models are registered with the Base metadata."""
    tables = Base.metadata.tables

    # Check that we have the expected tables
    expected_tables = {
        'usr_users',
        'usr_auth_identities',
        'usr_verification_codes',
        'job_schedules',
        'job_runs',
        'telegram_feedbacks',
        'telegram_settings',
        'telegram_command_audits',
        'telegram_broadcast_logs'
    }

    actual_tables = set(tables.keys())

    # Check that our expected tables are present
    missing_tables = expected_tables - actual_tables
    assert not missing_tables, f"Missing tables: {missing_tables}"


def test_renamed_tables_exist():
    """Test that the renamed tables exist with correct names."""
    tables = Base.metadata.tables

    # Check renamed user tables
    assert 'usr_users' in tables
    assert 'usr_auth_identities' in tables
    assert 'usr_verification_codes' in tables

    # Ensure old names don't exist
    assert 'users' not in tables
    assert 'auth_identity' not in tables
    assert 'telegram_verification_codes' not in tables


def test_new_job_tables_exist():
    """Test that the new job tables exist."""
    tables = Base.metadata.tables

    assert 'job_schedules' in tables
    assert 'job_runs' in tables


def test_foreign_key_references():
    """Test that foreign keys reference the correct table names."""
    tables = Base.metadata.tables

    # Check usr_auth_identities foreign key
    auth_table = tables['usr_auth_identities']
    user_id_fk = None
    for fk in auth_table.foreign_keys:
        if fk.parent.name == 'user_id':
            user_id_fk = fk
            break

    assert user_id_fk is not None
    assert user_id_fk.column.table.name == 'usr_users'

    # Check telegram_feedbacks foreign key
    feedback_table = tables['telegram_feedbacks']
    feedback_user_fk = None
    for fk in feedback_table.foreign_keys:
        if fk.parent.name == 'user_id':
            feedback_user_fk = fk
            break

    assert feedback_user_fk is not None
    assert feedback_user_fk.column.table.name == 'usr_users'


def test_table_ddl_generation():
    """Test that DDL can be generated for all tables."""
    # Create a mock engine to test DDL generation
    engine = create_engine("postgresql://", strategy='mock', executor=lambda sql, *_: None)

    # This should not raise any errors
    try:
        for table_name, table in Base.metadata.tables.items():
            ddl = CreateTable(table).compile(engine)
            assert ddl is not None
            assert len(str(ddl)) > 0
    except Exception as e:
        pytest.fail(f"DDL generation failed for table: {e}")


def test_corrected_data_types():
    """Test that corrected data types are properly defined."""
    from src.data.db.models.model_jobs import Run
    from src.data.db.models.model_users import AuthIdentity, VerificationCode
    from src.data.db.models.model_telegram import TelegramFeedback
    from sqlalchemy import BigInteger, Text, String, DateTime
    from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSONB

    # Test Run model data types
    run_table = Run.__table__

    # job_id should be BigInteger (was String(255))
    job_id_col = run_table.columns['job_id']
    assert isinstance(job_id_col.type, BigInteger), f"job_id should be BigInteger, got {type(job_id_col.type)}"

    # user_id should be BigInteger
    user_id_col = run_table.columns['user_id']
    assert isinstance(user_id_col.type, BigInteger), f"user_id should be BigInteger, got {type(user_id_col.type)}"

    # job_type should be Text (was String(50))
    job_type_col = run_table.columns['job_type']
    assert isinstance(job_type_col.type, Text), f"job_type should be Text, got {type(job_type_col.type)}"

    # status should be Text (was String(20))
    status_col = run_table.columns['status']
    assert isinstance(status_col.type, Text), f"status should be Text, got {type(status_col.type)}"

    # worker_id should exist as String(255)
    worker_id_col = run_table.columns['worker_id']
    assert isinstance(worker_id_col.type, String), f"worker_id should be String, got {type(worker_id_col.type)}"

    # Test AuthIdentity column mapping
    auth_table = AuthIdentity.__table__
    # identity_metadata attribute should map to "metadata" column
    metadata_col = auth_table.columns['metadata']
    assert metadata_col is not None, "metadata column should exist in auth_identities table"

    # Test VerificationCode new columns
    verification_table = VerificationCode.__table__
    provider_col = verification_table.columns['provider']
    assert isinstance(provider_col.type, String), f"provider should be String, got {type(provider_col.type)}"
    created_at_col = verification_table.columns['created_at']
    assert isinstance(created_at_col.type, DateTime), f"created_at should be DateTime, got {type(created_at_col.type)}"

    # Test TelegramFeedback new columns
    feedback_table = TelegramFeedback.__table__
    type_col = feedback_table.columns['type']
    assert isinstance(type_col.type, String), f"type should be String, got {type(type_col.type)}"
    message_col = feedback_table.columns['message']
    assert isinstance(message_col.type, Text), f"message should be Text, got {type(message_col.type)}"
    feedback_created_col = feedback_table.columns['created_at']
    assert isinstance(feedback_created_col.type, DateTime), f"created_at should be DateTime, got {type(feedback_created_col.type)}"
    feedback_status_col = feedback_table.columns['status']
    assert isinstance(feedback_status_col.type, String), f"status should be String, got {type(feedback_status_col.type)}"


def test_model_relationships():
    """Test that model classes have expected attributes."""
    from src.data.db.models.model_users import User, AuthIdentity, VerificationCode
    from src.data.db.models.model_jobs import Schedule, Run
    from src.data.db.models.model_telegram import TelegramFeedback

    # Test User model
    assert hasattr(User, '__tablename__')
    assert User.__tablename__ == 'usr_users'

    # Test AuthIdentity model
    assert hasattr(AuthIdentity, '__tablename__')
    assert AuthIdentity.__tablename__ == 'usr_auth_identities'
    # Test that identity_metadata attribute exists (maps to metadata column)
    assert hasattr(AuthIdentity, 'identity_metadata')

    # Test VerificationCode model
    assert hasattr(VerificationCode, '__tablename__')
    assert VerificationCode.__tablename__ == 'usr_verification_codes'
    # Test new columns added to VerificationCode
    assert hasattr(VerificationCode, 'provider')
    assert hasattr(VerificationCode, 'created_at')

    # Test Job models
    assert hasattr(Schedule, '__tablename__')
    assert Schedule.__tablename__ == 'job_schedules'

    assert hasattr(Run, '__tablename__')
    assert Run.__tablename__ == 'job_runs'
    # Test that Run model has the new worker_id field
    assert hasattr(Run, 'worker_id')

    # Test Telegram models
    assert hasattr(TelegramFeedback, '__tablename__')
    assert TelegramFeedback.__tablename__ == 'telegram_feedbacks'
    # Test new columns added to TelegramFeedback
    assert hasattr(TelegramFeedback, 'type')
    assert hasattr(TelegramFeedback, 'message')
    assert hasattr(TelegramFeedback, 'created_at')
    assert hasattr(TelegramFeedback, 'status')


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))