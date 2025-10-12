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
        'telegram_alerts',
        'telegram_feedbacks',
        'telegram_schedules',
        'telegram_verification_codes',
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

    # Check telegram_alerts foreign key
    alerts_table = tables['telegram_alerts']
    alert_user_fk = None
    for fk in alerts_table.foreign_keys:
        if fk.parent.name == 'user_id':
            alert_user_fk = fk
            break

    assert alert_user_fk is not None
    assert alert_user_fk.column.table.name == 'usr_users'


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


def test_model_relationships():
    """Test that model classes have expected attributes."""
    from src.data.db.models.model_users import User, AuthIdentity, VerificationCode
    from src.data.db.models.model_jobs import Schedule, Run
    from src.data.db.models.model_telegram import TelegramAlert, TelegramFeedback

    # Test User model
    assert hasattr(User, '__tablename__')
    assert User.__tablename__ == 'usr_users'

    # Test AuthIdentity model
    assert hasattr(AuthIdentity, '__tablename__')
    assert AuthIdentity.__tablename__ == 'usr_auth_identities'

    # Test VerificationCode model
    assert hasattr(VerificationCode, '__tablename__')
    assert VerificationCode.__tablename__ == 'usr_verification_codes'

    # Test Job models
    assert hasattr(Schedule, '__tablename__')
    assert Schedule.__tablename__ == 'job_schedules'

    assert hasattr(Run, '__tablename__')
    assert Run.__tablename__ == 'job_runs'

    # Test Telegram models
    assert hasattr(TelegramAlert, '__tablename__')
    assert TelegramAlert.__tablename__ == 'telegram_alerts'


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))