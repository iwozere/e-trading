"""
User Model Structure Tests

Tests for AuthIdentity and VerificationCode model structure to validate
the corrected column mappings and constraints without requiring database connection.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from sqlalchemy import Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from src.data.db.models.model_users import User, AuthIdentity, VerificationCode


class TestAuthIdentityModelStructure:
    """Test AuthIdentity model structure and definitions."""

    def test_auth_identity_table_name(self):
        """Test that AuthIdentity has correct table name."""
        assert AuthIdentity.__tablename__ == "usr_auth_identities"

    def test_auth_identity_columns(self):
        """Test that AuthIdentity has all required columns."""
        table = AuthIdentity.__table__
        column_names = [col.name for col in table.columns]

        expected_columns = ['id', 'user_id', 'provider', 'external_id', 'metadata', 'created_at']
        for col in expected_columns:
            assert col in column_names, f"Column {col} missing from AuthIdentity table"

    def test_auth_identity_metadata_column_mapping(self):
        """Test that identity_metadata attribute maps to metadata database column."""
        table = AuthIdentity.__table__

        # Check that the database column is named 'metadata'
        assert 'metadata' in [col.name for col in table.columns]

        # Check that the column uses JSONB type
        metadata_column = table.columns['metadata']
        assert isinstance(metadata_column.type, JSONB), f"Expected JSONB, got {type(metadata_column.type)}"

    def test_auth_identity_constraints(self):
        """Test that AuthIdentity has correct constraints."""
        table = AuthIdentity.__table__

        # Check unique constraint
        unique_constraints = [c for c in table.constraints if isinstance(c, UniqueConstraint)]
        assert len(unique_constraints) > 0, "AuthIdentity should have unique constraints"

        # Find the provider/external_id unique constraint
        provider_external_constraint = None
        for constraint in unique_constraints:
            if 'provider' in [col.name for col in constraint.columns] and 'external_id' in [col.name for col in constraint.columns]:
                provider_external_constraint = constraint
                break

        assert provider_external_constraint is not None, "Missing unique constraint on provider and external_id"
        assert provider_external_constraint.name == "uq_auth_identities_provider_external"

    def test_auth_identity_indexes(self):
        """Test that AuthIdentity has correct indexes."""
        table = AuthIdentity.__table__

        # Check that we have the expected indexes
        index_names = [idx.name for idx in table.indexes]
        expected_indexes = [
            "ix_auth_identities_provider_external",
            "ix_auth_identities_provider",
            "ix_auth_identities_user_id"
        ]

        for idx_name in expected_indexes:
            assert idx_name in index_names, f"Missing index: {idx_name}"

    def test_auth_identity_foreign_key(self):
        """Test that AuthIdentity has correct foreign key."""
        table = AuthIdentity.__table__

        # Check foreign key on user_id
        user_id_column = table.columns['user_id']
        foreign_keys = list(user_id_column.foreign_keys)
        assert len(foreign_keys) == 1, "user_id should have exactly one foreign key"

        fk = foreign_keys[0]
        assert fk.column.table.name == "usr_users", "Foreign key should reference usr_users table"
        assert fk.ondelete == "CASCADE", "Foreign key should have CASCADE delete"

    def test_auth_identity_convenience_methods(self):
        """Test that AuthIdentity has convenience methods."""
        assert hasattr(AuthIdentity, 'meta_get'), "AuthIdentity should have meta_get method"
        assert hasattr(AuthIdentity, 'meta_set'), "AuthIdentity should have meta_set method"

        # Test that methods are callable
        assert callable(AuthIdentity.meta_get), "meta_get should be callable"
        assert callable(AuthIdentity.meta_set), "meta_set should be callable"


class TestVerificationCodeModelStructure:
    """Test VerificationCode model structure and definitions."""

    def test_verification_code_table_name(self):
        """Test that VerificationCode has correct table name."""
        assert VerificationCode.__tablename__ == "usr_verification_codes"

    def test_verification_code_columns(self):
        """Test that VerificationCode has all required columns."""
        table = VerificationCode.__table__
        column_names = [col.name for col in table.columns]

        expected_columns = ['id', 'user_id', 'code', 'sent_time', 'provider', 'created_at']
        for col in expected_columns:
            assert col in column_names, f"Column {col} missing from VerificationCode table"

    def test_verification_code_provider_default(self):
        """Test that provider column has correct default value."""
        table = VerificationCode.__table__
        provider_column = table.columns['provider']

        # Check that provider has a default value
        assert provider_column.server_default is not None, "provider column should have a server default"

    def test_verification_code_indexes(self):
        """Test that VerificationCode has correct indexes."""
        table = VerificationCode.__table__

        # Check that we have the expected indexes
        index_names = [idx.name for idx in table.indexes]
        expected_indexes = ["ix_verification_codes_user_id"]

        for idx_name in expected_indexes:
            assert idx_name in index_names, f"Missing index: {idx_name}"

    def test_verification_code_foreign_key(self):
        """Test that VerificationCode has correct foreign key."""
        table = VerificationCode.__table__

        # Check foreign key on user_id
        user_id_column = table.columns['user_id']
        foreign_keys = list(user_id_column.foreign_keys)
        assert len(foreign_keys) == 1, "user_id should have exactly one foreign key"

        fk = foreign_keys[0]
        assert fk.column.table.name == "usr_users", "Foreign key should reference usr_users table"
        assert fk.ondelete == "CASCADE", "Foreign key should have CASCADE delete"

    def test_verification_code_column_types(self):
        """Test that VerificationCode columns have correct types."""
        table = VerificationCode.__table__

        # Check specific column types
        assert table.columns['code'].type.length == 32, "code column should be String(32)"
        assert table.columns['provider'].type.length == 20, "provider column should be String(20)"

        # Check that sent_time is Integer
        assert str(table.columns['sent_time'].type) == "INTEGER", "sent_time should be Integer type"


class TestUserModelIntegration:
    """Test integration between User and related models."""

    def test_user_table_name(self):
        """Test that User has correct table name."""
        assert User.__tablename__ == "usr_users"

    def test_model_relationships_exist(self):
        """Test that all user-related models exist and are importable."""
        # This test ensures all models can be imported without errors
        assert User is not None
        assert AuthIdentity is not None
        assert VerificationCode is not None

    def test_foreign_key_references_match(self):
        """Test that foreign key references point to correct tables."""
        # AuthIdentity should reference usr_users
        auth_table = AuthIdentity.__table__
        auth_fk = list(auth_table.columns['user_id'].foreign_keys)[0]
        assert auth_fk.column.table.name == "usr_users"

        # VerificationCode should reference usr_users
        verification_table = VerificationCode.__table__
        verification_fk = list(verification_table.columns['user_id'].foreign_keys)[0]
        assert verification_fk.column.table.name == "usr_users"


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "-rA"]))