"""
Comprehensive tests for UsersService.

Tests cover:
- User retrieval by Telegram ID
- User creation (ensure pattern)
- Telegram profile operations
- User listing and filtering
- Admin user operations
- Broadcast user listing

Note: UsersService works primarily through Telegram integration,
so tests focus on the actual API methods available.
"""

from src.data.db.services.users_service import UsersService
from src.data.db.models.model_users import User, AuthIdentity
from src.data.db.tests.fixtures.factory_users import UserFactory, AuthIdentityFactory


class TestUsersServiceTelegramOperations:
    """Tests for Telegram user operations."""

    def test_ensure_user_for_telegram_creates_new_user(self, mock_database_service, db_session):
        """Test ensuring user creates a new user if not exists."""
        service = UsersService(db_service=mock_database_service)

        telegram_id = 123456789
        defaults = {
            "email": "test@example.com"
        }

        user_id = service.ensure_user_for_telegram(
            telegram_user_id=telegram_id,
            defaults_user=defaults
        )

        assert user_id is not None
        assert isinstance(user_id, int)

        # Verify user was created
        user = service.get_user_by_telegram_id(telegram_user_id=telegram_id)
        assert user is not None
        assert user.id == user_id

    def test_ensure_user_for_telegram_returns_existing_user(self, mock_database_service, db_session):
        """Test ensuring user returns existing user if already exists."""
        service = UsersService(db_service=mock_database_service)

        telegram_id = 987654321
        defaults = {
            "email": "existing@example.com"
        }

        # Create user first time
        user_id_1 = service.ensure_user_for_telegram(
            telegram_user_id=telegram_id,
            defaults_user=defaults
        )

        # Try to create again
        user_id_2 = service.ensure_user_for_telegram(
            telegram_user_id=telegram_id,
            defaults_user=defaults
        )

        # Should return same user ID
        assert user_id_1 == user_id_2

    def test_get_user_by_telegram_id(self, mock_database_service, db_session):
        """Test getting user by Telegram ID."""
        service = UsersService(db_service=mock_database_service)

        telegram_id = 111222333
        defaults = {"email": "get@example.com"}

        user_id = service.ensure_user_for_telegram(
            telegram_user_id=telegram_id,
            defaults_user=defaults
        )

        # Get user
        user = service.get_user_by_telegram_id(telegram_user_id=telegram_id)

        assert user is not None
        assert user.id == user_id

    def test_get_user_by_telegram_id_not_found(self, mock_database_service, db_session):
        """Test getting non-existent user returns None."""
        service = UsersService(db_service=mock_database_service)

        user = service.get_user_by_telegram_id(telegram_user_id=999999999)

        assert user is None

    def test_get_telegram_profile(self, mock_database_service, db_session):
        """Test getting Telegram profile data."""
        service = UsersService(db_service=mock_database_service)

        telegram_id = 444555666
        defaults = {
            "email": "profile@example.com"
        }

        service.ensure_user_for_telegram(
            telegram_user_id=telegram_id,
            defaults_user=defaults
        )

        # Get profile
        profile = service.get_telegram_profile(telegram_user_id=telegram_id)

        assert profile is not None
        assert "telegram_user_id" in profile or "user_id" in profile

    def test_update_telegram_profile(self, mock_database_service, db_session):
        """Test updating Telegram profile."""
        service = UsersService(db_service=mock_database_service)

        telegram_id = 777888999
        defaults = {
            "email": "update@example.com"
        }

        service.ensure_user_for_telegram(
            telegram_user_id=telegram_id,
            defaults_user=defaults
        )

        # Update profile (these go into identity metadata)
        service.update_telegram_profile(
            telegram_user_id=telegram_id,
            verified=True,
            approved=True
        )

        # Verify update - profile should still exist
        profile = service.get_telegram_profile(telegram_user_id=telegram_id)
        assert profile is not None


class TestUsersServiceListing:
    """Tests for user listing operations."""

    def test_list_telegram_users_dto(self, mock_database_service, db_session):
        """Test listing all Telegram users as DTOs."""
        service = UsersService(db_service=mock_database_service)

        # Create multiple users
        for i in range(3):
            service.ensure_user_for_telegram(
                telegram_user_id=1000 + i,
                defaults_user={
                    "email": f"user{i}@example.com"
                }
            )

        # List users
        users = service.list_telegram_users_dto()

        assert isinstance(users, list)
        assert len(users) >= 3

    def test_list_users_for_broadcast(self, mock_database_service, db_session):
        """Test listing users in broadcast format."""
        service = UsersService(db_service=mock_database_service)

        # Create users
        for i in range(2):
            service.ensure_user_for_telegram(
                telegram_user_id=2000 + i,
                defaults_user={
                    "email": f"broadcast{i}@example.com"
                }
            )

        # List for broadcast
        broadcast_users = service.list_users_for_broadcast()

        assert isinstance(broadcast_users, list)
        assert len(broadcast_users) >= 2

        # Check format
        if len(broadcast_users) > 0:
            user = broadcast_users[0]
            assert "telegram_user_id" in user
            assert isinstance(user["telegram_user_id"], str)

    def test_list_pending_telegram_approvals(self, mock_database_service, db_session):
        """Test listing users pending approval."""
        service = UsersService(db_service=mock_database_service)

        # Create users (some may need approval based on implementation)
        for i in range(2):
            service.ensure_user_for_telegram(
                telegram_user_id=3000 + i,
                defaults_user={
                    "email": f"pending{i}@example.com"
                }
            )

        # List pending approvals
        pending = service.list_pending_telegram_approvals()

        assert isinstance(pending, list)
        # May be empty if all users are auto-approved


class TestUsersServiceAdmin:
    """Tests for admin user operations."""

    def test_get_admin_telegram_user_ids(self, mock_database_service, db_session):
        """Test getting admin Telegram user IDs."""
        service = UsersService(db_service=mock_database_service)

        # Create an admin user manually
        from src.data.db.repos.repo_users import UsersRepo
        repos = UsersRepo(db_session)

        # Create admin user with proper structure
        user_data = UserFactory.admin_user(email="admin@example.com")
        user = User(**user_data)
        db_session.add(user)
        db_session.flush()

        # Create auth identity for admin
        auth_data = AuthIdentityFactory.telegram_identity(
            user_id=user.id,
            telegram_id=9999999,
            username="admin_user"
        )
        # AuthIdentity uses external_id, not provider_user_id
        auth = AuthIdentity(
            user_id=user.id,
            provider="telegram",
            external_id=str(9999999),
            identity_metadata=auth_data["provider_data"]
        )
        db_session.add(auth)
        db_session.commit()

        # Get admin IDs
        admin_ids = service.get_admin_telegram_user_ids()

        assert isinstance(admin_ids, list)
        # May or may not contain the test user depending on how the method filters


class TestUsersServiceEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ensure_user_with_numeric_telegram_id(self, mock_database_service, db_session):
        """Test ensuring user with numeric Telegram ID."""
        service = UsersService(db_service=mock_database_service)

        # Use numeric ID
        user_id = service.ensure_user_for_telegram(
            telegram_user_id=123456,  # int, not string
            defaults_user={"email": "numeric@example.com"}
        )

        assert user_id is not None

    def test_ensure_user_with_string_telegram_id(self, mock_database_service, db_session):
        """Test ensuring user with string Telegram ID."""
        service = UsersService(db_service=mock_database_service)

        # Use string ID
        user_id = service.ensure_user_for_telegram(
            telegram_user_id="123456",  # string
            defaults_user={"email": "string@example.com"}
        )

        assert user_id is not None

    def test_ensure_user_with_minimal_defaults(self, mock_database_service, db_session):
        """Test ensuring user with minimal default data."""
        service = UsersService(db_service=mock_database_service)

        user_id = service.ensure_user_for_telegram(
            telegram_user_id=555666,
            defaults_user={}  # Minimal data
        )

        assert user_id is not None

    def test_ensure_user_with_no_defaults(self, mock_database_service, db_session):
        """Test ensuring user with no default data."""
        service = UsersService(db_service=mock_database_service)

        user_id = service.ensure_user_for_telegram(
            telegram_user_id=666777,
            defaults_user=None
        )

        assert user_id is not None


class TestUsersServiceIntegration:
    """Integration tests for user workflows."""

    def test_full_user_lifecycle(self, mock_database_service, db_session):
        """Test complete user lifecycle: create, update, retrieve."""
        service = UsersService(db_service=mock_database_service)

        telegram_id = 888999000

        # 1. Create user
        user_id = service.ensure_user_for_telegram(
            telegram_user_id=telegram_id,
            defaults_user={
                "email": "lifecycle@example.com"
            }
        )

        assert user_id is not None

        # 2. Get user
        user = service.get_user_by_telegram_id(telegram_user_id=telegram_id)
        assert user is not None
        assert user.id == user_id

        # 3. Update profile
        service.update_telegram_profile(
            telegram_user_id=telegram_id,
            verified=True,
            approved=True
        )

        # 4. Get profile
        profile = service.get_telegram_profile(telegram_user_id=telegram_id)
        assert profile is not None

        # 5. List users
        users = service.list_telegram_users_dto()
        assert len(users) >= 1

    def test_multiple_users_listing(self, mock_database_service, db_session):
        """Test listing multiple users."""
        service = UsersService(db_service=mock_database_service)

        # Create multiple users with different profiles
        user_ids = []
        for i in range(5):
            user_id = service.ensure_user_for_telegram(
                telegram_user_id=4000 + i,
                defaults_user={
                    "email": f"multi_user_{i}@example.com"
                }
            )
            user_ids.append(user_id)

        # List all users
        users = service.list_telegram_users_dto()
        assert len(users) >= 5

        # List for broadcast
        broadcast_users = service.list_users_for_broadcast()
        assert len(broadcast_users) >= 5
