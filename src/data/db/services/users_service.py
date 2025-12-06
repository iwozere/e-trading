from __future__ import annotations
from typing import Optional, Dict, Any, List
from src.data.db.services.base_service import BaseDBService, with_uow, handle_db_error
from src.data.db.core.constants import PROVIDER_TG
from src.data.db.models.model_users import User, AuthIdentity

class UsersService(BaseDBService):
    @with_uow
    @handle_db_error
    def get_user_by_telegram_id(self, telegram_user_id: str | int):
        return self.repos.users.get_user_by_telegram_id(telegram_user_id)

    @with_uow
    @handle_db_error
    def get_telegram_profile(self, telegram_user_id: str | int) -> Optional[Dict[str, Any]]:
        return self.repos.users.get_telegram_profile(telegram_user_id)

    @with_uow
    @handle_db_error
    def ensure_user_for_telegram(self, telegram_user_id: str | int, defaults_user: dict | None = None) -> int:
        """
        Ensures a user exists for the given Telegram ID.
        Returns the User ID.
        """
        tid = str(telegram_user_id)
        # Check if identity exists
        ident = self.repos.users.get_identity(provider=PROVIDER_TG, external_id=tid)
        if ident:
            return ident.user_id

        # Create new user
        defaults = defaults_user or {}
        if "role" not in defaults:
            defaults["role"] = "trader"
        u = User(**defaults)
        self.repos.users.create_user(u)

        # Create identity
        ident = AuthIdentity(
            user_id=u.id,
            provider=PROVIDER_TG,
            external_id=tid,
            identity_metadata={}
        )
        self.repos.users.create_identity(ident)

        return u.id

    @with_uow
    @handle_db_error
    def update_telegram_profile(self, telegram_user_id: str | int, **fields) -> None:
        """
        Merge fields into auth_identities.metadata for provider='telegram'.
        Supported keys: verified, approved, language, is_admin,
                        max_alerts, max_schedules, verification_code, code_sent_time
        You may also pass email=<str> to update users.email.
        """
        tid = str(telegram_user_id)
        ident = self.repos.users.get_identity(provider=PROVIDER_TG, external_id=tid)

        # If identity doesn't exist, we must create it (and the user) first
        if not ident:
            # We don't have defaults here, so we create a bare user
            defaults = {}
            if "email" in fields:
                defaults["email"] = fields["email"]
            u_id = self.ensure_user_for_telegram(tid, defaults_user=defaults)
            # Re-fetch identity attached to this session
            ident = self.repos.users.get_identity(provider=PROVIDER_TG, external_id=tid)
            # Fetch user explicitly if we need to update email
            u = self.repos.users.get_user_by_id(u_id)
        else:
            u = self.repos.users.get_user_by_id(ident.user_id)

        # Update user email if provided
        if "email" in fields:
            email = fields.pop("email")
            if u:
                u.email = email
                self.repos.users.create_user(u) # create_user acts as add/update if attached

        # Update metadata
        # Create a copy to ensure SQLAlchemy detects the change (handles in-place mutation issues)
        meta = dict(ident.identity_metadata or {})
        # Filter out None values to allow specific updates (or partial updates)
        meta.update({k: v for k, v in fields.items() if v is not None})
        ident.identity_metadata = meta
        self.repos.users.create_identity(ident) # flush

    @with_uow
    @handle_db_error
    def list_telegram_users_dto(self) -> List[Dict[str, Any]]:
        return self.repos.users.list_telegram_users_dto()

    @with_uow
    @handle_db_error
    def list_pending_telegram_approvals(self) -> List[Dict[str, Any]]:
        return self.repos.users.list_pending_telegram_approvals()

    @with_uow
    @handle_db_error
    def get_admin_telegram_user_ids(self) -> List[str]:
        return self.repos.users.get_admin_telegram_user_ids()

    def list_users_for_broadcast(self) -> List[Dict[str, Any]]:
        rows = self.list_telegram_users_dto()
        return [
            {
                "telegram_user_id": str(r.get("telegram_user_id")),
                "email": r.get("email"),
                "approved": r.get("approved"),
                "verified": r.get("verified"),
            }
            for r in rows
        ]

    @with_uow
    @handle_db_error
    def get_user_by_id(self, user_id: int):
        """Get user by ID."""
        return self.repos.users.get_user_by_id(user_id)

    @with_uow
    @handle_db_error
    def get_user_notification_channels(self, user_id: int) -> Optional[Dict[str, str]]:
        """
        Get user's notification channels (email + telegram_chat_id).

        Args:
            user_id: User ID

        Returns:
            Dictionary with email and telegram_chat_id, or None if user not found
        """
        return self.repos.users.get_user_notification_channels(user_id)


# Global service instance
users_service = UsersService()
