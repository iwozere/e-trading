# src/data/db/services/users_service.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from src.data.db.services.base_service import BaseDBService, with_uow, handle_db_error

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
        u = self.repos.users.ensure_user_for_telegram(telegram_user_id, defaults_user)
        return u.id

    @with_uow
    @handle_db_error
    def update_telegram_profile(self, telegram_user_id: str | int, **fields) -> None:
        self.repos.users.update_telegram_profile(telegram_user_id, **fields)

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
