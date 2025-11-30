# src/data/db/services/users_service.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from src.data.db.services.base_service import BaseDBService, with_uow

class UsersService(BaseDBService):
    @with_uow
    def get_user_by_telegram_id(self, repos, telegram_user_id: str | int):
        return repos.users.get_user_by_telegram_id(telegram_user_id)

    @with_uow
    def get_telegram_profile(self, repos, telegram_user_id: str | int) -> Optional[Dict[str, Any]]:
        return repos.users.get_telegram_profile(telegram_user_id)

    @with_uow
    def ensure_user_for_telegram(self, repos, telegram_user_id: str | int, defaults_user: dict | None = None) -> int:
        u = repos.users.ensure_user_for_telegram(telegram_user_id, defaults_user)
        return u.id

    @with_uow
    def update_telegram_profile(self, repos, telegram_user_id: str | int, **fields) -> None:
        repos.users.update_telegram_profile(telegram_user_id, **fields)

    @with_uow
    def list_telegram_users_dto(self, repos) -> List[Dict[str, Any]]:
        return repos.users.list_telegram_users_dto()

    @with_uow
    def list_pending_telegram_approvals(self, repos) -> List[Dict[str, Any]]:
        return repos.users.list_pending_telegram_approvals()

    @with_uow
    def get_admin_telegram_user_ids(self, repos) -> List[str]:
        return repos.users.get_admin_telegram_user_ids()

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
    def get_user_by_id(self, repos, user_id: int):
        """Get user by ID."""
        return repos.users.get_user_by_id(user_id)

    @with_uow
    def get_user_notification_channels(self, repos, user_id: int) -> Optional[Dict[str, str]]:
        """
        Get user's notification channels (email + telegram_chat_id).

        Args:
            user_id: User ID

        Returns:
            Dictionary with email and telegram_chat_id, or None if user not found
        """
        return repos.users.get_user_notification_channels(user_id)
