# src/data/db/services/users_service.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from src.data.db.services.base_service import BaseDBService

class UsersService(BaseDBService):
    @BaseDBService.with_uow
    def get_user_by_telegram_id(self, repos, telegram_user_id: str | int):
        return repos.users.get_user_by_telegram_id(telegram_user_id)

    @BaseDBService.with_uow
    def get_telegram_profile(self, repos, telegram_user_id: str | int) -> Optional[Dict[str, Any]]:
        return repos.users.get_telegram_profile(telegram_user_id)

    @BaseDBService.with_uow
    def ensure_user_for_telegram(self, repos, telegram_user_id: str | int, defaults_user: dict | None = None) -> int:
        u = repos.users.ensure_user_for_telegram(telegram_user_id, defaults_user)
        return u.id

    @BaseDBService.with_uow
    def update_telegram_profile(self, repos, telegram_user_id: str | int, **fields) -> None:
        repos.users.update_telegram_profile(telegram_user_id, **fields)

    @BaseDBService.with_uow
    def list_telegram_users_dto(self, repos) -> List[Dict[str, Any]]:
        return repos.users.list_telegram_users_dto()

    @BaseDBService.with_uow
    def list_pending_telegram_approvals(self, repos) -> List[Dict[str, Any]]:
        return repos.users.list_pending_telegram_approvals()

    @BaseDBService.with_uow
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
