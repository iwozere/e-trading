# src/data/db/services/users_service.py
from __future__ import annotations
from typing import Optional, Dict, Any, List

from src.data.db.services.database_service import database_service

# ---- Telegram-focused user facades (thin wrappers over UsersRepo) ----

def get_user_by_telegram_id(telegram_user_id: str | int):
    """Get User object by telegram user ID."""
    with database_service.uow() as r:
        return r.users.get_user_by_telegram_id(telegram_user_id)

def get_telegram_profile(telegram_user_id: str | int) -> Optional[Dict[str, Any]]:
    with database_service.uow() as r:
        return r.users.get_telegram_profile(telegram_user_id)

def ensure_user_for_telegram(telegram_user_id: str | int, defaults_user: dict | None = None) -> int:
    """Returns internal user_id after ensuring it exists for this Telegram account."""
    with database_service.uow() as r:
        u = r.users.ensure_user_for_telegram(telegram_user_id, defaults_user)
        return u.id

def update_telegram_profile(telegram_user_id: str | int, **fields) -> None:
    """
    Supported fields (mirrors repo): verified, approved, language, is_admin,
    max_alerts, max_schedules, verification_code, code_sent_time, email
    """
    with database_service.uow() as r:
        r.users.update_telegram_profile(telegram_user_id, **fields)

def list_telegram_users_dto() -> List[Dict[str, Any]]:
    with database_service.uow() as r:
        return r.users.list_telegram_users_dto()

def list_pending_telegram_approvals() -> List[Dict[str, Any]]:
    with database_service.uow() as r:
        return r.users.list_pending_telegram_approvals()

def get_admin_telegram_user_ids() -> List[str]:
    with database_service.uow() as r:
        return r.users.get_admin_telegram_user_ids()

# Convenience: the shape your bot likely expects for broadcast/status endpoints
def list_users_for_broadcast() -> List[Dict[str, Any]]:
    """
    Returns a compact DTO: [{telegram_user_id, email, approved, verified}]
    """
    rows = list_telegram_users_dto()
    return [
        {
            "telegram_user_id": str(r.get("telegram_user_id")),
            "email": r.get("email"),
            "approved": r.get("approved"),
            "verified": r.get("verified"),
        }
        for r in rows
    ]
