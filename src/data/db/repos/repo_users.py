# src/data/db/repo/repo_users.py
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from src.data.db.models.model_users import User, AuthIdentity, VerificationCode

PROVIDER_TG = "telegram"

class UsersRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    # ---------- generic helpers ----------
    def _get_identity(self, *, provider: str, external_id: str) -> Optional[AuthIdentity]:
        q = (select(AuthIdentity)
             .where(AuthIdentity.provider == provider,
                    AuthIdentity.external_id == str(external_id)))
        return self.s.execute(q).scalar_one_or_none()

    def _ensure_identity(self, *, provider: str, external_id: str, defaults_user: dict | None = None) -> Tuple[User, AuthIdentity]:
        row = (self.s.execute(
            select(User, AuthIdentity)
            .join(AuthIdentity, AuthIdentity.user_id == User.id)
            .where(AuthIdentity.provider == provider,
                   AuthIdentity.external_id == str(external_id))
        ).first())
        if row:
            u, ident = row
            if ident.identity_metadata is None:
                ident.identity_metadata = {}
            return u, ident

        u = User(**(defaults_user or {}))
        self.s.add(u); self.s.flush()
        ident = AuthIdentity(user_id=u.id, provider=provider,
                             external_id=str(external_id), identity_metadata={})
        self.s.add(ident); self.s.flush()
        return u, ident

    # ---------- telegram-specific wrappers ----------
    def get_user_by_telegram_id(self, telegram_user_id: str | int) -> Optional[User]:
        q = (
            select(User)
            .join(AuthIdentity, AuthIdentity.user_id == User.id)
            .where(AuthIdentity.provider == PROVIDER_TG,
                   AuthIdentity.external_id == str(telegram_user_id))
        )
        return self.s.execute(q).scalar_one_or_none()

    def ensure_user_for_telegram(self, telegram_user_id: str | int, defaults_user: dict | None = None) -> User:
        u, _ = self._ensure_identity(provider=PROVIDER_TG, external_id=str(telegram_user_id),
                                     defaults_user=defaults_user)
        return u

    def update_telegram_profile(self, telegram_user_id: str | int, **fields) -> None:
        """
        Merge fields into auth_identities.metadata for provider='telegram'.
        Supported keys: verified, approved, language, is_admin,
                        max_alerts, max_schedules, verification_code, code_sent_time
        You may also pass email=<str> to update users.email.
        """
        u, ident = self._ensure_identity(provider=PROVIDER_TG, external_id=str(telegram_user_id))
        # keep email on users table
        if "email" in fields:
            u.email = fields.pop("email")
        meta = ident.identity_metadata or {}
        meta.update({k: v for k, v in fields.items() if v is not None})
        ident.identity_metadata = meta
        self.s.flush()

    def get_telegram_profile(self, telegram_user_id: str | int) -> Optional[Dict[str, Any]]:
        u = self.get_user_by_telegram_id(telegram_user_id)
        if not u:
            return None
        ident = self._get_identity(provider=PROVIDER_TG, external_id=str(telegram_user_id))
        meta = (ident.identity_metadata or {}) if ident else {}
        return {
            "user_id": u.id,
            "telegram_user_id": ident.external_id if ident else str(telegram_user_id),
            "email": u.email,
            "verified": meta.get("verified"),
            "approved": meta.get("approved"),
            "language": meta.get("language"),
            "is_admin": meta.get("is_admin"),
            "max_alerts": meta.get("max_alerts"),
            "max_schedules": meta.get("max_schedules"),
            "verification_code": meta.get("verification_code"),
            "code_sent_time": meta.get("code_sent_time"),
        }

    def list_telegram_users_dto(self) -> List[Dict[str, Any]]:
        rows = self.s.execute(
            select(User, AuthIdentity)
            .join(AuthIdentity, AuthIdentity.user_id == User.id)
            .where(AuthIdentity.provider == PROVIDER_TG)
        ).all()
        out: List[Dict[str, Any]] = []
        for u, ident in rows:
            m = ident.identity_metadata or {}
            out.append({
                "user_id": u.id,
                "telegram_user_id": ident.external_id,
                "email": u.email,
                "verified": m.get("verified"),
                "approved": m.get("approved"),
                "language": m.get("language"),
                "is_admin": m.get("is_admin"),
                "max_alerts": m.get("max_alerts"),
                "max_schedules": m.get("max_schedules"),
                "verification_code": m.get("verification_code"),
                "code_sent_time": m.get("code_sent_time"),
            })
        return out

    def list_pending_telegram_approvals(self) -> List[Dict[str, Any]]:
        rows = self.list_telegram_users_dto()
        return [
            r for r in rows
            if r.get("verified") is True and r.get("approved") is not True
        ]

    def get_admin_telegram_user_ids(self) -> List[str]:
        rows = self.s.execute(
            select(AuthIdentity.external_id, AuthIdentity.identity_metadata)
            .where(AuthIdentity.provider == PROVIDER_TG)
        ).all()
        return [ext for ext, meta in rows if ext and (meta or {}).get("is_admin") is True]

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        q = select(User).where(User.id == user_id)
        return self.s.execute(q).scalar_one_or_none()

    def get_user_notification_channels(self, user_id: int) -> Optional[Dict[str, str]]:
        """
        Get user's notification channels (email + telegram_chat_id).

        Args:
            user_id: User ID

        Returns:
            Dictionary with email and telegram_chat_id, or None if user not found
            {
                "email": "user@example.com",
                "telegram_chat_id": "123456789"
            }
        """
        # Get user and telegram identity in one query
        q = (
            select(User, AuthIdentity)
            .outerjoin(AuthIdentity, (AuthIdentity.user_id == User.id) &
                      (AuthIdentity.provider == PROVIDER_TG))
            .where(User.id == user_id)
        )
        row = self.s.execute(q).first()

        if not row:
            return None

        user, telegram_identity = row

        result = {
            "email": user.email,
            "telegram_chat_id": telegram_identity.external_id if telegram_identity else None
        }

        return result

# -------------------- Verification --------------------
class VerificationRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def issue(self, user_id: int, *, code: str, sent_time: int) -> VerificationCode:
        row = VerificationCode(user_id=user_id, code=code, sent_time=sent_time)
        self.s.add(row); self.s.flush(); return row

    def count_last_hour_by_user_id(self, user_id: int, now_unix: int) -> int:
        cutoff = now_unix - 3600
        q = select(func.count(VerificationCode.id)).where(
            VerificationCode.user_id == user_id,
            VerificationCode.sent_time >= cutoff,
        )
        return int(self.s.execute(q).scalar_one() or 0)


