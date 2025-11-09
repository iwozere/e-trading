# model_users.py  (patched)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, DateTime, ForeignKey, text, UniqueConstraint, Index, CheckConstraint
from src.data.db.core.json_types import JsonType
from typing import Dict, Any, Optional

from src.data.db.core.base import Base


class User(Base):
    __tablename__ = "usr_users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str | None] = mapped_column(String(100), unique=True)
    role: Mapped[str] = mapped_column(String(20), server_default="trader")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
    last_login: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("role IN ('admin','trader','viewer')", name="ck_users_role"),
        # Optional: if you want the (non-unique) helper indexes exactly as in DB:
        Index("ix_users_email", "email"),
    )

    @property
    def username(self) -> Optional[str]:
        """Get username from email (part before @)."""
        if self.email:
            return self.email.split('@')[0]
        return None

    def verify_password(self, password: str) -> bool:
        """
        Verify password for user.

        For now, this is a simple implementation for default users.
        In production, this should use proper password hashing.
        """
        if not self.email:
            return False

        # For existing users, use simple password verification
        # This is temporary - in production use proper password hashing
        username = self.email.split('@')[0]

        # Allow the username as password for simplicity
        if password == username:
            return True

        # Also allow some common passwords for testing
        if password in ["password", "123456", "admin", "trader", "viewer"]:
            return True

        # For default users, allow role-based passwords
        if password == self.role:
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }

class AuthIdentity(Base):
    __tablename__ = "usr_auth_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("usr_users.id", ondelete="CASCADE"))
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    # IMPORTANT: attribute renamed; keep DB column name as "metadata"
    identity_metadata: Mapped[dict | None] = mapped_column("metadata", JsonType, nullable=True)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_auth_identities_provider_external"),
        Index("ix_auth_identities_provider_external", "provider", "external_id"),
        Index("ix_auth_identities_provider", "provider"),
        Index("ix_auth_identities_user_id", "user_id"),
    )

    # Convenience getters/setters
    def meta_get(self, key: str, default=None):
        return (self.identity_metadata or {}).get(key, default)

    def meta_set(self, key: str, value):
        md = dict(self.identity_metadata or {})
        md[key] = value
        self.identity_metadata = md

# telegram_verification_codes
class VerificationCode(Base):
    __tablename__ = "usr_verification_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("usr_users.id", ondelete="CASCADE"), nullable=False)
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    sent_time: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[Optional[str]] = mapped_column(String(20), server_default="'telegram'", nullable=True)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), nullable=True)

    __table_args__ = (
        Index("ix_verification_codes_user_id", "user_id"),
    )