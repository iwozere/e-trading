# model_users.py  (patched)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, DateTime, ForeignKey, JSON, text, UniqueConstraint, Index, CheckConstraint

from src.data.db.core.base import Base


class User(Base):
    __tablename__ = "users"
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

class AuthIdentity(Base):
    __tablename__ = "auth_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(255))
    # IMPORTANT: attribute renamed; keep DB column name as "metadata"
    identity_metadata: Mapped[dict | None] = mapped_column("metadata", JSON)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_auth_identities_provider_external"),
        Index("ix_auth_identities_provider_external", "provider", "external_id"),
        Index("idx_auth_identities_user", "user_id"),
    )

    # Convenience getters/setters
    def meta_get(self, key: str, default=None):
        return (self.metadata or {}).get(key, default)

    def meta_set(self, key: str, value):
        md = dict(self.metadata or {})
        md[key] = value
        self.metadata = md
