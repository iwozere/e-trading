"""
Database Models for Trading Web UI
---------------------------------

SQLAlchemy models for user authentication, strategy management,
and system configuration.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=True)
    role = Column(String(20), nullable=False, default="trader")  # admin, trader, viewer
    is_active = Column(Boolean, default=True)

    # Telegram integration fields
    telegram_user_id = Column(String(255), unique=True, index=True, nullable=True)
    telegram_verified = Column(Boolean, default=False)
    telegram_approved = Column(Boolean, default=False)
    telegram_language = Column(String(10), nullable=True)
    telegram_is_admin = Column(Boolean, default=False)
    telegram_max_alerts = Column(Integer, nullable=True)
    telegram_max_schedules = Column(Integer, nullable=True)
    telegram_verification_code = Column(String(32), nullable=True)
    telegram_code_sent_time = Column(Integer, nullable=True)

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    audit_logs = relationship("AuditLog", back_populates="user")

    def get_username(self) -> str:
        """Get username - use email if verified, otherwise telegram_user_id."""
        if self.email and (self.telegram_verified or self.telegram_approved):
            return self.email
        elif self.telegram_user_id:
            return f"telegram_{self.telegram_user_id}"
        else:
            return f"user_{self.id}"

    def verify_password(self, password: str) -> bool:
        """Temporary password verification - allow empty password or fixed value."""
        # For now, allow empty password or "temp" for 2FA transition
        return password == "" or password == "temp"

    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding sensitive data)."""
        return {
            "id": self.id,
            "username": self.get_username(),
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "telegram_user_id": self.telegram_user_id,
            "telegram_verified": self.telegram_verified,
            "telegram_approved": self.telegram_approved,
            "telegram_is_admin": self.telegram_is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class StrategyTemplate(Base):
    """Strategy template model for reusable configurations."""

    __tablename__ = "strategy_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    template_data = Column(JSON, nullable=False)
    is_public = Column(Boolean, default=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    creator = relationship("User")


class SystemConfig(Base):
    """System configuration model."""

    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class AuditLog(Base):
    """Audit log model for tracking user actions."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(100), nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="audit_logs")


class PerformanceSnapshot(Base):
    """Performance snapshot model for storing periodic performance data."""

    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    pnl = Column(JSON, nullable=False)  # {"total": 0.0, "realized": 0.0, "unrealized": 0.0}
    positions = Column(JSON, nullable=True)
    trades_count = Column(Integer, default=0)
    win_rate = Column(JSON, nullable=True)  # {"wins": 0, "losses": 0, "rate": 0.0}
    drawdown = Column(JSON, nullable=True)  # {"current": 0.0, "max": 0.0}
    metrics = Column(JSON, nullable=True)  # Additional performance metrics