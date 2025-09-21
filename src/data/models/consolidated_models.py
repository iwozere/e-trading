"""
Consolidated Database Models
---------------------------

Enhanced SQLAlchemy models for the consolidated database combining
web UI and Telegram user data with proper table prefixing.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, ForeignKey, Numeric, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any
import bcrypt

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

Base = declarative_base()


class User(Base):
    """
    Enhanced user model combining web UI and Telegram user data.

    This model serves as the central user entity for both web UI authentication
    and Telegram bot integration, eliminating the need for separate user tables.
    """

    __tablename__ = "users"

    # Core user fields
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=True)
    email = Column(String(100), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=True)
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
    webui_audit_logs = relationship("WebUIAuditLog", back_populates="user")
    webui_strategy_templates = relationship("WebUIStrategyTemplate", back_populates="creator")

    # Note: Telegram relationships are complex due to different foreign key structure
    # They will be handled through queries rather than direct relationships

    def verify_password(self, password: str) -> bool:
        """
        Verify password against stored hash.

        Args:
            password: Plain text password to verify

        Returns:
            bool: True if password matches
        """
        if not self.hashed_password:
            return False
        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password for storage.

        Args:
            password: Plain text password

        Returns:
            str: Hashed password
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def is_web_user(self) -> bool:
        """Check if user has web UI access (has username and password)."""
        return self.username is not None and self.hashed_password is not None

    def is_telegram_user(self) -> bool:
        """Check if user has Telegram integration."""
        return self.telegram_user_id is not None

    def is_linked_user(self) -> bool:
        """Check if user has both web UI and Telegram access."""
        return self.is_web_user() and self.is_telegram_user()

    def can_access_web_ui(self) -> bool:
        """Check if user can access web UI."""
        return self.is_web_user() and self.is_active

    def can_use_telegram_bot(self) -> bool:
        """Check if user can use Telegram bot."""
        return (self.is_telegram_user() and
                self.telegram_verified and
                self.telegram_approved and
                self.is_active)

    def to_dict(self, include_telegram: bool = True, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert user to dictionary.

        Args:
            include_telegram: Whether to include Telegram fields
            include_sensitive: Whether to include sensitive fields

        Returns:
            Dict: User data dictionary
        """
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_web_user": self.is_web_user(),
            "is_telegram_user": self.is_telegram_user(),
            "is_linked_user": self.is_linked_user()
        }

        if include_telegram:
            data.update({
                "telegram_user_id": self.telegram_user_id,
                "telegram_verified": self.telegram_verified,
                "telegram_approved": self.telegram_approved,
                "telegram_language": self.telegram_language,
                "telegram_is_admin": self.telegram_is_admin,
                "telegram_max_alerts": self.telegram_max_alerts,
                "telegram_max_schedules": self.telegram_max_schedules
            })

        if include_sensitive:
            data.update({
                "telegram_verification_code": self.telegram_verification_code,
                "telegram_code_sent_time": self.telegram_code_sent_time
            })

        return data

    def merge_telegram_data(self, telegram_data: Dict[str, Any]) -> bool:
        """
        Merge Telegram user data into this user record.

        Args:
            telegram_data: Dictionary containing Telegram user data

        Returns:
            bool: True if merge successful
        """
        try:
            # Map telegram_users fields to User model fields
            field_mapping = {
                'telegram_user_id': 'telegram_user_id',
                'email': 'email',  # Only update if current email is None
                'verification_code': 'telegram_verification_code',
                'code_sent_time': 'telegram_code_sent_time',
                'verified': 'telegram_verified',
                'approved': 'telegram_approved',
                'language': 'telegram_language',
                'is_admin': 'telegram_is_admin',
                'max_alerts': 'telegram_max_alerts',
                'max_schedules': 'telegram_max_schedules'
            }

            for telegram_field, user_field in field_mapping.items():
                if telegram_field in telegram_data:
                    value = telegram_data[telegram_field]

                    # Special handling for email - don't overwrite existing email
                    if telegram_field == 'email' and self.email is not None:
                        continue

                    setattr(self, user_field, value)

            # Update timestamps if provided
            if 'created_at' in telegram_data and self.created_at is None:
                self.created_at = telegram_data['created_at']

            if 'updated_at' in telegram_data:
                self.updated_at = telegram_data['updated_at']

            _logger.debug("Merged Telegram data for user %s", self.id)
            return True

        except Exception as e:
            _logger.error("Failed to merge Telegram data: %s", e)
            return False

    def __repr__(self):
        """String representation of user."""
        identifiers = []
        if self.username:
            identifiers.append(f"username='{self.username}'")
        if self.email:
            identifiers.append(f"email='{self.email}'")
        if self.telegram_user_id:
            identifiers.append(f"telegram_id='{self.telegram_user_id}'")

        return f"<User(id={self.id}, {', '.join(identifiers)})>"


class WebUIAuditLog(Base):
    """Web UI audit log model for tracking user actions."""

    __tablename__ = "webui_audit_logs"

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
    user = relationship("User", back_populates="webui_audit_logs")


class WebUIStrategyTemplate(Base):
    """Web UI strategy template model for reusable configurations."""

    __tablename__ = "webui_strategy_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    template_data = Column(JSON, nullable=False)
    is_public = Column(Boolean, default=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    creator = relationship("User", back_populates="webui_strategy_templates")


class WebUISystemConfig(Base):
    """Web UI system configuration model."""

    __tablename__ = "webui_system_config"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class WebUIPerformanceSnapshot(Base):
    """Web UI performance snapshot model for storing periodic performance data."""

    __tablename__ = "webui_performance_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    pnl = Column(JSON, nullable=False)  # {"total": 0.0, "realized": 0.0, "unrealized": 0.0}
    positions = Column(JSON, nullable=True)
    trades_count = Column(Integer, default=0)
    win_rate = Column(JSON, nullable=True)  # {"wins": 0, "losses": 0, "rate": 0.0}
    drawdown = Column(JSON, nullable=True)  # {"current": 0.0, "max": 0.0}
    metrics = Column(JSON, nullable=True)  # Additional performance metrics


# Telegram-related models (keeping existing structure but adding relationships)
class TelegramAlert(Base):
    """Telegram alert model."""

    __tablename__ = "telegram_alerts"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(50), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    price = Column(Numeric(20, 8), nullable=True)
    condition = Column(String(50), nullable=True)
    active = Column(Boolean, default=True, index=True)
    email = Column(Boolean, default=False)
    created = Column(String(40), nullable=True)
    alert_type = Column(String(20), default='price', index=True)
    timeframe = Column(String(10), default='15m')
    config_json = Column(Text, nullable=True)
    alert_action = Column(String(50), default='notify')
    re_arm_config = Column(Text, nullable=True)
    is_armed = Column(Boolean, default=True, index=True)
    last_price = Column(Numeric(20, 8), nullable=True)
    last_triggered_at = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_alerts_user_active', 'user_id', 'active'),
        Index('idx_alerts_armed', 'is_armed'),
    )

    # Note: Relationship will be established through telegram_user_id matching
    # user = relationship("User", foreign_keys="User.telegram_user_id", primaryjoin="TelegramAlert.user_id == User.telegram_user_id")


class TelegramSchedule(Base):
    """Telegram schedule model."""

    __tablename__ = "telegram_schedules"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(50), nullable=True)
    user_id = Column(String(255), nullable=False, index=True)
    scheduled_time = Column(String(20), nullable=False)
    period = Column(String(20), nullable=True)
    active = Column(Boolean, default=True, index=True)
    email = Column(Boolean, default=False)
    indicators = Column(Text, nullable=True)
    interval = Column(String(10), nullable=True)
    provider = Column(String(20), nullable=True)
    created = Column(String(40), nullable=True)
    schedule_type = Column(String(20), default='report', index=True)
    list_type = Column(String(50), nullable=True)
    config_json = Column(Text, nullable=True)
    schedule_config = Column(String(20), default='simple', index=True)

    # Note: Relationship will be established through telegram_user_id matching
    # user = relationship("User", foreign_keys="User.telegram_user_id", primaryjoin="TelegramSchedule.user_id == User.telegram_user_id")


class TelegramFeedback(Base):
    """Telegram feedback model."""

    __tablename__ = "telegram_feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # 'feedback' or 'feature_request'
    message = Column(Text, nullable=False)
    created = Column(String(40), nullable=True)
    status = Column(String(20), default='open')  # 'open', 'in_progress', 'closed'

    # Note: Relationship will be established through telegram_user_id matching
    # user = relationship("User", foreign_keys="User.telegram_user_id", primaryjoin="TelegramFeedback.user_id == User.telegram_user_id")