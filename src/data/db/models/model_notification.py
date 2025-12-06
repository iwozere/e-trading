"""
Notification Service Models

SQLAlchemy models for the notification service system.
Includes Message, DeliveryStatus, ChannelHealth, RateLimit, and ChannelConfig models.
"""

from __future__ import annotations
import datetime
from datetime import datetime as dt
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Integer, String, Boolean, DateTime, Text, BigInteger, CheckConstraint, UniqueConstraint, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship, Mapped, mapped_column
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src.data.db.core.base import Base
from src.data.db.core.json_types import JsonType
from src.data.db.models.model_system_health import SystemHealthStatus
# Logger will be set up by importing modules as needed


class MessagePriority(str, Enum):
    """Message priority enumeration."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MessageStatus(str, Enum):
    """Message status enumeration."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DeliveryStatus(str, Enum):
    """Delivery status enumeration."""
    PENDING = "PENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    BOUNCED = "BOUNCED"


class Message(Base):
    """Message model for notification messages queue."""

    __tablename__ = "msg_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    message_type: Mapped[str] = mapped_column(String(50), index=True)
    priority: Mapped[str] = mapped_column(String(20), default=MessagePriority.NORMAL.value)
    channels: Mapped[list[str]] = mapped_column(ARRAY(String))
    recipient_id: Mapped[str | None] = mapped_column(String(100), index=True)
    template_name: Mapped[str | None] = mapped_column(String(100))
    content: Mapped[dict] = mapped_column(JsonType())
    message_metadata: Mapped[dict | None] = mapped_column("metadata", JsonType())
    created_at: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now(), index=True)
    scheduled_for: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now(), index=True)
    status: Mapped[str] = mapped_column(String(20), default=MessageStatus.PENDING.value, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    last_error: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True))

    # Distributed processing lock fields
    locked_by: Mapped[str | None] = mapped_column(String(100), index=True)
    locked_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True), index=True)

    # Relationships
    delivery_statuses = relationship("MessageDeliveryStatus", back_populates="message", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint("priority IN ('LOW', 'NORMAL', 'HIGH', 'CRITICAL')", name="check_message_priority"),
        CheckConstraint("status IN ('PENDING', 'PROCESSING', 'DELIVERED', 'FAILED', 'CANCELLED')", name="check_message_status"),
        CheckConstraint("retry_count >= 0", name="check_retry_count_positive"),
        CheckConstraint("max_retries >= 0", name="check_max_retries_positive"),
    )

    def __repr__(self):
        return f"<Message(id={self.id}, type='{self.message_type}', priority='{self.priority}', status='{self.status}')>"

    @property
    def is_high_priority(self) -> bool:
        """Check if message is high priority (HIGH or CRITICAL)."""
        return self.priority in [MessagePriority.HIGH.value, MessagePriority.CRITICAL.value]

    @property
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries and self.status == MessageStatus.FAILED.value


class MessageDeliveryStatus(Base):
    """Delivery status model for tracking per-channel delivery."""

    __tablename__ = "msg_delivery_status"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    message_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("msg_messages.id", ondelete="CASCADE"), index=True)
    channel: Mapped[str] = mapped_column(String(50), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    delivered_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True), index=True)
    response_time_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    external_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now())

    # Relationships
    message = relationship("Message", back_populates="delivery_statuses")

    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('PENDING', 'SENT', 'DELIVERED', 'FAILED', 'BOUNCED')", name="check_delivery_status"),
        CheckConstraint("response_time_ms >= 0", name="check_response_time_positive"),
    )

    def __repr__(self):
        return f"<MessageDeliveryStatus(id={self.id}, message_id={self.message_id}, channel='{self.channel}', status='{self.status}')>"


class RateLimit(Base):
    """Rate limit model for per-user throttling."""

    __tablename__ = "msg_rate_limits"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    channel: Mapped[str] = mapped_column(String(50), index=True)
    tokens: Mapped[int] = mapped_column(Integer)
    last_refill: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now(), index=True)
    max_tokens: Mapped[int] = mapped_column(Integer)
    refill_rate: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "channel", name="unique_user_channel_rate_limit"),
        CheckConstraint("tokens >= 0", name="check_tokens_positive"),
        CheckConstraint("max_tokens > 0", name="check_max_tokens_positive"),
        CheckConstraint("refill_rate > 0", name="check_refill_rate_positive"),
    )

    def __repr__(self):
        return f"<RateLimit(id={self.id}, user_id='{self.user_id}', channel='{self.channel}', tokens={self.tokens})>"

    @property
    def is_available(self) -> bool:
        """Check if tokens are available."""
        return self.tokens > 0

    def consume_token(self) -> bool:
        """Consume a token if available."""
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False

    def refill_tokens(self, current_time: datetime) -> None:
        """Refill tokens based on time elapsed."""
        if self.last_refill is None:
            self.last_refill = current_time
            return

        time_diff = (current_time - self.last_refill).total_seconds()
        minutes_elapsed = time_diff / 60.0

        if minutes_elapsed >= 1.0:
            tokens_to_add = int(minutes_elapsed * self.refill_rate)
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = current_time


class ChannelConfig(Base):
    """Channel configuration model for plugin settings."""

    __tablename__ = "msg_channel_configs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    channel: Mapped[str] = mapped_column(String(50), unique=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    config: Mapped[dict] = mapped_column(JsonType())
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, default=60)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=30)
    created_at: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), index=True)

    # Constraints
    __table_args__ = (
        CheckConstraint("rate_limit_per_minute > 0", name="check_rate_limit_positive"),
        CheckConstraint("max_retries >= 0", name="check_max_retries_positive"),
        CheckConstraint("timeout_seconds > 0", name="check_timeout_positive"),
    )

    def __repr__(self):
        return f"<ChannelConfig(id={self.id}, channel='{self.channel}', enabled={self.enabled})>"


# Pydantic models for API validation
class MessageCreate(BaseModel):
    """Pydantic model for creating a message."""
    message_type: str = Field(..., min_length=1, max_length=50)
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    channels: List[str] = Field(..., min_length=1)
    recipient_id: Optional[str] = Field(None, max_length=100)
    template_name: Optional[str] = Field(None, max_length=100)
    content: Dict[str, Any] = Field(...)
    message_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    scheduled_for: Optional[dt] = None
    max_retries: int = Field(default=3, ge=0)

    @field_validator('channels')
    def validate_channels(cls, v):
        """Validate channels list."""
        if not v:
            raise ValueError('At least one channel must be specified')
        return v


class MessageUpdate(BaseModel):
    """Pydantic model for updating a message."""
    status: Optional[MessageStatus] = None
    retry_count: Optional[int] = Field(None, ge=0)
    last_error: Optional[str] = None
    processed_at: Optional[dt] = None


class MessageResponse(BaseModel):
    """Pydantic model for message API responses."""
    id: int
    message_type: str
    priority: MessagePriority
    channels: List[str]
    recipient_id: Optional[str]
    template_name: Optional[str]
    content: Dict[str, Any]
    message_metadata: Optional[Dict[str, Any]]
    created_at: dt
    scheduled_for: dt
    status: MessageStatus
    retry_count: int
    max_retries: int
    last_error: Optional[str]
    processed_at: Optional[dt]
    model_config = ConfigDict(from_attributes=True)


class DeliveryStatusCreate(BaseModel):
    """Pydantic model for creating delivery status."""
    message_id: int
    channel: str = Field(..., min_length=1, max_length=50)
    status: DeliveryStatus = Field(default=DeliveryStatus.PENDING)
    external_id: Optional[str] = Field(None, max_length=255)


class DeliveryStatusUpdate(BaseModel):
    """Pydantic model for updating delivery status."""
    status: Optional[DeliveryStatus] = None
    delivered_at: Optional[dt] = None
    response_time_ms: Optional[int] = Field(None, ge=0)
    error_message: Optional[str] = None
    external_id: Optional[str] = Field(None, max_length=255)


class DeliveryStatusResponse(BaseModel):
    """Pydantic model for delivery status API responses."""
    id: int
    message_id: int
    channel: str
    status: DeliveryStatus
    delivered_at: Optional[dt]
    response_time_ms: Optional[int]
    error_message: Optional[str]
    external_id: Optional[str]
    created_at: dt
    model_config = ConfigDict(from_attributes=True)


class ChannelHealthResponse(BaseModel):
    """Pydantic model for channel health API responses."""
    id: int
    channel: str
    status: SystemHealthStatus
    last_success: Optional[dt]
    last_failure: Optional[dt]
    failure_count: int
    avg_response_time_ms: Optional[int]
    error_message: Optional[str]
    checked_at: dt
    model_config = ConfigDict(from_attributes=True)


class RateLimitResponse(BaseModel):
    """Pydantic model for rate limit API responses."""
    id: int
    user_id: str
    channel: str
    tokens: int
    max_tokens: int
    refill_rate: int
    last_refill: dt
    created_at: dt
    model_config = ConfigDict(from_attributes=True)


class ChannelConfigCreate(BaseModel):
    """Pydantic model for creating channel config."""
    channel: str = Field(..., min_length=1, max_length=50)
    enabled: bool = Field(default=True)
    config: Dict[str, Any] = Field(...)
    rate_limit_per_minute: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=30, gt=0)


class ChannelConfigUpdate(BaseModel):
    """Pydantic model for updating channel config."""
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    rate_limit_per_minute: Optional[int] = Field(None, gt=0)
    max_retries: Optional[int] = Field(None, ge=0)
    timeout_seconds: Optional[int] = Field(None, gt=0)


class ChannelConfigResponse(BaseModel):
    """Pydantic model for channel config API responses."""
    id: int
    channel: str
    enabled: bool
    config: Dict[str, Any]
    rate_limit_per_minute: int
    max_retries: int
    timeout_seconds: int
    created_at: dt
    updated_at: dt
    model_config = ConfigDict(from_attributes=True)
