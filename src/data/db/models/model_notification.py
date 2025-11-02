"""
Notification Service Models

SQLAlchemy models for the notification service system.
Includes Message, DeliveryStatus, ChannelHealth, RateLimit, and ChannelConfig models.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, BigInteger, JSON,
    CheckConstraint, UniqueConstraint, Index, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
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

    id = Column(BigInteger, primary_key=True, index=True)
    message_type = Column(String(50), nullable=False, index=True)
    priority = Column(String(20), nullable=False, default=MessagePriority.NORMAL.value)
    channels = Column(ARRAY(String), nullable=False)
    recipient_id = Column(String(100), nullable=True, index=True)
    template_name = Column(String(100), nullable=True)
    content = Column(JsonType(), nullable=False)
    message_metadata = Column("metadata", JsonType(), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    status = Column(String(20), nullable=False, default=MessageStatus.PENDING.value, index=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    last_error = Column(Text, nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Distributed processing lock fields
    locked_by = Column(String(100), nullable=True, index=True)
    locked_at = Column(DateTime(timezone=True), nullable=True, index=True)

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

    id = Column(BigInteger, primary_key=True, index=True)
    message_id = Column(BigInteger, ForeignKey("msg_messages.id", ondelete="CASCADE"), nullable=False, index=True)
    channel = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True, index=True)
    response_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    external_id = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

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

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    channel = Column(String(50), nullable=False, index=True)
    tokens = Column(Integer, nullable=False)
    last_refill = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    max_tokens = Column(Integer, nullable=False)
    refill_rate = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

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

    id = Column(BigInteger, primary_key=True, index=True)
    channel = Column(String(50), nullable=False, unique=True)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    config = Column(JsonType(), nullable=False)
    rate_limit_per_minute = Column(Integer, nullable=False, default=60)
    max_retries = Column(Integer, nullable=False, default=3)
    timeout_seconds = Column(Integer, nullable=False, default=30)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now(), index=True)

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
    scheduled_for: Optional[datetime] = None
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
    processed_at: Optional[datetime] = None


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
    created_at: datetime
    scheduled_for: datetime
    status: MessageStatus
    retry_count: int
    max_retries: int
    last_error: Optional[str]
    processed_at: Optional[datetime]
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
    delivered_at: Optional[datetime] = None
    response_time_ms: Optional[int] = Field(None, ge=0)
    error_message: Optional[str] = None
    external_id: Optional[str] = Field(None, max_length=255)


class DeliveryStatusResponse(BaseModel):
    """Pydantic model for delivery status API responses."""
    id: int
    message_id: int
    channel: str
    status: DeliveryStatus
    delivered_at: Optional[datetime]
    response_time_ms: Optional[int]
    error_message: Optional[str]
    external_id: Optional[str]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ChannelHealthResponse(BaseModel):
    """Pydantic model for channel health API responses."""
    id: int
    channel: str
    status: SystemHealthStatus
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    failure_count: int
    avg_response_time_ms: Optional[int]
    error_message: Optional[str]
    checked_at: datetime
    model_config = ConfigDict(from_attributes=True)


class RateLimitResponse(BaseModel):
    """Pydantic model for rate limit API responses."""
    id: int
    user_id: str
    channel: str
    tokens: int
    max_tokens: int
    refill_rate: int
    last_refill: datetime
    created_at: datetime
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
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)
