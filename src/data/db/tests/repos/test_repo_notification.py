from __future__ import annotations
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from src.data.db.repos.repo_notification import (
    NotificationRepository,
    ChannelConfigRepository,
    RateLimitRepository,
    MessageRepository,
    DeliveryStatusRepository,
)
from src.data.db.models.model_notification import MessagePriority, MessageStatus, DeliveryStatus

UTC = timezone.utc


def test_channel_config_crud(db_session: Session):
    configs = ChannelConfigRepository(db_session)
    c = configs.create_channel_config({
        "channel": "email",
        "enabled": True,
        "config": {"provider": "ses"},
    })
    assert c.channel == "email"

    got = configs.get_channel_config("email")
    assert got and got.enabled is True

    updated = configs.update_channel_config("email", {"enabled": False})
    assert updated and updated.enabled is False

    enabled_names = configs.get_enabled_channels()
    assert "email" not in enabled_names

    assert configs.delete_channel_config("email") is True


def test_rate_limit_consume(db_session: Session):
    rates = RateLimitRepository(db_session)
    ok = rates.check_and_consume_token("userA", "telegram", {"max_tokens": 2, "refill_rate": 2})
    assert ok is True
    ok2 = rates.check_and_consume_token("userA", "telegram", {"max_tokens": 2, "refill_rate": 2})
    assert ok2 is True
    # 3rd should rate limit (no commit, but flush performed inside)
    ok3 = rates.check_and_consume_token("userA", "telegram", {"max_tokens": 2, "refill_rate": 2})
    assert ok3 is False


def test_messages_and_delivery_status(db_session: Session):
    messages = MessageRepository(db_session)
    deliveries = DeliveryStatusRepository(db_session)

    now = datetime.now(UTC)
    msg = messages.create_message({
        "message_type": "info",
        "recipient_id": "r1",
        "payload": {"text": "hello"},
        "priority": MessagePriority.NORMAL.value,
        "status": MessageStatus.PENDING.value,
        "scheduled_for": now - timedelta(minutes=1),
    })
    assert msg.id is not None

    # Pending fetch
    pend = messages.get_pending_messages(current_time=now)
    assert any(m.id == msg.id for m in pend)

    # Update message -> mark delivered
    messages.update_message(msg.id, {"status": MessageStatus.DELIVERED.value, "processed_at": now})

    # Delivery status record
    ds = deliveries.create_delivery_status({
        "message_id": msg.id,
        "channel": "telegram",
        "status": DeliveryStatus.DELIVERED.value,
        "response_time_ms": 120,
    })
    assert ds.id is not None

    stats = deliveries.get_delivery_statistics(days=1)
    assert stats["total_deliveries"] >= 1
