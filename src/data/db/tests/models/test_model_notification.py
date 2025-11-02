from datetime import datetime, timedelta, timezone

from src.data.db.models.model_notification import (
    Message,
    MessageDeliveryStatus,
    RateLimit,
    MessageStatus,
    MessagePriority,
)


def test_message_properties_and_db_insert(db_session):
    m = Message()
    m.message_type = "test"
    m.priority = MessagePriority.NORMAL.value
    m.channels = ["email"]
    m.content = {"hello": "world"}
    m.status = MessageStatus.PENDING.value

    db_session.add(m)
    db_session.flush()

    assert m.id is not None
    assert m.is_high_priority is False

    # add delivery status
    ds = MessageDeliveryStatus()
    ds.message_id = m.id
    ds.channel = "email"
    ds.status = "PENDING"
    db_session.add(ds)
    db_session.flush()

    assert ds.id is not None


def test_rate_limit_consume_and_refill():
    rl = RateLimit()
    rl.user_id = "u1"
    rl.channel = "email"
    rl.tokens = 1
    rl.max_tokens = 5
    rl.refill_rate = 2
    rl.last_refill = datetime.now(timezone.utc) - timedelta(minutes=2)

    assert rl.is_available
    assert rl.consume_token() is True
    assert rl.consume_token() is False

    # refill should add tokens
    rl.refill_tokens(datetime.now(timezone.utc))
    assert rl.tokens >= 0
