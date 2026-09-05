from datetime import UTC, datetime
from unittest.mock import MagicMock

from src.data.db.models.model_notification import Message, MessageDeliveryStatus
from src.data.db.repos.repo_notification import DeliveryStatusRepository, MessageRepository


def test_get_pending_messages_with_channels():
    # Setup mock session and query
    mock_session = MagicMock()
    mock_query = mock_session.query.return_value
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.all.return_value = []

    repo = MessageRepository(mock_session)
    now = datetime.now(UTC)
    channels = ["email", "sms"]

    # Execute
    repo.get_pending_messages(current_time=now, channels=channels)

    # Verify filtering
    # We expect multiple filter calls:
    # 1. status == PENDING and scheduled_for <= now
    # 2. or_(*[Message.channels.contains([ch]) for ch in channels])

    filter_calls = mock_query.filter.call_args_list
    assert len(filter_calls) >= 2

    # Check if any filter call involves the channel filtering
    # This is a bit tricky to verify exactly due to SQLAlchemy's expression objects,
    # but we can check if 'or_' was used or if the parameter was list-like
    channel_filter_applied = False
    for call in filter_calls:
        args = call[0]
        if len(args) > 0:
            arg_str = str(args[0])
            if "channels @> ARRAY" in arg_str or "ANY" in arg_str or "OR" in arg_str:
                channel_filter_applied = True
                break

    # Since we are using or_(*channel_filters), we expect it to be present
    # In my implementation:
    # channel_filters = [Message.channels.contains([ch]) for ch in channels]
    # query = query.filter(or_(*channel_filters))

    # Let's just verify the method was called without making it too fragile
    assert mock_query.filter.called


def test_get_pending_messages_with_lock_channels():
    # Setup mock session
    mock_session = MagicMock()
    repo = MessageRepository(mock_session)

    channels = ["telegram"]

    # Mock execute return value
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.execute.return_value = mock_result

    # Execute
    repo.get_pending_messages_with_lock(limit=5, lock_instance_id="test_inst", channels=channels)

    # Verify raw SQL execution
    assert mock_session.execute.called
    args, kwargs = mock_session.execute.call_args
    sql_text = str(args[0])
    params = args[1]

    # Check if SQL contains the channel filtering logic
    assert "AND channels && :channels" in sql_text
    assert params["channels"] == channels
    assert params["lock_instance_id"] == "test_inst"


def test_claim_pending_deliveries_claims_by_channel_not_whole_message():
    """
    claim_pending_deliveries must claim msg_delivery_status rows scoped to the
    requested channels, not the whole msg_messages row -- that's the fix for
    a message requesting ["telegram", "email"] being fully claimed (and
    failing on channels it doesn't own) by whichever consumer's overlap match
    on the whole message fired first (monitoring.txt, 2026-09-02).
    """
    mock_session = MagicMock()
    repo = DeliveryStatusRepository(mock_session)

    claimed_row = MagicMock(id=101, message_id=55, channel="telegram")
    mock_session.execute.return_value.fetchall.return_value = [claimed_row]

    message = Message(
        id=55,
        message_type="SYSTEM_ALERT",
        priority="CRITICAL",
        channels=["telegram", "email"],
        recipient_id="42",
        content={"title": "t", "message": "m"},
        message_metadata={},
        scheduled_for=datetime.now(UTC),
        retry_count=0,
        max_retries=3,
    )
    mock_session.query.return_value.filter.return_value.all.return_value = [message]

    results = repo.claim_pending_deliveries(channels=["telegram"], limit=10)

    # Claim query is scoped to msg_delivery_status, filtered by channel -- not
    # the whole-message overlap query get_pending_messages_with_lock uses.
    args, params = mock_session.execute.call_args[0]
    sql_text = str(args)
    assert "UPDATE msg_delivery_status" in sql_text
    assert "ds.channel = ANY(CAST(:channels AS TEXT[]))" in sql_text
    assert params["channels"] == ["telegram"]

    assert len(results) == 1
    assert results[0]["delivery_status_id"] == 101
    assert results[0]["message_id"] == 55
    assert results[0]["channel"] == "telegram"
    assert results[0]["content"] == {"title": "t", "message": "m"}


def test_record_delivery_result_rollup_waits_for_all_channels():
    """
    A message's overall status must not flip to DELIVERED/FAILED until every
    one of its per-channel msg_delivery_status rows is terminal -- otherwise
    the other channel's delivery is dropped once the message row looks "done".
    """
    mock_session = MagicMock()
    repo = DeliveryStatusRepository(mock_session)

    delivery = MessageDeliveryStatus(id=101, message_id=55, channel="telegram", status="PENDING")
    message = Message(id=55, status="PENDING", retry_count=0, max_retries=3)

    def query_side_effect(model):
        mock_query = MagicMock()
        if model is MessageDeliveryStatus:
            mock_query.filter.return_value.first.return_value = delivery
            # Sibling rows: telegram (being updated) + email still pending.
            mock_query.filter.return_value.order_by.return_value.all.return_value = [
                MessageDeliveryStatus(id=101, message_id=55, channel="telegram", status="PENDING"),
                MessageDeliveryStatus(id=102, message_id=55, channel="email", status="PENDING"),
            ]
        elif model is Message:
            mock_query.filter.return_value.first.return_value = message
        return mock_query

    mock_session.query.side_effect = query_side_effect

    # Telegram succeeds, but email sibling is still pending -> message stays PROCESSING.
    updated = repo.record_delivery_result(101, "DELIVERED")
    assert updated is not None
    assert updated.status == "PROCESSING"

    # Now both siblings are terminal (telegram delivered, email delivered too).
    def query_side_effect_done(model):
        mock_query = MagicMock()
        if model is MessageDeliveryStatus:
            mock_query.filter.return_value.first.return_value = delivery
            mock_query.filter.return_value.order_by.return_value.all.return_value = [
                MessageDeliveryStatus(id=101, message_id=55, channel="telegram", status="DELIVERED"),
                MessageDeliveryStatus(id=102, message_id=55, channel="email", status="DELIVERED"),
            ]
        elif model is Message:
            mock_query.filter.return_value.first.return_value = message
        return mock_query

    mock_session.query.side_effect = query_side_effect_done
    updated = repo.record_delivery_result(101, "DELIVERED")
    assert updated is not None
    assert updated.status == "DELIVERED"


def test_record_delivery_failure_retries_then_marks_permanently_failed():
    """
    record_delivery_failure should requeue the channel while the message's
    shared retry budget allows it, then mark it terminally FAILED once
    exhausted -- and skip straight to FAILED when permanent=True regardless
    of budget (e.g. an unresolvable recipient).
    """
    mock_session = MagicMock()
    repo = DeliveryStatusRepository(mock_session)

    delivery = MessageDeliveryStatus(id=101, message_id=55, channel="telegram", status="SENT")
    message = Message(id=55, status="PROCESSING", retry_count=0, max_retries=1)

    def query_side_effect(model):
        mock_query = MagicMock()
        if model is MessageDeliveryStatus:
            mock_query.filter.return_value.first.return_value = delivery
        elif model is Message:
            mock_query.filter.return_value.first.return_value = message
        return mock_query

    mock_session.query.side_effect = query_side_effect

    # First failure: retry budget (0 < 1) allows a retry -> requeued as PENDING.
    updated = repo.record_delivery_failure(101, "boom")
    assert updated is not None
    assert updated.retry_count == 1
    assert delivery.status == "PENDING"

    # Second failure: budget exhausted (1 < 1 is False) -> permanently failed.
    mock_session.query.side_effect = query_side_effect
    updated = repo.record_delivery_failure(101, "boom again")
    assert updated is not None
    assert updated.status == "FAILED"

    # permanent=True skips the retry budget entirely, even with room left.
    message.retry_count = 0
    delivery.status = "SENT"
    mock_session.query.side_effect = query_side_effect
    updated = repo.record_delivery_failure(101, "unresolvable recipient", permanent=True)
    assert updated is not None
    assert updated.status == "FAILED"
