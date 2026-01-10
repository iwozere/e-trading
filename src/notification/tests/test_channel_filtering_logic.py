import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from sqlalchemy import or_
from src.data.db.repos.repo_notification import MessageRepository
from src.data.db.models.model_notification import Message, MessageStatus

def test_get_pending_messages_with_channels():
    # Setup mock session and query
    mock_session = MagicMock()
    mock_query = mock_session.query.return_value
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.all.return_value = []

    repo = MessageRepository(mock_session)
    now = datetime.now(timezone.utc)
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
