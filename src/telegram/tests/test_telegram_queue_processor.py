"""
Tests for TelegramQueueProcessor delivery error handling.

Covers permanent-failure classification (unreachable chats must not burn the retry
budget) and the plain-text fallback when Telegram rejects message HTML.
"""

import asyncio
from pathlib import Path
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramNetworkError
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.telegram.services import telegram_queue_processor as tqp  # noqa: E402


def _bad_request(description: str) -> TelegramBadRequest:
    """
    Build a TelegramBadRequest as aiogram raises it.

    Args:
        description: Telegram error description.

    Returns:
        TelegramBadRequest instance.
    """
    return TelegramBadRequest(method=Mock(), message=description)


@pytest.fixture
def queue_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the shared message queue client with a mock."""
    client = Mock()
    monkeypatch.setattr(tqp, "get_message_queue_client", lambda: client)
    return client


@pytest.fixture
def bot() -> Mock:
    """Mock aiogram Bot whose send_message is awaitable."""
    mock_bot = Mock()
    mock_bot.send_message = AsyncMock()
    return mock_bot


@pytest.fixture
def processor(bot: Mock, queue_client: Mock) -> tqp.TelegramQueueProcessor:
    """TelegramQueueProcessor wired to the mocked bot and queue client."""
    assert queue_client is not None  # requested so the client is patched before construction
    return tqp.TelegramQueueProcessor(bot)


def _delivery(text: str = "hello") -> Dict[str, Any]:
    """
    Build a claimed-delivery dict with an explicit chat ID.

    Args:
        text: Message body.

    Returns:
        Claimed-delivery dict as returned by claim_pending_deliveries.
    """
    return {
        "delivery_status_id": 7,
        "message_id": 42,
        "recipient_id": "5",
        "message_metadata": {"telegram_chat_id": "123456789"},
        "content": {"message": text},
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_bad_request("Bad Request: chat not found"), True),
        (_bad_request("Bad Request: user is deactivated"), True),
        (TelegramForbiddenError(method=Mock(), message="Forbidden: bot was blocked by the user"), True),
        (_bad_request("Bad Request: message is too long"), False),
        (TelegramNetworkError(method=Mock(), message="timeout"), False),
    ],
)
def test_is_permanent_telegram_error(error: Exception, expected: bool) -> None:
    """Unreachable-recipient errors are permanent; transient or content errors are not."""
    assert tqp.TelegramQueueProcessor._is_permanent_telegram_error(error) is expected  # type: ignore[arg-type]


def test_chat_not_found_records_permanent_failure(
    processor: tqp.TelegramQueueProcessor, bot: Mock, queue_client: Mock
) -> None:
    """A 'chat not found' delivery is failed permanently instead of being retried."""
    bot.send_message.side_effect = _bad_request("Bad Request: chat not found")

    asyncio.run(processor._process_message(_delivery()))

    queue_client.record_delivery_failure.assert_called_once()
    assert queue_client.record_delivery_failure.call_args.kwargs["permanent"] is True
    queue_client.record_delivery_success.assert_not_called()


def test_transient_error_records_retryable_failure(
    processor: tqp.TelegramQueueProcessor, bot: Mock, queue_client: Mock
) -> None:
    """Non-permanent Telegram errors keep using the retry budget."""
    bot.send_message.side_effect = _bad_request("Bad Request: message is too long")

    asyncio.run(processor._process_message(_delivery()))

    assert queue_client.record_delivery_failure.call_args.kwargs["permanent"] is False


def test_unparseable_html_falls_back_to_plain_text(
    processor: tqp.TelegramQueueProcessor, bot: Mock, queue_client: Mock
) -> None:
    """Text Telegram rejects as HTML is resent without parse_mode and recorded as delivered."""
    parse_error = _bad_request('Bad Request: can\'t parse entities: Unsupported start tag "message" at byte offset 288')
    bot.send_message.side_effect = [parse_error, None]
    send = bot.send_message

    asyncio.run(processor._process_message(_delivery("Usage: /feedback <message>")))

    assert send.await_count == 2
    assert send.await_args_list[0].kwargs["parse_mode"] == "HTML"
    assert send.await_args_list[1].kwargs["parse_mode"] is None
    queue_client.record_delivery_success.assert_called_once_with(7)
    queue_client.record_delivery_failure.assert_not_called()
