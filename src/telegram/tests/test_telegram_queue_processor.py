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


def test_split_text_keeps_short_text_whole() -> None:
    """Text within the limit is returned as one chunk, unchanged."""
    assert tqp._split_text("a\nb", limit=10) == ["a\nb"]


def test_split_text_breaks_at_line_ends() -> None:
    """Long text is split between lines, every chunk within the limit, nothing lost."""
    text = "\n".join(f"line {i:02d}" for i in range(10))  # 10 lines of 7 chars

    chunks = tqp._split_text(text, limit=20)

    assert all(len(c) <= 20 for c in chunks)
    assert "\n".join(chunks) == text
    assert chunks[0] == "line 00\nline 01"


def test_split_text_cuts_overlong_line() -> None:
    """A single line longer than the limit is cut hard."""
    assert tqp._split_text("x" * 25, limit=10) == ["x" * 10, "x" * 10, "x" * 5]


def test_split_text_counts_emoji_as_two_units() -> None:
    """Limits are in UTF-16 code units, as Telegram counts them."""
    chunks = tqp._split_text("\U0001f53a" * 6, limit=4)

    assert chunks == ["\U0001f53a" * 2] * 3


def test_long_message_sent_in_chunks(processor: tqp.TelegramQueueProcessor, bot: Mock, queue_client: Mock) -> None:
    """A message over Telegram's limit is delivered as several messages instead of failing."""
    text = "\n".join("row " + "x" * 90 for _ in range(100))  # ~9.5k chars

    asyncio.run(processor._process_message(_delivery(text)))

    sent = [call.kwargs["text"] for call in bot.send_message.await_args_list]
    assert len(sent) == 3
    assert all(tqp._utf16_len(s) <= tqp._TEXT_LIMIT for s in sent)
    assert "\n".join(sent) == text
    queue_client.record_delivery_success.assert_called_once_with(7)


def test_long_caption_sent_as_separate_text(processor: tqp.TelegramQueueProcessor, bot: Mock) -> None:
    """Text over the caption limit is not used as a caption but sent as its own message."""
    bot.send_document = AsyncMock()
    text = "y" * (tqp._CAPTION_LIMIT + 1)

    asyncio.run(processor._send_with_attachments(123, text, {"a.txt": b"data"}))

    assert bot.send_document.await_args.kwargs["caption"] is None
    assert bot.send_message.await_args.kwargs["text"] == text
