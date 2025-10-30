import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiogram.types import Message
import src.telegram.bot as bot_module

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.telegram_service_instance")
@patch("src.telegram.bot.telegram_service")
async def test_register_valid_email(mock_telegram_service, mock_telegram_service_instance):
    # Set up the service instance mock
    mock_telegram_service_instance.count_codes_last_hour.return_value = 0

    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.chat = MagicMock()
    message.chat.id = 123
    message.message_id = 456
    message.text = "/register user@example.com"
    await bot_module.cmd_register(message)
    assert message.answer.await_count == 1
    args, kwargs = message.answer.await_args
    assert "Verification code sent to user@example.com" in args[0]

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.telegram_service")
async def test_register_rate_limit(mock_telegram_service):
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/register user@example.com"
    mock_telegram_service.count_codes_last_hour.return_value = 5
    await bot_module.cmd_register(message)
    args, kwargs = message.answer.await_args
    assert "Too many verification attempts" in args[0]

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
async def test_register_invalid_email():
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/register notanemail"
    await bot_module.cmd_register(message)
    args, kwargs = message.answer.await_args
    assert "Usage: /register email@example.com" in args[0]

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.telegram_service")
async def test_verify_valid_code(mock_telegram_service):
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/verify 123456"
    mock_telegram_service.verify_code.return_value = True
    await bot_module.cmd_verify(message)
    args, kwargs = message.answer.await_args
    assert "Email verified successfully" in args[0]

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.telegram_service")
async def test_verify_invalid_code(mock_telegram_service):
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/verify 654321"
    mock_telegram_service.verify_code.return_value = False
    await bot_module.cmd_verify(message)
    args, kwargs = message.answer.await_args
    assert "Invalid or expired code" in args[0]

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
async def test_verify_invalid_format():
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/verify abcdef"
    await bot_module.cmd_verify(message)
    args, kwargs = message.answer.await_args
    assert "Usage: /verify CODE" in args[0]

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.telegram_service")
async def test_info_verified_user(mock_telegram_service):
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    mock_telegram_service.get_user_status.return_value = {"email": "user@example.com", "verified": True, "language": "en"}
    await bot_module.cmd_info(message)
    args, kwargs = message.answer.await_args
    assert "Email: user@example.com" in args[0]
    assert "Verified: Yes" in args[0]

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.telegram_service")
async def test_info_unverified_user(mock_telegram_service):
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    mock_telegram_service.get_user_status.return_value = {"email": "user@example.com", "verified": False, "language": "en"}
    await bot_module.cmd_info(message)
    args, kwargs = message.answer.await_args
    assert "Email: user@example.com" in args[0]
    assert "Verified: No" in args[0]

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.telegram_service")
async def test_info_no_user(mock_telegram_service):
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    mock_telegram_service.get_user_status.return_value = None
    await bot_module.cmd_info(message)
    args, kwargs = message.answer.await_args
    assert "Email: (not set)" in args[0]
    assert "Verified: No" in args[0]
