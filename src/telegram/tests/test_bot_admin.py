import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.telegram import bot as bot_module

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.is_admin_user", return_value=True)
@patch("src.telegram.bot.telegram_service")
async def test_admin_help(mock_telegram_service, mock_is_admin):
    message = AsyncMock()
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = "admin"
    message.text = "/admin help"
    await bot_module.cmd_admin(message)
    message.answer.assert_awaited_with(bot_module.ADMIN_HELP_TEXT, parse_mode="HTML")

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.is_admin_user", return_value=True)
@patch("src.telegram.bot.telegram_service")
async def test_admin_listusers(mock_telegram_service, mock_is_admin):
    message = AsyncMock()
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = "admin"
    message.text = "/admin listusers"
    mock_telegram_service.list_users.return_value = [("user1", "a@b.com"), ("user2", None)]
    await bot_module.cmd_admin(message)
    assert message.answer.await_count == 1
    args, kwargs = message.answer.await_args
    assert "user1 - a@b.com" in args[0]
    assert "user2 - (no email)" in args[0]
    assert kwargs["parse_mode"] == "HTML"

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.is_admin_user", return_value=True)
@patch("src.telegram.bot.telegram_service")
async def test_admin_setlimit_global(mock_telegram_service, mock_is_admin):
    message = AsyncMock()
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = "admin"
    message.text = "/admin setlimit alerts 10"
    await bot_module.cmd_admin(message)
    mock_telegram_service.set_setting.assert_called_with("max_alerts", "10")
    message.answer.assert_awaited_with("Set global max_alerts to 10.")

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.is_admin_user", return_value=True)
@patch("src.telegram.bot.telegram_service")
async def test_admin_setlimit_per_user(mock_telegram_service, mock_is_admin):
    message = AsyncMock()
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = "admin"
    message.text = "/admin setlimit schedules 7 user123"
    await bot_module.cmd_admin(message)
    mock_telegram_service.set_user_limit.assert_called_with("user123", "max_schedules", 7)
    message.answer.assert_awaited_with("Set max_schedules for user user123 to 7.")

@pytest.mark.asyncio
@patch("src.telegram.bot.notification_manager", new=AsyncMock())
@patch("src.telegram.bot.is_admin_user", return_value=True)
@patch("src.telegram.bot.telegram_service")
async def test_admin_users(mock_telegram_service, mock_is_admin):
    message = AsyncMock()
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = "admin"
    message.text = "/admin users"
    mock_telegram_service.list_users.return_value = [("user1", "a@b.com")]
    await bot_module.cmd_admin(message)
    args, kwargs = message.answer.await_args
    assert "user1 - a@b.com" in args[0]
    assert kwargs["parse_mode"] == "HTML"
