from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.telegram.handlers import admin as admin_handler

# Constants formerly in bot.py but now likely in business_logic or similar
# Based on old test, it expects admin_handler.ADMIN_HELP_TEXT
ADMIN_HELP_TEXT = """<b>Admin Commands</b>
/admin help - Show this help
/admin listusers - List all users
/admin setlimit <type> <limit> [user_id] - Set limits
"""


@pytest.mark.asyncio
@patch("src.telegram.handlers.admin.get_notification_client")
@patch("src.telegram.screener.business_logic.is_admin_user", return_value=True)
@patch("src.telegram.handlers.common.get_service_instances")
async def test_admin_help(mock_get_services, mock_is_admin, mock_get_client):
    mock_client = AsyncMock()
    mock_get_client.return_value = mock_client

    message = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 12345
    message.text = "/admin help"

    # Mock services
    mock_telegram_service = MagicMock()
    mock_get_services.return_value = (mock_telegram_service, None)

    await admin_handler.cmd_admin(message)

    # Verify notification was sent (process_admin_command uses send_notification)
    assert mock_client.send_notification.called


@pytest.mark.asyncio
@patch("src.telegram.handlers.admin.get_notification_client", new=AsyncMock())
@patch("src.telegram.screener.business_logic.is_admin_user", return_value=True)
@patch("src.telegram.handlers.common.get_service_instances")
@patch("src.telegram.screener.notifications.process_admin_command", new=AsyncMock())
async def test_admin_listusers(mock_get_services, mock_is_admin):
    message = AsyncMock()
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 12345
    message.text = "/admin listusers"

    mock_telegram_service = MagicMock()
    mock_get_services.return_value = (mock_telegram_service, None)

    await admin_handler.cmd_admin(message)
    assert message.from_user.id == 12345


@pytest.mark.asyncio
@patch("src.telegram.handlers.admin.get_notification_client", new=AsyncMock())
@patch("src.telegram.screener.business_logic.is_admin_user", return_value=True)
@patch("src.telegram.handlers.common.get_service_instances")
async def test_admin_setlimit_global(mock_get_services, mock_is_admin):
    message = AsyncMock()
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 12345
    message.text = "/admin setlimit alerts 10"

    mock_telegram_service = MagicMock()
    mock_get_services.return_value = (mock_telegram_service, None)

    await admin_handler.cmd_admin(message)
    assert message.from_user.id == 12345
