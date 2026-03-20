import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiogram.types import Message
from src.telegram.handlers import account as account_handler

@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_register_valid_email(mock_common_get_services, mock_get_services):
    mock_telegram_service = MagicMock()
    mock_telegram_service.count_codes_last_hour.return_value = 0
    mock_telegram_service.get_user_status.return_value = None
    mock_get_services.return_value = (mock_telegram_service, None)
    mock_common_get_services.return_value = (mock_telegram_service, None)

    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/register user@example.com"
    
    await account_handler.cmd_register(message)
    
    assert message.reply.called
    call_args = message.reply.call_args[0][0]
    assert "Registration Successful" in call_args

@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_register_rate_limit(mock_common_get_services, mock_get_services):
    mock_telegram_service = MagicMock()
    mock_telegram_service.count_codes_last_hour.return_value = 5
    mock_get_services.return_value = (mock_telegram_service, None)
    mock_common_get_services.return_value = (mock_telegram_service, None)

    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/register user@example.com"
    
    await account_handler.cmd_register(message)
    assert "Rate Limit Exceeded" in message.reply.call_args[0][0]

@pytest.mark.asyncio
async def test_register_invalid_email():
    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/register notanemail"
    
    with patch("src.telegram.handlers.account.get_service_instances") as mock_get_services, \
         patch("src.telegram.handlers.common.get_service_instances") as mock_common_get_services:
        mock_get_services.return_value = (MagicMock(), None)
        mock_common_get_services.return_value = (mock_get_services.return_value[0], None)
        await account_handler.cmd_register(message)
    
    assert "Please provide a valid" in message.reply.call_args[0][0]

@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_verify_valid_code(mock_common_get_services, mock_get_services):
    mock_telegram_service = MagicMock()
    mock_telegram_service.verify_code.return_value = True
    mock_telegram_service.get_user_status.return_value = {"email": "a@b.com"}
    mock_get_services.return_value = (mock_telegram_service, None)
    mock_common_get_services.return_value = (mock_telegram_service, None)

    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/verify 123456"
    
    await account_handler.cmd_verify(message)
    assert "Verified Successfully" in message.reply.call_args[0][0]

@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_verify_invalid_code(mock_common_get_services, mock_get_services):
    mock_telegram_service = MagicMock()
    mock_telegram_service.verify_code.return_value = False
    mock_telegram_service.get_user_status.return_value = {"email": "a@b.com"}
    mock_get_services.return_value = (mock_telegram_service, None)
    mock_common_get_services.return_value = (mock_telegram_service, None)

    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/verify 654321"
    
    await account_handler.cmd_verify(message)
    assert "Invalid or Expired" in message.reply.call_args[0][0]

@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_verify_invalid_format(mock_common_get_services, mock_get_services):
    mock_telegram_service = MagicMock()
    mock_get_services.return_value = (mock_telegram_service, None)
    mock_common_get_services.return_value = (mock_telegram_service, None)
    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/verify abcdef"
    
    await account_handler.cmd_verify(message)
    assert "Usage: /verify CODE" in message.reply.call_args[0][0]

@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_info_verified_user(mock_common_get_services, mock_get_services):
    mock_telegram_service = MagicMock()
    mock_telegram_service.get_user_status.return_value = {
        "email": "user@example.com", "verified": True, "approved": True, "is_admin": False, "language": "en"
    }
    mock_get_services.return_value = (mock_telegram_service, None)
    mock_common_get_services.return_value = (mock_telegram_service, None)

    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/info"
    
    await account_handler.cmd_info(message)
    assert "user@example.com" in message.reply.call_args[0][0]
    assert "Verified: ✅ Yes" in message.reply.call_args[0][0]

@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_info_unverified_user(mock_common_get_services, mock_get_services):
    mock_telegram_service = MagicMock()
    mock_telegram_service.get_user_status.return_value = {
        "email": "user@example.com", "verified": False, "approved": False, "is_admin": False, "language": "en"
    }
    mock_get_services.return_value = (mock_telegram_service, None)
    mock_common_get_services.return_value = (mock_telegram_service, None)

    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/info"
    
    await account_handler.cmd_info(message)
    assert "Verified: ❌ No" in message.reply.call_args[0][0]

@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_info_no_user(mock_common_get_services, mock_get_services):
    mock_telegram_service = MagicMock()
    mock_telegram_service.get_user_status.return_value = None
    mock_get_services.return_value = (mock_telegram_service, None)
    mock_common_get_services.return_value = (mock_telegram_service, None)

    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/info"
    
    await account_handler.cmd_info(message)
    assert "Email: (not set)" in message.reply.call_args[0][0]
