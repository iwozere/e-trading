import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.telegram.screener.immediate_handlers import (
    process_info_command_immediate,
    process_register_command_immediate,
    process_verify_command_immediate,
    process_unknown_command_immediate
)


class TestImmediateHandlers:
    """Test immediate command handlers that respond without notification service."""

    @pytest.fixture
    def mock_message(self):
        """Create a mock Telegram message object."""
        message = MagicMock()
        message.answer = AsyncMock()
        message.text = "/info"
        message.chat.id = 12345
        message.message_id = 67890
        return message

    @pytest.mark.asyncio
    async def test_process_info_command_immediate_success(self, mock_message):
        """Test successful info command processing."""
        with patch('src.telegram.screener.immediate_handlers.handle_command') as mock_handle:
            mock_handle.return_value = {
                "status": "ok",
                "message": "User info retrieved successfully"
            }

            result = await process_info_command_immediate("123456789", mock_message)

            # Verify the command was processed
            assert result["status"] == "ok"

            # Verify immediate response was sent
            mock_message.answer.assert_called_once_with("User info retrieved successfully")

    @pytest.mark.asyncio
    async def test_process_info_command_immediate_error(self, mock_message):
        """Test info command processing with error."""
        with patch('src.telegram.screener.immediate_handlers.handle_command') as mock_handle:
            mock_handle.return_value = {
                "status": "error",
                "message": "User not found"
            }

            result = await process_info_command_immediate("123456789", mock_message)

            # Verify the error was handled
            assert result["status"] == "error"

            # Verify error response was sent immediately
            mock_message.answer.assert_called_once_with("❌ User not found")

    @pytest.mark.asyncio
    async def test_process_register_command_immediate_success(self, mock_message):
        """Test successful register command processing."""
        mock_message.text = "/register test@example.com"

        with patch('src.telegram.screener.immediate_handlers.handle_command') as mock_handle:
            mock_handle.return_value = {
                "status": "ok",
                "message": "Registration successful",
                "email_verification": {
                    "code": "123456",
                    "email": "test@example.com"
                }
            }

            with patch('src.telegram.telegram_bot.notification_client') as mock_client:
                mock_client.send_notification = AsyncMock()

                args = ["/register", "test@example.com"]
                result = await process_register_command_immediate("123456789", args, mock_message)

                # Verify the command was processed
                assert result["status"] == "ok"

                # Verify immediate response was sent
                mock_message.answer.assert_called_once_with("Registration successful")

    @pytest.mark.asyncio
    async def test_process_verify_command_immediate_success(self, mock_message):
        """Test successful verify command processing."""
        mock_message.text = "/verify 123456"

        with patch('src.telegram.screener.immediate_handlers.handle_command') as mock_handle:
            mock_handle.return_value = {
                "status": "ok",
                "message": "Email verification successful"
            }

            args = ["/verify", "123456"]
            result = await process_verify_command_immediate("123456789", args, mock_message)

            # Verify the command was processed
            assert result["status"] == "ok"

            # Verify immediate response was sent
            mock_message.answer.assert_called_once_with("Email verification successful")

    @pytest.mark.asyncio
    async def test_process_unknown_command_immediate(self, mock_message):
        """Test unknown command processing."""
        mock_message.text = "/unknown_command"

        result = await process_unknown_command_immediate("123456789", mock_message, "Help text")

        # Verify the command was processed
        assert result["status"] == "ok"

        # Verify immediate response was sent
        expected_message = "❓ Unknown command: /unknown_command\n\nI don't recognize this command. Please use /help to see all available commands."
        mock_message.answer.assert_called_once_with(expected_message)

    @pytest.mark.asyncio
    async def test_immediate_handler_exception_handling(self, mock_message):
        """Test exception handling in immediate handlers."""
        with patch('src.telegram.screener.immediate_handlers.handle_command') as mock_handle:
            mock_handle.side_effect = Exception("Test exception")

            result = await process_info_command_immediate("123456789", mock_message)

            # Verify error was handled gracefully
            assert result["status"] == "error"

            # Verify error message was sent immediately
            mock_message.answer.assert_called_once_with("❌ Error retrieving your information. Please try again.")


if __name__ == "__main__":
    pytest.main([__file__])