#!/usr/bin/env python3
"""
Test script to verify Telegram notification system with multiple reports.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))

import asyncio
from src.frontend.telegram.command_parser import parse_command
from src.frontend.telegram.screener.business_logic import handle_command
from src.notification.async_notification_manager import initialize_notification_manager
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER, SMTP_PASSWORD

async def test_telegram_notifications():
    """Test Telegram notifications with multiple reports."""

    print("Testing Telegram notifications with multiple reports...")

    # Initialize notification manager
    notification_manager = await initialize_notification_manager(
        telegram_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        email_api_key=SMTP_PASSWORD,
        email_sender=SMTP_USER,
        email_receiver=SMTP_USER
    )

    # Test command
    test_command = "/report AAPL MSFT VT"
    print(f"Testing command: {test_command}")

    # Parse the command
    parsed = parse_command(test_command)
    parsed.args["telegram_user_id"] = "test_user_123"

    # Handle the command
    print("Handling command...")
    result = handle_command(parsed)

    print(f"Result status: {result.get('status')}")
    print(f"Number of reports: {len(result.get('reports', []))}")

    if result.get("status") == "ok" and "reports" in result:
        # Create a mock message object
        class MockMessage:
            def __init__(self):
                self.chat = MockChat()
                self.message_id = 123

        class MockChat:
            def __init__(self):
                self.id = TELEGRAM_CHAT_ID

        mock_message = MockMessage()

        # Test the notification processing
        from src.frontend.telegram.screener.notifications import process_report_notifications

        print("Processing notifications...")
        await process_report_notifications(result, notification_manager, mock_message, None)

        print("Notification processing completed!")
    else:
        print(f"Error: {result.get('message')}")

    # Clean up
    await notification_manager.stop()

if __name__ == "__main__":
    asyncio.run(test_telegram_notifications())
