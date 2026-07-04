import asyncio
import unittest
from unittest.mock import MagicMock, patch

from src.notification.async_notification_manager import (
    NotificationPriority,
    NotificationType,
    initialize_notification_manager,
)

from config.donotshare.donotshare import SMTP_USER

to_addr = "akossyrev@gmail.com"


class TestAsyncNotificationManager(unittest.TestCase):
    @patch(
        "src.notification.async_notification_manager.AsyncNotificationManager.send_notification", new_callable=MagicMock
    )
    def test_send_email_notification(self, mock_send_notification):
        async def run_test():
            notification_manager = await initialize_notification_manager(email_sender=SMTP_USER, email_receiver=to_addr)
            await notification_manager.send_notification(
                notification_type=NotificationType.INFO,
                title="Test Subject",
                message="Test Body",
                priority=NotificationPriority.NORMAL,
                data={},
                source="test_emailer",
                channels=["email"],
            )
            mock_send_notification.assert_called_once()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
