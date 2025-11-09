import tempfile
import os
import asyncio
from src.notification.async_notification_manager import initialize_notification_manager, NotificationType, NotificationPriority

def test_email_with_attachment():
    # Create a dummy file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"Test attachment content")
        temp_file.flush()
        attachment_path = temp_file.name

    async def send_email():
        notification_manager = await initialize_notification_manager(
            email_sender="sender@example.com",
            email_receiver="receiver@example.com"
        )
        await notification_manager.send_notification(
            notification_type=NotificationType.INFO,
            title="Test Email with Attachment",
            message="This is a test email with an attachment.",
            priority=NotificationPriority.NORMAL,
            data={},
            source="test_email_with_attachment",
            channels=["email"],
            attachments=[attachment_path]
        )

    try:
        asyncio.run(send_email())
        print("Test email with attachment sent.")
    finally:
        os.unlink(attachment_path)