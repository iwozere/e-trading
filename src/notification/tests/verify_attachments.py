import asyncio
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from src.notification.channels.email_channel import EmailChannel


async def test_attachment_resolution() -> None:
    """Test resolution of email attachments with different input formats."""
    print("Testing attachment resolution logic...")

    # Mock config
    channel_config = {
        "max_attachment_size_mb": 10,
        "smtp_host": "localhost",
        "smtp_port": 587,
        "smtp_username": "test@example.com",
        "smtp_password": "test",
        "from_email": "test@example.com",
    }
    channel = EmailChannel("email", channel_config)

    # Create dummy files
    p1 = Path("test_p2.csv")
    p1.write_text("ticker,phase\nSNBR,2", encoding="utf-8")
    p2 = Path("test_sent.csv")
    p2.write_text("ticker,sentiment\nSNBR,0.8", encoding="utf-8")

    try:
        # 1. Test nested format (the old broken one)
        attachments_nested = {"files": [str(p1), str(p2)]}
        print(f"Testing nested format: {attachments_nested}")

        msg1 = MIMEMultipart()
        await channel._add_attachments(msg1, attachments_nested)
        payload1 = msg1.get_payload()
        assert isinstance(payload1, list)
        print(f"Nested format resulted in {len(payload1)} attachments")
        assert len(payload1) == 2

        # 2. Test flat format (the new preferred one)
        attachments_flat = {p1.name: str(p1), p2.name: str(p2)}
        print(f"Testing flat format: {attachments_flat}")

        msg2 = MIMEMultipart()
        await channel._add_attachments(msg2, attachments_flat)
        payload2 = msg2.get_payload()
        assert isinstance(payload2, list)
        print(f"Flat format resulted in {len(payload2)} attachments")
        assert len(payload2) == 2

        # 3. Test wrapped format (the current DB structure)
        attachments_wrapped = {
            p1.name: {"path": str(p1), "type": "file_path"},
            p2.name: {"path": str(p2), "type": "file_path"},
        }
        print(f"Testing wrapped format: {attachments_wrapped}")

        msg3 = MIMEMultipart()
        await channel._add_attachments(msg3, attachments_wrapped)
        payload3 = msg3.get_payload()
        assert isinstance(payload3, list)
        print(f"Wrapped format resulted in {len(payload3)} attachments")
        assert len(payload3) == 2

        print("Attachment resolution tests passed!")

    finally:
        # Cleanup
        if p1.exists():
            p1.unlink()
        if p2.exists():
            p2.unlink()


if __name__ == "__main__":
    asyncio.run(test_attachment_resolution())

