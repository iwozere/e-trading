import asyncio
import json
from pathlib import Path
from src.notification.channels.email_channel import EmailChannel
from src.notification.channels.base import MessageContent
from src.data.db.services.users_service import users_service
from src.notification.service.config import config

async def test_attachment_resolution():
    print("Testing attachment resolution logic...")

    # Mock config
    channel_config = config.channels.email.dict()
    channel = EmailChannel("email", channel_config)

    # Create dummy files
    p1 = Path("test_p2.csv")
    p1.write_text("ticker,phase\nSNBR,2")
    p2 = Path("test_sent.csv")
    p2.write_text("ticker,sentiment\nSNBR,0.8")

    try:
        # 1. Test nested format (the old broken one)
        attachments_nested = {"files": [str(p1), str(p2)]}
        print(f"Testing nested format: {attachments_nested}")

        # We'll just test the internal _add_attachments logic by mocking MIMEMultipart
        class MockMIME:
            def __init__(self): self.parts = []
            def attach(self, part): self.parts.append(part)

        msg = MockMIME()
        await channel._add_attachments(msg, attachments_nested)
        print(f"Nested format resulted in {len(msg.parts)} attachments")
        assert len(msg.parts) == 2

        # 2. Test flat format (the new preferred one)
        attachments_flat = {p1.name: str(p1), p2.name: str(p2)}
        print(f"Testing flat format: {attachments_flat}")

        msg = MockMIME()
        await channel._add_attachments(msg, attachments_flat)
        print(f"Flat format resulted in {len(msg.parts)} attachments")
        assert len(msg.parts) == 2

        print("✅ Attachment resolution tests passed!")

    finally:
        # Cleanup
        if p1.exists(): p1.unlink()
        if p2.exists(): p2.unlink()

if __name__ == "__main__":
    asyncio.run(test_attachment_resolution())
