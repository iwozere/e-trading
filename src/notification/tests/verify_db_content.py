import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.db.models.model_notification import Message
from src.data.db.services.database_service import get_database_service


def verify_db():
    with get_database_service().uow() as r:
        msg = r.s.query(Message).order_by(Message.id.desc()).first()
        if msg:
            print(f"ID: {msg.id}")
            print(f"Content: {msg.content}")
            print(f"Channels: {msg.channels}")

            attachments = msg.content.get("attachments", {})
            if attachments:
                print(f"✅ Attachments found: {list(attachments.keys())}")
            else:
                print("❌ No attachments found in message content")
        else:
            print("❌ No messages found in DB")


if __name__ == "__main__":
    verify_db()
