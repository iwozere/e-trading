import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.db.services.base_service import uow
from src.data.db.models.model_notification import Message

def verify_db():
    with uow() as r:
        msg = r.s.query(Message).order_by(Message.id.desc()).first()
        if msg:
            print(f"ID: {msg.id}")
            print(f"Content: {msg.content}")
            print(f"Channels: {msg.channels}")

            attachments = msg.content.get('attachments', {})
            if attachments:
                print(f"✅ Attachments found: {list(attachments.keys())}")
            else:
                print("❌ No attachments found in message content")
        else:
            print("❌ No messages found in DB")

if __name__ == "__main__":
    verify_db()
