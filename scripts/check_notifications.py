import os
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.database_service import get_database_service
from src.data.db.models.model_notification import Message, MessageDeliveryStatus

def check_messages():
    db_service = get_database_service()
    with db_service.uow() as uow:
        # Get last 5 messages
        messages = uow.s.query(Message).order_by(Message.id.desc()).limit(5).all()

        print(f"{'ID':<5} | {'Type':<12} | {'Recipient':<15} | {'Status':<10} | {'Channels'}")
        print("-" * 70)

        for msg in messages:
            print(f"{msg.id:<5} | {msg.message_type:<12} | {msg.recipient_id:<15} | {msg.status:<10} | {msg.channels}")

            # Get delivery statuses
            statuses = uow.s.query(MessageDeliveryStatus).filter(MessageDeliveryStatus.message_id == msg.id).all()
            for s in statuses:
                print(f"  -> Channel: {s.channel:<10} | Status: {s.status:<10} | Error: {s.last_error}")
            print("-" * 70)

if __name__ == "__main__":
    check_messages()
