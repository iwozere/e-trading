import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.db.services.database_service import get_database_service
from src.data.db.models.model_notification import Message, DeliveryStatus

def check_delivery():
    db = get_database_service()
    with db.uow() as r:
        # Get the latest message
        msg = r.s.query(Message).order_by(Message.id.desc()).first()
        if not msg:
            print("No messages found.")
            return

        print(f"--- Message ID: {msg.id} ---")
        print(f"Recipient: {msg.recipient_id}")
        print(f"Channels: {msg.channels}")
        print(f"Status: {msg.status}")

        # Get delivery statuses
        deliveries = r.s.query(DeliveryStatus).filter(DeliveryStatus.message_id == msg.id).all()
        if not deliveries:
            print("No delivery attempts found in msg_delivery_status.")
        else:
            print(f"\nDelivery Attempts ({len(deliveries)}):")
            for d in deliveries:
                print(f"- Channel: {d.channel}")
                print(f"  Status: {d.status}")
                print(f"  Error: {d.error_message}")
                print(f"  External ID: {d.external_id}")

if __name__ == "__main__":
    check_delivery()
