import sys
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime, timezone

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Mocking parts of the system to avoid database dependency
from src.notification.service.database_optimization import OptimizedMessageRepository
from src.data.db.models.model_notification import MessagePriority

def test_repro_type_error():
    print("Testing OptimizedMessageRepository.get_pending_messages() for TypeError...")

    # Mock SQLAlchemy session
    mock_session = MagicMock()

    # Instantiate repository
    repo = OptimizedMessageRepository(mock_session)

    # Test parameters
    current_time = datetime.now(timezone.utc)
    channels = ["email"]
    limit = 5

    try:
        # This call should no longer raise TypeError
        repo.get_pending_messages(
            current_time=current_time,
            priority=MessagePriority.CRITICAL,
            channels=channels,
            limit=limit
        )
        print("✅ SUCCESS: get_pending_messages called successfully with channels argument.")
    except TypeError as e:
        print(f"❌ FAILURE: TypeError raised: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ℹ️ Received expected non-TypeError: {type(e).__name__}: {e}")
        print("✅ SUCCESS: The TypeError was averted.")

if __name__ == "__main__":
    test_repro_type_error()
