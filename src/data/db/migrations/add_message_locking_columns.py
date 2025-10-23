"""
Database Migration: Add Message Locking Columns

Adds distributed processing lock columns to the msg_messages table
for database-centric notification service architecture.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from sqlalchemy import text
from src.data.db.services.database_service import get_database_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def upgrade():
    """Add locking columns to msg_messages table."""
    db_service = get_database_service()

    with db_service.uow() as uow:
        # Add locked_by column
        uow.s.execute(text("""
            ALTER TABLE msg_messages
            ADD COLUMN IF NOT EXISTS locked_by VARCHAR(100)
        """))

        # Add locked_at column
        uow.s.execute(text("""
            ALTER TABLE msg_messages
            ADD COLUMN IF NOT EXISTS locked_at TIMESTAMP WITH TIME ZONE
        """))

        # Add indexes for efficient querying
        uow.s.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_msg_messages_locked_by
            ON msg_messages(locked_by)
        """))

        uow.s.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_msg_messages_locked_at
            ON msg_messages(locked_at)
        """))

        # Add composite index for pending message polling
        uow.s.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_msg_messages_pending_poll
            ON msg_messages(status, scheduled_for, locked_by, locked_at)
            WHERE status = 'PENDING'
        """))

        _logger.info("Successfully added message locking columns and indexes")


def downgrade():
    """Remove locking columns from msg_messages table."""
    db_service = get_database_service()

    with db_service.uow() as uow:
        # Drop indexes
        uow.s.execute(text("""
            DROP INDEX IF EXISTS idx_msg_messages_pending_poll
        """))

        uow.s.execute(text("""
            DROP INDEX IF EXISTS idx_msg_messages_locked_at
        """))

        uow.s.execute(text("""
            DROP INDEX IF EXISTS idx_msg_messages_locked_by
        """))

        # Drop columns
        uow.s.execute(text("""
            ALTER TABLE msg_messages
            DROP COLUMN IF EXISTS locked_at
        """))

        uow.s.execute(text("""
            ALTER TABLE msg_messages
            DROP COLUMN IF EXISTS locked_by
        """))

        _logger.info("Successfully removed message locking columns and indexes")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "downgrade":
        downgrade()
    else:
        upgrade()