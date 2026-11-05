"""drop_ss_sentiment_history

Revision ID: 016_drop_ss_sentiment_history
Revises: 015_legacy_schema_fixes
Create Date: 2026-11-05

Drops the legacy P04 table ss_sentiment_history and its retention function
cleanup_old_sentiment_history(), both created by the hand-run SQL migration
src/ml/pipeline/p04_short_squeeze/data/migrations/001_add_sentiment_metrics.sql.
No code reads or writes the table, no job calls the function, and no model
describes either. The downgrade recreates both empty, as that SQL defined them.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "016_drop_ss_sentiment_history"
down_revision = "015_legacy_schema_fixes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop the unused table and its cleanup function."""
    op.execute("DROP FUNCTION IF EXISTS cleanup_old_sentiment_history()")
    op.execute("DROP TABLE IF EXISTS ss_sentiment_history")


def downgrade() -> None:
    """Recreate the table (empty) and the cleanup function as SQL migration 001 defined them."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ss_sentiment_history (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            mentions_count INTEGER NOT NULL DEFAULT 0,
            unique_authors INTEGER NOT NULL DEFAULT 0,
            sentiment_avg FLOAT NOT NULL DEFAULT 0.0,
            virality_avg FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_ticker_date UNIQUE (ticker, date)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_ss_sentiment_history_ticker_date ON ss_sentiment_history (ticker, date DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ss_sentiment_history_date ON ss_sentiment_history (date DESC)")
    op.execute(
        """
        CREATE OR REPLACE FUNCTION cleanup_old_sentiment_history()
        RETURNS void AS $$
        BEGIN
            DELETE FROM ss_sentiment_history
            WHERE date < CURRENT_DATE - INTERVAL '30 days';
        END;
        $$ LANGUAGE plpgsql
        """
    )
