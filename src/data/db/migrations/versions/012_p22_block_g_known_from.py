"""p22_block_g_known_from

Revision ID: 012_p22_block_g_known_from
Revises: 011_p22_deal_review_fields
Create Date: 2026-10-30

Data-only backfill. `ingest/activist_positions.py` and `ingest/process_events.py` stamped
`known_from` with the ingest time instead of the filing date that spec §4.7/§8.3 require. That was
never a lookahead leak (ingest time is always after the filing), but it hid every historical 13D and
8-K process event from the backtest until the day it happened to be ingested. Both ingests now write
the filing date at 00:00 UTC (the `financial_facts` convention); this rewrites the rows already stored
the same way. Only rows whose `known_from` falls after the filing date are touched, so re-running it,
or running it after the code fix, changes nothing.

Process events use `event_date`, which the ingest sets to the 8-K filing date. Downgrade is a no-op:
the original ingest timestamps carry no information worth restoring.
"""

from alembic import op

revision = "012_p22_block_g_known_from"
down_revision = "011_p22_deal_review_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE p22_activist_position
        SET known_from = (filed_date::timestamp AT TIME ZONE 'UTC')
        WHERE known_from IS NULL OR known_from > (filed_date::timestamp AT TIME ZONE 'UTC')
        """
    )
    op.execute(
        """
        UPDATE p22_corporate_process_event
        SET known_from = (event_date::timestamp AT TIME ZONE 'UTC')
        WHERE accession_no IS NOT NULL
          AND (known_from IS NULL OR known_from > (event_date::timestamp AT TIME ZONE 'UTC'))
        """
    )


def downgrade() -> None:
    pass
