"""p22_trial_dates

Revision ID: 018_p22_trial_dates
Revises: 017_signal_ledger_exit_policies
Create Date: 2026-11-08

Adds `start_date` and `first_posted_date` to `p22_trial`, so P22 Block B can read a company's
trials as of a past date (a trial counts only once it was public). Both are filled by the next
P22 Trial Normalization run; `first_posted_date` also needs the next CT.gov ingest, which now
fetches it.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "018_p22_trial_dates"
down_revision = "017_signal_ledger_exit_policies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the two nullable date columns."""
    op.add_column("p22_trial", sa.Column("start_date", sa.Date(), nullable=True))
    op.add_column("p22_trial", sa.Column("first_posted_date", sa.Date(), nullable=True))


def downgrade() -> None:
    """Drop the two date columns."""
    op.drop_column("p22_trial", "first_posted_date")
    op.drop_column("p22_trial", "start_date")
