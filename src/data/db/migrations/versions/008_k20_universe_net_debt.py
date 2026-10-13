"""k20_universe_net_debt

Revision ID: 008_k20_universe_net_debt
Revises: 007_p22_deal_candidate_type
Create Date: 2026-10-13

Adds nullable `k20_universe.net_debt` (total_debt - total_cash). P20 Kestrel Sleeve A's
§4.1 balance-sheet filter accepts "net cash OR net debt/EBITDA < 3 OR interest coverage > 4x",
but nothing ever produced a net-cash input, and a net-cash company with negative EBITDA (common
for turnarounds) can't be expressed through `net_debt_ebitda` alone. Does NOT modify any other
table or column.
"""

import sqlalchemy as sa
from alembic import op

revision = "008_k20_universe_net_debt"
down_revision = "007_p22_deal_candidate_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("k20_universe", sa.Column("net_debt", sa.Numeric(), nullable=True))


def downgrade() -> None:
    op.drop_column("k20_universe", "net_debt")
