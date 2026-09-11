"""p22_deal_candidate_type

Revision ID: 007_p22_deal_candidate_type
Revises: 006_delivery_status_updated_at
Create Date: 2026-09-10

Adds `'deal_candidate'` to `ck_p22_review_item_type`. `run_deal_candidates_ingest.py`
(added 2026-09-08, `deal_candidates.py`) writes `item_type="deal_candidate"` rows, but the
check constraint from migration 003 was never widened for it — every run has been failing
in production with `psycopg2.errors.CheckViolation` on `ck_p22_review_item_type` since the
job's cron first fired. Does NOT modify any other table or constraint.
"""

from alembic import op

revision = "007_p22_deal_candidate_type"
down_revision = "006_delivery_status_updated_at"
branch_labels = None
depends_on = None

_OLD_TYPES = "'entity_match','process_event','activist_intent','partnership_structure','deal_type'"
_NEW_TYPES = _OLD_TYPES + ",'deal_candidate'"


def upgrade() -> None:
    op.drop_constraint("ck_p22_review_item_type", "p22_review_item", type_="check")
    op.create_check_constraint(
        "ck_p22_review_item_type",
        "p22_review_item",
        f"item_type IN ({_NEW_TYPES})",
    )


def downgrade() -> None:
    op.drop_constraint("ck_p22_review_item_type", "p22_review_item", type_="check")
    op.create_check_constraint(
        "ck_p22_review_item_type",
        "p22_review_item",
        f"item_type IN ({_OLD_TYPES})",
    )
