"""k20_sentiment_wiki_source

Revision ID: 009_k20_sentiment_wiki_source
Revises: 008_k20_universe_net_debt
Create Date: 2026-10-14

Adds `'wikipedia'` to `ck_k20_sentiment_source`. P20 Kestrel's retail-attention poll moved
from Google Trends (pytrends — never installed in production, so it never wrote a row) to
the Wikimedia pageviews API (`sentiment/wiki_pageviews_poll.py`), which writes
`source='wikipedia'`. `'trends'` stays allowed so the constraint stays a pure widening.
Does NOT modify any other table or constraint.
"""

from alembic import op

revision = "009_k20_sentiment_wiki_source"
down_revision = "008_k20_universe_net_debt"
branch_labels = None
depends_on = None

_OLD_SOURCES = "'gdelt','stocktwits','reddit','apewisdom','trends','av_news'"
_NEW_SOURCES = _OLD_SOURCES + ",'wikipedia'"


def upgrade() -> None:
    op.drop_constraint("ck_k20_sentiment_source", "k20_sentiment_daily", type_="check")
    op.create_check_constraint(
        "ck_k20_sentiment_source",
        "k20_sentiment_daily",
        f"source IN ({_NEW_SOURCES})",
    )


def downgrade() -> None:
    op.execute("DELETE FROM k20_sentiment_daily WHERE source = 'wikipedia'")
    op.drop_constraint("ck_k20_sentiment_source", "k20_sentiment_daily", type_="check")
    op.create_check_constraint(
        "ck_k20_sentiment_source",
        "k20_sentiment_daily",
        f"source IN ({_OLD_SOURCES})",
    )
