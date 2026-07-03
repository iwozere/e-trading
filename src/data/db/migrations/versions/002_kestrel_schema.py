"""kestrel_schema

Revision ID: 002_kestrel
Revises: 032f9959e8cf
Create Date: 2026-07-03

Adds all k20_* tables for the P20 Kestrel pipeline.
Does NOT modify any existing tables.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002_kestrel"
down_revision = "032f9959e8cf"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "k20_universe",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("exchange", sa.Text(), nullable=True),
        sa.Column("sector", sa.Text(), nullable=True),
        sa.Column("industry", sa.Text(), nullable=True),
        sa.Column("mcap", sa.Numeric(), nullable=True),
        sa.Column("adv_20d", sa.Numeric(), nullable=True),
        sa.Column("revenue_yoy_growth", sa.Numeric(), nullable=True),
        sa.Column("gross_margin", sa.Numeric(), nullable=True),
        sa.Column("net_debt_ebitda", sa.Numeric(), nullable=True),
        sa.Column("interest_coverage", sa.Numeric(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('active','delisted','suspended')", name="ck_k20_universe_status"
        ),
        sa.PrimaryKeyConstraint("ticker"),
    )
    op.create_index("idx_k20_universe_status", "k20_universe", ["status"])

    op.create_table(
        "k20_company_aliases",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("alias_type", sa.Text(), nullable=False),
        sa.Column("normalized_alias", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "alias_type IN ('legal_name','short_name','brand','former_name')",
            name="ck_k20_alias_type",
        ),
        sa.PrimaryKeyConstraint("ticker", "alias"),
    )
    op.create_index("idx_k20_aliases_normalized", "k20_company_aliases", ["normalized_alias"])

    op.create_table(
        "k20_alias_blocklist",
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=True),
        sa.Column("match_policy", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "match_policy IN ('legal_name_only','name_plus_context','never')",
            name="ck_k20_blocklist_policy",
        ),
        sa.PrimaryKeyConstraint("alias"),
    )

    op.create_table(
        "k20_signals",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("signal_type", sa.Text(), nullable=False),
        sa.Column("value", sa.Numeric(), nullable=True),
        sa.Column("sleeve", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("ticker", "date", "signal_type"),
    )
    op.create_index("idx_k20_signals_ticker_date", "k20_signals", ["ticker", "date"])
    op.create_index("idx_k20_signals_type", "k20_signals", ["signal_type"])

    op.create_table(
        "k20_sentiment_daily",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("mentions", sa.Numeric(), nullable=True),
        sa.Column("avg_tone", sa.Numeric(), nullable=True),
        sa.Column("tone_std", sa.Numeric(), nullable=True),
        sa.Column("pos_score", sa.Numeric(), nullable=True),
        sa.Column("neg_score", sa.Numeric(), nullable=True),
        sa.Column("bullish_ratio", sa.Numeric(), nullable=True),
        sa.Column("top_domains", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("mention_z20", sa.Numeric(), nullable=True),
        sa.Column("tone_z20", sa.Numeric(), nullable=True),
        sa.CheckConstraint(
            "source IN ('gdelt','stocktwits','reddit','apewisdom','trends','av_news')",
            name="ck_k20_sentiment_source",
        ),
        sa.PrimaryKeyConstraint("ticker", "date", "source"),
    )
    op.create_index("idx_k20_sentiment_ticker_date", "k20_sentiment_daily", ["ticker", "date"])

    # k20_llm_runs must come before k20_watchlist (FK dependency)
    op.create_table(
        "k20_llm_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("ticker", sa.Text(), nullable=True),
        sa.Column("task_type", sa.Text(), nullable=False),
        sa.Column("input_ref", sa.Text(), nullable=True),
        sa.Column("output_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=True),
        sa.Column("tokens_out", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=True),
        sa.Column("verdict", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_type", "input_ref", name="uq_k20_llm_runs_task_ref"),
    )
    op.create_index("idx_k20_llm_runs_ticker", "k20_llm_runs", ["ticker"])
    op.create_index("idx_k20_llm_runs_task", "k20_llm_runs", ["task_type"])
    op.create_index("idx_k20_llm_runs_ts", "k20_llm_runs", ["ts"])

    op.create_table(
        "k20_catalysts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=True),
        sa.Column("confidence", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("catalyst_detail", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("state", sa.Text(), nullable=False, server_default="upcoming"),
        sa.Column("t10_alerted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("t3_alerted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("datechange_alerted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('upcoming','date_changed','passed','cancelled')",
            name="ck_k20_catalyst_state",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_k20_catalysts_ticker", "k20_catalysts", ["ticker"])
    op.create_index("idx_k20_catalysts_event_date", "k20_catalysts", ["event_date"])

    op.create_table(
        "k20_watchlist",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("sleeve", sa.Text(), nullable=False),
        sa.Column("score", sa.Numeric(), nullable=True),
        sa.Column("llm_verdict", sa.Text(), nullable=True),
        sa.Column("dossier_run_id", sa.BigInteger(), nullable=True),
        sa.Column("thesis_short", sa.Text(), nullable=True),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("state", sa.Text(), nullable=False, server_default="screening"),
        sa.CheckConstraint(
            "state IN ('screening','candidate','active_position','rejected','expired')",
            name="ck_k20_watchlist_state",
        ),
        sa.ForeignKeyConstraint(["dossier_run_id"], ["k20_llm_runs.id"], name="fk_k20_watchlist_dossier"),
        sa.PrimaryKeyConstraint("ticker", "sleeve"),
    )
    op.create_index("idx_k20_watchlist_state", "k20_watchlist", ["state"])
    op.create_index("idx_k20_watchlist_score", "k20_watchlist", ["score"])

    op.create_table(
        "k20_positions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("sleeve", sa.Text(), nullable=False),
        sa.Column("entry_date", sa.Date(), nullable=True),
        sa.Column("entry_px", sa.Numeric(12, 4), nullable=True),
        sa.Column("size_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("stop_px", sa.Numeric(12, 4), nullable=True),
        sa.Column("t1_px", sa.Numeric(12, 4), nullable=True),
        sa.Column("t2_px", sa.Numeric(12, 4), nullable=True),
        sa.Column("trail_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("realized_thirds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_k20_positions_ticker", "k20_positions", ["ticker"])
    op.create_index("idx_k20_positions_status", "k20_positions", ["status"])

    op.create_table(
        "k20_request_budget",
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("quota", sa.Integer(), nullable=False),
        sa.Column("used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("notes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("source", "date"),
    )

    op.create_table(
        "k20_job_runs",
        sa.Column("job", sa.Text(), nullable=False),
        sa.Column("run_date", sa.Date(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rows_out", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running','ok','failed','skipped')", name="ck_k20_job_run_status"
        ),
        sa.PrimaryKeyConstraint("job", "run_date"),
    )
    op.create_index("idx_k20_job_runs_status", "k20_job_runs", ["status"])

    op.create_table(
        "k20_alerts_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("ticker", sa.Text(), nullable=True),
        sa.Column("trigger", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("channel", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_k20_alerts_ts", "k20_alerts_log", ["ts"])
    op.create_index("idx_k20_alerts_ticker", "k20_alerts_log", ["ticker"])


def downgrade() -> None:
    op.drop_table("k20_alerts_log")
    op.drop_table("k20_job_runs")
    op.drop_table("k20_request_budget")
    op.drop_table("k20_positions")
    op.drop_table("k20_watchlist")
    op.drop_table("k20_catalysts")
    op.drop_table("k20_llm_runs")
    op.drop_table("k20_sentiment_daily")
    op.drop_table("k20_signals")
    op.drop_table("k20_alias_blocklist")
    op.drop_table("k20_company_aliases")
    op.drop_table("k20_universe")
