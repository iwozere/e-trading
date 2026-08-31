"""p22_biotech_ma_schema

Revision ID: 003_p22_biotech_ma
Revises: 002_kestrel
Create Date: 2026-08-30

Adds all p22_* tables for the P22 Biotech M&A pipeline (spec §3.2, §3.4,
§7.2). Does NOT modify any existing tables.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "003_p22_biotech_ma"
down_revision = "002_kestrel"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "p22_company",
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("cik", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=True),
        sa.Column("exchange", sa.Text(), nullable=True),
        sa.Column("sic_code", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("delisted_date", sa.Date(), nullable=True),
        sa.Column("role", sa.Text(), nullable=True),
        sa.CheckConstraint("role IN ('target','acquirer','both')", name="ck_p22_company_role"),
        sa.PrimaryKeyConstraint("company_id"),
        sa.UniqueConstraint("cik", name="uq_p22_company_cik"),
    )
    op.create_index("idx_p22_company_ticker", "p22_company", ["ticker"])
    op.create_index("idx_p22_company_sic", "p22_company", ["sic_code"])

    op.create_table(
        "p22_company_alias",
        sa.Column("alias_id", sa.BigInteger(), nullable=False),
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("known_from", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_company_alias_company_id_p22_company"),
        sa.PrimaryKeyConstraint("alias_id"),
        sa.UniqueConstraint("company_id", "alias", "source", name="uq_p22_company_alias"),
    )
    op.create_index("idx_p22_company_alias_alias", "p22_company_alias", ["alias"])

    op.create_table(
        "p22_financial_fact",
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("metric", sa.Text(), nullable=False),
        sa.Column("value", sa.Numeric(), nullable=True),
        sa.Column("unit", sa.Text(), nullable=False, server_default="USD"),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("valid_from", sa.Date(), nullable=True),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.Column("known_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_financial_fact_company_id_p22_company"),
        sa.PrimaryKeyConstraint("fact_id"),
    )
    op.create_index("idx_p22_financial_fact_company_metric", "p22_financial_fact", ["company_id", "metric"])
    op.create_index("idx_p22_financial_fact_known_from", "p22_financial_fact", ["known_from"])

    op.create_table(
        "p22_asset",
        sa.Column("asset_id", sa.BigInteger(), nullable=False),
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("modality", sa.Text(), nullable=True),
        sa.Column("target_protein", sa.Text(), nullable=True),
        sa.Column("therapeutic_area", sa.Text(), nullable=False),
        sa.Column("indication", sa.Text(), nullable=True),
        sa.Column("is_lead", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_asset_company_id_p22_company"),
        sa.PrimaryKeyConstraint("asset_id"),
    )
    op.create_index("idx_p22_asset_company", "p22_asset", ["company_id"])

    op.create_table(
        "p22_trial",
        sa.Column("nct_id", sa.Text(), nullable=False),
        sa.Column("asset_id", sa.BigInteger(), nullable=True),
        sa.Column("phase", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("enrollment", sa.Integer(), nullable=True),
        sa.Column("primary_completion_date", sa.Date(), nullable=True),
        sa.Column("uses_biomarker_selection", sa.Boolean(), nullable=True),
        sa.Column("is_randomized", sa.Boolean(), nullable=True),
        sa.Column("has_active_comparator", sa.Boolean(), nullable=True),
        sa.Column("primary_endpoint_text", sa.Text(), nullable=True),
        sa.Column("endpoint_changed_midtrial", sa.Boolean(), nullable=True),
        sa.Column("countries", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("known_from", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["asset_id"], ["p22_asset.asset_id"], name="fk_p22_trial_asset_id_p22_asset"),
        sa.PrimaryKeyConstraint("nct_id"),
    )
    op.create_index("idx_p22_trial_asset", "p22_trial", ["asset_id"])

    op.create_table(
        "p22_patent_expiry",
        sa.Column("patent_expiry_id", sa.BigInteger(), nullable=False),
        sa.Column("acquirer_id", sa.BigInteger(), nullable=False),
        sa.Column("product_name", sa.Text(), nullable=True),
        sa.Column("application_no", sa.Text(), nullable=True),
        sa.Column("therapeutic_area", sa.Text(), nullable=True),
        sa.Column("loe_date", sa.Date(), nullable=False),
        sa.Column("ttm_revenue_usd", sa.Numeric(), nullable=True),
        sa.Column("exclusivity_type", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["acquirer_id"], ["p22_company.company_id"], name="fk_p22_patent_expiry_acquirer_id_p22_company"),
        sa.CheckConstraint(
            "exclusivity_type IN ('patent','orphan','ped','bla_12yr') OR exclusivity_type IS NULL",
            name="ck_p22_patent_expiry_exclusivity_type",
        ),
        sa.CheckConstraint(
            "source IN ('orange_book','purple_book','manual') OR source IS NULL",
            name="ck_p22_patent_expiry_source",
        ),
        sa.PrimaryKeyConstraint("patent_expiry_id"),
    )
    op.create_index("idx_p22_patent_expiry_acquirer", "p22_patent_expiry", ["acquirer_id"])
    op.create_index("idx_p22_patent_expiry_loe_date", "p22_patent_expiry", ["loe_date"])

    op.create_table(
        "p22_deal",
        sa.Column("deal_id", sa.BigInteger(), nullable=False),
        sa.Column("target_id", sa.BigInteger(), nullable=False),
        sa.Column("acquirer_id", sa.BigInteger(), nullable=True),
        sa.Column("announcement_date", sa.Date(), nullable=False),
        sa.Column("completion_date", sa.Date(), nullable=True),
        sa.Column("upfront_per_share", sa.Numeric(), nullable=True),
        sa.Column("has_cvr", sa.Boolean(), nullable=True),
        sa.Column("cvr_max_per_share", sa.Numeric(), nullable=True),
        sa.Column("cvr_realized_per_share", sa.Numeric(), nullable=True),
        sa.Column("premium_1d", sa.Numeric(), nullable=True),
        sa.Column("premium_30d", sa.Numeric(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("deal_type", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["target_id"], ["p22_company.company_id"], name="fk_p22_deal_target_id_p22_company"),
        sa.ForeignKeyConstraint(["acquirer_id"], ["p22_company.company_id"], name="fk_p22_deal_acquirer_id_p22_company"),
        sa.CheckConstraint(
            "status IN ('announced','completed','terminated') OR status IS NULL",
            name="ck_p22_deal_status",
        ),
        sa.CheckConstraint(
            "deal_type IN ('strategic_acquisition','reverse_merger','shell_transaction',"
            "'liquidation','asset_sale','pe_take_private') OR deal_type IS NULL",
            name="ck_p22_deal_type",
        ),
        sa.PrimaryKeyConstraint("deal_id"),
    )
    op.create_index("idx_p22_deal_target", "p22_deal", ["target_id"])
    op.create_index("idx_p22_deal_announcement_date", "p22_deal", ["announcement_date"])

    op.create_table(
        "p22_corporate_process_event",
        sa.Column("event_id", sa.BigInteger(), nullable=False),
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column("strength", sa.Text(), nullable=True),
        sa.Column("advisor_name", sa.Text(), nullable=True),
        sa.Column("accession_no", sa.Text(), nullable=True),
        sa.Column("matched_phrase", sa.Text(), nullable=True),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("known_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_process_event_company_id_p22_company"),
        sa.CheckConstraint(
            "state IN ('rumored','disclosed_open','concluded_deal','concluded_no_deal')",
            name="ck_p22_process_event_state",
        ),
        sa.CheckConstraint(
            "scope IN ('whole_company','asset_only','unclear') OR scope IS NULL",
            name="ck_p22_process_event_scope",
        ),
        sa.CheckConstraint(
            "strength IN ('strong','moderate') OR strength IS NULL",
            name="ck_p22_process_event_strength",
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("idx_p22_process_event_company", "p22_corporate_process_event", ["company_id"])
    op.create_index("idx_p22_process_event_verified", "p22_corporate_process_event", ["is_verified"])

    op.create_table(
        "p22_activist_position",
        sa.Column("position_id", sa.BigInteger(), nullable=False),
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("filer_cik", sa.Text(), nullable=False),
        sa.Column("filer_name", sa.Text(), nullable=True),
        sa.Column("filer_type", sa.Text(), nullable=True),
        sa.Column("form_type", sa.Text(), nullable=False),
        sa.Column("pct_of_class", sa.Numeric(), nullable=True),
        sa.Column("stated_intent", sa.Text(), nullable=True),
        sa.Column("amendment_seq", sa.Integer(), nullable=True),
        sa.Column("filed_date", sa.Date(), nullable=False),
        sa.Column("known_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_activist_position_company_id_p22_company"),
        sa.CheckConstraint(
            "filer_type IN ('activist','crossover_fund','strategic_corporate','other') OR filer_type IS NULL",
            name="ck_p22_activist_filer_type",
        ),
        sa.CheckConstraint(
            "form_type IN ('SC 13D','SC 13D/A','SC 13G','SC 13G/A')",
            name="ck_p22_activist_form_type",
        ),
        sa.CheckConstraint(
            "stated_intent IN ('passive','engagement','board_seats','sale_demand') OR stated_intent IS NULL",
            name="ck_p22_activist_stated_intent",
        ),
        sa.PrimaryKeyConstraint("position_id"),
    )
    op.create_index("idx_p22_activist_company", "p22_activist_position", ["company_id"])
    op.create_index("idx_p22_activist_known_from", "p22_activist_position", ["known_from"])

    op.create_table(
        "p22_partnership_structure",
        sa.Column("structure_id", sa.BigInteger(), nullable=False),
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("partner_id", sa.BigInteger(), nullable=True),
        sa.Column("asset_id", sa.BigInteger(), nullable=True),
        sa.Column("structure_type", sa.Text(), nullable=False),
        sa.Column("partner_equity_pct", sa.Numeric(), nullable=True),
        sa.Column("agreement_date", sa.Date(), nullable=True),
        sa.Column("option_trigger", sa.Text(), nullable=True),
        sa.Column("is_redacted", sa.Boolean(), nullable=True),
        sa.Column("entry_method", sa.Text(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("known_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_partnership_company_id_p22_company"),
        sa.ForeignKeyConstraint(["partner_id"], ["p22_company.company_id"], name="fk_p22_partnership_partner_id_p22_company"),
        sa.ForeignKeyConstraint(["asset_id"], ["p22_asset.asset_id"], name="fk_p22_partnership_asset_id_p22_asset"),
        sa.CheckConstraint(
            "structure_type IN ('acquisition_option','rofn_rofr','equity_plus_commercial','license_only')",
            name="ck_p22_partnership_structure_type",
        ),
        sa.CheckConstraint(
            "entry_method IN ('manual','keyword_detected')",
            name="ck_p22_partnership_entry_method",
        ),
        sa.PrimaryKeyConstraint("structure_id"),
    )
    op.create_index("idx_p22_partnership_company", "p22_partnership_structure", ["company_id"])
    op.create_index("idx_p22_partnership_partner", "p22_partnership_structure", ["partner_id"])

    op.create_table(
        "p22_score_run",
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_table(
        "p22_score",
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("composite", sa.Numeric(), nullable=True),
        sa.Column("rank_by_composite", sa.Integer(), nullable=True),
        sa.Column("expected_value", sa.Numeric(), nullable=True),
        sa.Column("rank_by_expected_value", sa.Integer(), nullable=True),
        sa.Column("p_deal_24m", sa.Numeric(), nullable=True),
        sa.Column("expected_return_if_deal", sa.Numeric(), nullable=True),
        sa.Column("tier", sa.SmallInteger(), nullable=True),
        sa.Column("tier_reason", sa.Text(), nullable=True),
        sa.Column("subscores", postgresql.JSONB(), nullable=True),
        sa.Column("contributions", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["p22_score_run.run_id"], name="fk_p22_score_run_id_p22_score_run"),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_score_company_id_p22_company"),
        sa.CheckConstraint("tier BETWEEN 0 AND 3 OR tier IS NULL", name="ck_p22_score_tier"),
        sa.PrimaryKeyConstraint("run_id", "company_id"),
    )
    op.create_index("idx_p22_score_rank_ev", "p22_score", ["run_id", "rank_by_expected_value"])
    op.create_index("idx_p22_score_rank_composite", "p22_score", ["run_id", "rank_by_composite"])

    op.create_table(
        "p22_review_item",
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("item_type", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("evidence_url", sa.Text(), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("reviewed_by", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "item_type IN ('entity_match','process_event','activist_intent',"
            "'partnership_structure','deal_type')",
            name="ck_p22_review_item_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending','confirmed','rejected','needs_info')",
            name="ck_p22_review_item_status",
        ),
        sa.PrimaryKeyConstraint("item_id"),
    )
    op.create_index("idx_p22_review_item_status_priority", "p22_review_item", ["status", "priority"])

    op.create_table(
        "p22_fetch_failure",
        sa.Column("failure_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("entity", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.PrimaryKeyConstraint("failure_id"),
    )
    op.create_index("idx_p22_fetch_failure_source", "p22_fetch_failure", ["source"])
    op.create_index("idx_p22_fetch_failure_resolved", "p22_fetch_failure", ["resolved"])


def downgrade() -> None:
    op.drop_table("p22_fetch_failure")
    op.drop_table("p22_review_item")
    op.drop_table("p22_score")
    op.drop_table("p22_score_run")
    op.drop_table("p22_partnership_structure")
    op.drop_table("p22_activist_position")
    op.drop_table("p22_corporate_process_event")
    op.drop_table("p22_deal")
    op.drop_table("p22_patent_expiry")
    op.drop_table("p22_trial")
    op.drop_table("p22_asset")
    op.drop_table("p22_financial_fact")
    op.drop_table("p22_company_alias")
    op.drop_table("p22_company")
