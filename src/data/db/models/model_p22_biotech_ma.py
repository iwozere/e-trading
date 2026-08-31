"""
SQLAlchemy ORM models for all p22_* tables (P22 Biotech M&A pipeline).

Only p22_* tables are defined here. No existing tables are modified.

Bitemporal discipline (spec §3.1) — non-negotiable: `valid_from`/`valid_to`
mark when a fact was true in the world, `known_from` marks when the pipeline
learned it. All feature computation queries filter `known_from <= as_of_date`.
`valid_to IS NULL` means "still true." Restatements are new rows with the
prior row's `valid_to` closed, never an in-place `UPDATE ... SET value` — see
`src/data/db/repos/repo_p22_biotech_ma.py:upsert_financial_fact_bitemporal`,
the single write path that enforces this.

This module is schema-only for M1 (spec §9: "bitemporal schema live"). Tables
belonging to M2+ milestones (asset, trial, patent_expiry, deal,
corporate_process_event, activist_position, partnership_structure, score_run,
score, price_daily, corporate_action) exist here but are not yet written to
by any M1 ingest job — `p22_price_daily`/`p22_corporate_action` (spec §2.0.7,
added v0.6) wait on the deferred market-data vendor decision (§2.4).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from src.data.db.core.base import Base
from src.data.db.core.json_types import JsonType

# ---------------------------------------------------------------------------
# Core entity tables (spec §3.2)
# ---------------------------------------------------------------------------


class P22Company(Base):
    """A company — target, acquirer, or both. Universe basis: spec §2.0."""

    __tablename__ = "p22_company"

    company_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cik: Mapped[str | None] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    ticker: Mapped[str | None] = mapped_column(Text)
    exchange: Mapped[str | None] = mapped_column(Text)
    sic_code: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool | None] = mapped_column(Boolean)
    delisted_date: Mapped[date | None] = mapped_column(Date)
    role: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("role IN ('target','acquirer','both')", name="ck_p22_company_role"),
        Index("idx_p22_company_ticker", "ticker"),
        Index("idx_p22_company_sic", "sic_code"),
    )


class P22CompanyAlias(Base):
    """CT.gov sponsor strings, FDA applicant names, etc. — spec §3.3 entity resolution."""

    __tablename__ = "p22_company_alias"

    alias_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), nullable=False)
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    known_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("company_id", "alias", "source", name="uq_p22_company_alias"),
        Index("idx_p22_company_alias_alias", "alias"),
    )


class P22FinancialFact(Base):
    """
    Bitemporal financial facts — cash, burn, debt, shares, vendor-sourced
    market cap, etc. (spec §2.1, §2.4, §3.2). The single generic write path
    is `P22Repo.upsert_financial_fact_bitemporal`; nothing else writes here.
    """

    __tablename__ = "p22_financial_fact"

    fact_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), nullable=False)
    metric: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[float | None] = mapped_column(Numeric)
    unit: Mapped[str] = mapped_column(Text, nullable=False, default="USD")
    period_end: Mapped[date | None] = mapped_column(Date)
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    known_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("idx_p22_financial_fact_company_metric", "company_id", "metric"),
        Index("idx_p22_financial_fact_known_from", "known_from"),
        # Partial-uniqueness on "one open row per (company, metric)" is enforced
        # in the repo layer (close valid_to before inserting), not here — a DB
        # constraint can't express "at most one NULL valid_to per group" portably.
    )


class P22Asset(Base):
    """A drug program (spec §3.2, §4.2 Block B)."""

    __tablename__ = "p22_asset"

    asset_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), nullable=False)
    name: Mapped[str | None] = mapped_column(Text)
    modality: Mapped[str | None] = mapped_column(Text)
    target_protein: Mapped[str | None] = mapped_column(Text)
    therapeutic_area: Mapped[str] = mapped_column(Text, nullable=False)
    indication: Mapped[str | None] = mapped_column(Text)
    is_lead: Mapped[bool | None] = mapped_column(Boolean)

    __table_args__ = (Index("idx_p22_asset_company", "company_id"),)


class P22Trial(Base):
    """A clinical trial (spec §2.2, §3.2)."""

    __tablename__ = "p22_trial"

    nct_id: Mapped[str] = mapped_column(Text, primary_key=True)
    asset_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("p22_asset.asset_id"))
    phase: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(Text)
    enrollment: Mapped[int | None] = mapped_column(Integer)
    primary_completion_date: Mapped[date | None] = mapped_column(Date)
    uses_biomarker_selection: Mapped[bool | None] = mapped_column(Boolean)
    is_randomized: Mapped[bool | None] = mapped_column(Boolean)
    has_active_comparator: Mapped[bool | None] = mapped_column(Boolean)
    primary_endpoint_text: Mapped[str | None] = mapped_column(Text)
    endpoint_changed_midtrial: Mapped[bool | None] = mapped_column(Boolean)
    countries: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    known_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("idx_p22_trial_asset", "asset_id"),)


class P22PatentExpiry(Base):
    """Acquirer-side patent/exclusivity expiry (spec §2.3, §4.1 Block A)."""

    __tablename__ = "p22_patent_expiry"

    patent_expiry_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    acquirer_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), nullable=False)
    product_name: Mapped[str | None] = mapped_column(Text)
    application_no: Mapped[str | None] = mapped_column(Text)
    therapeutic_area: Mapped[str | None] = mapped_column(Text)
    loe_date: Mapped[date] = mapped_column(Date, nullable=False)
    ttm_revenue_usd: Mapped[float | None] = mapped_column(Numeric)
    exclusivity_type: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "exclusivity_type IN ('patent','orphan','ped','bla_12yr') OR exclusivity_type IS NULL",
            name="ck_p22_patent_expiry_exclusivity_type",
        ),
        CheckConstraint(
            "source IN ('orange_book','purple_book','manual') OR source IS NULL",
            name="ck_p22_patent_expiry_source",
        ),
        Index("idx_p22_patent_expiry_acquirer", "acquirer_id"),
        Index("idx_p22_patent_expiry_loe_date", "loe_date"),
    )


class P22Deal(Base):
    """Acquisition labels for the backtest (spec §2.5)."""

    __tablename__ = "p22_deal"

    deal_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    target_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), nullable=False)
    acquirer_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"))
    announcement_date: Mapped[date] = mapped_column(Date, nullable=False)
    completion_date: Mapped[date | None] = mapped_column(Date)
    upfront_per_share: Mapped[float | None] = mapped_column(Numeric)
    has_cvr: Mapped[bool | None] = mapped_column(Boolean)
    cvr_max_per_share: Mapped[float | None] = mapped_column(Numeric)
    cvr_realized_per_share: Mapped[float | None] = mapped_column(Numeric)  # spec §10, populated as outcomes known
    premium_1d: Mapped[float | None] = mapped_column(Numeric)
    premium_30d: Mapped[float | None] = mapped_column(Numeric)
    status: Mapped[str | None] = mapped_column(Text)
    deal_type: Mapped[str | None] = mapped_column(Text)  # spec §2.5 mandatory exclusions

    __table_args__ = (
        CheckConstraint(
            "status IN ('announced','completed','terminated') OR status IS NULL",
            name="ck_p22_deal_status",
        ),
        CheckConstraint(
            "deal_type IN ('strategic_acquisition','reverse_merger','shell_transaction',"
            "'liquidation','asset_sale','pe_take_private') OR deal_type IS NULL",
            name="ck_p22_deal_type",
        ),
        Index("idx_p22_deal_target", "target_id"),
        Index("idx_p22_deal_announcement_date", "announcement_date"),
    )


# ---------------------------------------------------------------------------
# Block G — revealed process signals (spec §2.6, §3.2, §4.7)
# ---------------------------------------------------------------------------


class P22CorporateProcessEvent(Base):
    """Strategic-alternatives disclosures (spec §2.6.1)."""

    __tablename__ = "p22_corporate_process_event"

    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), nullable=False)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str | None] = mapped_column(Text)
    strength: Mapped[str | None] = mapped_column(Text)
    advisor_name: Mapped[str | None] = mapped_column(Text)
    accession_no: Mapped[str | None] = mapped_column(Text)
    matched_phrase: Mapped[str | None] = mapped_column(Text)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    known_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "state IN ('rumored','disclosed_open','concluded_deal','concluded_no_deal')",
            name="ck_p22_process_event_state",
        ),
        CheckConstraint(
            "scope IN ('whole_company','asset_only','unclear') OR scope IS NULL",
            name="ck_p22_process_event_scope",
        ),
        CheckConstraint(
            "strength IN ('strong','moderate') OR strength IS NULL",
            name="ck_p22_process_event_strength",
        ),
        Index("idx_p22_process_event_company", "company_id"),
        Index("idx_p22_process_event_verified", "is_verified"),
    )


class P22ActivistPosition(Base):
    """Schedule 13D/13G activist and strategic-toehold positions (spec §2.6.2)."""

    __tablename__ = "p22_activist_position"

    position_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), nullable=False)
    filer_cik: Mapped[str] = mapped_column(Text, nullable=False)
    filer_name: Mapped[str | None] = mapped_column(Text)
    filer_type: Mapped[str | None] = mapped_column(Text)
    form_type: Mapped[str] = mapped_column(Text, nullable=False)
    pct_of_class: Mapped[float | None] = mapped_column(Numeric)
    stated_intent: Mapped[str | None] = mapped_column(Text)
    amendment_seq: Mapped[int | None] = mapped_column(Integer)
    filed_date: Mapped[date] = mapped_column(Date, nullable=False)
    known_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "filer_type IN ('activist','crossover_fund','strategic_corporate','other') OR filer_type IS NULL",
            name="ck_p22_activist_filer_type",
        ),
        CheckConstraint(
            "form_type IN ('SC 13D','SC 13D/A','SC 13G','SC 13G/A')",
            name="ck_p22_activist_form_type",
        ),
        CheckConstraint(
            "stated_intent IN ('passive','engagement','board_seats','sale_demand') OR stated_intent IS NULL",
            name="ck_p22_activist_stated_intent",
        ),
        Index("idx_p22_activist_company", "company_id"),
        # 13D is due within 5 business days of crossing 5% — known_from must be
        # the filing date, never the crossing date (spec §4.7 bitemporal caution).
        Index("idx_p22_activist_known_from", "known_from"),
    )


class P22PartnershipStructure(Base):
    """Incumbent-partner / option-to-acquire structures (spec §2.6.3)."""

    __tablename__ = "p22_partnership_structure"

    structure_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), nullable=False)
    partner_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"))
    asset_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("p22_asset.asset_id"))
    structure_type: Mapped[str] = mapped_column(Text, nullable=False)
    partner_equity_pct: Mapped[float | None] = mapped_column(Numeric)
    agreement_date: Mapped[date | None] = mapped_column(Date)
    option_trigger: Mapped[str | None] = mapped_column(Text)
    is_redacted: Mapped[bool | None] = mapped_column(Boolean)
    entry_method: Mapped[str] = mapped_column(Text, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    known_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "structure_type IN ('acquisition_option','rofn_rofr','equity_plus_commercial','license_only')",
            name="ck_p22_partnership_structure_type",
        ),
        CheckConstraint(
            "entry_method IN ('manual','keyword_detected')",
            name="ck_p22_partnership_entry_method",
        ),
        Index("idx_p22_partnership_company", "company_id"),
        Index("idx_p22_partnership_partner", "partner_id"),
    )


# ---------------------------------------------------------------------------
# Price archive and corporate actions (spec §2.0.7, added v0.6)
# ---------------------------------------------------------------------------
# Raw OHLCV, never rewritten; adjustment happens at read time
# (`P22Repo.get_adjusted_close`, `ingest/price_archive.py`), never at write
# time — storing adjusted series makes every split a retroactive rewrite of
# the whole history, which corrupts point-in-time market cap (computed
# against as-filed, unadjusted `dei:EntityCommonStockSharesOutstanding`) and
# leaks 2023 split information into a 2019 `as_of` decision. See
# docs/Design.md and docs/implementation-plan.md §2.0.7.


class P22PriceDaily(Base):
    """RAW daily OHLCV, as traded. Never rewritten (spec §2.0.7)."""

    __tablename__ = "p22_price_daily"

    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    vendor: Mapped[str] = mapped_column(Text, primary_key=True)  # 'ibkr'|'fmp'|'yfinance'
    open_raw: Mapped[float | None] = mapped_column(Numeric)
    high_raw: Mapped[float | None] = mapped_column(Numeric)
    low_raw: Mapped[float | None] = mapped_column(Numeric)
    close_raw: Mapped[float | None] = mapped_column(Numeric)
    volume_raw: Mapped[int | None] = mapped_column(BigInteger)
    known_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("idx_p22_price_daily_company_date", "company_id", "trade_date"),)


class P22CorporateAction(Base):
    """Splits, reverse splits, dividends, spinoffs, ticker changes (spec §2.0.7)."""

    __tablename__ = "p22_corporate_action"

    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), primary_key=True)
    ex_date: Mapped[date] = mapped_column(Date, primary_key=True)
    action_type: Mapped[str] = mapped_column(Text, primary_key=True)
    ratio: Mapped[float | None] = mapped_column(Numeric)  # 4.0 for 4:1 fwd, 0.05 for 1:20 reverse
    cash_amount: Mapped[float | None] = mapped_column(Numeric)
    new_ticker: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text, nullable=False)  # 'sec_8k'|'fmp'|'yfinance'|'ibkr'|'manual'
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    known_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "action_type IN ('split','reverse_split','dividend','spinoff','ticker_change')",
            name="ck_p22_corporate_action_type",
        ),
        Index("idx_p22_corporate_action_company", "company_id"),
    )


# ---------------------------------------------------------------------------
# Scoring (spec §3.2, §5) — schema only, unpopulated until M4+
# ---------------------------------------------------------------------------


class P22ScoreRun(Base):
    """One scoring run (spec §3.2)."""

    __tablename__ = "p22_score_run"

    run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    config_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class P22Score(Base):
    """
    Per-(run, company) score. Two ranks, never a bare `rank` column (spec
    §5.4) — `rank_by_expected_value` is the default view.
    """

    __tablename__ = "p22_score"

    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_score_run.run_id"), primary_key=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("p22_company.company_id"), primary_key=True)

    composite: Mapped[float | None] = mapped_column(Numeric)
    rank_by_composite: Mapped[int | None] = mapped_column(Integer)
    expected_value: Mapped[float | None] = mapped_column(Numeric)
    rank_by_expected_value: Mapped[int | None] = mapped_column(Integer)
    p_deal_24m: Mapped[float | None] = mapped_column(Numeric)
    expected_return_if_deal: Mapped[float | None] = mapped_column(Numeric)

    tier: Mapped[int | None] = mapped_column(SmallInteger)
    tier_reason: Mapped[str | None] = mapped_column(Text)
    subscores: Mapped[dict | None] = mapped_column(JsonType)
    contributions: Mapped[dict | None] = mapped_column(JsonType)

    __table_args__ = (
        CheckConstraint("tier BETWEEN 0 AND 3 OR tier IS NULL", name="ck_p22_score_tier"),
        Index("idx_p22_score_rank_ev", "run_id", "rank_by_expected_value"),
        Index("idx_p22_score_rank_composite", "run_id", "rank_by_composite"),
    )


# ---------------------------------------------------------------------------
# Manual review queue (spec §3.4)
# ---------------------------------------------------------------------------


class P22ReviewItem(Base):
    """
    Entity-resolution candidates, process-event confirmations, activist-
    intent classification, partnership structures, deal-type classification —
    load-bearing manual inputs, not edge cases (spec §3.4).
    """

    __tablename__ = "p22_review_item"

    item_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    item_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JsonType, nullable=False)
    evidence_url: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)
    # Not in the spec §3.4 SQL sketch, added here (like p22_review_item/p22_fetch_failure
    # themselves) because "queue depth and median age by item_type are reported in every
    # run" (§3.4) is unanswerable without an insertion timestamp.
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "item_type IN ('entity_match','process_event','activist_intent',"
            "'partnership_structure','deal_type')",
            name="ck_p22_review_item_type",
        ),
        CheckConstraint(
            "status IN ('pending','confirmed','rejected','needs_info')",
            name="ck_p22_review_item_status",
        ),
        Index("idx_p22_review_item_status_priority", "status", "priority"),
    )


# ---------------------------------------------------------------------------
# Observability (spec §7.2) — not in the spec's §3.2 SQL sketch, but §7.2
# requires every failed fetch (after retries) logged and surfaced in the run
# report, so this table exists from M1 rather than being retrofitted later.
# ---------------------------------------------------------------------------


class P22FetchFailure(Base):
    """A source fetch that failed after all retries (spec §7.2)."""

    __tablename__ = "p22_fetch_failure"

    failure_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    entity: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    error_message: Mapped[str | None] = mapped_column(Text)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("idx_p22_fetch_failure_source", "source"),
        Index("idx_p22_fetch_failure_resolved", "resolved"),
    )
