"""
Tests for P22 Biotech M&A DB models and migration.

These tests verify the ORM model shapes and column constraints without
requiring a live database connection. Mirrors
src/ml/pipeline/p20_kestrel/tests/test_db_models.py.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.models.model_p22_biotech_ma import (
    P22ActivistPosition,
    P22Asset,
    P22Company,
    P22CompanyAlias,
    P22CorporateProcessEvent,
    P22Deal,
    P22FetchFailure,
    P22FinancialFact,
    P22PartnershipStructure,
    P22PatentExpiry,
    P22ReviewItem,
    P22Score,
    P22ScoreRun,
    P22Trial,
)

_ALL_MODELS = [
    P22Company,
    P22CompanyAlias,
    P22FinancialFact,
    P22Asset,
    P22Trial,
    P22PatentExpiry,
    P22Deal,
    P22CorporateProcessEvent,
    P22ActivistPosition,
    P22PartnershipStructure,
    P22ScoreRun,
    P22Score,
    P22ReviewItem,
    P22FetchFailure,
]


def test_model_table_names():
    """All P22 models have p22_ prefix."""
    for model in _ALL_MODELS:
        assert model.__tablename__.startswith("p22_"), (
            f"{model.__name__} tablename '{model.__tablename__}' must start with p22_"
        )


def test_no_duplicate_table_names():
    names = [m.__tablename__ for m in _ALL_MODELS]
    assert len(names) == len(set(names))


def test_p22_company_columns():
    columns = {c.name for c in P22Company.__table__.columns}
    for required in ("company_id", "cik", "name", "ticker", "exchange", "sic_code", "role"):
        assert required in columns, f"p22_company missing column: {required}"


def test_p22_financial_fact_bitemporal_columns():
    """The bitemporal invariant (spec §3.1) requires these four columns."""
    columns = {c.name for c in P22FinancialFact.__table__.columns}
    for required in ("valid_from", "valid_to", "known_from", "source_id"):
        assert required in columns, f"p22_financial_fact missing bitemporal column: {required}"


def test_p22_score_has_two_ranks_never_bare_rank():
    """Spec §5.4: never a bare `rank` column."""
    columns = {c.name for c in P22Score.__table__.columns}
    assert "rank_by_composite" in columns
    assert "rank_by_expected_value" in columns
    assert "rank" not in columns


def test_p22_review_item_verification_gate_columns():
    """Spec §3.4 / §4.7: review items must be status-gated before use."""
    columns = {c.name for c in P22ReviewItem.__table__.columns}
    for required in ("item_type", "status", "payload"):
        assert required in columns


def test_p22_activist_position_and_process_event_have_verification_or_known_from():
    """Spec §4.7 verification gate + bitemporal caution."""
    process_columns = {c.name for c in P22CorporateProcessEvent.__table__.columns}
    assert "is_verified" in process_columns
    assert "known_from" in process_columns

    activist_columns = {c.name for c in P22ActivistPosition.__table__.columns}
    assert "known_from" in activist_columns
    assert "filed_date" in activist_columns
