"""
Tests for P20 Kestrel DB models and migration.

These tests verify the ORM model shapes and column constraints
without requiring a live database connection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.models.model_kestrel import (
    K20AlertsLog,
    K20AliasBlocklist,
    K20Catalyst,
    K20CompanyAlias,
    K20JobRun,
    K20LLMRun,
    K20Position,
    K20RequestBudget,
    K20SentimentDaily,
    K20Signal,
    K20Universe,
    K20Watchlist,
)


def test_model_table_names():
    """All P20 models have k20_ prefix."""
    models = [
        K20Universe, K20Signal, K20SentimentDaily, K20Catalyst,
        K20Watchlist, K20Position, K20LLMRun, K20RequestBudget,
        K20JobRun, K20AlertsLog, K20CompanyAlias, K20AliasBlocklist,
    ]
    for model in models:
        assert model.__tablename__.startswith("k20_"), (
            f"{model.__name__} tablename '{model.__tablename__}' must start with k20_"
        )


def test_k20_universe_columns():
    """Universe table has expected key columns."""
    columns = {c.name for c in K20Universe.__table__.columns}
    for required in ("id", "ticker", "mcap", "status", "updated_at"):
        assert required in columns, f"k20_universe missing column: {required}"


def test_k20_signal_columns():
    """Signal table has required columns."""
    columns = {c.name for c in K20Signal.__table__.columns}
    for required in ("id", "ticker", "date", "signal_type", "value"):
        assert required in columns, f"k20_signals missing column: {required}"


def test_k20_watchlist_columns():
    """Watchlist table has required columns."""
    columns = {c.name for c in K20Watchlist.__table__.columns}
    for required in ("id", "ticker", "sleeve", "state", "score"):
        assert required in columns, f"k20_watchlist missing column: {required}"


def test_k20_position_columns():
    """Position table has required columns."""
    columns = {c.name for c in K20Position.__table__.columns}
    for required in ("id", "ticker", "sleeve", "entry_px", "stop_px", "t1_px"):
        assert required in columns, f"k20_positions missing column: {required}"


def test_k20_llm_run_columns():
    """LLM run table has required columns."""
    columns = {c.name for c in K20LLMRun.__table__.columns}
    for required in ("id", "task_type", "input_ref", "model", "cost_usd"):
        assert required in columns, f"k20_llm_runs missing column: {required}"


def test_k20_job_run_columns():
    """Job run table has required columns."""
    columns = {c.name for c in K20JobRun.__table__.columns}
    for required in ("id", "job", "run_date", "status"):
        assert required in columns, f"k20_job_runs missing column: {required}"


def test_k20_catalyst_columns():
    """Catalyst table has required columns."""
    columns = {c.name for c in K20Catalyst.__table__.columns}
    for required in ("id", "ticker", "event_type", "event_date", "state"):
        assert required in columns, f"k20_catalysts missing column: {required}"
