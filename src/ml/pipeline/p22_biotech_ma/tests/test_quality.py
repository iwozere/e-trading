"""Tests for features/quality.py (spec §8.2). No live DB — repo is a MagicMock."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pandera.pandas as pa
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.quality import (
    assert_every_company_has_a_verified_alias,
    block_c_schema,
    loe_date_schema,
)


def test_block_c_schema_accepts_valid_rows():
    df = pd.DataFrame({"cash_runway_months": [0.0, 60.0, 120.0, None], "enterprise_value": [-500.0, 0.0, 1e9, None]})
    validated = block_c_schema().validate(df)
    assert len(validated) == 4


def test_block_c_schema_rejects_cash_runway_months_out_of_range():
    df = pd.DataFrame({"cash_runway_months": [121.0], "enterprise_value": [100.0]})
    with pytest.raises(pa.errors.SchemaError):
        block_c_schema().validate(df)


def test_block_c_schema_rejects_negative_cash_runway_months():
    df = pd.DataFrame({"cash_runway_months": [-1.0], "enterprise_value": [100.0]})
    with pytest.raises(pa.errors.SchemaError):
        block_c_schema().validate(df)


def test_block_c_schema_allows_negative_enterprise_value():
    """Negative EV is legitimate (spec §8.2) — must not be rejected."""
    df = pd.DataFrame({"cash_runway_months": [10.0], "enterprise_value": [-1_000_000.0]})
    validated = block_c_schema().validate(df)
    assert validated["enterprise_value"].iloc[0] == -1_000_000.0


def test_block_c_schema_extra_columns_allowed():
    df = pd.DataFrame({"cash_runway_months": [10.0], "enterprise_value": [100.0], "company_id": [7]})
    validated = block_c_schema().validate(df)
    assert "company_id" in validated.columns


def test_loe_date_schema_accepts_in_range():
    df = pd.DataFrame({"loe_date": pd.to_datetime(["1990-01-01", "2030-06-01"])})
    schema = pa.DataFrameSchema({"loe_date": loe_date_schema(date(2024, 6, 1))})
    schema.validate(df)  # must not raise


def test_loe_date_schema_rejects_before_1990():
    df = pd.DataFrame({"loe_date": pd.to_datetime(["1985-01-01"])})
    schema = pa.DataFrameSchema({"loe_date": loe_date_schema(date(2024, 6, 1))})
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)


def test_loe_date_schema_rejects_more_than_25_years_out():
    df = pd.DataFrame({"loe_date": pd.to_datetime(["2060-01-01"])})
    schema = pa.DataFrameSchema({"loe_date": loe_date_schema(date(2024, 6, 1))})
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)


def test_assert_every_company_has_a_verified_alias_passes_when_none_missing():
    repo = MagicMock()
    repo.get_companies_without_verified_alias.return_value = []
    assert_every_company_has_a_verified_alias(repo)  # must not raise


def test_assert_every_company_has_a_verified_alias_raises_listing_offenders():
    repo = MagicMock()
    repo.get_companies_without_verified_alias.return_value = [3, 1, 2]
    with pytest.raises(AssertionError, match=r"\[1, 2, 3\]"):
        assert_every_company_has_a_verified_alias(repo)
