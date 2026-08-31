"""Tests for features/registry.py."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.registry import get_feature, list_features, register_feature


def test_register_and_get_feature_round_trip():
    @register_feature("test_block.some_feature")
    def _some_feature(company_id, as_of, ctx):
        return 1.0

    assert get_feature("test_block.some_feature") is _some_feature
    assert "test_block.some_feature" in list_features()


def test_register_feature_duplicate_name_raises():
    @register_feature("test_block.duplicate_feature")
    def _first(company_id, as_of, ctx):
        return 1.0

    with pytest.raises(ValueError):

        @register_feature("test_block.duplicate_feature")
        def _second(company_id, as_of, ctx):
            return 2.0


def test_get_feature_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        get_feature("test_block.does_not_exist")
