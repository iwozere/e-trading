"""Tests for features/block_e.py (spec §4.5)."""

import sys
from datetime import date
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.block_e import is_foreign_domiciled
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext

_AS_OF = date(2026, 9, 8)
_COMPANY_ID = 7


class _FakeRepo:
    def __init__(self, facts: Dict[str, float]):
        self._facts = facts

    def get_financial_facts_as_of(self, company_id: int, metric: str, as_of_date: date):
        del company_id, as_of_date
        if metric in self._facts:
            return [{"value": self._facts[metric]}]
        return []


def test_is_foreign_domiciled_none_when_not_normalized_yet():
    ctx = FeatureContext(as_of=_AS_OF, repo=_FakeRepo({}))
    assert is_foreign_domiciled(_COMPANY_ID, _AS_OF, ctx) is None


def test_is_foreign_domiciled_real_value_true():
    ctx = FeatureContext(as_of=_AS_OF, repo=_FakeRepo({"is_foreign_domiciled": 1.0}))
    assert is_foreign_domiciled(_COMPANY_ID, _AS_OF, ctx) == 1.0


def test_is_foreign_domiciled_real_value_false():
    ctx = FeatureContext(as_of=_AS_OF, repo=_FakeRepo({"is_foreign_domiciled": 0.0}))
    assert is_foreign_domiciled(_COMPANY_ID, _AS_OF, ctx) == 0.0
