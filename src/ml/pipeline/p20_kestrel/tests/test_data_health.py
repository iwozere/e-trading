"""Tests for P20 Kestrel data health check logic."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.reporting.data_health import (
    check_av_budget,
    check_llm_budget,
)


def test_check_av_budget_ok(monkeypatch):
    """Returns None when budget is under 90%."""
    from src.ml.pipeline.p20_kestrel.reporting import data_health
    monkeypatch.setattr(
        data_health, "get_or_create_budget",
        lambda source, run_date, quota: {"used": 10}
    )
    result = check_av_budget(date.today())
    assert result is None


def test_check_av_budget_warn(monkeypatch):
    """Returns warning string when budget ≥ 90%."""
    from src.ml.pipeline.p20_kestrel.reporting import data_health
    monkeypatch.setattr(
        data_health, "get_or_create_budget",
        lambda source, run_date, quota: {"used": 19}
    )
    monkeypatch.setattr(data_health, "AV_DAILY_QUOTA", 20)
    result = check_av_budget(date.today())
    assert result is not None
    assert "AV budget" in result


def test_check_llm_budget_ok(monkeypatch):
    """Returns None when LLM spend is under 80%."""
    from src.ml.pipeline.p20_kestrel.reporting import data_health
    monkeypatch.setattr(data_health, "get_llm_monthly_spend", lambda: 10.0)
    monkeypatch.setattr(data_health, "LLM_MONTHLY_BUDGET_USD", 100.0)
    result = check_llm_budget()
    assert result is None


def test_check_llm_budget_warn(monkeypatch):
    """Returns warning string when LLM spend ≥ 80%."""
    from src.ml.pipeline.p20_kestrel.reporting import data_health
    monkeypatch.setattr(data_health, "get_llm_monthly_spend", lambda: 85.0)
    monkeypatch.setattr(data_health, "LLM_MONTHLY_BUDGET_USD", 100.0)
    result = check_llm_budget()
    assert result is not None
    assert "LLM spend" in result


def test_check_llm_budget_exception(monkeypatch):
    """Returns None on exception (graceful fail)."""
    from src.ml.pipeline.p20_kestrel.reporting import data_health

    def _raise():
        raise RuntimeError("db error")

    monkeypatch.setattr(data_health, "get_llm_monthly_spend", _raise)
    result = check_llm_budget()
    assert result is None
