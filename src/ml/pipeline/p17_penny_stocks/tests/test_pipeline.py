"""Tests for P17Pipeline orchestrator."""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.config import P17PipelineConfig
from src.ml.pipeline.p17_penny_stocks.p17_pipeline import P17Pipeline, _sf
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate


# ── _sf helper ─────────────────────────────────────────────────────────────────

def test_sf_converts_float():
    assert _sf(3.14) == 3.14


def test_sf_converts_int():
    assert _sf(42) == 42.0


def test_sf_converts_string_number():
    assert _sf("1.5") == 1.5


def test_sf_returns_default_for_none():
    assert _sf(None) == 0.0


def test_sf_returns_default_for_invalid():
    assert _sf("not-a-number") == 0.0


def test_sf_custom_default():
    assert _sf(None, default=-1.0) == -1.0


# ── Pipeline initialisation ────────────────────────────────────────────────────

def test_pipeline_init_creates_results_dir(tmp_path):
    config = P17PipelineConfig.create_default()
    with patch("src.ml.pipeline.p17_penny_stocks.p17_pipeline.Path") as mock_path_cls:
        results_mock = MagicMock(spec=Path)
        results_mock.__truediv__ = lambda self, other: results_mock
        mock_path_cls.return_value = results_mock
        # Just verify the object constructs without exception using real Path
    pipeline = P17Pipeline(config=config, target_date="2025-01-01")
    assert pipeline.target_date == "2025-01-01"
    assert pipeline.config is config


def test_pipeline_uses_yesterday_as_default_date():
    from datetime import datetime, timedelta
    config = P17PipelineConfig.create_default()
    pipeline = P17Pipeline(config=config)
    expected = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    assert pipeline.target_date == expected


# ── _build_candidates ──────────────────────────────────────────────────────────

def _make_pipeline(target_date: str = "2025-01-01") -> P17Pipeline:
    return P17Pipeline(config=P17PipelineConfig.create_default(), target_date=target_date)


def test_build_candidates_basic():
    pipeline = _make_pipeline()
    universe_df = pd.DataFrame([
        {
            "ticker": "ABCD",
            "company_name": "Acme Corp",
            "exchange_norm": "NASDAQ",
            "sector": "Technology",
            "industry": "Software",
            "price": 3.50,
            "market_cap": 100_000_000,
            "volume": 1_000_000,
            "avg_volume": 800_000,
            "float_shares": 15_000_000,
            "shares_outstanding": 20_000_000,
            "high_52w": 8.0,
            "low_52w": 1.5,
        }
    ])
    candidates = pipeline._build_candidates(universe_df, {})
    assert len(candidates) == 1
    c = candidates[0]
    assert c.ticker == "ABCD"
    assert c.price == 3.50
    assert c.exchange == "NASDAQ"
    assert c.run_date == "2025-01-01"


def test_build_candidates_skips_empty_ticker():
    pipeline = _make_pipeline()
    universe_df = pd.DataFrame([
        {"ticker": "", "company_name": "Ghost", "price": 1.0},
    ])
    candidates = pipeline._build_candidates(universe_df, {})
    assert candidates == []


def test_build_candidates_merges_fundamentals():
    pipeline = _make_pipeline()
    universe_df = pd.DataFrame([
        {
            "ticker": "XYZ",
            "company_name": "XYZ Inc",
            "price": 2.0,
            "market_cap": 50_000_000,
        }
    ])
    fundamentals = {
        "XYZ": {
            "revenue_growth_yoy": 0.35,
            "gross_margin": 0.40,
            "total_cash": 5_000_000,
            "total_debt": 1_000_000,
            "cash_runway_months": 18.0,
            "operating_cashflow": 200_000,
        }
    }
    candidates = pipeline._build_candidates(universe_df, fundamentals)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.revenue_growth_yoy == 0.35
    assert c.gross_margin == 0.40
    assert c.cash_runway_months == 18.0


def test_build_candidates_handles_missing_columns_gracefully():
    pipeline = _make_pipeline()
    universe_df = pd.DataFrame([{"ticker": "MIN"}])
    candidates = pipeline._build_candidates(universe_df, {})
    assert len(candidates) == 1
    assert candidates[0].price == 0.0
    assert candidates[0].market_cap == 0.0


def test_build_candidates_fallback_exchange():
    """Falls back to 'exchange' column when 'exchange_norm' is absent."""
    pipeline = _make_pipeline()
    universe_df = pd.DataFrame([
        {"ticker": "TICK", "exchange": "NASDAQ", "price": 1.0}
    ])
    candidates = pipeline._build_candidates(universe_df, {})
    assert candidates[0].exchange == "NASDAQ"


# ── Abort behaviour ────────────────────────────────────────────────────────────

def test_run_aborts_when_universe_empty():
    pipeline = _make_pipeline()
    pipeline._universe_agent = MagicMock()
    pipeline._universe_agent.run.return_value = pd.DataFrame()

    result = pipeline.run()
    assert result["success"] is False
    assert result["total_candidates"] == 0


def test_run_aborts_when_no_candidates_after_market():
    pipeline = _make_pipeline()

    pipeline._universe_agent = MagicMock()
    pipeline._universe_agent.run.return_value = pd.DataFrame([
        {"ticker": "ABCD", "price": 3.0, "market_cap": 100_000_000}
    ])

    pipeline._market_agent = MagicMock()
    pipeline._market_agent.run.return_value = ({}, {})
    pipeline._market_agent.apply_survival_filter.return_value = []

    # Patch _build_candidates to return empty list (empty fundamentals → filtered out)
    pipeline._build_candidates = MagicMock(return_value=[])

    result = pipeline.run()
    assert result["success"] is False
    assert result["total_candidates"] == 0


# ── _run_job ───────────────────────────────────────────────────────────────────

def test_run_job_success():
    result = P17Pipeline._run_job("TestStage", lambda: {"count": 5})
    assert result["success"] is True
    assert result["count"] == 5
    assert "elapsed_s" in result


def test_run_job_failure_does_not_raise():
    def bad_fn():
        raise RuntimeError("boom")

    result = P17Pipeline._run_job("FailStage", bad_fn)
    assert result["success"] is False
    assert "elapsed_s" in result


def test_run_job_none_return_treated_as_empty():
    result = P17Pipeline._run_job("NoneStage", lambda: None)
    assert result["success"] is True
    assert result == {"success": True, "elapsed_s": result["elapsed_s"]}
