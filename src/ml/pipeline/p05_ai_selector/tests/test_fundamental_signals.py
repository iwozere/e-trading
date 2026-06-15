"""Tests for fundamental signal scoring."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.ml.pipeline.p05_ai_selector.signals.fundamental import score_fundamentals, build_sector_medians


def _make_fund(
    pe_ratio=20.0,
    profit_margin=0.20,
    debt_to_equity=1.0,
    revenue_growth=0.15,
    dividend_yield=0.02,
    sector="Technology",
) -> dict:
    return {
        "pe_ratio": pe_ratio,
        "profit_margin": profit_margin,
        "debt_to_equity": debt_to_equity,
        "revenue_growth": revenue_growth,
        "dividend_yield": dividend_yield,
        "sector": sector,
    }


class TestScoreFundamentals:
    def test_none_fundamentals_returns_zero(self):
        """None input returns (0.0, {}) immediately."""
        score, breakdown = score_fundamentals(None, {})
        assert score == 0.0
        assert breakdown == {}

    def test_high_margin_scores(self):
        """profit_margin=0.25 (25%) satisfies the quality threshold (>15%)."""
        fund = _make_fund(profit_margin=0.25)
        score, breakdown = score_fundamentals(fund, {})
        assert breakdown["quality_signal"] is True
        assert score >= 10

    def test_low_margin_no_quality_score(self):
        """profit_margin=0.10 (10%) does not satisfy the quality threshold."""
        fund = _make_fund(profit_margin=0.10)
        score, breakdown = score_fundamentals(fund, {})
        assert breakdown["quality_signal"] is False

    def test_sector_median_pe_comparison(self):
        """PE below sector median awards value points."""
        fund = _make_fund(pe_ratio=15.0, sector="Technology")
        sector_medians = {"Technology": {"median_pe": 25.0}}
        score, breakdown = score_fundamentals(fund, sector_medians)
        assert breakdown["value_signal"] is True
        assert score >= 10

    def test_pe_above_sector_median_no_value(self):
        """PE above sector median: no value signal."""
        fund = _make_fund(pe_ratio=40.0, sector="Technology")
        sector_medians = {"Technology": {"median_pe": 25.0}}
        score, breakdown = score_fundamentals(fund, sector_medians)
        assert breakdown["value_signal"] is False

    def test_revenue_growth_as_fraction(self):
        """revenue_growth=0.15 (15% as fraction) triggers growth signal."""
        fund = _make_fund(revenue_growth=0.15)
        score, breakdown = score_fundamentals(fund, {})
        assert breakdown["growth_signal"] is True


class TestBuildSectorMedians:
    def test_computes_median_correctly(self):
        """Median PE is computed per sector."""
        funds = {
            "AAPL": _make_fund(pe_ratio=20.0, sector="Technology"),
            "MSFT": _make_fund(pe_ratio=30.0, sector="Technology"),
            "JPM":  _make_fund(pe_ratio=12.0, sector="Financial Services"),
        }
        medians = build_sector_medians(funds)
        assert medians["Technology"]["median_pe"] == 25.0
        assert medians["Financial Services"]["median_pe"] == 12.0

    def test_none_fundamentals_skipped(self):
        """None entries are skipped without error."""
        funds = {"AAPL": _make_fund(pe_ratio=20.0, sector="Technology"), "MSFT": None}
        medians = build_sector_medians(funds)
        assert "Technology" in medians
