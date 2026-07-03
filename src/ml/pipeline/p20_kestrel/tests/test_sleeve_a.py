"""Tests for P20 Kestrel Sleeve A scoring logic."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.screening.sleeve_a import (
    _passes_hard_filters,
    _score_interim,
)


def _universe(mcap=5_000_000_000, adv=50_000_000, **kw):
    return {
        "ticker": "TST",
        "mcap": mcap,
        "adv_20d": adv,
        "net_debt_ebitda": kw.get("net_debt_ebitda", 2.0),
        "interest_coverage": kw.get("interest_coverage", 5.0),
        "revenue_yoy_growth": kw.get("revenue_yoy_growth", 0.05),
        "gross_margin": kw.get("gross_margin", 0.40),
    }


def _signals(drawdown=-0.55, **kw):
    return {
        "drawdown_from_2y_high": drawdown,
        "price_vs_50dma": kw.get("price_vs_50dma", 1.0),
        "sma_50_rising": kw.get("sma_50_rising", 0),
        "net_cash": kw.get("net_cash", None),
        "insider_buy_value_90d": kw.get("insider_buy_value_90d", 5_000_000),
        "form4_buy_count_90d": kw.get("form4_buy_count_90d", 3),
        "crowding_score": kw.get("crowding_score", 0.3),
    }


def test_passes_hard_filters_basic():
    """A standard candidate passes all hard filters (returns None)."""
    assert _passes_hard_filters(_universe(), _signals()) is None


def test_fails_drawdown_too_shallow():
    """Drawdown above -40% is filtered out."""
    assert _passes_hard_filters(_universe(), _signals(drawdown=-0.30)) is not None


def test_fails_drawdown_too_deep():
    """Drawdown below -75% is filtered out (wipeout risk)."""
    assert _passes_hard_filters(_universe(), _signals(drawdown=-0.80)) is not None


def test_fails_mcap_too_small():
    """Mcap below $500M is filtered out."""
    assert _passes_hard_filters(_universe(mcap=400_000_000), _signals()) is not None


def test_fails_adv_too_low():
    """ADV below $10M is filtered out."""
    assert _passes_hard_filters(_universe(adv=8_000_000), _signals()) is not None


def test_passes_net_cash():
    """Net cash satisfies balance-sheet filter."""
    row = _universe(net_debt_ebitda=None, interest_coverage=None)
    sigs = _signals(net_cash=1_000_000)
    assert _passes_hard_filters(row, sigs) is None


def test_fails_balance_sheet_all_bad():
    """Fails when no balance-sheet pass criterion is met."""
    row = _universe(net_debt_ebitda=5.0, interest_coverage=1.5)
    sigs = _signals(net_cash=None)
    assert _passes_hard_filters(row, sigs) is not None


def test_score_interim_with_insider_buys():
    """Insider buys contribute positive score."""
    sigs = _signals(insider_buy_value_90d=10_000_000)
    result = _score_interim(sigs)
    assert result["score"] > 0


def test_score_interim_max_100():
    """Score never exceeds 100."""
    sigs = _signals(insider_buy_value_90d=50_000_000, sma_50_rising=1)
    result = _score_interim(sigs)
    assert result["score"] <= 100


def test_score_interim_non_negative():
    """Score is non-negative even with no positive signals."""
    sigs = _signals(insider_buy_value_90d=0, price_vs_50dma=0, sma_50_rising=0)
    result = _score_interim(sigs)
    assert result["score"] >= 0


def test_score_interim_returns_components():
    """Result dict includes 'score', 'score_partial', 'components', 'interim_mode'."""
    result = _score_interim(_signals())
    assert "score" in result
    assert "score_partial" in result
    assert "components" in result
    assert "interim_mode" in result
