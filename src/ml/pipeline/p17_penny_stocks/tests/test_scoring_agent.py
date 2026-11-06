"""Tests for P17 ScoringAgent."""

from typing import Any, Dict
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.agents.scoring_agent import ScoringAgent, _interp
from src.ml.pipeline.p17_penny_stocks.config import P17ScoringConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate


def _make_agent() -> ScoringAgent:
    ss_agent = MagicMock()
    ss_agent.compute_score.return_value = 0.0
    return ScoringAgent(P17ScoringConfig(), ss_agent)


def _make_candidate(**kwargs: Any) -> Candidate:
    defaults: Dict[str, Any] = dict(
        ticker="TEST",
        price=5.0,
        market_cap=100_000_000,
        float_shares=10_000_000,
        relative_volume=1.0,
        price_20d_return=0.0,
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


# ── _interp ────────────────────────────────────────────────────────────────────


def test_interp_at_low_boundary():
    assert _interp(0.0, 0.0, 0.5, 1.0) == 0.0


def test_interp_below_low():
    assert _interp(-1.0, 0.0, 0.5, 1.0) == 0.0


def test_interp_at_high_boundary():
    assert _interp(1.0, 0.0, 0.5, 1.0) == 100.0


def test_interp_above_high():
    assert _interp(2.0, 0.0, 0.5, 1.0) == 100.0


def test_interp_at_midpoint():
    assert _interp(0.5, 0.0, 0.5, 1.0) == 50.0


def test_interp_lower_segment():
    result = _interp(0.25, 0.0, 0.5, 1.0)
    assert abs(result - 25.0) < 1e-9


def test_interp_upper_segment():
    result = _interp(0.75, 0.0, 0.5, 1.0)
    assert abs(result - 75.0) < 1e-9


# ── Momentum score ─────────────────────────────────────────────────────────────


def test_momentum_score_zero_return():
    agent = _make_agent()
    c = _make_candidate(price_20d_return=0.0)
    agent._score(c)
    assert c.momentum_score == 50.0


def test_momentum_score_negative_return():
    agent = _make_agent()
    c = _make_candidate(price_20d_return=-0.20)
    agent._score(c)
    assert c.momentum_score == 0.0


def test_momentum_score_max_return():
    agent = _make_agent()
    c = _make_candidate(price_20d_return=0.50)
    agent._score(c)
    assert c.momentum_score == 100.0


def test_momentum_score_euphoric_spike():
    """Returns >= 300% are scored 0 (euphoric spike — already blown)."""
    agent = _make_agent()
    c = _make_candidate(price_20d_return=3.0)
    agent._score(c)
    assert c.momentum_score == 0.0


# ── Volume score ───────────────────────────────────────────────────────────────


def test_volume_score_low_rvol():
    agent = _make_agent()
    c = _make_candidate(relative_volume=0.5)
    agent._score(c)
    assert c.volume_score == 0.0


def test_volume_score_mid_rvol():
    agent = _make_agent()
    c = _make_candidate(relative_volume=2.0)
    agent._score(c)
    assert c.volume_score == 50.0


def test_volume_score_high_rvol():
    agent = _make_agent()
    c = _make_candidate(relative_volume=5.0)
    agent._score(c)
    assert c.volume_score == 100.0


# ── Technical score ────────────────────────────────────────────────────────────


def test_technical_score_no_signals():
    agent = _make_agent()
    c = _make_candidate()
    agent._score(c)
    assert c.technical_score == 0.0


def test_technical_score_breakout_20d_counts_double():
    agent = _make_agent()
    c = _make_candidate(breakout_20d=True)
    agent._score(c)
    assert c.technical_score == 40.0  # 2 signals × 20 = 40


def test_technical_score_all_signals():
    agent = _make_agent()
    c = _make_candidate(
        breakout_20d=True,
        breakout_50d=True,
        bb_squeeze=True,
        above_sma50=True,
    )
    agent._score(c)
    assert c.technical_score == 100.0  # 5 signals × 20 = 100, capped


# ── Fundamentals score ─────────────────────────────────────────────────────────


def test_fundamentals_score_unknown_revenue():
    """Unknown revenue growth scores neutral (50), not a penalty (see _fundamentals_score)."""
    agent = _make_agent()
    c = _make_candidate(revenue_growth_yoy=None)
    agent._score(c)
    assert c.fundamentals_score == 50.0


def test_fundamentals_score_positive_cashflow_bonus():
    agent = _make_agent()
    c = _make_candidate(
        revenue_growth_yoy=0.25,
        operating_cashflow=1_000_000,
    )
    agent._score(c)
    # At midpoint (0.25) score = 50; +10 bonus for CF = 60
    assert abs(c.fundamentals_score - 60.0) < 1e-6


def test_fundamentals_score_negative_margin_penalty():
    agent = _make_agent()
    c = _make_candidate(
        revenue_growth_yoy=0.25,
        gross_margin=-0.05,
    )
    agent._score(c)
    # At midpoint (0.25) score = 50; -20 penalty = 30
    assert abs(c.fundamentals_score - 30.0) < 1e-6


# ── Dilution penalty ───────────────────────────────────────────────────────────


def test_dilution_penalty_reduces_final_score():
    agent = _make_agent()
    c = _make_candidate(
        relative_volume=5.0,
        price_20d_return=0.50,
        dilution_penalty=20.0,
    )
    agent._score(c)
    assert c.final_score == c.weighted_score - 20.0


def test_final_score_floored_at_zero():
    agent = _make_agent()
    c = _make_candidate(dilution_penalty=999.0)
    agent._score(c)
    assert c.final_score == 0.0


# ── Tier assignment ────────────────────────────────────────────────────────────


def test_tier_a_requires_mandatory_conditions():
    agent = _make_agent()
    ss = MagicMock()
    ss.compute_score.return_value = 100.0
    agent._ss_agent = ss

    c = _make_candidate(
        relative_volume=5.0,
        price_20d_return=0.50,
        above_sma50=True,
        breakout_20d=True,
        revenue_growth_yoy=0.50,
    )
    agent._score(c)
    assert c.tier == "A"


def test_tier_b_when_mandatory_conditions_missing():
    """High score but no breakout → drops to B."""
    agent = _make_agent()
    ss = MagicMock()
    ss.compute_score.return_value = 100.0
    agent._ss_agent = ss

    c = _make_candidate(
        relative_volume=5.0,
        price_20d_return=0.50,
        above_sma50=False,  # mandatory condition missing
        breakout_20d=False,
        revenue_growth_yoy=0.50,
    )
    agent._score(c)
    assert c.tier == "B"


def test_tier_w_low_score():
    agent = _make_agent()
    c = _make_candidate(relative_volume=0.5, price_20d_return=-0.20)
    agent._score(c)
    assert c.tier == "W"


# ── Sorting ────────────────────────────────────────────────────────────────────


def test_run_sorts_candidates_descending():
    agent = _make_agent()
    c1 = _make_candidate(ticker="A", price_20d_return=-0.20)
    c2 = _make_candidate(ticker="B", relative_volume=5.0, price_20d_return=0.50)
    c3 = _make_candidate(ticker="C", price_20d_return=0.10)

    result = agent.run([c1, c2, c3])
    scores = [c.final_score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_run_returns_same_list():
    agent = _make_agent()
    candidates = [_make_candidate(ticker="X")]
    result = agent.run(candidates)
    assert result is candidates


# ── Extension gate ─────────────────────────────────────────────────────────────


def _squeeze_agent() -> ScoringAgent:
    agent = _make_agent()
    ss = MagicMock()
    ss.compute_score.return_value = 100.0
    agent._ss_agent = ss
    return agent


def _tier_a_candidate(**kwargs: Any) -> Candidate:
    base: Dict[str, Any] = dict(
        relative_volume=5.0,
        price_20d_return=0.50,
        above_sma50=True,
        breakout_20d=True,
        revenue_growth_yoy=0.50,
    )
    base.update(kwargs)
    return _make_candidate(**base)


def test_extended_5d_return_demotes_tier_a_to_c():
    agent = _squeeze_agent()
    c = _tier_a_candidate(price_5d_return=0.45)
    agent._score(c)
    assert c.extended
    assert c.tier == "C"
    assert not c.explosive_candidate
    assert any(s.startswith("extended_5d_") for s in c.signals)


def test_extended_atr_stretch_demotes_tier_a_to_c():
    agent = _squeeze_agent()
    c = _tier_a_candidate(price_5d_return=0.10, extension_atr=3.5)
    agent._score(c)
    assert c.extended and c.tier == "C"


def test_not_extended_keeps_tier_a():
    agent = _squeeze_agent()
    c = _tier_a_candidate(price_5d_return=0.10, extension_atr=2.0)
    agent._score(c)
    assert not c.extended and c.tier == "A"


def test_extended_low_score_is_not_promoted():
    agent = _make_agent()
    c = _make_candidate(price_5d_return=0.60)
    agent._score(c)
    assert c.extended and c.tier == "W"


# ── Pre-breakout setup ─────────────────────────────────────────────────────────


def _coiled(**kwargs: Any) -> Candidate:
    base: Dict[str, Any] = dict(
        relative_volume=2.0,
        price_5d_return=0.02,
        price_20d_return=0.05,
        range_10d_pct=0.10,
        dist_to_50d_high=-0.02,
        bb_squeeze=True,
        accumulation_days=7,
        obv_slope=0.2,
    )
    base.update(kwargs)
    return _make_candidate(**base)


def test_setup_flags_tight_coil_near_high():
    agent = _squeeze_agent()
    c = _coiled()
    agent._score(c)
    assert c.setup_score >= 90.0
    assert c.setup_candidate
    assert any(s.startswith("setup_") for s in c.signals)


def test_setup_rejected_when_already_moving():
    agent = _squeeze_agent()
    c = _coiled(price_5d_return=0.20)
    agent._score(c)
    assert not c.setup_candidate


def test_setup_rejected_far_below_high():
    agent = _squeeze_agent()
    c = _coiled(dist_to_50d_high=-0.30)
    agent._score(c)
    assert not c.setup_candidate


def test_setup_rejected_on_heavy_dilution():
    agent = _squeeze_agent()
    c = _coiled(dilution_penalty=40.0)
    agent._score(c)
    assert not c.setup_candidate


def test_setup_requires_ohlcv_features():
    """No OHLCV (range 0) → no evidence of a coil, never a setup."""
    agent = _squeeze_agent()
    c = _coiled(range_10d_pct=0.0)
    agent._score(c)
    assert c.setup_score < 60.0 and not c.setup_candidate


def test_setup_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        P17ScoringConfig(setup_weight_coil=0.5)
