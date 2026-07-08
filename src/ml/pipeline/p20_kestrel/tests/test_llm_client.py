"""Tests for P20 Kestrel LLM client — cost calculation logic."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.llm.client import _PRICING, _compute_cost


def test_compute_cost_haiku():
    """Cost computation for haiku model."""
    cost = _compute_cost("claude-haiku-4-5-20251001", tokens_in=1000, tokens_out=500)
    assert cost > 0
    pricing = _PRICING["claude-haiku-4-5-20251001"]
    expected = 1000 * pricing["in"] + 500 * pricing["out"]
    assert abs(cost - expected) < 1e-9


def test_compute_cost_sonnet():
    """Cost computation for sonnet model."""
    cost = _compute_cost("claude-sonnet-4-6", tokens_in=2000, tokens_out=1000)
    assert cost > 0


def test_compute_cost_zero_tokens():
    """Zero-token call returns zero cost."""
    cost = _compute_cost("claude-haiku-4-5-20251001", tokens_in=0, tokens_out=0)
    assert cost == 0.0


def test_compute_cost_unknown_model():
    """Unknown model falls back to sonnet pricing."""
    cost_known = _compute_cost("claude-sonnet-4-6", tokens_in=100, tokens_out=100)
    cost_unknown = _compute_cost("claude-unknown-9000", tokens_in=100, tokens_out=100)
    assert cost_unknown == cost_known


def test_pricing_dict_has_required_keys():
    """Pricing dict has entries for haiku and sonnet."""
    assert "claude-haiku-4-5-20251001" in _PRICING
    assert "claude-sonnet-4-6" in _PRICING
    for prices in _PRICING.values():
        assert "in" in prices
        assert "out" in prices
        assert prices["in"] > 0
        assert prices["out"] > 0
