"""Tests for Stage3LLMSynthesizer — mocks the Anthropic client."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.ml.pipeline.p05_ai_selector.stages.stage3_llm_synthesizer import Stage3LLMSynthesizer


def _make_stage2_df(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "ticker": f"TICK{i}",
            "asset_type": "equity",
            "last_price": 100.0 + i,
            "market_cap_b": 5.0,
            "total_score": 100.0 - i * 5,
            "momentum_score": 30.0,
            "fundamental_score": 20.0,
            "p18_score": 0.0,
            "volume_surge_ratio": 1.2,
            "earnings_flag": False,
            "earnings_date": "",
            "fundamentals_available": True,
            "signal_breakdown": "{}",
        })
    return pd.DataFrame(rows)


def _make_mock_pick(rank: int = 1, confidence: int = 7) -> dict:
    return {
        "rank": rank,
        "ticker": f"TICK{rank-1}",
        "confidence": confidence,
        "bias": "long",
        "thesis": "A compelling thesis.",
        "risk_factors": ["Macro headwinds"],
        "time_horizon": "3-6 months",
        "exit_strategy": {
            "add_conditions": ["Dip to $90 on volume"],
            "hold_conditions": ["Revenue growth > 10%"],
            "thesis_breakers": ["Revenue misses by >5%", "CEO departure"],
            "profit_targets": [
                {"price_level": 120, "action": "Trim 25%", "note": "lock gains"}
            ],
            "time_horizon_note": "Patience is the edge here.",
        },
    }


def _make_mock_claude_response(picks: list, notification_override: bool = False) -> MagicMock:
    tool_block = MagicMock()
    tool_block.input = {
        "picks": picks,
        "market_context": "Markets are showing resilience.",
        "notification_override": notification_override,
    }
    usage = MagicMock()
    usage.input_tokens = 8000
    usage.output_tokens = 3000
    response = MagicMock()
    response.content = [tool_block]
    response.usage = usage
    return response


@pytest.fixture()
def synthesizer():
    with patch("src.ml.pipeline.p05_ai_selector.stages.stage3_llm_synthesizer.Stage3LLMSynthesizer.__init__",
               lambda self, api_key="", model="claude-sonnet-4-6": setattr(self, "_api_key", "fake") or
               setattr(self, "_model", "claude-sonnet-4-6")):
        s = Stage3LLMSynthesizer.__new__(Stage3LLMSynthesizer)
        s._api_key = "fake-key"
        s._model = "claude-sonnet-4-6"
        return s


class TestBuildDataPackets:
    def test_build_data_packets_structure(self, synthesizer):
        """Each packet has the required top-level keys from spec §7.1."""
        df = _make_stage2_df(3)
        packets = synthesizer._build_data_packets(df)

        assert len(packets) == 3
        required_keys = {"ticker", "asset_type", "price", "technicals", "fundamentals",
                         "institutional_flow", "contextual", "deterministic_score"}
        for packet in packets:
            assert required_keys.issubset(packet.keys())


class TestParseResponse:
    def test_parse_response_valid(self, synthesizer):
        """Valid tool response is parsed correctly."""
        picks = [_make_mock_pick(i + 1) for i in range(5)]
        raw = {"tool_input": {
            "picks": picks,
            "market_context": "Positive backdrop.",
            "notification_override": False,
        }, "tokens_used": 11000}

        result = synthesizer._parse_response(raw)
        assert len(result["picks"]) == 5
        assert result["market_context"] == "Positive backdrop."
        assert result["tokens_used"] == 11000

    def test_parse_response_missing_picks_raises(self, synthesizer):
        """Missing 'picks' key raises ValueError."""
        raw = {"tool_input": {"market_context": "ok"}, "tokens_used": 0}
        with pytest.raises(ValueError, match="missing 'picks'"):
            synthesizer._parse_response(raw)

    def test_notification_override_extracted(self, synthesizer):
        """notification_override=True is passed through."""
        picks = [_make_mock_pick(i + 1) for i in range(5)]
        raw = {"tool_input": {
            "picks": picks,
            "market_context": "",
            "notification_override": True,
        }, "tokens_used": 5000}
        result = synthesizer._parse_response(raw)
        assert result["notification_override"] is True

    def test_high_confidence_sets_override(self, synthesizer):
        """A pick with confidence >= 9 auto-sets notification_override even if LLM returned False."""
        picks = [_make_mock_pick(i + 1, confidence=9 if i == 0 else 7) for i in range(5)]
        raw = {"tool_input": {
            "picks": picks,
            "market_context": "",
            "notification_override": False,
        }, "tokens_used": 5000}
        result = synthesizer._parse_response(raw)
        assert result["notification_override"] is True
