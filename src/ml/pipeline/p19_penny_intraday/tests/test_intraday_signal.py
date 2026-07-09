"""Tests for the P19 IntradaySignal model."""

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict

from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal


def _sig(**kw: Any) -> IntradaySignal:
    base: Dict[str, Any] = dict(ticker="ILLR", ts=datetime(2026, 6, 24, 14, 30, tzinfo=UTC), price=4.46)
    base.update(kw)
    return IntradaySignal(**base)


def test_triggered_flag():
    assert _sig().triggered is False
    assert _sig(trigger_reason="price_thrust").triggered is True


def test_to_dict_flattens_lists_and_sentiment():
    s = _sig(
        catalyst_signals=["catalyst_tier1_news_2026-06-24"],
        sentiment={"mentions": 42.0, "finbert": 0.8},
        trigger_reason="price_thrust+volume",
    )
    d = s.to_dict()
    assert d["ticker"] == "ILLR"
    assert d["catalyst_signals"] == "catalyst_tier1_news_2026-06-24"
    assert d["sentiment"] == "mentions=42.0;finbert=0.8"
    assert d["trigger_reason"] == "price_thrust+volume"
    assert d["ts"].startswith("2026-06-24T14:30")


def test_defaults_safe_for_shadow_row():
    d = _sig().to_dict()
    # un-triggered shadow row is still fully serialisable
    assert d["catalyst_signals"] == "" and d["sentiment"] == ""
    assert d["eod_close"] is None and d["volume_is_delayed"] is True
