"""Tests for strategy_pack signal model."""

from src.strategy_pack.models import PackSignal, make_idempotency_key


def test_idempotency_key_stable():
    k1 = make_idempotency_key("SP-2", "SPY", "2026-04-16T00:00:00", "BUY", "A")
    k2 = make_idempotency_key("SP-2", "SPY", "2026-04-16T00:00:00", "BUY", "A")
    assert k1 == k2
    assert len(k1) == 40


def test_pack_signal_fills_idempotency():
    s = PackSignal(
        strategy_id="SP-2",
        variant="A",
        symbol="SPY",
        signal="STATUS",
        bar_timeframe="1d",
        bar_close_ts="2026-04-16T00:00:00",
        price=100.0,
        reason_code="test",
    )
    assert s.idempotency_key
    d = s.to_jsonl_dict()
    assert d["strategy_id"] == "SP-2"
    assert "idempotency_key" in d
