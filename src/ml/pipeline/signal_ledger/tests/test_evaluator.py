"""Tests for forward-return evaluation of single signals."""

from __future__ import annotations

from datetime import date

import pytest

from src.ml.pipeline.signal_ledger.evaluator import FLAG_PRICE_JUMP, STATUS_COMPLETE, STATUS_NO_DATA, STATUS_PENDING, evaluate_signal
from src.ml.pipeline.signal_ledger.tests.helpers import flat, panel, ramp, sessions

N = 30
IDX = sessions(N)  # 2026-01-05 (Mon) ...


def _panels(ticker_open, ticker_close):
    opens = panel(IDX, {"ABC": ticker_open, "VT": flat(N), "SPY": flat(N), "IWM": flat(N)})
    closes = panel(IDX, {"ABC": ticker_close, "VT": flat(N, 101.0), "SPY": flat(N), "IWM": flat(N)})
    return opens, closes


def test_entry_is_next_session_open_and_exit_is_close_h_sessions_later() -> None:
    opens, closes = _panels(flat(N), ramp(N, 100.0, 1.0))
    signal = {"signal_date": IDX[0].date(), "ticker": "ABC"}

    out = evaluate_signal(signal, opens, closes, today=date(2026, 3, 1))

    assert out["entry_date"] == IDX[1].date()
    assert out["entry_price"] == 100.0
    assert out["ret_1d"] == pytest.approx(closes["ABC"].iloc[1] / 100.0 - 1)
    assert out["ret_20d"] == pytest.approx(closes["ABC"].iloc[20] / 100.0 - 1)
    assert out["vt_20d"] == pytest.approx(0.01)
    assert out["status"] == STATUS_COMPLETE


def test_weekend_signal_enters_monday() -> None:
    opens, closes = _panels(flat(N), flat(N))
    out = evaluate_signal({"signal_date": date(2026, 1, 10), "ticker": "ABC"}, opens, closes, today=date(2026, 3, 1))
    assert out["entry_date"] == date(2026, 1, 12)


def test_partial_horizons_stay_pending() -> None:
    opens, closes = _panels(flat(N), flat(N))
    out = evaluate_signal({"signal_date": IDX[N - 8].date(), "ticker": "ABC"}, opens, closes, today=date(2026, 3, 1))
    assert out["status"] == STATUS_PENDING
    assert out["ret_5d"] == 0.0
    assert out["ret_10d"] is None


def test_signal_on_last_session_has_no_entry_yet() -> None:
    opens, closes = _panels(flat(N), flat(N))
    out = evaluate_signal({"signal_date": IDX[-1].date(), "ticker": "ABC"}, opens, closes, today=date(2026, 3, 1))
    assert out["status"] == STATUS_PENDING
    assert out["entry_date"] is None


def test_missing_ticker_becomes_no_data_after_grace_period() -> None:
    opens, closes = _panels(flat(N), flat(N))
    signal = {"signal_date": IDX[0].date(), "ticker": "GONE"}
    assert evaluate_signal(signal, opens, closes, today=IDX[3].date())["status"] == STATUS_PENDING
    assert evaluate_signal(signal, opens, closes, today=date(2026, 3, 1))["status"] == STATUS_NO_DATA


def test_unadjusted_reverse_split_is_flagged() -> None:
    closes_abc = flat(N, 0.1)
    for i in range(6, N):
        closes_abc[i] = 3.0  # 1:30 reverse split Yahoo did not adjust
    opens, closes = _panels(flat(N, 0.1), closes_abc)
    out = evaluate_signal({"signal_date": IDX[0].date(), "ticker": "ABC"}, opens, closes, today=date(2026, 3, 1))
    assert out["data_flag"] == FLAG_PRICE_JUMP


def test_normal_moves_are_not_flagged() -> None:
    opens, closes = _panels(flat(N), ramp(N, 100.0, 3.0))
    out = evaluate_signal({"signal_date": IDX[0].date(), "ticker": "ABC"}, opens, closes, today=date(2026, 3, 1))
    assert out["data_flag"] is None


def test_delisted_ticker_exits_at_last_traded_close() -> None:
    closes_abc = flat(N, 50.0)
    for i in range(4, N):
        closes_abc[i] = float("nan")
    opens, closes = _panels(flat(N), closes_abc)
    out = evaluate_signal({"signal_date": IDX[0].date(), "ticker": "ABC"}, opens, closes, today=date(2026, 3, 1))
    assert out["ret_20d"] == pytest.approx(-0.5)
