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
    assert out["status"] == STATUS_PENDING  # 60/120/250-session horizons not known yet
    assert out["ret_60d"] is None


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


LONG = 260
LONG_IDX = sessions(LONG)


def _long_panels(ticker_close, highs=None, lows=None):
    opens = panel(LONG_IDX, {"ABC": flat(LONG), "VT": flat(LONG), "SPY": flat(LONG), "IWM": flat(LONG)})
    closes = panel(LONG_IDX, {"ABC": ticker_close, "VT": flat(LONG), "SPY": flat(LONG), "IWM": flat(LONG)})
    high_frame = panel(LONG_IDX, {"ABC": highs}) if highs is not None else None
    low_frame = panel(LONG_IDX, {"ABC": lows}) if lows is not None else None
    return opens, closes, high_frame, low_frame


def test_signal_completes_once_the_250_session_horizon_is_known() -> None:
    opens, closes, _, _ = _long_panels(ramp(LONG, 100.0, 0.1))
    out = evaluate_signal({"signal_date": LONG_IDX[0].date(), "ticker": "ABC"}, opens, closes, today=date(2027, 6, 1))
    assert out["status"] == STATUS_COMPLETE
    assert out["ret_250d"] == pytest.approx(closes["ABC"].iloc[250] / 100.0 - 1)
    assert out["vt_250d"] == pytest.approx(0.0)


def test_late_price_jump_voids_only_the_long_horizons() -> None:
    closes_abc = flat(LONG)
    for i in range(80, LONG):
        closes_abc[i] = 1_000.0
    opens, closes, _, _ = _long_panels(closes_abc)
    out = evaluate_signal({"signal_date": LONG_IDX[0].date(), "ticker": "ABC"}, opens, closes, today=date(2027, 6, 1))
    assert out["data_flag"] is None
    assert out["ret_60d"] == pytest.approx(0.0)
    assert out["ret_120d"] is None and out["ret_250d"] is None


def test_mae_mfe_use_highs_and_lows_with_open_close_fallback() -> None:
    highs = [float("nan")] * LONG
    lows = [float("nan")] * LONG
    highs[5], lows[7] = 130.0, 85.0
    closes_abc = flat(LONG)
    closes_abc[10] = 80.0  # lows[10] is missing, so the open/close range supplies this session's low
    opens, closes, high_frame, low_frame = _long_panels(closes_abc, highs, lows)
    out = evaluate_signal({"signal_date": LONG_IDX[0].date(), "ticker": "ABC"}, opens, closes, today=date(2027, 6, 1),
                          highs=high_frame, lows=low_frame)
    assert out["mfe_20d"] == pytest.approx(0.30)
    assert out["mae_20d"] == pytest.approx(-0.20)


def test_mae_mfe_without_high_low_panels() -> None:
    closes_abc = flat(LONG)
    closes_abc[3] = 110.0
    opens, closes, _, _ = _long_panels(closes_abc)
    out = evaluate_signal({"signal_date": LONG_IDX[0].date(), "ticker": "ABC"}, opens, closes, today=date(2027, 6, 1))
    assert out["mfe_20d"] == pytest.approx(0.10)
    assert out["mae_20d"] == pytest.approx(0.0)


def test_ffill_carries_last_value_and_keeps_leading_gaps() -> None:
    import numpy as np

    from src.ml.pipeline.signal_ledger.evaluator import _ffill

    nan = float("nan")
    out = _ffill(np.array([nan, 1.0, nan, nan, 3.0, nan]))
    assert np.isnan(out[0])
    assert list(out[1:]) == [1.0, 1.0, 1.0, 3.0, 3.0]
