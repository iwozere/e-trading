"""Tests for P20 Kestrel risk checker position alert logic."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.risk.risk_checker import _check_position


def _pos(entry=100.0, stop=75.0, t1=135.0, t2=160.0, realized_thirds=0, **kw):
    return {
        "ticker": kw.get("ticker", "TST"),
        "entry_px": entry,
        "stop_px": stop,
        "t1_px": t1,
        "t2_px": t2,
        "realized_thirds": realized_thirds,
    }


def test_no_alert_in_normal_range():
    """No alert when price is between stop and T1."""
    pos = _pos()
    assert _check_position(pos, 110.0) is None


def test_stop_hit():
    """Alert fires when close <= stop_px."""
    pos = _pos(stop=75.0)
    alert = _check_position(pos, 74.0)
    assert alert is not None
    assert alert["trigger"] == "stop_hit"


def test_stop_hit_exactly_at_stop():
    """Alert fires when close == stop_px (boundary inclusive)."""
    pos = _pos(stop=75.0)
    alert = _check_position(pos, 75.0)
    assert alert is not None
    assert alert["trigger"] == "stop_hit"


def test_t1_target_hit():
    """T1 alert fires when realized_thirds == 0 and close >= t1."""
    pos = _pos(t1=135.0, realized_thirds=0)
    alert = _check_position(pos, 136.0)
    assert alert is not None
    assert alert["trigger"] == "t1_target"


def test_t1_not_hit_when_already_scaled():
    """T1 alert does not fire if first third already realized."""
    pos = _pos(t1=135.0, realized_thirds=1)
    assert _check_position(pos, 136.0) is None


def test_t2_target_hit():
    """T2 alert fires when realized_thirds == 1 and close >= t2."""
    pos = _pos(t2=160.0, realized_thirds=1)
    alert = _check_position(pos, 161.0)
    assert alert is not None
    assert alert["trigger"] == "t2_target"


def test_t2_not_hit_when_first_third_unrealized():
    """T2 alert does not fire if realized_thirds is still 0."""
    pos = _pos(t2=160.0, realized_thirds=0)
    # At T2 price but stop is lower and T1 hasn't fired yet
    assert _check_position(pos, 161.0) is None


def test_intraday_loss_alert():
    """Intraday loss alert fires when close < entry * (1 - 12%)."""
    pos = _pos(entry=100.0, stop=75.0)  # stop at 75, loss alert at 88
    alert = _check_position(pos, 87.0)
    assert alert is not None
    assert alert["trigger"] == "intraday_loss"
    assert alert["pct_loss"] < -0.10


def test_no_alert_just_above_intraday_loss_threshold():
    """No intraday loss alert when loss is just below 12%."""
    pos = _pos(entry=100.0, stop=75.0)
    assert _check_position(pos, 89.0) is None


def test_stop_takes_priority_over_loss():
    """Stop hit takes priority over intraday loss check."""
    pos = _pos(entry=100.0, stop=85.0)
    alert = _check_position(pos, 84.0)  # both stop and intraday-loss triggered
    assert alert is not None
    assert alert["trigger"] == "stop_hit"


def test_no_alert_when_price_is_none():
    """No alert when close price is unavailable."""
    assert _check_position(_pos(), None) is None


def test_no_alert_when_no_stop_set():
    """No stop alert when stop_px is None."""
    pos = {**_pos(), "stop_px": None}
    assert _check_position(pos, 50.0) is None
