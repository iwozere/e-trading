"""Tests for P19 intraday metrics (pure)."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.config import P19TriggerConfig
from src.ml.pipeline.p19_penny_intraday.metrics import (
    classify_momentum_tier,
    compute_same_day_labels,
    compute_signal,
    session_fraction,
)
from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal
from src.ml.pipeline.p19_penny_intraday.models.watchlist_entry import WatchlistEntry

_ET = ZoneInfo("America/New_York")


def _utc(hh, mm):
    """UTC instant for a given ET wall-clock on a summer weekday."""
    return datetime(2026, 6, 24, hh, mm, tzinfo=_ET).astimezone(UTC)


def test_session_fraction_bounds():
    assert session_fraction(_utc(9, 30)) == 0.05  # at open → floor
    assert abs(session_fraction(_utc(12, 45)) - 0.5) < 1e-6  # mid session
    assert session_fraction(_utc(16, 0)) == 1.0  # close
    assert session_fraction(_utc(20, 0)) == 1.0  # after hours


def test_compute_signal_pct_and_rvol():
    e = WatchlistEntry(ticker="AAA", source="p17", tier="B", avg_volume_30d=1_000_000, prior_close=2.0)
    q = {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}
    s = compute_signal(e, q, _utc(16, 0), lot_size=100)  # fraction 1.0
    assert abs(s.pct_from_open - 0.2) < 1e-9  # 3/2.5 - 1
    assert abs(s.pct_from_prev_close - 0.5) < 1e-9  # 3/2 - 1
    # 3000 lots × 100 = 300k shares; expected = 1e6 × 1.0 → rvol 0.3
    assert abs(s.rvol_so_far - 0.3) < 1e-3
    assert s.dollar_volume_so_far == 3.0 * 300000
    assert s.day_volume == 300000 and s.source == "p17"


def test_compute_signal_no_baseline_volume_safe():
    e = WatchlistEntry(ticker="GAP", source="gapper")  # avg_volume_30d defaults 0
    q = {"last": 1.0, "open": 0.9, "high": 1.1, "low": 0.8, "prev_close": 0.85, "volume": 500}
    s = compute_signal(e, q, _utc(11, 0), lot_size=100)
    assert s.rvol_so_far == 0.0  # no baseline → no div-by-zero
    assert abs(s.pct_from_prev_close - (1.0 / 0.85 - 1)) < 1e-9


# ── classify_momentum_tier (v2, log-only) ────────────────────────────────────


def _sig(**kw) -> IntradaySignal:
    return IntradaySignal(ticker="AAA", ts=_utc(11, 0), **kw)


def test_no_trigger_is_t0():
    cfg = P19TriggerConfig()
    score, tier = classify_momentum_tier(_sig(rvol_so_far=0.5, pct_from_open=0.01, dollar_volume_so_far=10_000), cfg)
    assert tier == "T0"


def test_gate_requires_both_volume_and_price_by_default():
    cfg = P19TriggerConfig(rvol_trigger=5.0, move_trigger_pct=0.20, dollar_volume_floor=50_000)
    # Price thrust alone, no volume -> gate fails.
    price_only = _sig(rvol_so_far=0.5, pct_from_open=0.30, dollar_volume_so_far=100_000)
    assert classify_momentum_tier(price_only, cfg)[1] == "T0"
    # Volume alone, no price thrust, no catalyst -> gate fails.
    vol_only = _sig(rvol_so_far=10.0, pct_from_open=0.01, dollar_volume_so_far=100_000)
    assert classify_momentum_tier(vol_only, cfg)[1] == "T0"
    # Both -> fires.
    both = _sig(rvol_so_far=10.0, pct_from_open=0.30, dollar_volume_so_far=100_000)
    assert classify_momentum_tier(both, cfg)[1] != "T0"


def test_fresh_catalyst_satisfies_gate_without_price_thrust():
    cfg = P19TriggerConfig(rvol_trigger=5.0, move_trigger_pct=0.20, dollar_volume_floor=50_000)
    s = _sig(rvol_so_far=10.0, pct_from_open=0.01, dollar_volume_so_far=100_000, fresh_catalyst=True)
    assert classify_momentum_tier(s, cfg)[1] != "T0"


def test_explosive_move_reaches_t3():
    cfg = P19TriggerConfig(rvol_trigger=5.0, move_trigger_pct=0.20, dollar_volume_floor=50_000)
    s = _sig(rvol_so_far=25.0, pct_from_open=1.0, dollar_volume_so_far=1_000_000)  # 5x both thresholds
    score, tier = classify_momentum_tier(s, cfg)
    assert tier == "T3"
    assert score == 100.0


def test_elevated_move_reaches_t1_not_t3():
    cfg = P19TriggerConfig(rvol_trigger=5.0, move_trigger_pct=0.20, dollar_volume_floor=50_000)
    s = _sig(rvol_so_far=5.5, pct_from_open=0.21, dollar_volume_so_far=100_000)  # just past the gate
    score, tier = classify_momentum_tier(s, cfg)
    assert tier == "T1"
    assert 0.0 < score < 40.0


def test_loose_gate_when_require_volume_and_price_false():
    cfg = P19TriggerConfig(rvol_trigger=5.0, move_trigger_pct=0.20, dollar_volume_floor=50_000, require_volume_and_price=False)
    price_only = _sig(rvol_so_far=0.5, pct_from_open=0.30, dollar_volume_so_far=100_000)
    assert classify_momentum_tier(price_only, cfg)[1] != "T0"


# ── compute_same_day_labels ───────────────────────────────────────────────


def _poll(ts, price, day_high, day_low, tier="T0"):
    return {"ts": ts, "price": price, "day_high": day_high, "day_low": day_low, "momentum_tier": tier}


def test_no_polls_returns_all_none():
    labels = compute_same_day_labels([], {"open": 1.0, "high": 1.5, "low": 0.9, "close": 1.2})
    assert labels == {"high_time": None, "close_retention": None, "mae_from_alert": None, "mfe_from_alert": None}


def test_close_retention_computed_from_eod_alone():
    polls = [_poll("2026-06-24T14:30:00+00:00", 2.5, 2.5, 2.5)]
    eod = {"open": 2.0, "high": 4.0, "low": 1.9, "close": 3.5}  # retained 75% of the move
    labels = compute_same_day_labels(polls, eod)
    assert abs(labels["close_retention"] - 0.75) < 1e-9


def test_flat_day_no_divide_by_zero():
    polls = [_poll("2026-06-24T14:30:00+00:00", 1.0, 1.0, 1.0)]
    eod = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
    labels = compute_same_day_labels(polls, eod)
    assert labels["close_retention"] is None


def test_high_time_matches_first_poll_reaching_eod_high():
    polls = [
        _poll("2026-06-24T14:00:00+00:00", 2.0, 2.0, 2.0),
        _poll("2026-06-24T14:30:00+00:00", 5.0, 5.0, 2.0),  # peak reached here
        _poll("2026-06-24T15:00:00+00:00", 3.0, 5.0, 2.0),  # still shows the peak, faded since
    ]
    eod = {"open": 2.0, "high": 5.0, "low": 2.0, "close": 3.0}
    labels = compute_same_day_labels(polls, eod)
    assert labels["high_time"] == "10:30"  # 14:30 UTC -> 10:30 ET (summer)


def test_mae_mfe_from_alert_price_no_trigger_crossed():
    polls = [_poll("2026-06-24T14:30:00+00:00", 2.05, 2.1, 2.0, tier="T0")]  # never crosses T1
    eod = {"open": 2.0, "high": 2.1, "low": 2.0, "close": 2.05}
    labels = compute_same_day_labels(polls, eod)
    assert labels["mae_from_alert"] is None and labels["mfe_from_alert"] is None


def test_mae_mfe_computed_from_alert_price_onward():
    polls = [
        _poll("2026-06-24T14:00:00+00:00", 2.0, 2.0, 2.0, tier="T0"),  # pre-trigger, excluded from MAE/MFE
        _poll("2026-06-24T14:30:00+00:00", 5.0, 5.0, 3.2, tier="T3"),  # alert price = 5.0
        _poll("2026-06-24T15:00:00+00:00", 3.0, 5.0, 2.8, tier="T3"),
    ]
    eod = {"open": 2.0, "high": 5.0, "low": 2.5, "close": 3.0}
    labels = compute_same_day_labels(polls, eod)
    assert abs(labels["mfe_from_alert"] - (5.0 / 5.0 - 1.0)) < 1e-9  # no upside past the alert price
    # min low across [alert poll onward] + eod low = min(3.2, 2.8, 2.5) = 2.5 -> -50% from the 5.0 alert price
    assert abs(labels["mae_from_alert"] - (2.5 / 5.0 - 1.0)) < 1e-9
