"""Tests for P19 intraday metrics (pure)."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.metrics import compute_signal, session_fraction
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
