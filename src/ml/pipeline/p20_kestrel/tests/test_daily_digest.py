"""Tests for P20 Kestrel daily digest builder."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.reporting.daily_digest import build_digest


def test_build_digest_returns_string(monkeypatch):
    """build_digest returns a non-empty string."""
    import src.ml.pipeline.p20_kestrel.reporting.daily_digest as dd

    monkeypatch.setattr(dd, "get_latest_signal", lambda *_: None)
    monkeypatch.setattr(dd, "get_open_positions", lambda: [])
    monkeypatch.setattr(dd, "get_catalysts_in_window", lambda **_: [])
    monkeypatch.setattr(dd, "get_watchlist", lambda **_: [])

    result = build_digest(date(2026, 7, 2))
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_digest_contains_date(monkeypatch):
    """Digest header contains the date."""
    import src.ml.pipeline.p20_kestrel.reporting.daily_digest as dd

    monkeypatch.setattr(dd, "get_latest_signal", lambda *_: None)
    monkeypatch.setattr(dd, "get_open_positions", lambda: [])
    monkeypatch.setattr(dd, "get_catalysts_in_window", lambda **_: [])
    monkeypatch.setattr(dd, "get_watchlist", lambda **_: [])

    result = build_digest(date(2026, 7, 2))
    assert "2026-07-02" in result


def test_build_digest_sections(monkeypatch):
    """Digest contains all expected sections."""
    import src.ml.pipeline.p20_kestrel.reporting.daily_digest as dd

    monkeypatch.setattr(dd, "get_latest_signal", lambda *_: None)
    monkeypatch.setattr(dd, "get_open_positions", lambda: [])
    monkeypatch.setattr(dd, "get_catalysts_in_window", lambda **_: [])
    monkeypatch.setattr(dd, "get_watchlist", lambda **_: [])

    result = build_digest(date(2026, 7, 2))
    assert "Regime" in result
    assert "Open Positions" in result
    assert "Catalysts" in result
    assert "Candidates" in result


def test_build_digest_with_open_position(monkeypatch):
    """Digest shows position info when positions exist."""
    import src.ml.pipeline.p20_kestrel.reporting.daily_digest as dd

    monkeypatch.setattr(dd, "get_latest_signal", lambda *_: 55.0)
    monkeypatch.setattr(
        dd,
        "get_open_positions",
        lambda: [
            {
                "ticker": "AAPL",
                "sleeve": "A",
                "entry_px": 50.0,
                "stop_px": 40.0,
                "t1_px": 65.0,
            }
        ],
    )
    monkeypatch.setattr(dd, "get_catalysts_in_window", lambda **_: [])
    monkeypatch.setattr(dd, "get_watchlist", lambda **_: [])

    result = build_digest(date(2026, 7, 2))
    assert "AAPL" in result


def test_build_digest_warns_on_inactive_sleeve_b1_b2(monkeypatch):
    """
    Data Health must warn while PDUFA_CALENDAR_AVAILABLE / SPINOFF_MONITOR_AVAILABLE
    are False (their default) -- otherwise B1/B2 silently produce zero candidates
    forever with no signal anywhere that they're structurally inactive, same as
    happened in production from at least 2026-08-10 through 2026-08-26.
    """
    import src.ml.pipeline.p20_kestrel.reporting.daily_digest as dd

    monkeypatch.setattr(dd, "get_latest_signal", lambda *_: None)
    monkeypatch.setattr(dd, "get_open_positions", lambda: [])
    monkeypatch.setattr(dd, "get_catalysts_in_window", lambda **_: [])
    monkeypatch.setattr(dd, "get_watchlist", lambda **_: [])
    monkeypatch.setattr(dd, "PDUFA_CALENDAR_AVAILABLE", False)
    monkeypatch.setattr(dd, "SPINOFF_MONITOR_AVAILABLE", False)

    result = build_digest(date(2026, 7, 2))
    assert "PDUFA calendar not sourced" in result
    assert "spin-off monitor not built" in result
    assert "gap 10.2" in result


def test_build_digest_silent_when_sleeve_b1_b2_available(monkeypatch):
    """Once both flags flip True, the corresponding warnings must disappear."""
    import src.ml.pipeline.p20_kestrel.reporting.daily_digest as dd

    monkeypatch.setattr(dd, "get_latest_signal", lambda *_: None)
    monkeypatch.setattr(dd, "get_open_positions", lambda: [])
    monkeypatch.setattr(dd, "get_catalysts_in_window", lambda **_: [])
    monkeypatch.setattr(dd, "get_watchlist", lambda **_: [])
    monkeypatch.setattr(dd, "PDUFA_CALENDAR_AVAILABLE", True)
    monkeypatch.setattr(dd, "SPINOFF_MONITOR_AVAILABLE", True)

    result = build_digest(date(2026, 7, 2))
    assert "PDUFA calendar not sourced" not in result
    assert "spin-off monitor not built" not in result
