"""Integration test: /pos add → confirm_add → risk_checker roundtrip.

Mocks DB calls to verify that the full command parse → position insert →
watchlist update → risk check flow works end-to-end without a live database.
"""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.pos.pos_commands import (
    PosCommandError,
    _parse_add,
    confirm_add,
    echo_card,
    handle_command,
)
from src.ml.pipeline.p20_kestrel.risk.risk_checker import _check_position

# ---------------------------------------------------------------------------
# /pos add parse
# ---------------------------------------------------------------------------


def test_parse_add_defaults():
    """/pos add with no optional args applies default stop/T1/T2/trail."""
    pos = _parse_add("/pos add AAPL A 150 2.0")
    assert pos["ticker"] == "AAPL"
    assert pos["sleeve"] == "A"
    assert pos["entry_px"] == 150.0
    assert pos["size_pct"] == 2.0
    assert pos["stop_px"] == round(150.0 * 0.75, 4)
    assert pos["t1_px"] == round(150.0 * 1.35, 4)
    assert pos["t2_px"] == round(150.0 * 1.60, 4)
    assert pos["trail_pct"] == 20.0
    assert pos["status"] == "pending"


def test_parse_add_explicit_overrides():
    """/pos add with explicit stop/t1/t2 uses provided values."""
    pos = _parse_add("/pos add NVDA B 500 1.5 stop=440 t1=600 t2=700 trail=15")
    assert pos["stop_px"] == 440.0
    assert pos["t1_px"] == 600.0
    assert pos["t2_px"] == 700.0
    assert pos["trail_pct"] == 15.0


def test_parse_add_invalid_syntax_raises():
    """Missing required args raises PosCommandError."""
    try:
        _parse_add("/pos add AAPL")
        assert False, "Should have raised"
    except PosCommandError:
        pass


# ---------------------------------------------------------------------------
# echo_card
# ---------------------------------------------------------------------------


def test_echo_card_contains_ticker_and_prices():
    """Echo card string includes ticker, entry, stop, and T1/T2."""
    pos = _parse_add("/pos add MSFT A 300 2.0")
    card = echo_card(pos)
    assert "MSFT" in card
    assert "300" in card
    assert "225" in card  # stop at 75% = 225


# ---------------------------------------------------------------------------
# confirm_add
# ---------------------------------------------------------------------------


def test_confirm_add_inserts_position_and_updates_watchlist():
    """confirm_add() calls insert_position and upsert_watchlist with correct args."""
    pending = _parse_add("/pos add TSLA C 200 1.0")

    with (
        patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.insert_position", return_value=42) as mock_insert,
        patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.upsert_watchlist") as mock_watchlist,
    ):
        confirmed = confirm_add(pending)

    assert confirmed["id"] == 42
    assert confirmed["status"] == "open"
    mock_insert.assert_called_once()
    mock_watchlist.assert_called_once_with(
        {
            "ticker": "TSLA",
            "sleeve": "C",
            "state": "active_position",
        }
    )


# ---------------------------------------------------------------------------
# Full roundtrip: parse → confirm → risk_check
# ---------------------------------------------------------------------------


def test_pos_add_then_risk_check_stop_hit():
    """After confirm_add, risk_check correctly fires a stop_hit alert."""
    pending = _parse_add("/pos add AAPL A 100 2.0")  # stop at 75.0

    with (
        patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.insert_position", return_value=1),
        patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.upsert_watchlist"),
    ):
        confirmed = confirm_add(pending)

    # Simulate price falling to stop
    alert = _check_position(confirmed, 74.0)
    assert alert is not None
    assert alert["trigger"] == "stop_hit"
    assert alert["ticker"] == "AAPL"


def test_pos_add_then_risk_check_t1_hit():
    """After confirm_add, risk_check fires t1_target when price hits T1."""
    pending = _parse_add("/pos add GOOG A 100 2.0")  # t1 at 135.0

    with (
        patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.insert_position", return_value=2),
        patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.upsert_watchlist"),
    ):
        confirmed = confirm_add(pending)

    alert = _check_position(confirmed, 136.0)
    assert alert is not None
    assert alert["trigger"] == "t1_target"


def test_pos_add_then_risk_check_no_alert_in_range():
    """After confirm_add, no alert fires when price is in the normal range."""
    pending = _parse_add("/pos add META A 100 2.0")  # stop=75, t1=135

    with (
        patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.insert_position", return_value=3),
        patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.upsert_watchlist"),
    ):
        confirmed = confirm_add(pending)

    assert _check_position(confirmed, 110.0) is None


# ---------------------------------------------------------------------------
# handle_command dispatch
# ---------------------------------------------------------------------------


def test_handle_command_list_empty():
    """'/pos list' returns 'No open positions.' when DB is empty."""
    with patch("src.ml.pipeline.p20_kestrel.pos.pos_commands.get_open_positions", return_value=[]):
        reply, data = handle_command("/pos list")

    assert "No open positions" in reply
    assert data is None


def test_handle_command_add_returns_pending():
    """'/pos add' returns echo card text and pending position dict."""
    reply, pending = handle_command("/pos add AAPL A 100 2.0")
    assert pending is not None
    assert pending["ticker"] == "AAPL"
    assert "AAPL" in reply


def test_handle_command_unknown_raises():
    """Unknown /pos sub-command raises PosCommandError."""
    try:
        handle_command("/pos unknown")
        assert False, "Should have raised"
    except PosCommandError:
        pass
