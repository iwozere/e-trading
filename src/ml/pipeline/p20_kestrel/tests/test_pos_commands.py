"""Tests for P20 Kestrel /pos command parser."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.pos.pos_commands import (
    PosCommandError,
    _parse_add,
    echo_card,
)


def test_parse_add_basic():
    """/pos add AAPL A 100 2 parses correctly with defaults."""
    pos = _parse_add("/pos add AAPL A 100.00 2.0")
    assert pos["ticker"] == "AAPL"
    assert pos["sleeve"] == "A"
    assert pos["entry_px"] == 100.0
    assert pos["size_pct"] == 2.0
    assert pos["stop_px"] == pytest.approx(75.0, rel=1e-3)
    assert pos["t1_px"] == pytest.approx(135.0, rel=1e-3)
    assert pos["t2_px"] == pytest.approx(160.0, rel=1e-3)
    assert pos["trail_pct"] == 20.0


def test_parse_add_with_overrides():
    """/pos add with explicit stop/t1/t2 overrides."""
    pos = _parse_add("/pos add TSLA B 200 1.5 stop=150 t1=280 t2=350")
    assert pos["stop_px"] == 150.0
    assert pos["t1_px"] == 280.0
    assert pos["t2_px"] == 350.0


def test_parse_add_lowercase_ticker():
    """Ticker is uppercased from lowercase input."""
    pos = _parse_add("/pos add aapl a 50 1")
    assert pos["ticker"] == "AAPL"
    assert pos["sleeve"] == "A"


def test_parse_add_invalid_sleeve():
    """Sleeve D is not valid — should raise PosCommandError."""
    with pytest.raises(PosCommandError):
        _parse_add("/pos add AAPL D 100 2")


def test_parse_add_missing_fields():
    """Missing required fields raises PosCommandError."""
    with pytest.raises(PosCommandError):
        _parse_add("/pos add AAPL A")


def test_echo_card_contains_ticker():
    """Echo card includes ticker and key fields."""
    pos = _parse_add("/pos add NVDA C 500 1")
    card = echo_card(pos)
    assert "NVDA" in card
    assert "Entry" in card
    assert "Stop" in card
    assert "T1" in card


def test_parse_add_sleeve_c():
    """Sleeve C is valid."""
    pos = _parse_add("/pos add SPY C 450 3")
    assert pos["sleeve"] == "C"
