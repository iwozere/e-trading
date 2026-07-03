"""Tests for P20 Kestrel Sleeve B event catalyst screens (B1, B2, B3)."""

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.screening.sleeve_b import (
    screen_b1,
    screen_b2,
    screen_b3_activist,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _catalyst(ticker="DRUG", event_type="pdufa", days_out=30):
    event_date = date.today() + timedelta(days=days_out)
    return {
        "ticker": ticker,
        "event_type": event_type,
        "event_date": event_date,
        "state": "upcoming",
    }


def _universe(ticker="DRUG", mcap=500_000_000):
    return {"ticker": ticker, "mcap": mcap}


def _spinoff(ticker="SPINCO", days_ago=35):
    return {
        "ticker": ticker,
        "event_type": "spinoff",
        "event_date": date.today() - timedelta(days=days_ago),
    }


# ---------------------------------------------------------------------------
# B1: FDA run-up
# ---------------------------------------------------------------------------

def test_b1_returns_fda_candidate():
    """Standard PDUFA 30 days out with valid mcap is a B1 candidate."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_catalysts_in_window",
              return_value=[_catalyst()]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe()),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_latest_signal",
              return_value=None),
    ):
        result = screen_b1(date.today())

    assert len(result) == 1
    assert result[0]["sub_sleeve"] == "B1"
    assert result[0]["ticker"] == "DRUG"


def test_b1_excludes_non_fda_event_type():
    """Non-FDA event types (e.g. earnings) are excluded from B1."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_catalysts_in_window",
              return_value=[_catalyst(event_type="earnings")]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe()),
    ):
        result = screen_b1(date.today())

    assert result == []


def test_b1_excludes_mcap_too_large():
    """Mcap above $10B cap is excluded."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_catalysts_in_window",
              return_value=[_catalyst()]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe(mcap=15_000_000_000)),
    ):
        result = screen_b1(date.today())

    assert result == []


def test_b1_excludes_mcap_too_small():
    """Mcap below $300M floor is excluded."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_catalysts_in_window",
              return_value=[_catalyst()]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe(mcap=100_000_000)),
    ):
        result = screen_b1(date.today())

    assert result == []


def test_b1_crowding_spike_skips_near_event():
    """Crowding z-score > 3 causes skip anywhere in the entry window."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_catalysts_in_window",
              return_value=[_catalyst(days_out=15)]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe()),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_latest_signal",
              return_value=3.5),
    ):
        result = screen_b1(date.today())

    assert result == []


def test_b1_no_crowding_data_does_not_skip():
    """Missing crowding signal (None) does not cause skip."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_catalysts_in_window",
              return_value=[_catalyst(days_out=15)]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe()),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_latest_signal",
              return_value=None),
    ):
        result = screen_b1(date.today())

    assert len(result) == 1


def test_b1_excludes_inside_t10():
    """Candidates closer than T-10 are outside the entry window."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_catalysts_in_window",
              return_value=[_catalyst(days_out=5)]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe()),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_latest_signal",
              return_value=None),
    ):
        result = screen_b1(date.today())

    assert result == []


# ---------------------------------------------------------------------------
# B2: Spin-offs
# ---------------------------------------------------------------------------

def test_b2_returns_spinoff_in_window():
    """Spin-off 35 days ago with valid mcap is a B2 candidate."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_past_spinoffs",
              return_value=[_spinoff()]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe(ticker="SPINCO", mcap=300_000_000)),
    ):
        result = screen_b2(date.today())

    assert len(result) == 1
    assert result[0]["sub_sleeve"] == "B2"
    assert result[0]["ticker"] == "SPINCO"
    assert result[0]["days_since_spin"] == 35


def test_b2_excludes_micro_cap():
    """Spin-off below $150M mcap floor is excluded."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_past_spinoffs",
              return_value=[_spinoff()]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=_universe(ticker="SPINCO", mcap=100_000_000)),
    ):
        result = screen_b2(date.today())

    assert result == []


def test_b2_excludes_unknown_ticker():
    """Spin-off ticker not in k20_universe is excluded."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_past_spinoffs",
              return_value=[_spinoff()]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_universe_row",
              return_value=None),
    ):
        result = screen_b2(date.today())

    assert result == []


def test_b2_empty_when_no_spinoffs():
    """No spin-offs in window produces empty result."""
    with patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_past_spinoffs",
               return_value=[]):
        result = screen_b2(date.today())

    assert result == []


# ---------------------------------------------------------------------------
# B3: Activist
# ---------------------------------------------------------------------------

def test_b3_returns_activist_ticker():
    """Ticker with activist_13d signal > 0 is a B3 candidate."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_active_tickers",
              return_value=["TRGT"]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_latest_signal",
              return_value=1.0),
    ):
        result = screen_b3_activist(date.today())

    assert len(result) == 1
    assert result[0]["sub_sleeve"] == "B3"
    assert result[0]["ticker"] == "TRGT"


def test_b3_excludes_zero_signal():
    """Activist signal of 0 is not a B3 candidate."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_active_tickers",
              return_value=["TRGT"]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_latest_signal",
              return_value=0.0),
    ):
        result = screen_b3_activist(date.today())

    assert result == []


def test_b3_excludes_none_signal():
    """Missing activist signal produces no B3 candidates."""
    with (
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_active_tickers",
              return_value=["TRGT"]),
        patch("src.ml.pipeline.p20_kestrel.screening.sleeve_b.get_latest_signal",
              return_value=None),
    ):
        result = screen_b3_activist(date.today())

    assert result == []
