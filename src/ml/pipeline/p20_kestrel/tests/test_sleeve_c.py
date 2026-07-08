"""Tests for P20 Kestrel Sleeve C (Momentum) logic."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.screening.sleeve_c import _compute_rs_score


def test_rs_score_basic():
    """RS = 0.5 × r3m + 0.5 × r6m."""
    sig_map = {"return_3m": 0.20, "return_6m": 0.40}
    rs = _compute_rs_score(sig_map)
    assert rs is not None
    assert abs(rs - 0.30) < 1e-6


def test_rs_score_missing_r3m():
    """Returns None when 3m return is missing."""
    assert _compute_rs_score({"return_6m": 0.30}) is None


def test_rs_score_missing_r6m():
    """Returns None when 6m return is missing."""
    assert _compute_rs_score({"return_3m": 0.10}) is None


def test_rs_score_negative():
    """RS handles negative returns."""
    sig_map = {"return_3m": -0.10, "return_6m": -0.20}
    rs = _compute_rs_score(sig_map)
    assert rs is not None
    assert rs < 0


def test_rs_score_empty_signals():
    """Returns None for empty signal map."""
    assert _compute_rs_score({}) is None


def test_regime_filter_fail_open(monkeypatch):
    """Regime check returns True (fail-open) when no SPY signal is available."""
    import src.ml.pipeline.p20_kestrel.screening.sleeve_c as sleeve_c

    monkeypatch.setattr(sleeve_c, "get_latest_signal", lambda *_: None)
    assert sleeve_c._regime_allows_new_entry() is True


def test_regime_filter_blocks_below_200dma(monkeypatch):
    """Regime check returns False when SPY below 200DMA."""
    import src.ml.pipeline.p20_kestrel.screening.sleeve_c as sleeve_c

    monkeypatch.setattr(sleeve_c, "get_latest_signal", lambda *_: 0.0)
    assert sleeve_c._regime_allows_new_entry() is False


def test_regime_filter_allows_above_200dma(monkeypatch):
    """Regime check returns True when SPY above 200DMA."""
    import src.ml.pipeline.p20_kestrel.screening.sleeve_c as sleeve_c

    monkeypatch.setattr(sleeve_c, "get_latest_signal", lambda *_: 1.0)
    assert sleeve_c._regime_allows_new_entry() is True
