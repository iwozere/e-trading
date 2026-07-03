"""Tests for P20 Kestrel Sleeve C (Momentum) logic."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.screening.sleeve_c import _compute_rs_score


def test_rs_score_basic():
    """RS = 0.5 × r3m + 0.5 × r6m."""
    signals = [
        {"signal_type": "return_3m", "value": 0.20},
        {"signal_type": "return_6m", "value": 0.40},
    ]
    rs = _compute_rs_score(signals)
    assert rs is not None
    assert abs(rs - 0.30) < 1e-6


def test_rs_score_missing_r3m():
    """Returns None when 3m return is missing."""
    assert _compute_rs_score([{"signal_type": "return_6m", "value": 0.30}]) is None


def test_rs_score_missing_r6m():
    """Returns None when 6m return is missing."""
    assert _compute_rs_score([{"signal_type": "return_3m", "value": 0.10}]) is None


def test_rs_score_negative():
    """RS handles negative returns."""
    signals = [
        {"signal_type": "return_3m", "value": -0.10},
        {"signal_type": "return_6m", "value": -0.20},
    ]
    rs = _compute_rs_score(signals)
    assert rs is not None
    assert rs < 0


def test_rs_score_empty_signals():
    """Returns None for empty signal list."""
    assert _compute_rs_score([]) is None


def test_regime_filter_fail_open(monkeypatch):
    """Regime check returns True (fail-open) when no SPY signal is available."""
    import src.ml.pipeline.p20_kestrel.screening.sleeve_c as sleeve_c
    monkeypatch.setattr(sleeve_c, "get_latest_signal", lambda *_: None)
    assert sleeve_c._regime_allows_new_entry() is True


def test_regime_filter_blocks_below_200dma(monkeypatch):
    """Regime check returns False when SPY below 200DMA."""
    import src.ml.pipeline.p20_kestrel.screening.sleeve_c as sleeve_c
    monkeypatch.setattr(sleeve_c, "get_latest_signal", lambda *_: {"value": 0.3})
    assert sleeve_c._regime_allows_new_entry() is False


def test_regime_filter_allows_above_200dma(monkeypatch):
    """Regime check returns True when SPY above 200DMA."""
    import src.ml.pipeline.p20_kestrel.screening.sleeve_c as sleeve_c
    monkeypatch.setattr(sleeve_c, "get_latest_signal", lambda *_: {"value": 0.8})
    assert sleeve_c._regime_allows_new_entry() is True
