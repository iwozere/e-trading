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


def test_run_falls_back_to_signal_adv_20d(monkeypatch):
    """
    Regression guard: adv_20d is never populated on the k20_universe row
    itself (eod_ingest.py only ever writes it as a k20_signals row) — run()
    must fall back to the signals dict, same as sleeve_a.py's
    _passes_hard_filters does, or every ticker is rejected before RS is ever
    computed (as happened in production for every trading day from at least
    2026-08-10 through 2026-08-25).
    """
    import src.ml.pipeline.p20_kestrel.screening.sleeve_c as sleeve_c

    sig_map = {
        "adv_20d": 25_000_000,  # only source of truth — universe row omits it
        "price_vs_50dma": 1.0,
        "price_vs_200dma": 1.0,
        "sma_50": 110.0,
        "sma_200": 100.0,
        "return_3m": 0.20,
        "return_6m": 0.40,
    }
    upserted_watchlist = []

    monkeypatch.setattr(sleeve_c, "get_latest_signal", lambda *_: 1.0)  # regime open
    monkeypatch.setattr(sleeve_c, "get_active_tickers", lambda: ["TST"])
    monkeypatch.setattr(sleeve_c, "get_universe_row", lambda *_: {"ticker": "TST"})
    monkeypatch.setattr(sleeve_c, "get_signals_for_date", lambda *_: sig_map)
    monkeypatch.setattr(sleeve_c, "upsert_signals", lambda *_: None)
    monkeypatch.setattr(sleeve_c, "upsert_watchlist", lambda row: upserted_watchlist.append(row))

    result = sleeve_c.run()

    assert result["rs_computed"] == 1
    assert result["candidates"] == 1
    assert upserted_watchlist and upserted_watchlist[0]["ticker"] == "TST"


def test_run_rejects_when_adv_20d_missing_everywhere(monkeypatch):
    """Sanity complement: still correctly rejects when adv_20d is genuinely absent."""
    import src.ml.pipeline.p20_kestrel.screening.sleeve_c as sleeve_c

    sig_map = {
        "price_vs_50dma": 1.0,
        "price_vs_200dma": 1.0,
        "sma_50": 110.0,
        "sma_200": 100.0,
        "return_3m": 0.20,
        "return_6m": 0.40,
    }

    monkeypatch.setattr(sleeve_c, "get_latest_signal", lambda *_: 1.0)
    monkeypatch.setattr(sleeve_c, "get_active_tickers", lambda: ["TST"])
    monkeypatch.setattr(sleeve_c, "get_universe_row", lambda *_: {"ticker": "TST"})
    monkeypatch.setattr(sleeve_c, "get_signals_for_date", lambda *_: sig_map)
    monkeypatch.setattr(sleeve_c, "upsert_signals", lambda *_: None)
    monkeypatch.setattr(sleeve_c, "upsert_watchlist", lambda *_: None)

    result = sleeve_c.run()

    assert result["rs_computed"] == 0
    assert result["candidates"] == 0
