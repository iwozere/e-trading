"""Tests for DataManager fundamentals enrichment (nested FMP profile + Yahoo cap=0)."""

from src.data.data_manager import DataManager


def test_market_cap_from_nested_fmp_profile():
    dm = DataManager.__new__(DataManager)
    payload = {
        "symbol": "YSPY",
        "profile": {"marketCap": 5_011_000, "averageVolume": 31_000},
        "metrics": {},
    }
    assert dm._market_cap_from_fundamentals_payload(payload) == 5_011_000.0


def test_enrich_combined_from_provider_data():
    dm = DataManager.__new__(DataManager)
    combined = {"ticker": "YSPY", "avg_volume": 30_000.0, "_metadata": {"field_sources": {}}}
    provider_data = {
        "yahoo": {"ticker": "YSPY", "market_cap": 0.0, "avg_volume": 30_000.0},
        "fmp": {"symbol": "YSPY", "profile": {"marketCap": 4_991_266}},
    }
    out = dm._enrich_combined_fundamentals_output(combined, provider_data=provider_data)
    assert out["market_cap"] == 4_991_266.0
    assert out["avg_volume"] == 30_000.0
