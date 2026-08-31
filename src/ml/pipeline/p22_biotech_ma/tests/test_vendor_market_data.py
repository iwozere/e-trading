"""Tests for ingest/vendor_market_data.py — the deferred-vendor stub contract."""

import sys
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.vendor_market_data import NullMarketDataProvider


@pytest.mark.parametrize(
    "method_name,kwargs",
    [
        ("get_market_cap", {"ticker": "XYZ", "as_of": date(2024, 1, 1)}),
        ("get_shares_outstanding", {"ticker": "XYZ", "as_of": date(2024, 1, 1)}),
        (
            "get_segment_revenue",
            {"ticker": "XYZ", "as_of": date(2024, 1, 1), "therapeutic_area": "oncology_solid"},
        ),
        ("get_historical_price", {"ticker": "XYZ", "as_of": date(2024, 1, 1)}),
    ],
)
def test_null_provider_raises_not_implemented(method_name, kwargs):
    """
    Every method must raise loudly, not return None — a silent None could be
    mistaken for "no data available" and corrupt a hard gate like
    dilution_gate into always passing or always failing.
    """
    provider = NullMarketDataProvider()
    method = getattr(provider, method_name)
    with pytest.raises(NotImplementedError):
        method(**kwargs)
