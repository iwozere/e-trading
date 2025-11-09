# ---------------------------------------------------------------------------
# tests/test_registry.py
# Normal unit tests for registry module (no importlib)
# ---------------------------------------------------------------------------
import pytest
from pathlib import Path
import sys

# Add project root (adjust if your registry.py lives elsewhere)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Import the registry directly
from src.indicators.registry import INDICATOR_META

def test_indicator_meta_exists_and_not_empty():
    assert isinstance(INDICATOR_META, dict), "INDICATOR_META must be a dict"
    assert len(INDICATOR_META) > 0, "INDICATOR_META should not be empty"

def test_registry_contains_expected_keys_subset():
    keys = set(INDICATOR_META.keys())
    expected_subset = {
        # technical
        "rsi","ema","sma","macd","bbands","adx","plus_di","minus_di","atr","stoch","obv",
        # fundamentals
        "pe","forward_pe","pb","ps","peg","roe","roa","de_ratio","current_ratio","quick_ratio",
        "div_yield","payout_ratio","market_cap","enterprise_value"
    }
    missing = expected_subset - keys
    assert not missing, f"Missing indicator keys: {missing}"

def test_indicator_meta_entry_shape_and_values():
    for name, meta in INDICATOR_META.items():
        # kind
        assert getattr(meta, "kind", None) in ("tech", "fund"), f"{name} invalid kind: {getattr(meta, 'kind', None)}"
        # outputs
        outputs = getattr(meta, "outputs", None)
        assert isinstance(outputs, list) and len(outputs) >= 1, f"{name} must define outputs list"
        assert all(isinstance(o, str) and o for o in outputs), f"{name} outputs must be non-empty strings"
        # inputs (optional, but if present must be list[str])
        inputs = getattr(meta, "inputs", None)
        if inputs is not None:
            assert isinstance(inputs, list), f"{name} inputs must be a list when present"
            assert all(isinstance(i, str) and i for i in inputs), f"{name} inputs must be non-empty strings"
        # params (optional, dict)
        params = getattr(meta, "params", None)
        if params is not None:
            assert isinstance(params, dict), f"{name} params must be a dict when present"
        # providers
        providers = getattr(meta, "providers", None)
        assert isinstance(providers, list) and len(providers) >= 1, f"{name} must have at least one provider"
        assert all(isinstance(p, str) and p for p in providers), f"{name} providers must be non-empty strings"

@pytest.mark.parametrize("tech_key", ["rsi","ema","sma","macd","bbands","atr","adx","stoch","obv"])
def test_tech_indicators_have_ohlcv_inputs(tech_key):
    meta = INDICATOR_META[tech_key]
    assert meta.kind == "tech"
    # Most tech indicators need at least 'close' in inputs; allow exceptions but require list to exist
    assert isinstance(meta.inputs, list) and len(meta.inputs) >= 1
    assert "close" in meta.inputs or any(k in meta.inputs for k in ("high","low","open","volume"))

@pytest.mark.parametrize(
    "fund_key",
    ["pe","forward_pe","pb","ps","peg","roe","roa","de_ratio","current_ratio","quick_ratio","div_yield","payout_ratio"]
)
def test_fundamentals_have_single_value_output(fund_key):
    meta = INDICATOR_META[fund_key]
    assert meta.kind == "fund"
    assert isinstance(meta.outputs, list) and len(meta.outputs) == 1
    # Fundamentals should come from a fundamentals provider
    assert any("fund" in p or "fundamental" in p for p in meta.providers), \
        f"{fund_key} should be provided by fundamentals provider"

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))