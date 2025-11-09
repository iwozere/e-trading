# ---------------------------------------------------------------------------
# tests/test_service.py
# Uses REAL adapters (no monkeypatch). Keep your adapters importable.
# ---------------------------------------------------------------------------
import os
from pathlib import Path
import asyncio
import numpy as np
import pandas as pd
import pytest
import sys

# Ensure project root (where registry.py, service.py, models.py live) is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))


# Import your real modules
import src.indicators.models as models
import src.indicators.service as service

# Try to import real adapters' dependencies to decide whether to run or skip
# If your adapters don't require these libs, feel free to remove these checks.
talib = None
pandas_ta = None
try:
    import talib as talib  # type: ignore
except Exception:
    talib = None
try:
    import pandas_ta as pandas_ta  # type: ignore
except Exception:
    pandas_ta = None

IndicatorSpec = models.IndicatorSpec
IndicatorBatchConfig = models.IndicatorBatchConfig
IndicatorResultSet = models.IndicatorResultSet
TickerIndicatorsRequest = models.TickerIndicatorsRequest

def _sample_df():
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": np.linspace(10, 15, len(idx)),
        "high": np.linspace(10.5, 15.5, len(idx)),
        "low":  np.linspace(9.5, 14.5, len(idx)),
        "close":np.linspace(10.2, 15.2, len(idx)),
        "volume":np.linspace(100, 200, len(idx)).astype(int),
    }, index=idx)

@pytest.mark.skipif(talib is None and pandas_ta is None, reason="Neither TA-Lib nor pandas_ta available for real adapters")
def test_compute_from_dataframe_with_real_adapters():
    svc = service.IndicatorService()
    cfg = IndicatorBatchConfig(
        timeframe=None,  # keep original DF timeframe
        indicators=[
            IndicatorSpec(name="rsi", output="rsi_value"),
            IndicatorSpec(name="ema", output="ema_10", params={"length": 10}),
            IndicatorSpec(name="macd", output="macd_hist"),
        ],
    )
    df = _sample_df()
    out = svc.compute(df, cfg, fund_params=None)
    # presence
    assert set(["rsi_value", "ema_10"]).issubset(out.columns)
    # MACD returns multiple outputs; service may suffix columns; check partial match
    assert any("macd" in c for c in out.columns), "MACD outputs not found"
    # length aligns to input
    assert len(out) == len(df)
    # numeric columns are numeric
    assert out["rsi_value"].dtype.kind in "fc"
    assert out["ema_10"].dtype.kind in "fc"

@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION_TESTS") != "1", reason="Set RUN_INTEGRATION_TESTS=1 to run integration test against real data providers")
def test_compute_for_ticker_with_real_stack():
    # This test will import and use whatever your service & adapters need:
    # - src.common.get_ohlcv must be implemented and accessible
    # - adapters must be importable and functional
    svc = service.IndicatorService()
    req = TickerIndicatorsRequest(
        ticker="AAPL",
        timeframe="1D",
        period="3M",
        indicators=["rsi", "ema", "macd"],
        include_recommendations=False,
    )
    res = asyncio.get_event_loop().run_until_complete(svc.compute_for_ticker(req))
    assert isinstance(res, IndicatorResultSet)
    assert res.ticker == "AAPL"
    assert any("rsi" in k for k in res.technical.keys())
    assert any(("ema" in k) or ("macd" in k) for k in res.technical.keys())

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
