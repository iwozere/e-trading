import pandas as pd
import pytest

from src.ml.pipeline.p13_bdsh.models import P13Config
from src.ml.pipeline.p13_bdsh.vix_scaling_engine import VIXScalingEngine


@pytest.fixture
def mock_config():
    return P13Config(
        entry_tiers={
            "Tier 1": {"z_threshold": 1.5, "allocation": 0.33},
            "Tier 2": {"z_threshold": 2.5, "allocation": 0.33},
            "Tier 3": {"z_threshold": 3.5, "allocation": 0.34},
        },
        exit_z_threshold=0.0,
        vix_lookback=20,
        initial_capital=100000.0,
        slippage_pct=0.001,
        atr_multiplier=2.0,
    )


@pytest.fixture
def engine(mock_config):
    return VIXScalingEngine(mock_config)


def test_compute_target_exposure(engine):
    # Neutral
    assert engine._compute_target_exposure(0.5) == 0.0
    # Tier 1
    assert engine._compute_target_exposure(1.6) == 0.33
    # Tier 2
    assert engine._compute_target_exposure(2.6) == 0.66
    # Tier 3
    assert engine._compute_target_exposure(3.6) == 1.0
    # Exit
    assert engine._compute_target_exposure(-0.1) == 0.0


def test_calculate_vix_zscore(engine):
    vix = pd.Series([10.0] * 19 + [12.0])  # 20 days
    z = engine.calculate_vix_zscore(vix)
    assert not pd.isna(z.iloc[-1])
    # Constant series would have 0 std, let's test a simple variation
    vix2 = pd.Series([10, 20] * 10)
    z2 = engine.calculate_vix_zscore(vix2)
    assert len(z2) == 20


def test_calculate_atr(engine):
    df = pd.DataFrame({"high": [102, 103, 104], "low": [100, 101, 102], "close": [101, 102, 103]})
    atr = engine.calculate_atr(df, period=2)
    assert not pd.isna(atr.iloc[-1])
    assert atr.iloc[-1] == 2.0  # (103-101=2, 104-102=2) mean=2


def test_backtest_stop_loss(engine):
    # Override config for this test to avoid NaN ATR with small sample
    engine.config.atr_period = 2
    engine.config.atr_multiplier = 2.0

    # Synthetic data to trigger stop loss
    # Start at 100, Tier 1 entry (z=2.0), then drop to 80
    dates = pd.date_range("2023-01-01", periods=5)
    prices = pd.Series([100.0, 100.0, 100.0, 80.0, 80.0], index=dates)
    vix_z = pd.Series([0.0, 2.0, 2.0, 2.0, 2.0], index=dates)

    ticker_df = pd.DataFrame({"high": prices + 1, "low": prices - 1, "close": prices}, index=dates)

    # ATR will be 2.0 (High-Low=2). SL at 100 - (2*2) = 96.
    # Price 80 (index 3) should trigger SL.

    results = engine.run_backtest(ticker_df, vix_z)

    assert results["In_Cooldown"].iloc[3] == True
    assert results["Target_Exposure"].iloc[3] == 0.0
    assert "stop_loss" in engine.markers
    assert len(engine.markers["stop_loss"]) > 0
