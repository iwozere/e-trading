# pyright: reportCallIssue=false
"""
Integration test: full buy → hold → sell cycle via an entry/exit mixin pair.

Uses a thin bt.Strategy wrapper (no DB, no provider selector, no config files)
to keep external dependencies at zero.  Runs Backtrader's Cerebro with
synthetic OHLCV data and real TALib indicators created by IndicatorFactory.

Pair under test:
  - Entry: RSIOrBBEntryMixin  (OR: RSI oversold OR price ≤ BB lower)
  - Exit:  FixedRatioExitMixin (take-profit at +5%, stop-loss at 50%)

The synthetic data declines for 50 bars (RSI → deeply oversold) then
recovers, guaranteeing entry is triggered and the 5% take-profit exits it.
"""

import sys
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.entry.rsi_or_bb_entry_mixin import RSIOrBBEntryMixin
from src.strategy.exit.fixed_ratio_exit_mixin import FixedRatioExitMixin
from src.strategy.indicator_factory import IndicatorFactory

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 250) -> pd.DataFrame:
    """
    50 bars of -1.2%/bar decline → RSI goes deeply oversold.
    200 bars of +0.4%/bar recovery → 5% take-profit triggered quickly.
    """
    np.random.seed(7)
    prices = [100.0]
    for i in range(n - 1):
        if i < 50:
            pct = -0.012 + np.random.normal(0, 0.002)
        else:
            pct = +0.004 + np.random.normal(0, 0.002)
        prices.append(max(1.0, prices[-1] * (1 + pct)))

    arr = np.array(prices)
    spread = np.random.uniform(0.002, 0.008, n)
    df = pd.DataFrame(
        {
            "open": arr * (1 + np.random.uniform(-0.001, 0.001, n)),
            "high": arr * (1 + spread),
            "low": arr * (1 - spread),
            "close": arr,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    return df


# ---------------------------------------------------------------------------
# Thin test strategy
# ---------------------------------------------------------------------------


class MixinTestStrategy(bt.Strategy):
    """
    Minimal Backtrader strategy that wires an entry and exit mixin pair.

    Deliberately avoids BaseStrategy's DB/provider dependencies so the test
    runs without any external services.
    """

    def __init__(self):
        self.indicators: dict = {}
        self.completed_trades: int = 0

        # Instantiate mixin pair
        self.entry_mixin = RSIOrBBEntryMixin(
            params={
                "rsi_period": 14,
                "rsi_oversold": 30,
                "bb_period": 20,
                "bb_dev": 2.0,
                "use_bb_touch": True,
                "cooldown_bars": 0,
            }
        )
        self.exit_mixin = FixedRatioExitMixin(
            params={
                "take_profit": 0.05,
                "stop_loss": 0.50,  # very wide — prevents stop-out during recovery
            }
        )

        # Create real TALib indicators from the entry mixin's blueprint
        ind_configs = RSIOrBBEntryMixin.get_indicator_config(self.entry_mixin.params)
        self.indicators = IndicatorFactory.create_indicators(self.data, ind_configs)

        # Attach both mixins to this strategy
        self.entry_mixin.init_entry(self)
        self.exit_mixin.init_exit(self)

    def next(self):
        if not self.position:
            if self.entry_mixin.should_enter():
                self.buy(size=1)
        else:
            if self.exit_mixin.should_exit():
                self.close()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.completed_trades += 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuyHoldSellCycle:
    def _run(self, cash: float = 10_000.0) -> MixinTestStrategy:
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(cash)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.adddata(bt.feeds.PandasData(dataname=_make_ohlcv()))
        cerebro.addstrategy(MixinTestStrategy)
        results = cerebro.run()
        return results[0]

    def test_at_least_one_trade_completes(self):
        strat = self._run()
        assert strat.completed_trades >= 1, f"Expected ≥1 completed trade, got {strat.completed_trades}"

    def test_broker_has_cash_after_run(self):
        """Broker value should be non-zero — strategy didn't blow up."""
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(10_000.0)
        cerebro.adddata(bt.feeds.PandasData(dataname=_make_ohlcv()))
        cerebro.addstrategy(MixinTestStrategy)
        cerebro.run()
        assert cerebro.broker.get_value() > 0

    def test_mixin_pair_attaches_correctly(self):
        """Mixins must be attached to the strategy after cerebro.run()."""
        strat = self._run()
        assert strat.entry_mixin.strategy is strat
        assert strat.exit_mixin.strategy is strat

    def test_indicators_populated(self):
        """IndicatorFactory must populate at least entry_rsi and entry_bb_lower."""
        strat = self._run()
        assert "entry_rsi" in strat.indicators
        assert "entry_bb_lower" in strat.indicators
        assert "entry_bb_middle" in strat.indicators


class TestMixinLifecycle:
    """Verify the mixin lifecycle (init → entry → exit) in isolation."""

    def test_entry_mixin_instantiates(self):
        mixin = RSIOrBBEntryMixin()
        assert mixin.get_minimum_lookback() >= 14

    def test_exit_mixin_instantiates(self):
        mixin = FixedRatioExitMixin(params={"take_profit": 0.05, "stop_loss": 0.03})
        assert mixin.params["take_profit"] == pytest.approx(0.05)

    def test_entry_mixin_no_signal_without_strategy(self):
        """should_enter must return False (not raise) when not attached."""
        mixin = RSIOrBBEntryMixin()
        # Not attached — are_indicators_ready() returns False → should_enter() = False
        assert mixin.should_enter() is False

    def test_exit_mixin_no_signal_without_strategy(self):
        mixin = FixedRatioExitMixin()
        # Not attached — should_exit checks self.strategy.position which is None
        try:
            result = mixin.should_exit()
            assert result is False
        except (AttributeError, RuntimeError):
            pass  # Either graceful False or AttributeError is acceptable
