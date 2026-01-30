#!/usr/bin/env python3
"""
Screener Strategy Bridge
------------------------
A wrapper that runs a Backtrader strategy in 'one-shot' mode to extract
the latest signals and indicator values without full trade simulation.
"""

import backtrader as bt
import pandas as pd
from typing import Dict, Any, Type, Optional
import logging
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class ScreenerStrategyBridge:
    """
    Acts as a 'Virtual Cerebro' to run any BaseStrategy derivative
    on a DataFrame and harvest its state at the last bar.
    """

    def __init__(self, strategy_class: Type[BaseStrategy], strategy_config: Dict[str, Any]):
        """
        Initialize the bridge with a strategy and its parameters.

        Args:
            strategy_class: The strategy class to instantiate.
            strategy_config: Dictionary containing 'parameters' for the strategy.
        """
        self.strategy_class = strategy_class
        self.strategy_config = strategy_config

    def run(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes the strategy on the provided history.

        Args:
            symbol: Ticker symbol.
            df: Historical OHLCV data.

        Returns:
            Dictionary with the latest signal, regime, and indicators.
        """
        if df.empty:
            return {"symbol": symbol, "error": "Empty data"}

        cerebro = bt.Cerebro()

        # 1. Add Data
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=symbol)

        # 2. Add Strategy
        # Ensure optimization_mode is set to skip DB logging
        sc = self.strategy_config.copy()
        sc['optimization_mode'] = True
        sc['enable_database_logging'] = False

        cerebro.addstrategy(
            self.strategy_class,
            strategy_config=sc,
            symbol=symbol
        )

        # 3. Setup minimal broker
        cerebro.broker.setcash(100000.0)

        try:
            # Run the strategy
            # runonce=True and preload=True for performance
            results = cerebro.run(runonce=True, preload=True)
            strat = results[0]

            # 4. Extract data from the last bar
            return self._harvest_results(strat, symbol)

        except Exception as e:
            _logger.exception("Error in StrategyBridge for %s: %s", symbol, e)
            return {"symbol": symbol, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _harvest_results(self, strategy: BaseStrategy, symbol: str) -> Dict[str, Any]:
        """Extracts the state of the strategy at the current (last) bar."""

        # Basic Info
        results = {
            "symbol": symbol,
            "timestamp": strategy.data.datetime.datetime(0).isoformat(),
            "price": strategy.data.close[0],
            "indicators": {},
            "signal": "neutral"
        }

        # Extract Indicators
        # strategy.indicators stores the line objects created via IndicatorFactory
        if hasattr(strategy, 'indicators'):
            for alias, indicator in strategy.indicators.items():
                try:
                    # Get current value (last bar)
                    val = indicator[0]
                    results["indicators"][alias] = float(val) if not pd.isna(val) else None
                except (IndexError, TypeError):
                    results["indicators"][alias] = None

        # Extract ML state (HMM-LSTM specific)
        if hasattr(strategy, 'entry_mixin'):
            em = strategy.entry_mixin
            ml_state = {
                "regime": getattr(em, 'current_regime', None),
                "confidence": float(getattr(em, 'regime_confidence', 0.0)),
                "prediction": float(getattr(em, 'current_prediction', 0.0))
            }
            results["ml_analysis"] = ml_state

            # Determine overall signal from mixin state if possible
            # We can also call em.should_enter() to see if it triggers right now
            try:
                if em.should_enter():
                    results["signal"] = "buy"
                # Note: Exit mixin could also be checked here for 'sell' signals
            except Exception as e:
                _logger.debug("Could not determine dynamic signal for %s: %s", symbol, e)

        return results
