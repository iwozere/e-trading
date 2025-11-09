"""
Composite Strategy Manager Module

This module handles strategy orchestration and composite signal generation.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from src.strategy.future.strategy_core import (
    BaseStrategy,
    StrategySignal,
    CompositeSignal,
    SignalAggregator,
    AggregationMethod,
    MarketRegimeDetector
)
from src.strategy.future.multi_timeframe_engine import TimeframeSyncer, MultiTimeframeStrategy

from src.notification.logger import setup_logger
logger = setup_logger(__name__)


class StrategyComposer:
    """Orchestrates multiple strategies and aggregates their signals."""

    def __init__(self, strategies: List[BaseStrategy],
                 aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_VOTING,
                 consensus_threshold: float = 0.6):
        """
        Initialize the strategy composer.

        Args:
            strategies: List of strategy instances
            aggregation_method: Method for aggregating signals
            consensus_threshold: Threshold for consensus voting
        """
        self.strategies = strategies
        self.aggregator = SignalAggregator(aggregation_method, consensus_threshold)
        self.performance_history = {}
        self.regime_detector = MarketRegimeDetector()

    def aggregate_signals(self) -> CompositeSignal:
        """
        Generate signals from all strategies and aggregate them.

        Returns:
            CompositeSignal: Aggregated signal from all strategies
        """
        signals = []

        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal()
                signals.append(signal)
                logger.debug("Generated signal from %s: %s (confidence: %s)", strategy.name, signal.signal_type, signal.confidence)
            except Exception as e:
                logger.exception("Error generating signal from %s: ", strategy.name)
                # Add a hold signal with zero confidence for failed strategies
                signals.append(StrategySignal(
                    strategy_name=strategy.name,
                    signal_type="hold",
                    confidence=0.0,
                    weight=strategy.weight,
                    timestamp=datetime.now(),
                    metadata={"error": str(e)}
                ))

        return self.aggregator.aggregate_signals(signals)

    def get_strategy_signals(self) -> Dict[str, StrategySignal]:
        """
        Get individual signals from all strategies.

        Returns:
            Dict[str, StrategySignal]: Dictionary mapping strategy names to signals
        """
        signals = {}
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal()
                signals[strategy.name] = signal
            except Exception as e:
                logger.exception("Error getting signal from %s: ", strategy.name)
                signals[strategy.name] = StrategySignal(
                    strategy_name=strategy.name,
                    signal_type="hold",
                    confidence=0.0,
                    weight=strategy.weight,
                    timestamp=datetime.now(),
                    metadata={"error": str(e)}
                )
        return signals

    def update_performance(self, strategy_name: str, performance_metrics: Dict[str, float]):
        """Update performance metrics for a strategy."""
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []

        self.performance_history[strategy_name].append({
            'timestamp': datetime.now(),
            'metrics': performance_metrics
        })

    def get_strategy_performance(self, strategy_name: str, lookback_periods: int = 20) -> Dict[str, float]:
        """Get performance metrics for a strategy."""
        if strategy_name not in self.performance_history:
            return {}

        history = self.performance_history[strategy_name][-lookback_periods:]
        if not history:
            return {}

        # Calculate average metrics
        metrics = {}
        for key in history[0]['metrics'].keys():
            values = [h['metrics'].get(key, 0) for h in history]
            metrics[key] = sum(values) / len(values)

        return metrics


class AdvancedStrategyFramework:
    """Main framework for managing composite strategies."""

    def __init__(self, config_path: str = "config/strategy/"):
        """
        Initialize the advanced strategy framework.

        Args:
            config_path: Path to strategy configuration files
        """
        self.config_path = config_path
        self.composite_strategies = {}
        self.multi_timeframe_strategies = {}
        self.dynamic_switching_config = {}
        self.performance_tracker = {}

        # Load configurations
        self._load_configs()

        # Initialize strategy components
        self.initialize_composite_strategies()
        self.initialize_multi_timeframe_strategies()
        self.initialize_dynamic_switching()

    def _load_configs(self) -> Dict[str, Any]:
        """Load strategy configurations from files."""
        configs = {}

        config_files = [
            "composite_strategies.json",
            "multi_timeframe.json",
            "dynamic_switching.json"
        ]

        for config_file in config_files:
            file_path = os.path.join(self.config_path, config_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        configs[config_file.replace('.json', '')] = json.load(f)
                    logger.info("Loaded configuration from %s", config_file)
                except Exception:
                    logger.exception("Error loading%s: ", config_file)
            else:
                logger.warning("Configuration file not found: %s", config_file)

        return configs

    def initialize_composite_strategies(self):
        """Initialize composite strategies from configuration."""
        config = self._load_configs().get("composite_strategies", {})

        for strategy_name, strategy_config in config.items():
            try:
                # Create individual strategies
                sub_strategies = []
                for sub_strategy_config in strategy_config.get("strategies", []):
                    strategy = self._create_strategy_from_config(sub_strategy_config)
                    if strategy:
                        sub_strategies.append(strategy)

                # Create composer
                aggregation_method = AggregationMethod(
                    strategy_config.get("aggregation_method", "weighted_voting")
                )
                consensus_threshold = strategy_config.get("consensus_threshold", 0.6)

                composer = StrategyComposer(
                    strategies=sub_strategies,
                    aggregation_method=aggregation_method,
                    consensus_threshold=consensus_threshold
                )

                self.composite_strategies[strategy_name] = {
                    "composer": composer,
                    "config": strategy_config
                }

                logger.info("Initialized composite strategy: %s", strategy_name)

            except Exception:
                logger.exception("Error initializing composite strategy %s: ", strategy.name)

    def initialize_multi_timeframe_strategies(self):
        """Initialize multi-timeframe strategies from configuration."""
        config = self._load_configs().get("multi_timeframe", {})

        for strategy_name, strategy_config in config.items():
            try:
                # Create timeframe syncer
                timeframes = strategy_config.get("timeframes", {})
                primary_tf = timeframes.get("primary", "1h")
                secondary_tfs = timeframes.get("secondary", ["4h", "1d"])

                syncer = TimeframeSyncer(primary_tf, secondary_tfs)

                # Create multi-timeframe strategy
                strategy = MultiTimeframeStrategy(
                    name=strategy_name,
                    syncer=syncer,
                    config=strategy_config
                )

                self.multi_timeframe_strategies[strategy_name] = {
                    "strategy": strategy,
                    "syncer": syncer,
                    "config": strategy_config
                }

                logger.info("Initialized multi-timeframe strategy: %s", strategy_name)

            except Exception:
                logger.exception("Error initializing multi-timeframe strategy%s: ", strategy.name)

    def initialize_dynamic_switching(self):
        """Initialize dynamic strategy switching configuration."""
        config = self._load_configs().get("dynamic_switching", {})
        self.dynamic_switching_config = config
        logger.info("Initialized dynamic strategy switching configuration")

    def _create_strategy_from_config(self, config: Dict[str, Any]) -> Optional[BaseStrategy]:
        """Create a strategy instance from configuration."""
        strategy_type = config.get("type")
        strategy_name = config.get("name", "unnamed")
        weight = config.get("weight", 1.0)

        if strategy_type == "rsi_momentum":
            return self._create_rsi_momentum_strategy(strategy_name, weight, config)
        elif strategy_type == "supertrend":
            return self._create_supertrend_strategy(strategy_name, weight, config)
        elif strategy_type == "bollinger_bands":
            return self._create_bollinger_bands_strategy(strategy_name, weight, config)
        elif strategy_type == "macd":
            return self._create_macd_strategy(strategy_name, weight, config)
        elif strategy_type == "atr_breakout":
            return self._create_atr_breakout_strategy(strategy_name, weight, config)
        else:
            logger.warning("Unknown strategy type: %s", strategy_type)
            return None

    def _create_rsi_momentum_strategy(self, name: str, weight: float, config: Dict) -> BaseStrategy:
        """Create RSI momentum strategy."""
        class RSIMomentumStrategy(BaseStrategy):
            def __init__(self, name, weight, params):
                super().__init__(name, weight)
                self.params = params

            def generate_signal(self) -> StrategySignal:
                if self.data is None or len(self.data) < 20:
                    return StrategySignal(
                        strategy_name=self.name,
                        signal_type="hold",
                        confidence=0.0,
                        weight=self.weight,
                        timestamp=datetime.now(),
                        metadata={"reason": "insufficient_data"}
                    )

                # Calculate RSI
                rsi_period = self.params.get("rsi_period", 14)
                rsi_oversold = self.params.get("rsi_oversold", 30)
                rsi_overbought = self.params.get("rsi_overbought", 70)

                delta = self.data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                current_rsi = rsi.iloc[-1]

                # Generate signal
                if current_rsi < rsi_oversold:
                    signal_type = "buy"
                    confidence = min(1.0, (rsi_oversold - current_rsi) / rsi_oversold)
                elif current_rsi > rsi_overbought:
                    signal_type = "sell"
                    confidence = min(1.0, (current_rsi - rsi_overbought) / (100 - rsi_overbought))
                else:
                    signal_type = "hold"
                    confidence = 0.0

                return StrategySignal(
                    strategy_name=self.name,
                    signal_type=signal_type,
                    confidence=confidence,
                    weight=self.weight,
                    timestamp=datetime.now(),
                    metadata={"rsi": current_rsi}
                )

        return RSIMomentumStrategy(name, weight, config.get("params", {}))

    def _create_supertrend_strategy(self, name: str, weight: float, config: Dict) -> BaseStrategy:
        """Create SuperTrend strategy."""
        class SuperTrendStrategy(BaseStrategy):
            def __init__(self, name, weight, params):
                super().__init__(name, weight)
                self.params = params

            def generate_signal(self) -> StrategySignal:
                if self.data is None or len(self.data) < 20:
                    return StrategySignal(
                        strategy_name=self.name,
                        signal_type="hold",
                        confidence=0.0,
                        weight=self.weight,
                        timestamp=datetime.now(),
                        metadata={"reason": "insufficient_data"}
                    )

                # Simplified SuperTrend calculation
                atr_period = self.params.get("atr_period", 14)
                multiplier = self.params.get("multiplier", 3.0)

                import numpy as np
                high_low = self.data['high'] - self.data['low']
                high_close = np.abs(self.data['high'] - self.data['close'].shift())
                low_close = np.abs(self.data['low'] - self.data['close'].shift())

                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(atr_period).mean()

                basic_upper = (self.data['high'] + self.data['low']) / 2 + multiplier * atr
                basic_lower = (self.data['high'] + self.data['low']) / 2 - multiplier * atr

                current_price = self.data['close'].iloc[-1]
                current_upper = basic_upper.iloc[-1]
                current_lower = basic_lower.iloc[-1]

                if current_price > current_upper:
                    signal_type = "buy"
                    confidence = 0.8
                elif current_price < current_lower:
                    signal_type = "sell"
                    confidence = 0.8
                else:
                    signal_type = "hold"
                    confidence = 0.0

                return StrategySignal(
                    strategy_name=self.name,
                    signal_type=signal_type,
                    confidence=confidence,
                    weight=self.weight,
                    timestamp=datetime.now(),
                    metadata={"atr": atr.iloc[-1]}
                )

        return SuperTrendStrategy(name, weight, config.get("params", {}))

    def _create_bollinger_bands_strategy(self, name: str, weight: float, config: Dict) -> BaseStrategy:
        """Create Bollinger Bands strategy."""
        class BollingerBandsStrategy(BaseStrategy):
            def __init__(self, name, weight, params):
                super().__init__(name, weight)
                self.params = params

            def generate_signal(self) -> StrategySignal:
                if self.data is None or len(self.data) < 20:
                    return StrategySignal(
                        strategy_name=self.name,
                        signal_type="hold",
                        confidence=0.0,
                        weight=self.weight,
                        timestamp=datetime.now(),
                        metadata={"reason": "insufficient_data"}
                    )

                # Calculate Bollinger Bands
                period = self.params.get("period", 20)
                std_dev = self.params.get("std_dev", 2.0)

                sma = self.data['close'].rolling(period).mean()
                std = self.data['close'].rolling(period).std()
                upper_band = sma + (std * std_dev)
                lower_band = sma - (std * std_dev)

                current_price = self.data['close'].iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]

                # Generate signal
                if current_price < current_lower:
                    signal_type = "buy"
                    confidence = min(1.0, (current_lower - current_price) / current_lower)
                elif current_price > current_upper:
                    signal_type = "sell"
                    confidence = min(1.0, (current_price - current_upper) / current_upper)
                else:
                    signal_type = "hold"
                    confidence = 0.0

                return StrategySignal(
                    strategy_name=self.name,
                    signal_type=signal_type,
                    confidence=confidence,
                    weight=self.weight,
                    timestamp=datetime.now(),
                    metadata={"bb_position": (current_price - current_lower) / (current_upper - current_lower)}
                )

        return BollingerBandsStrategy(name, weight, config.get("params", {}))

    def _create_macd_strategy(self, name: str, weight: float, config: Dict) -> BaseStrategy:
        """Create MACD strategy."""
        class MACDStrategy(BaseStrategy):
            def __init__(self, name, weight, params):
                super().__init__(name, weight)
                self.params = params

            def generate_signal(self) -> StrategySignal:
                if self.data is None or len(self.data) < 30:
                    return StrategySignal(
                        strategy_name=self.name,
                        signal_type="hold",
                        confidence=0.0,
                        weight=self.weight,
                        timestamp=datetime.now(),
                        metadata={"reason": "insufficient_data"}
                    )

                # Calculate MACD
                fast_ema = self.params.get("fast_ema", 12)
                slow_ema = self.params.get("slow_ema", 26)
                signal_ema = self.params.get("signal_ema", 9)

                ema_fast = self.data['close'].ewm(span=fast_ema).mean()
                ema_slow = self.data['close'].ewm(span=slow_ema).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_ema).mean()
                histogram = macd_line - signal_line

                current_macd = macd_line.iloc[-1]
                current_signal = signal_line.iloc[-1]
                current_histogram = histogram.iloc[-1]
                prev_histogram = histogram.iloc[-2]

                # Generate signal
                if current_macd > current_signal and current_histogram > prev_histogram:
                    signal_type = "buy"
                    confidence = min(1.0, abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0.5)
                elif current_macd < current_signal and current_histogram < prev_histogram:
                    signal_type = "sell"
                    confidence = min(1.0, abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0.5)
                else:
                    signal_type = "hold"
                    confidence = 0.0

                return StrategySignal(
                    strategy_name=self.name,
                    signal_type=signal_type,
                    confidence=confidence,
                    weight=self.weight,
                    timestamp=datetime.now(),
                    metadata={"macd": current_macd, "signal": current_signal, "histogram": current_histogram}
                )

        return MACDStrategy(name, weight, config.get("params", {}))

    def _create_atr_breakout_strategy(self, name: str, weight: float, config: Dict) -> BaseStrategy:
        """Create ATR breakout strategy."""
        class ATRBreakoutStrategy(BaseStrategy):
            def __init__(self, name, weight, params):
                super().__init__(name, weight)
                self.params = params

            def generate_signal(self) -> StrategySignal:
                if self.data is None or len(self.data) < 20:
                    return StrategySignal(
                        strategy_name=self.name,
                        signal_type="hold",
                        confidence=0.0,
                        weight=self.weight,
                        timestamp=datetime.now(),
                        metadata={"reason": "insufficient_data"}
                    )

                # Calculate ATR
                atr_period = self.params.get("atr_period", 14)
                breakout_multiplier = self.params.get("breakout_multiplier", 1.5)

                import numpy as np
                high_low = self.data['high'] - self.data['low']
                high_close = np.abs(self.data['high'] - self.data['close'].shift())
                low_close = np.abs(self.data['low'] - self.data['close'].shift())

                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(atr_period).mean()

                current_atr = atr.iloc[-1]
                current_price = self.data['close'].iloc[-1]
                prev_price = self.data['close'].iloc[-2]

                breakout_threshold = current_atr * breakout_multiplier
                price_change = abs(current_price - prev_price)

                # Generate signal
                if price_change > breakout_threshold and current_price > prev_price:
                    signal_type = "buy"
                    confidence = min(1.0, price_change / breakout_threshold)
                elif price_change > breakout_threshold and current_price < prev_price:
                    signal_type = "sell"
                    confidence = min(1.0, price_change / breakout_threshold)
                else:
                    signal_type = "hold"
                    confidence = 0.0

                return StrategySignal(
                    strategy_name=self.name,
                    signal_type=signal_type,
                    confidence=confidence,
                    weight=self.weight,
                    timestamp=datetime.now(),
                    metadata={"atr": current_atr, "price_change": price_change, "breakout_threshold": breakout_threshold}
                )

        return ATRBreakoutStrategy(name, weight, config.get("params", {}))

    @staticmethod
    def _get_primary_data(data_feeds):
        for key in ("1h", "primary"):
            df = data_feeds.get(key)
            if df is not None:
                return df
        return None

    def get_composite_signal(self, strategy_name: str, data_feeds: Dict[str, pd.DataFrame]) -> CompositeSignal:
        """Get composite signal from a specific strategy."""
        if strategy_name in self.composite_strategies:
            composer = self.composite_strategies[strategy_name]["composer"]

            # Set data for all strategies
            for strategy in composer.strategies:
                if "timeframe" in strategy.name:
                    # Multi-timeframe strategy
                    pass  # Data is set separately
                else:
                    # Single timeframe strategy - use primary data
                    primary_data = self._get_primary_data(data_feeds)
                    if primary_data is not None:
                        strategy.set_data(primary_data)

            return composer.aggregate_signals()

        elif strategy_name in self.multi_timeframe_strategies:
            strategy_info = self.multi_timeframe_strategies[strategy_name]
            strategy = strategy_info["strategy"]
            syncer = strategy_info["syncer"]

            # Set data feeds for the syncer
            for timeframe, data in data_feeds.items():
                syncer.add_data_feed(timeframe, data)

            return CompositeSignal(
                signal_type=strategy.generate_signal().signal_type,
                confidence=strategy.generate_signal().confidence,
                contributing_strategies=[strategy_name],
                timestamp=datetime.now(),
                metadata={}
            )

        else:
            logger.error("Strategy not found: %s", strategy_name)
            return CompositeSignal(
                signal_type="hold",
                confidence=0.0,
                contributing_strategies=[],
                timestamp=datetime.now(),
                metadata={"error": "strategy_not_found"}
            )

    def get_dynamic_strategy(self, data_feeds: Dict[str, pd.DataFrame]) -> str:
        """Get the best strategy based on current market conditions."""
        if not self.dynamic_switching_config:
            return list(self.composite_strategies.keys())[0] if self.composite_strategies else "default"

        # Use primary data for regime detection
        primary_data = self._get_primary_data(data_feeds)
        if primary_data is None:
            return list(self.composite_strategies.keys())[0] if self.composite_strategies else "default"

        # Detect market regime
        regime_detector = MarketRegimeDetector()
        regime = regime_detector.detect_regime(primary_data)

        # Map regime to strategy
        regime_mapping = self.dynamic_switching_config.get("regime_mapping", {})
        strategy_name = regime_mapping.get(regime.value, "default")

        if strategy_name in self.composite_strategies or strategy_name in self.multi_timeframe_strategies:
            return strategy_name
        else:
            return list(self.composite_strategies.keys())[0] if self.composite_strategies else "default"

    def execute_strategy(self, strategy_name: str, data_feeds: Dict[str, pd.DataFrame]) -> CompositeSignal:
        """Execute a specific strategy with given data feeds."""
        try:
            return self.get_composite_signal(strategy_name, data_feeds)
        except Exception as e:
            logger.exception("Error executing strategy %s: ", strategy_name)
            return CompositeSignal(
                signal_type="hold",
                confidence=0.0,
                contributing_strategies=[],
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    def update_performance(self, strategy_name: str, performance_metrics: Dict[str, float]):
        """Update performance metrics for a strategy."""
        if strategy_name not in self.performance_tracker:
            self.performance_tracker[strategy_name] = []

        self.performance_tracker[strategy_name].append({
            'timestamp': datetime.now(),
            'metrics': performance_metrics
        })

    def get_strategy_performance(self, strategy_name: str, lookback_periods: int = 20) -> Dict[str, float]:
        """Get performance metrics for a strategy."""
        if strategy_name not in self.performance_tracker:
            return {}

        history = self.performance_tracker[strategy_name][-lookback_periods:]
        if not history:
            return {}

        # Calculate average metrics
        metrics = {}
        for key in history[0]['metrics'].keys():
            values = [h['metrics'].get(key, 0) for h in history]
            metrics[key] = sum(values) / len(values)

        return metrics
