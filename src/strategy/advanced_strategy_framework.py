"""
Advanced Strategy Framework

This module implements advanced trading strategies including:
- Composite strategies (combining multiple strategies)
- Multi-timeframe strategies
- Dynamic strategy switching
- Portfolio optimization strategies
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import backtrader as bt
from dataclasses import dataclass
from enum import Enum
import logging

from src.entry.entry_mixin_factory import EntryMixinFactory
from src.exit.exit_mixin_factory import ExitMixinFactory
from src.indicator.indicator_factory import IndicatorFactory

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating signals from multiple strategies."""
    WEIGHTED_VOTING = "weighted_voting"
    CONSENSUS = "consensus"
    MAJORITY = "majority"
    WEIGHTED_AVERAGE = "weighted_average"


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_VOLATILE = "trending_volatile"
    TRENDING_STABLE = "trending_stable"
    RANGING_VOLATILE = "ranging_volatile"
    RANGING_STABLE = "ranging_stable"
    CRISIS = "crisis"


@dataclass
class StrategySignal:
    """Represents a trading signal from a strategy."""
    strategy_name: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    weight: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class CompositeSignal:
    """Represents an aggregated signal from multiple strategies."""
    signal_type: str
    confidence: float
    contributing_strategies: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]


class MarketRegimeDetector:
    """Detects market regimes based on volatility and trend conditions."""
    
    def __init__(self, 
                 volatility_threshold: float = 0.02,
                 trend_strength_threshold: float = 0.6,
                 lookback_period: int = 20):
        self.volatility_threshold = volatility_threshold
        self.trend_strength_threshold = trend_strength_threshold
        self.lookback_period = lookback_period
    
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime based on volatility and trend conditions.
        """
        if len(data) < self.lookback_period:
            return MarketRegime.RANGING_STABLE
        
        # Calculate volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate trend strength using linear regression
        prices = data['close'].values
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = np.polyfit(x, prices, 1, full=True)
        trend_strength = abs(r_value[0]) if len(r_value) > 0 else 0
        
        # Determine regime
        if trend_strength > self.trend_strength_threshold:
            if volatility > self.volatility_threshold:
                return MarketRegime.TRENDING_VOLATILE
            else:
                return MarketRegime.TRENDING_STABLE
        else:
            if volatility > self.volatility_threshold:
                return MarketRegime.RANGING_VOLATILE
            else:
                return MarketRegime.RANGING_STABLE


class SignalAggregator:
    """Aggregates signals from multiple strategies using various methods."""
    
    def __init__(self, method: AggregationMethod, consensus_threshold: float = 0.6):
        self.method = method
        self.consensus_threshold = consensus_threshold
    
    def aggregate_signals(self, signals: List[StrategySignal]) -> CompositeSignal:
        """
        Aggregate multiple strategy signals into a single composite signal.
        """
        if not signals:
            return CompositeSignal(
                signal_type="hold",
                confidence=0.0,
                contributing_strategies=[],
                timestamp=datetime.now(),
                metadata={}
            )
        
        if self.method == AggregationMethod.WEIGHTED_VOTING:
            return self._weighted_voting(signals)
        elif self.method == AggregationMethod.CONSENSUS:
            return self._consensus_voting(signals)
        elif self.method == AggregationMethod.MAJORITY:
            return self._majority_voting(signals)
        elif self.method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(signals)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def _weighted_voting(self, signals: List[StrategySignal]) -> CompositeSignal:
        """Weighted voting based on strategy weights and confidence."""
        buy_weight = 0.0
        sell_weight = 0.0
        hold_weight = 0.0
        
        for signal in signals:
            weight = signal.weight * signal.confidence
            if signal.signal_type == "buy":
                buy_weight += weight
            elif signal.signal_type == "sell":
                sell_weight += weight
            else:
                hold_weight += weight
        
        # Determine signal type
        max_weight = max(buy_weight, sell_weight, hold_weight)
        if max_weight == buy_weight:
            signal_type = "buy"
            confidence = buy_weight / sum([s.weight for s in signals])
        elif max_weight == sell_weight:
            signal_type = "sell"
            confidence = sell_weight / sum([s.weight for s in signals])
        else:
            signal_type = "hold"
            confidence = hold_weight / sum([s.weight for s in signals])
        
        return CompositeSignal(
            signal_type=signal_type,
            confidence=confidence,
            contributing_strategies=[s.strategy_name for s in signals],
            timestamp=datetime.now(),
            metadata={"weights": {"buy": buy_weight, "sell": sell_weight, "hold": hold_weight}}
        )
    
    def _consensus_voting(self, signals: List[StrategySignal]) -> CompositeSignal:
        """Consensus voting requiring agreement above threshold."""
        buy_signals = [s for s in signals if s.signal_type == "buy"]
        sell_signals = [s for s in signals if s.signal_type == "sell"]
        
        total_weight = sum(s.weight for s in signals)
        buy_weight = sum(s.weight for s in buy_signals)
        sell_weight = sum(s.weight for s in sell_signals)
        
        buy_ratio = buy_weight / total_weight if total_weight > 0 else 0
        sell_ratio = sell_weight / total_weight if total_weight > 0 else 0
        
        if buy_ratio >= self.consensus_threshold:
            signal_type = "buy"
            confidence = buy_ratio
        elif sell_ratio >= self.consensus_threshold:
            signal_type = "sell"
            confidence = sell_ratio
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return CompositeSignal(
            signal_type=signal_type,
            confidence=confidence,
            contributing_strategies=[s.strategy_name for s in signals],
            timestamp=datetime.now(),
            metadata={"ratios": {"buy": buy_ratio, "sell": sell_ratio}}
        )
    
    def _majority_voting(self, signals: List[StrategySignal]) -> CompositeSignal:
        """Simple majority voting."""
        buy_count = sum(1 for s in signals if s.signal_type == "buy")
        sell_count = sum(1 for s in signals if s.signal_type == "sell")
        hold_count = sum(1 for s in signals if s.signal_type == "hold")
        
        total = len(signals)
        if buy_count > sell_count and buy_count > hold_count:
            signal_type = "buy"
            confidence = buy_count / total
        elif sell_count > buy_count and sell_count > hold_count:
            signal_type = "sell"
            confidence = sell_count / total
        else:
            signal_type = "hold"
            confidence = hold_count / total
        
        return CompositeSignal(
            signal_type=signal_type,
            confidence=confidence,
            contributing_strategies=[s.strategy_name for s in signals],
            timestamp=datetime.now(),
            metadata={"counts": {"buy": buy_count, "sell": sell_count, "hold": hold_count}}
        )
    
    def _weighted_average(self, signals: List[StrategySignal]) -> CompositeSignal:
        """Weighted average of signal confidences."""
        # Convert signals to numerical values
        signal_values = []
        weights = []
        
        for signal in signals:
            if signal.signal_type == "buy":
                value = signal.confidence
            elif signal.signal_type == "sell":
                value = -signal.confidence
            else:
                value = 0
            
            signal_values.append(value)
            weights.append(signal.weight)
        
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_avg = sum(v * w for v, w in zip(signal_values, weights)) / total_weight
        else:
            weighted_avg = 0
        
        # Determine signal type
        if weighted_avg > 0.1:
            signal_type = "buy"
            confidence = abs(weighted_avg)
        elif weighted_avg < -0.1:
            signal_type = "sell"
            confidence = abs(weighted_avg)
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return CompositeSignal(
            signal_type=signal_type,
            confidence=confidence,
            contributing_strategies=[s.strategy_name for s in signals],
            timestamp=datetime.now(),
            metadata={"weighted_average": weighted_avg}
        )


class MultiTimeframeStrategy:
    """Implements multi-timeframe strategy logic."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeframes = config.get("timeframes", {})
        self.strategy_config = config.get("strategy_config", {})
        self.rules = config.get("rules", {})
        
        # Initialize indicator factories for different timeframes
        self.indicator_factories = {}
        self.data_feeds = {}
    
    def initialize_timeframes(self, data_feeds: Dict[str, pd.DataFrame]):
        """Initialize data feeds for different timeframes."""
        self.data_feeds = data_feeds
        
        for timeframe_name, data in data_feeds.items():
            self.indicator_factories[timeframe_name] = IndicatorFactory(
                data=data,
                use_talib=True
            )
    
    def get_trend_direction(self) -> str:
        """Get trend direction from higher timeframe."""
        trend_config = self.strategy_config.get("trend_analysis", {})
        trend_timeframe = self.timeframes.get("trend_timeframe", "4h")
        
        if trend_timeframe not in self.data_feeds:
            return "neutral"
        
        data = self.data_feeds[trend_timeframe]
        method = trend_config.get("method", "supertrend")
        params = trend_config.get("params", {})
        
        if method == "supertrend":
            return self._get_supertrend_direction(data, params)
        elif method == "ema_crossover":
            return self._get_ema_crossover_direction(data, params)
        else:
            return "neutral"
    
    def _get_supertrend_direction(self, data: pd.DataFrame, params: Dict) -> str:
        """Get trend direction using SuperTrend indicator."""
        # Simplified SuperTrend calculation
        atr_period = params.get("atr_period", 14)
        multiplier = params.get("multiplier", 3.0)
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean()
        
        # Calculate SuperTrend
        basic_upper = (data['high'] + data['low']) / 2 + multiplier * atr
        basic_lower = (data['high'] + data['low']) / 2 - multiplier * atr
        
        # Determine trend
        current_price = data['close'].iloc[-1]
        current_upper = basic_upper.iloc[-1]
        current_lower = basic_lower.iloc[-1]
        
        if current_price > current_upper:
            return "bullish"
        elif current_price < current_lower:
            return "bearish"
        else:
            return "neutral"
    
    def _get_ema_crossover_direction(self, data: pd.DataFrame, params: Dict) -> str:
        """Get trend direction using EMA crossover."""
        fast_ema = params.get("fast_ema", 12)
        slow_ema = params.get("slow_ema", 26)
        
        ema_fast = data['close'].ewm(span=fast_ema).mean()
        ema_slow = data['close'].ewm(span=slow_ema).mean()
        
        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        prev_fast = ema_fast.iloc[-2]
        prev_slow = ema_slow.iloc[-2]
        
        # Check for crossover
        if current_fast > current_slow and prev_fast <= prev_slow:
            return "bullish"
        elif current_fast < current_slow and prev_fast >= prev_slow:
            return "bearish"
        elif current_fast > current_slow:
            return "bullish"
        else:
            return "bearish"
    
    def generate_entry_signal(self) -> StrategySignal:
        """Generate entry signal based on lower timeframe."""
        entry_config = self.strategy_config.get("entry_signals", {})
        entry_timeframe = self.timeframes.get("entry_timeframe", "1h")
        
        if entry_timeframe not in self.data_feeds:
            return StrategySignal(
                strategy_name="multi_timeframe",
                signal_type="hold",
                confidence=0.0,
                weight=1.0,
                timestamp=datetime.now(),
                metadata={}
            )
        
        data = self.data_feeds[entry_timeframe]
        method = entry_config.get("method", "rsi_volume")
        params = entry_config.get("params", {})
        
        # Check trend confirmation
        trend_direction = self.get_trend_direction()
        if self.rules.get("entry_only_in_trend_direction", False):
            if trend_direction == "neutral":
                return StrategySignal(
                    strategy_name="multi_timeframe",
                    signal_type="hold",
                    confidence=0.0,
                    weight=1.0,
                    timestamp=datetime.now(),
                    metadata={"reason": "no_trend_direction"}
                )
        
        # Generate entry signal
        if method == "rsi_volume":
            return self._generate_rsi_volume_signal(data, params, trend_direction)
        elif method == "atr_breakout":
            return self._generate_atr_breakout_signal(data, params, trend_direction)
        else:
            return StrategySignal(
                strategy_name="multi_timeframe",
                signal_type="hold",
                confidence=0.0,
                weight=1.0,
                timestamp=datetime.now(),
                metadata={"reason": "unknown_method"}
            )
    
    def _generate_rsi_volume_signal(self, data: pd.DataFrame, params: Dict, trend_direction: str) -> StrategySignal:
        """Generate signal using RSI and volume."""
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)
        volume_threshold = params.get("volume_threshold", 1.5)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate volume ratio
        avg_volume = data['volume'].rolling(20).mean()
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        
        current_rsi = rsi.iloc[-1]
        
        # Generate signal
        if trend_direction == "bullish" and current_rsi < rsi_oversold and volume_ratio > volume_threshold:
            signal_type = "buy"
            confidence = min(1.0, (rsi_oversold - current_rsi) / rsi_oversold * volume_ratio / volume_threshold)
        elif trend_direction == "bearish" and current_rsi > rsi_overbought and volume_ratio > volume_threshold:
            signal_type = "sell"
            confidence = min(1.0, (current_rsi - rsi_overbought) / (100 - rsi_overbought) * volume_ratio / volume_threshold)
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return StrategySignal(
            strategy_name="multi_timeframe",
            signal_type=signal_type,
            confidence=confidence,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={
                "rsi": current_rsi,
                "volume_ratio": volume_ratio,
                "trend_direction": trend_direction
            }
        )
    
    def _generate_atr_breakout_signal(self, data: pd.DataFrame, params: Dict, trend_direction: str) -> StrategySignal:
        """Generate signal using ATR breakout."""
        atr_period = params.get("atr_period", 14)
        breakout_multiplier = params.get("breakout_multiplier", 1.5)
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean()
        
        # Calculate breakout levels
        current_atr = atr.iloc[-1]
        current_price = data['close'].iloc[-1]
        prev_high = data['high'].iloc[-2]
        prev_low = data['low'].iloc[-2]
        
        upper_breakout = prev_high + breakout_multiplier * current_atr
        lower_breakout = prev_low - breakout_multiplier * current_atr
        
        # Generate signal
        if trend_direction == "bullish" and current_price > upper_breakout:
            signal_type = "buy"
            confidence = min(1.0, (current_price - upper_breakout) / (breakout_multiplier * current_atr))
        elif trend_direction == "bearish" and current_price < lower_breakout:
            signal_type = "sell"
            confidence = min(1.0, (lower_breakout - current_price) / (breakout_multiplier * current_atr))
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return StrategySignal(
            strategy_name="multi_timeframe",
            signal_type=signal_type,
            confidence=confidence,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={
                "atr": current_atr,
                "upper_breakout": upper_breakout,
                "lower_breakout": lower_breakout,
                "trend_direction": trend_direction
            }
        )


class AdvancedStrategyFramework:
    """
    Advanced Strategy Framework implementing composite strategies, 
    multi-timeframe support, and dynamic switching.
    """
    
    def __init__(self, config_path: str = "config/strategy/"):
        self.config_path = config_path
        self.configs = self._load_configs()
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.signal_aggregator = None
        self.multi_timeframe_strategies = {}
        self.composite_strategies = {}
        self.dynamic_switching_strategies = {}
        
        # Performance tracking
        self.performance_history = []
        self.current_strategy = None
        self.strategy_switches = []
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load all strategy configurations."""
        configs = {}
        
        config_files = [
            "composite_strategies.json",
            "multi_timeframe.json", 
            "dynamic_switching.json",
            "portfolio_optimization.json"
        ]
        
        for config_file in config_files:
            try:
                with open(f"{self.config_path}{config_file}", 'r') as f:
                    configs[config_file.replace('.json', '')] = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Config file not found: {config_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing config file {config_file}: {e}")
        
        return configs
    
    def initialize_composite_strategies(self):
        """Initialize composite strategies from configuration."""
        composite_configs = self.configs.get("composite_strategies", {})
        
        for strategy_name, config in composite_configs.get("composite_strategies", {}).items():
            self.composite_strategies[strategy_name] = {
                "config": config,
                "aggregator": SignalAggregator(
                    method=AggregationMethod(config.get("aggregation_method", "weighted_voting")),
                    consensus_threshold=config.get("consensus_threshold", 0.6)
                )
            }
    
    def initialize_multi_timeframe_strategies(self):
        """Initialize multi-timeframe strategies from configuration."""
        mtf_configs = self.configs.get("multi_timeframe", {})
        
        for strategy_name, config in mtf_configs.get("multi_timeframe_strategies", {}).items():
            self.multi_timeframe_strategies[strategy_name] = MultiTimeframeStrategy(config)
    
    def initialize_dynamic_switching(self):
        """Initialize dynamic switching strategies from configuration."""
        switching_configs = self.configs.get("dynamic_switching", {})
        
        for strategy_name, config in switching_configs.get("dynamic_switching_strategies", {}).items():
            self.dynamic_switching_strategies[strategy_name] = config
    
    def get_composite_signal(self, strategy_name: str, data_feeds: Dict[str, pd.DataFrame]) -> CompositeSignal:
        """
        Generate composite signal from multiple strategies.
        """
        if strategy_name not in self.composite_strategies:
            raise ValueError(f"Composite strategy '{strategy_name}' not found")
        
        strategy_config = self.composite_strategies[strategy_name]
        config = strategy_config["config"]
        aggregator = strategy_config["aggregator"]
        
        # Generate signals from all sub-strategies
        signals = []
        for sub_strategy in config.get("strategies", []):
            signal = self._generate_sub_strategy_signal(sub_strategy, data_feeds)
            signals.append(signal)
        
        # Aggregate signals
        composite_signal = aggregator.aggregate_signals(signals)
        
        return composite_signal
    
    def _generate_sub_strategy_signal(self, sub_strategy: Dict, data_feeds: Dict[str, pd.DataFrame]) -> StrategySignal:
        """Generate signal from a sub-strategy."""
        strategy_name = sub_strategy.get("name", "unknown")
        weight = sub_strategy.get("weight", 1.0)
        timeframe = sub_strategy.get("timeframe", "1h")
        params = sub_strategy.get("params", {})
        
        # Get data for the specified timeframe
        if timeframe not in data_feeds:
            return StrategySignal(
                strategy_name=strategy_name,
                signal_type="hold",
                confidence=0.0,
                weight=weight,
                timestamp=datetime.now(),
                metadata={"error": "timeframe_not_available"}
            )
        
        data = data_feeds[timeframe]
        
        # Generate signal based on strategy type
        if strategy_name == "rsi_momentum":
            return self._generate_rsi_momentum_signal(data, params, strategy_name, weight)
        elif strategy_name == "supertrend_trend":
            return self._generate_supertrend_signal(data, params, strategy_name, weight)
        elif strategy_name == "bollinger_bands_mean_reversion":
            return self._generate_bollinger_bands_signal(data, params, strategy_name, weight)
        elif strategy_name == "macd_momentum":
            return self._generate_macd_signal(data, params, strategy_name, weight)
        elif strategy_name == "atr_breakout":
            return self._generate_atr_breakout_signal(data, params, strategy_name, weight)
        else:
            return StrategySignal(
                strategy_name=strategy_name,
                signal_type="hold",
                confidence=0.0,
                weight=weight,
                timestamp=datetime.now(),
                metadata={"error": "unknown_strategy"}
            )
    
    def _generate_rsi_momentum_signal(self, data: pd.DataFrame, params: Dict, strategy_name: str, weight: float) -> StrategySignal:
        """Generate RSI momentum signal."""
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)
        volume_threshold = params.get("volume_threshold", 1.5)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate volume ratio
        avg_volume = data['volume'].rolling(20).mean()
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        
        current_rsi = rsi.iloc[-1]
        
        # Generate signal
        if current_rsi < rsi_oversold and volume_ratio > volume_threshold:
            signal_type = "buy"
            confidence = min(1.0, (rsi_oversold - current_rsi) / rsi_oversold * volume_ratio / volume_threshold)
        elif current_rsi > rsi_overbought and volume_ratio > volume_threshold:
            signal_type = "sell"
            confidence = min(1.0, (current_rsi - rsi_overbought) / (100 - rsi_overbought) * volume_ratio / volume_threshold)
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return StrategySignal(
            strategy_name=strategy_name,
            signal_type=signal_type,
            confidence=confidence,
            weight=weight,
            timestamp=datetime.now(),
            metadata={"rsi": current_rsi, "volume_ratio": volume_ratio}
        )
    
    def _generate_supertrend_signal(self, data: pd.DataFrame, params: Dict, strategy_name: str, weight: float) -> StrategySignal:
        """Generate SuperTrend signal."""
        period = params.get("supertrend_period", 10)
        multiplier = params.get("supertrend_multiplier", 3.0)
        atr_period = params.get("atr_period", 14)
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean()
        
        # Calculate SuperTrend
        basic_upper = (data['high'] + data['low']) / 2 + multiplier * atr
        basic_lower = (data['high'] + data['low']) / 2 - multiplier * atr
        
        current_price = data['close'].iloc[-1]
        current_upper = basic_upper.iloc[-1]
        current_lower = basic_lower.iloc[-1]
        
        # Generate signal
        if current_price > current_upper:
            signal_type = "buy"
            confidence = min(1.0, (current_price - current_upper) / (multiplier * atr.iloc[-1]))
        elif current_price < current_lower:
            signal_type = "sell"
            confidence = min(1.0, (current_lower - current_price) / (multiplier * atr.iloc[-1]))
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return StrategySignal(
            strategy_name=strategy_name,
            signal_type=signal_type,
            confidence=confidence,
            weight=weight,
            timestamp=datetime.now(),
            metadata={"atr": atr.iloc[-1], "upper": current_upper, "lower": current_lower}
        )
    
    def _generate_bollinger_bands_signal(self, data: pd.DataFrame, params: Dict, strategy_name: str, weight: float) -> StrategySignal:
        """Generate Bollinger Bands mean reversion signal."""
        bb_period = params.get("bb_period", 20)
        bb_stddev = params.get("bb_stddev", 2.0)
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 25)
        rsi_overbought = params.get("rsi_overbought", 75)
        
        # Calculate Bollinger Bands
        sma = data['close'].rolling(bb_period).mean()
        std = data['close'].rolling(bb_period).std()
        upper_band = sma + (bb_stddev * std)
        lower_band = sma - (bb_stddev * std)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_price = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Generate signal
        if current_price < current_lower and current_rsi < rsi_oversold:
            signal_type = "buy"
            confidence = min(1.0, (current_lower - current_price) / (bb_stddev * std.iloc[-1]) * (rsi_oversold - current_rsi) / rsi_oversold)
        elif current_price > current_upper and current_rsi > rsi_overbought:
            signal_type = "sell"
            confidence = min(1.0, (current_price - current_upper) / (bb_stddev * std.iloc[-1]) * (current_rsi - rsi_overbought) / (100 - rsi_overbought))
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return StrategySignal(
            strategy_name=strategy_name,
            signal_type=signal_type,
            confidence=confidence,
            weight=weight,
            timestamp=datetime.now(),
            metadata={"rsi": current_rsi, "upper_band": current_upper, "lower_band": current_lower}
        )
    
    def _generate_macd_signal(self, data: pd.DataFrame, params: Dict, strategy_name: str, weight: float) -> StrategySignal:
        """Generate MACD momentum signal."""
        macd_fast = params.get("macd_fast", 12)
        macd_slow = params.get("macd_slow", 26)
        macd_signal = params.get("macd_signal", 9)
        volume_threshold = params.get("volume_threshold", 1.2)
        
        # Calculate MACD
        ema_fast = data['close'].ewm(span=macd_fast).mean()
        ema_slow = data['close'].ewm(span=macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_signal).mean()
        histogram = macd_line - signal_line
        
        # Calculate volume ratio
        avg_volume = data['volume'].rolling(20).mean()
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else 0
        
        # Generate signal
        if current_macd > current_signal and current_histogram > prev_histogram and volume_ratio > volume_threshold:
            signal_type = "buy"
            confidence = min(1.0, abs(current_histogram) / abs(current_macd) * volume_ratio / volume_threshold)
        elif current_macd < current_signal and current_histogram < prev_histogram and volume_ratio > volume_threshold:
            signal_type = "sell"
            confidence = min(1.0, abs(current_histogram) / abs(current_macd) * volume_ratio / volume_threshold)
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return StrategySignal(
            strategy_name=strategy_name,
            signal_type=signal_type,
            confidence=confidence,
            weight=weight,
            timestamp=datetime.now(),
            metadata={"macd": current_macd, "signal": current_signal, "histogram": current_histogram, "volume_ratio": volume_ratio}
        )
    
    def _generate_atr_breakout_signal(self, data: pd.DataFrame, params: Dict, strategy_name: str, weight: float) -> StrategySignal:
        """Generate ATR breakout signal."""
        atr_period = params.get("atr_period", 14)
        breakout_multiplier = params.get("breakout_multiplier", 2.0)
        volume_confirmation = params.get("volume_confirmation", True)
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean()
        
        # Calculate breakout levels
        current_atr = atr.iloc[-1]
        current_price = data['close'].iloc[-1]
        prev_high = data['high'].iloc[-2]
        prev_low = data['low'].iloc[-2]
        
        upper_breakout = prev_high + breakout_multiplier * current_atr
        lower_breakout = prev_low - breakout_multiplier * current_atr
        
        # Check volume confirmation
        volume_confirmed = True
        if volume_confirmation:
            avg_volume = data['volume'].rolling(20).mean()
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
            volume_confirmed = volume_ratio > 1.2
        
        # Generate signal
        if current_price > upper_breakout and volume_confirmed:
            signal_type = "buy"
            confidence = min(1.0, (current_price - upper_breakout) / (breakout_multiplier * current_atr))
        elif current_price < lower_breakout and volume_confirmed:
            signal_type = "sell"
            confidence = min(1.0, (lower_breakout - current_price) / (breakout_multiplier * current_atr))
        else:
            signal_type = "hold"
            confidence = 0.0
        
        return StrategySignal(
            strategy_name=strategy_name,
            signal_type=signal_type,
            confidence=confidence,
            weight=weight,
            timestamp=datetime.now(),
            metadata={"atr": current_atr, "upper_breakout": upper_breakout, "lower_breakout": lower_breakout}
        )
    
    def get_dynamic_strategy(self, data_feeds: Dict[str, pd.DataFrame]) -> str:
        """
        Determine which strategy to use based on current market conditions.
        """
        # Detect market regime
        primary_data = list(data_feeds.values())[0] if data_feeds else pd.DataFrame()
        if len(primary_data) == 0:
            return "momentum_trend_composite"  # Default strategy
        
        regime = self.regime_detector.detect_regime(primary_data)
        
        # Get strategy mapping from configuration
        switching_configs = self.configs.get("dynamic_switching", {})
        for strategy_name, config in switching_configs.get("dynamic_switching_strategies", {}).items():
            if strategy_name == "market_regime_adaptive":
                strategies = config.get("strategies", {})
                
                if regime == MarketRegime.TRENDING_VOLATILE:
                    return strategies.get("trending_volatile", {}).get("name", "momentum_trend_composite")
                elif regime == MarketRegime.TRENDING_STABLE:
                    return strategies.get("trending_stable", {}).get("name", "trend_following_mtf")
                elif regime == MarketRegime.RANGING_VOLATILE:
                    return strategies.get("ranging_volatile", {}).get("name", "volatility_breakout")
                elif regime == MarketRegime.RANGING_STABLE:
                    return strategies.get("ranging_stable", {}).get("name", "mean_reversion_momentum")
        
        return "momentum_trend_composite"  # Default fallback
    
    def execute_strategy(self, strategy_name: str, data_feeds: Dict[str, pd.DataFrame]) -> CompositeSignal:
        """
        Execute a strategy and return the composite signal.
        """
        # Check if it's a composite strategy
        if strategy_name in self.composite_strategies:
            return self.get_composite_signal(strategy_name, data_feeds)
        
        # Check if it's a multi-timeframe strategy
        elif strategy_name in self.multi_timeframe_strategies:
            mtf_strategy = self.multi_timeframe_strategies[strategy_name]
            mtf_strategy.initialize_timeframes(data_feeds)
            signal = mtf_strategy.generate_entry_signal()
            
            # Convert to composite signal
            return CompositeSignal(
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                contributing_strategies=[signal.strategy_name],
                timestamp=signal.timestamp,
                metadata=signal.metadata
            )
        
        # Check if it's a dynamic switching strategy
        elif strategy_name in self.dynamic_switching_strategies:
            # Get the appropriate strategy based on current conditions
            actual_strategy = self.get_dynamic_strategy(data_feeds)
            return self.execute_strategy(actual_strategy, data_feeds)
        
        else:
            # Default to composite strategy
            return self.get_composite_signal("momentum_trend_composite", data_feeds)
    
    def update_performance(self, strategy_name: str, performance_metrics: Dict[str, float]):
        """Update performance tracking for strategy switching decisions."""
        self.performance_history.append({
            "strategy": strategy_name,
            "timestamp": datetime.now(),
            "metrics": performance_metrics
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_strategy_performance(self, strategy_name: str, lookback_periods: int = 20) -> Dict[str, float]:
        """Get performance metrics for a specific strategy."""
        strategy_history = [
            entry for entry in self.performance_history[-lookback_periods:]
            if entry["strategy"] == strategy_name
        ]
        
        if not strategy_history:
            return {"sharpe_ratio": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
        
        # Calculate average metrics
        avg_metrics = {}
        for key in strategy_history[0]["metrics"].keys():
            values = [entry["metrics"][key] for entry in strategy_history if key in entry["metrics"]]
            avg_metrics[key] = np.mean(values) if values else 0.0
        
        return avg_metrics 