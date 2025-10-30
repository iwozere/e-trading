"""
Strategy Core Module

This module contains the base abstractions for all trading strategies.
Provides the foundation for strategy implementation and risk management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
import pandas as pd

from src.model.strategy import StrategySignal, MarketRegime, AggregationMethod, CompositeSignal

class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.data = None
        self.indicators = {}

    @abstractmethod
    def generate_signal(self) -> StrategySignal:
        """
        Generate a trading signal.

        Returns:
            StrategySignal: Signal with asset, action, and confidence
        """

    def set_data(self, data: pd.DataFrame):
        """Set the data for the strategy."""
        self.data = data

    def add_indicator(self, name: str, indicator: Any):
        """Add an indicator to the strategy."""
        self.indicators[name] = indicator

    def get_indicator(self, name: str) -> Optional[Any]:
        """Get an indicator by name."""
        return self.indicators.get(name)


class RiskManager(ABC):
    """Base class for risk management."""

    @abstractmethod
    def validate_position(self, signal: StrategySignal) -> bool:
        """
        Validate if a position should be taken based on risk parameters.

        Args:
            signal: The strategy signal to validate

        Returns:
            bool: True if position is valid, False otherwise
        """

    @abstractmethod
    def calculate_position_size(self, signal: StrategySignal, capital: float) -> float:
        """
        Calculate the position size based on risk parameters.

        Args:
            signal: The strategy signal
            capital: Available capital

        Returns:
            float: Position size
        """


class DataLoader(Protocol):
    """Protocol for data loading components."""

    def load_multiple_timeframes(self) -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes."""
        ...


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

        # Calculate trend strength using linear regression R-squared
        import numpy as np
        prices = data['close'].values
        x = np.arange(len(prices))

        # Fit linear regression
        coeffs = np.polyfit(x, prices, 1)
        p = np.poly1d(coeffs)

        # Calculate R-squared
        y_pred = p(x)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        trend_strength = r_squared

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
        total_weight = sum(s.weight for s in signals)
        weighted_confidence = sum(s.weight * s.confidence for s in signals)

        # Determine dominant signal type by weighted voting
        buy_weight = sum(s.weight * s.confidence for s in signals if s.signal_type == "buy")
        sell_weight = sum(s.weight * s.confidence for s in signals if s.signal_type == "sell")
        hold_weight = sum(s.weight * s.confidence for s in signals if s.signal_type == "hold")

        if buy_weight > sell_weight and buy_weight > hold_weight:
            signal_type = "buy"
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            signal_type = "sell"
        else:
            signal_type = "hold"

        confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

        return CompositeSignal(
            signal_type=signal_type,
            confidence=confidence,
            contributing_strategies=[s.strategy_name for s in signals],
            timestamp=datetime.now(),
            metadata={"weighted_confidence": weighted_confidence, "total_weight": total_weight}
        )
