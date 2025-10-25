"""
Multi-Timeframe Engine Module

This module handles data aggregation and timeframe synchronization for multi-timeframe strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from src.notification.logger import setup_logger

from src.strategy.future.strategy_core import BaseStrategy, StrategySignal

logger = setup_logger(__name__)


class TimeframeSyncer:
    """Handles synchronization of data across multiple timeframes."""

    def __init__(self, primary_tf: str, secondary_tfs: List[str]):
        """
        Initialize the timeframe synchronizer.

        Args:
            primary_tf: Primary timeframe (e.g., '1h')
            secondary_tfs: List of secondary timeframes (e.g., ['4h', '1d'])
        """
        self.primary_tf = primary_tf
        self.timeframes = [primary_tf] + secondary_tfs
        self.data_feeds = {}
        self.indicator_factories = {}

    def add_data_feed(self, timeframe: str, data: pd.DataFrame):
        """Add a data feed for a specific timeframe."""
        self.data_feeds[timeframe] = data
        logger.debug("Added data feed for timeframe %s: %d rows", timeframe, len(data))

    def resample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to a target timeframe.

        Args:
            data: Input DataFrame with OHLCV data
            target_timeframe: Target timeframe string

        Returns:
            pd.DataFrame: Resampled data
        """
        if target_timeframe not in self.data_feeds:
            # Resample the input data to target timeframe
            resampled = data.resample(target_timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            return resampled
        else:
            return self.data_feeds[target_timeframe]

    def get_all_timeframes_data(self) -> Dict[str, pd.DataFrame]:
        """Get data for all timeframes."""
        return self.data_feeds.copy()

    def align_timeframes(self, reference_timeframe: str = None) -> Dict[str, pd.DataFrame]:
        """
        Align all timeframes to a reference timeframe.

        Args:
            reference_timeframe: Reference timeframe (defaults to primary)

        Returns:
            Dict[str, pd.DataFrame]: Aligned data for all timeframes
        """
        if reference_timeframe is None:
            reference_timeframe = self.primary_tf

        if reference_timeframe not in self.data_feeds:
            raise ValueError(f"Reference timeframe {reference_timeframe} not found in data feeds")

        reference_data = self.data_feeds[reference_timeframe]
        aligned_data = {}

        for timeframe, data in self.data_feeds.items():
            if timeframe == reference_timeframe:
                aligned_data[timeframe] = data
            else:
                # Resample to reference timeframe
                aligned_data[timeframe] = self.resample_data(data, reference_timeframe)

        return aligned_data


class MultiTimeframeStrategy(BaseStrategy):
    """Base class for multi-timeframe strategies."""

    def __init__(self, name: str, syncer: TimeframeSyncer, config: Dict[str, Any]):
        """
        Initialize multi-timeframe strategy.

        Args:
            name: Strategy name
            syncer: Timeframe synchronizer
            config: Strategy configuration
        """
        super().__init__(name)
        self.syncer = syncer
        self.config = config
        self.timeframes = config.get("timeframes", {})
        self.strategy_config = config.get("strategy_config", {})
        self.rules = config.get("rules", {})

    def get_trend_direction(self) -> str:
        """Get trend direction from higher timeframe."""
        trend_config = self.strategy_config.get("trend_analysis", {})
        trend_timeframe = self.timeframes.get("trend_timeframe", "4h")

        if trend_timeframe not in self.syncer.data_feeds:
            return "neutral"

        data = self.syncer.data_feeds[trend_timeframe]
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

    def generate_signal(self) -> StrategySignal:
        """Generate entry signal based on lower timeframe."""
        entry_config = self.strategy_config.get("entry_signals", {})
        entry_timeframe = self.timeframes.get("entry_timeframe", "1h")

        if entry_timeframe not in self.syncer.data_feeds:
            return StrategySignal(
                strategy_name=self.name,
                signal_type="hold",
                confidence=0.0,
                weight=self.weight,
                timestamp=datetime.now(),
                metadata={"reason": "no_entry_timeframe_data"}
            )

        data = self.syncer.data_feeds[entry_timeframe]
        method = entry_config.get("method", "rsi_volume")
        params = entry_config.get("params", {})

        # Check trend confirmation
        trend_direction = self.get_trend_direction()
        if self.rules.get("entry_only_in_trend_direction", False):
            if trend_direction == "neutral":
                return StrategySignal(
                    strategy_name=self.name,
                    signal_type="hold",
                    confidence=0.0,
                    weight=self.weight,
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
                strategy_name=self.name,
                signal_type="hold",
                confidence=0.0,
                weight=self.weight,
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
            strategy_name=self.name,
            signal_type=signal_type,
            confidence=confidence,
            weight=self.weight,
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
        prev_price = data['close'].iloc[-2]

        breakout_threshold = current_atr * breakout_multiplier
        price_change = abs(current_price - prev_price)

        # Generate signal
        if trend_direction == "bullish" and price_change > breakout_threshold and current_price > prev_price:
            signal_type = "buy"
            confidence = min(1.0, price_change / breakout_threshold)
        elif trend_direction == "bearish" and price_change > breakout_threshold and current_price < prev_price:
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
            metadata={
                "atr": current_atr,
                "price_change": price_change,
                "breakout_threshold": breakout_threshold,
                "trend_direction": trend_direction
            }
        )
