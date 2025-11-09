"""
Advanced Backtrader Strategy

This module provides a Backtrader strategy class that integrates with the
Advanced Strategy Framework for composite strategies, multi-timeframe support,
and dynamic switching.
"""

import backtrader as bt
import pandas as pd
from typing import Dict, Optional, Any
from src.notification.logger import setup_logger

from src.strategy.future.composite_strategy_manager import AdvancedStrategyFramework, CompositeSignal
from src.strategy.entry.entry_mixin_factory import EntryMixinFactory
from src.strategy.exit.exit_mixin_factory import ExitMixinFactory

logger = setup_logger(__name__)


class AdvancedBacktraderStrategy(bt.Strategy):
    """
    Advanced Backtrader strategy that integrates with the Advanced Strategy Framework.

    This strategy supports:
    - Composite strategies (combining multiple strategies)
    - Multi-timeframe analysis
    - Dynamic strategy switching
    - Portfolio optimization
    """

    params = (
        ('strategy_name', 'momentum_trend_composite'),
        ('use_dynamic_switching', True),
        ('max_position_size', 0.1),
        ('stop_loss_pct', 0.02),
        ('take_profit_pct', 0.04),
        ('use_trailing_stop', True),
        ('trailing_stop_pct', 0.01),
    )

    def __init__(self):
        """Initialize the advanced strategy."""
        super().__init__()

        # Initialize the advanced strategy framework
        self.advanced_framework = AdvancedStrategyFramework()
        self.advanced_framework.initialize_composite_strategies()
        self.advanced_framework.initialize_multi_timeframe_strategies()
        self.advanced_framework.initialize_dynamic_switching()

        # Strategy state
        self.current_strategy = self.p.strategy_name
        self.last_signal = None
        self.position_entry_price = None
        self.trailing_stop_price = None

        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_value': 0.0
        }

        # Initialize data feeds for multi-timeframe analysis
        self.data_feeds = {}
        self._initialize_data_feeds()

        # Initialize entry and exit mixins
        self.entry_mixins = {}
        self.exit_mixins = {}
        self._initialize_mixins()

        logger.info("Advanced strategy initialized with strategy: %s", self.current_strategy)

    def _initialize_data_feeds(self):
        """Initialize data feeds for different timeframes."""
        # Primary data feed (from Backtrader)
        primary_data = {
            'datetime': [d.datetime.datetime() for d in self.datas[0]],
            'open': [d.open[0] for d in self.datas[0]],
            'high': [d.high[0] for d in self.datas[0]],
            'low': [d.low[0] for d in self.datas[0]],
            'close': [d.close[0] for d in self.datas[0]],
            'volume': [d.volume[0] for d in self.datas[0]]
        }

        # Create DataFrame for primary timeframe
        self.data_feeds['1h'] = pd.DataFrame(primary_data)
        self.data_feeds['1h']['datetime'] = pd.to_datetime(self.data_feeds['1h']['datetime'])
        self.data_feeds['1h'].set_index('datetime', inplace=True)

        # Create resampled timeframes (simplified - in production, you'd use actual multi-timeframe data)
        if len(self.data_feeds['1h']) > 0:
            # 15-minute timeframe (aggregate 4 1-hour bars)
            self.data_feeds['15m'] = self._resample_data(self.data_feeds['1h'], '15T')

            # 4-hour timeframe (aggregate 4 1-hour bars)
            self.data_feeds['4h'] = self._resample_data(self.data_feeds['1h'], '4H')

            # Daily timeframe (aggregate 24 1-hour bars)
            self.data_feeds['1d'] = self._resample_data(self.data_feeds['1h'], '1D')

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to different timeframe (simplified implementation)."""
        if len(data) < 4:  # Need at least 4 bars for resampling
            return data.copy()

        # Simple resampling - in production, use proper OHLCV aggregation
        resampled = data.copy()

        if timeframe == '15T':
            # Take every 4th bar for 15-minute
            resampled = data.iloc[::4].copy()
        elif timeframe == '4H':
            # Take every 4th bar for 4-hour
            resampled = data.iloc[::4].copy()
        elif timeframe == '1D':
            # Take every 24th bar for daily
            resampled = data.iloc[::24].copy()

        return resampled

    def _initialize_mixins(self):
        """Initialize entry and exit mixins."""
        # Initialize entry mixins based on strategy configuration
        entry_factory = EntryMixinFactory()
        exit_factory = ExitMixinFactory()

        # Get strategy configuration
        strategy_config = self._get_strategy_config()

        if strategy_config:
            # Initialize entry mixins
            for strategy in strategy_config.get('strategies', []):
                strategy_name = strategy.get('name', '')
                if strategy_name:
                    try:
                        entry_mixin = entry_factory.create_entry_mixin(strategy_name, strategy.get('params', {}))
                        self.entry_mixins[strategy_name] = entry_mixin
                    except Exception as e:
                        logger.warning("Could not initialize entry mixin for %s: %s", strategy_name, e)

            # Initialize exit mixins
            exit_config = strategy_config.get('risk_management', {})
            if exit_config:
                try:
                    exit_mixin = exit_factory.create_exit_mixin('atr_exit', exit_config)
                    self.exit_mixins['atr_exit'] = exit_mixin
                except Exception as e:
                    logger.warning("Could not initialize exit mixin: %s", e)

    def _get_strategy_config(self) -> Optional[Dict]:
        """Get configuration for the current strategy."""
        composite_configs = self.advanced_framework.configs.get("composite_strategies", {})
        return composite_configs.get("composite_strategies", {}).get(self.current_strategy)

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        # Update data feeds with latest data
        self._update_data_feeds()

        # Check if we should switch strategies (dynamic switching)
        if self.p.use_dynamic_switching:
            new_strategy = self.advanced_framework.get_dynamic_strategy(self.data_feeds)
            if new_strategy != self.current_strategy:
                logger.info("Switching strategy from %s to %s", self.current_strategy, new_strategy)
                self.current_strategy = new_strategy
                self._initialize_mixins()  # Reinitialize mixins for new strategy

        # Generate trading signal using advanced framework
        try:
            composite_signal = self.advanced_framework.execute_strategy(
                self.current_strategy,
                self.data_feeds
            )
            self.last_signal = composite_signal

            # Execute trading logic based on signal
            self._execute_trading_logic(composite_signal)

        except Exception:
            logger.exception("Error generating trading signal: ")

        # Update performance metrics
        self._update_performance_metrics()

        # Update trailing stop
        if self.p.use_trailing_stop and self.position:
            self._update_trailing_stop()

    def _update_data_feeds(self):
        """Update data feeds with latest market data."""
        if len(self.datas) == 0:
            return

        # Update primary data feed
        current_data = {
            'datetime': self.datas[0].datetime.datetime(),
            'open': self.datas[0].open[0],
            'high': self.datas[0].high[0],
            'low': self.datas[0].low[0],
            'close': self.datas[0].close[0],
            'volume': self.datas[0].volume[0]
        }

        # Add to primary timeframe
        new_row = pd.DataFrame([current_data])
        new_row['datetime'] = pd.to_datetime(new_row['datetime'])
        new_row.set_index('datetime', inplace=True)

        self.data_feeds['1h'] = pd.concat([self.data_feeds['1h'], new_row])

        # Keep only recent data (last 1000 bars)
        for timeframe in self.data_feeds:
            if len(self.data_feeds[timeframe]) > 1000:
                self.data_feeds[timeframe] = self.data_feeds[timeframe].tail(1000)

        # Update resampled timeframes
        if len(self.data_feeds['1h']) > 0:
            self.data_feeds['15m'] = self._resample_data(self.data_feeds['1h'], '15T')
            self.data_feeds['4h'] = self._resample_data(self.data_feeds['1h'], '4H')
            self.data_feeds['1d'] = self._resample_data(self.data_feeds['1h'], '1D')

    def _execute_trading_logic(self, signal: CompositeSignal):
        """
        Execute trading logic based on the composite signal.
        """
        if not signal or signal.confidence < 0.3:  # Minimum confidence threshold
            return

        current_price = self.datas[0].close[0]

        # Entry logic
        if signal.signal_type == "buy" and not self.position:
            self._execute_buy_signal(signal, current_price)

        elif signal.signal_type == "sell" and not self.position:
            self._execute_sell_signal(signal, current_price)

        # Exit logic
        elif signal.signal_type == "sell" and self.position.size > 0:
            self._execute_exit_signal(signal, current_price)

        elif signal.signal_type == "buy" and self.position.size < 0:
            self._execute_exit_signal(signal, current_price)

    def _execute_buy_signal(self, signal: CompositeSignal, current_price: float):
        """Execute buy signal."""
        # Calculate position size
        position_size = self._calculate_position_size(signal.confidence)

        # Calculate stop loss and take profit
        stop_loss = current_price * (1 - self.p.stop_loss_pct)
        take_profit = current_price * (1 + self.p.take_profit_pct)

        # Execute buy order
        self.buy(size=position_size)

        # Store trade information
        self.position_entry_price = current_price
        self.trailing_stop_price = stop_loss

        logger.info("BUY signal executed: Price=%.4f, Size=%s, Stop=%.4f, TP=%.4f", current_price, position_size, stop_loss, take_profit)

    def _execute_sell_signal(self, signal: CompositeSignal, current_price: float):
        """Execute sell signal."""
        # Calculate position size
        position_size = self._calculate_position_size(signal.confidence)

        # Calculate stop loss and take profit
        stop_loss = current_price * (1 + self.p.stop_loss_pct)
        take_profit = current_price * (1 - self.p.take_profit_pct)

        # Execute sell order
        self.sell(size=position_size)

        # Store trade information
        self.position_entry_price = current_price
        self.trailing_stop_price = stop_loss

        logger.info("SELL signal executed: Price=%.4f, Size=%s, Stop=%.4f, TP=%.4f", current_price, position_size, stop_loss, take_profit)

    def _execute_exit_signal(self, signal: CompositeSignal, current_price: float):
        """Execute exit signal."""
        if self.position:
            self.close()
            logger.info("EXIT signal executed: Price=%.4f, Size=%s, PnL=%.2f", current_price, self.position.size, self.position.pnl)

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and risk management."""
        # Base position size from max_position_size parameter
        base_size = self.p.max_position_size

        # Adjust based on confidence
        adjusted_size = base_size * confidence

        # Ensure minimum and maximum bounds
        min_size = 0.01  # 1% minimum
        max_size = self.p.max_position_size

        return max(min_size, min(adjusted_size, max_size))

    def _update_trailing_stop(self):
        """Update trailing stop for open positions."""
        if not self.position or not self.trailing_stop_price:
            return

        current_price = self.datas[0].close[0]

        if self.position.size > 0:  # Long position
            new_stop = current_price * (1 - self.p.trailing_stop_pct)
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
            elif current_price <= self.trailing_stop_price:
                self.close()
                logger.info("Trailing stop triggered: Price=%.4f, Stop=%.4f", current_price, self.trailing_stop_price)

        elif self.position.size < 0:  # Short position
            new_stop = current_price * (1 + self.p.trailing_stop_pct)
            if new_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_stop
            elif current_price >= self.trailing_stop_price:
                self.close()
                logger.info("Trailing stop triggered: Price=%.4f, Stop=%.4f", current_price, self.trailing_stop_price)

    def _update_performance_metrics(self):
        """Update performance metrics."""
        # Update peak value and drawdown
        current_value = self.broker.getvalue()

        if current_value > self.performance_metrics['peak_value']:
            self.performance_metrics['peak_value'] = current_value

        if self.performance_metrics['peak_value'] > 0:
            current_drawdown = (current_value - self.performance_metrics['peak_value']) / self.performance_metrics['peak_value']
            self.performance_metrics['current_drawdown'] = current_drawdown

            if current_drawdown < self.performance_metrics['max_drawdown']:
                self.performance_metrics['max_drawdown'] = current_drawdown

    def notify_trade(self, trade):
        """Called when a trade is completed."""
        if trade.isclosed:
            # Update trade statistics
            self.performance_metrics['total_trades'] += 1

            if trade.pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1

            self.performance_metrics['total_pnl'] += trade.pnl

            # Calculate win rate
            win_rate = (self.performance_metrics['winning_trades'] /
                       self.performance_metrics['total_trades']) if self.performance_metrics['total_trades'] > 0 else 0

            # Calculate Sharpe ratio (simplified)
            if self.performance_metrics['total_trades'] > 0:
                avg_pnl = self.performance_metrics['total_pnl'] / self.performance_metrics['total_trades']
                # This is a simplified Sharpe calculation - in production, use proper risk-free rate and volatility
                sharpe_ratio = avg_pnl / max(abs(self.performance_metrics['max_drawdown']), 0.001)
            else:
                sharpe_ratio = 0.0

            # Store trade in history
            trade_record = {
                'datetime': self.datas[0].datetime.datetime(),
                'strategy': self.current_strategy,
                'pnl': trade.pnl,
                'size': trade.size,
                'price': trade.price,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self.performance_metrics['max_drawdown']
            }
            self.trade_history.append(trade_record)

            # Update performance in advanced framework
            self.advanced_framework.update_performance(
                self.current_strategy,
                {
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate,
                    'max_drawdown': self.performance_metrics['max_drawdown']
                }
            )

            logger.info("Trade closed: PnL=%.2f, Win Rate=%.2f%%, Total Trades=%d, Win Trades=%d, Loss Trades=%d", trade.pnl, win_rate*100, self.performance_metrics['total_trades'], self.performance_metrics['winning_trades'], self.performance_metrics['losing_trades'])

    def stop(self):
        """Called when the strategy stops."""
        logger.info("Advanced strategy stopped")
        logger.info("Final performance: Total PnL=%.2f, Sharpe=%.2f, Max Drawdown=%.2f", self.performance_metrics['total_pnl'], self.performance_metrics['sharpe_ratio'], self.performance_metrics['max_drawdown'])

    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy performance and configuration."""
        return {
            'strategy_name': self.current_strategy,
            'performance_metrics': self.performance_metrics.copy(),
            'trade_history': self.trade_history[-10:],  # Last 10 trades
            'data_feeds': list(self.data_feeds.keys()),
            'entry_mixins': list(self.entry_mixins.keys()),
            'exit_mixins': list(self.exit_mixins.keys()),
            'last_signal': self.last_signal.__dict__ if self.last_signal else None
        }
