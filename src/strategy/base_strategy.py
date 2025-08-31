"""
Base Backtrader Strategy Module

This module provides a base class for all Backtrader-based trading strategies.
It extracts common functionality like trade tracking, position management,
and performance monitoring that is shared across different strategy implementations.
"""

from typing import Dict, Any
import backtrader as bt

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BaseStrategy(bt.Strategy):
    """
    Base class for all Backtrader-based trading strategies.

    Provides common functionality for:
    - Trade tracking and management
    - Position sizing
    - Performance monitoring
    - Configuration management
    - Error handling
    """

    params = (
        ("strategy_config", None), # Strategy configuration
        ("position_size", 0.1),    # Default position size as fraction of capital
        ("symbol", ""),            # Trading symbol
        ("timeframe", ""),         # Trading timeframe
    )

    def __init__(self):
        """Initialize base strategy components."""
        super().__init__()



        # Configuration
        self.config = self.p.strategy_config or {}
        self.symbol = self.p.symbol
        self.timeframe = self.p.timeframe

        # Position and trade tracking
        self.current_trade = None
        self.current_exit_reason = None
        self.entry_price = None
        self.highest_profit = 0.0

        # Trade history
        self.trades = []
        self.equity_curve = []
        self.equity_dates = []

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0

        # Position sizing
        self.base_position_size = self.config.get('position_size', self.p.position_size)
        self.max_position_size = self.config.get('max_position_size', 0.2)
        self.min_position_size = self.config.get('min_position_size', 0.05)

        _logger.debug("BaseBacktraderStrategy initialized")

    def start(self):
        """Called once at the start of the strategy."""
        _logger.debug("BaseBacktraderStrategy.start called")
        self._initialize_strategy()

    def _initialize_strategy(self):
        """Initialize strategy-specific components. Override in subclasses."""
        pass

    def prenext(self):
        """Skip bars until we have enough data."""
        pass

    def next(self):
        """Main strategy logic. Override in subclasses."""
        # Update equity curve
        self._update_equity_curve()

        # Call subclass-specific logic
        self._execute_strategy_logic()

    def _execute_strategy_logic(self):
        """Execute strategy-specific logic. Override in subclasses."""
        pass

    def _update_equity_curve(self):
        """Update equity curve tracking."""
        try:
            current_equity = self.broker.getvalue()
            self.equity_curve.append(current_equity)

            # Safely get current date, handling edge cases
            try:
                current_date = self.data.num2date(0)
                self.equity_dates.append(current_date)
            except (ValueError, IndexError) as e:
                # Fallback to current datetime if num2date fails
                from datetime import datetime
                current_date = datetime.now()
                self.equity_dates.append(current_date)
                _logger.debug("Using fallback date for equity curve: %s", current_date)

            # Update peak equity and drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity

            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

        except Exception as e:
            _logger.exception("Error updating equity curve")

    def _calculate_position_size(self, confidence: float = 1.0, risk_multiplier: float = 1.0) -> float:
        """
        Calculate position size based on available capital and risk parameters.

        Args:
            confidence: Confidence level (0.0 to 1.0) affecting position size
            risk_multiplier: Risk multiplier for position sizing

        Returns:
            float: Position size as fraction of capital
        """
        try:
            # Base position size
            position_size = self.base_position_size * confidence * risk_multiplier

            # Apply limits
            position_size = max(self.min_position_size, min(self.max_position_size, position_size))

            return position_size

        except Exception as e:
            _logger.exception("Error calculating position size")
            return self.min_position_size

    def _calculate_shares(self, position_size: float) -> float:
        """
        Calculate number of shares based on position size and current price.

        Args:
            position_size: Position size as fraction of capital

        Returns:
            float: Number of shares to trade
        """
        try:
            cash = self.broker.get_cash()
            price = self.data.close[0]

            if price <= 0:
                return 0.0

            shares = (cash * position_size) / price
            return shares

        except Exception as e:
            _logger.exception("Error calculating shares")
            return 0.0

    def _enter_position(self, direction: str, confidence: float = 1.0,
                       risk_multiplier: float = 1.0, reason: str = ""):
        """
        Enter a position with the specified parameters.

        Args:
            direction: 'long' or 'short'
            confidence: Confidence level for position sizing
            risk_multiplier: Risk multiplier for position sizing
            reason: Reason for entry
        """
        try:
            if self.position.size != 0:
                _logger.debug("Already in position, skipping entry")
                return

            position_size = self._calculate_position_size(confidence, risk_multiplier)
            shares = self._calculate_shares(position_size)

            if shares < 1:
                _logger.debug("Insufficient capital for position")
                return

            if direction.lower() == 'long':
                order = self.buy(size=shares)
                _logger.info("LONG entry - Size: %.3f, Price: %.4f, Reason: %s",
                                position_size, self.data.close[0], reason)
            elif direction.lower() == 'short':
                order = self.sell(size=shares)
                _logger.info("SHORT entry - Size: %.3f, Price: %.4f, Reason: %s",
                                position_size, self.data.close[0], reason)
            else:
                _logger.error("Invalid direction: %s", direction)
                return

            if order:
                self.entry_price = self.data.close[0]
                self.highest_profit = 0.0

        except Exception as e:
            _logger.exception("Error entering position")

    def _exit_position(self, reason: str = ""):
        """
        Exit current position.

        Args:
            reason: Reason for exit
        """
        try:
            if self.position.size == 0:
                return

            self.close()
            _logger.info("Position exit - Reason: %s", reason)

            # Reset trade tracking
            self.entry_price = None
            self.highest_profit = 0.0

        except Exception as e:
            _logger.exception("Error exiting position")

    def _update_trade_tracking(self):
        """Update trade tracking metrics."""
        try:
            if self.position.size != 0 and self.entry_price:
                current_price = self.data.close[0]

                # Calculate current PnL
                if self.position.size > 0:  # Long position
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                else:  # Short position
                    pnl_pct = (self.entry_price - current_price) / self.entry_price

                # Update highest profit for trailing stop
                if pnl_pct > self.highest_profit:
                    self.highest_profit = pnl_pct

        except Exception as e:
            _logger.exception("Error updating trade tracking")

    def notify_trade(self, trade):
        """Handle trade notifications and update metrics."""
        try:
            _logger.info(
                "Trade notification - Status: %s, Size: %s, PnL: %s, Price: %s",
                'CLOSED' if trade.isclosed else 'OPEN',
                trade.size, trade.pnl, trade.price
            )

            if trade.isclosed:
                # Calculate trade metrics
                duration_days = trade.dtclose - trade.dtopen
                duration_minutes = duration_days * 24 * 60

                # Calculate PnL
                entry_value = self.entry_price * abs(trade.size) if self.entry_price else 0
                exit_value = self.data.close[0] * abs(trade.size)
                gross_pnl = exit_value - entry_value if trade.size > 0 else entry_value - exit_value
                net_pnl = gross_pnl - trade.commission

                # Update trade record
                trade_record = {
                    "entry_time": self.data.num2date(trade.dtopen),
                    "exit_time": self.data.num2date(trade.dtclose),
                    "entry_price": self.entry_price,
                    "exit_price": self.data.close[0],
                    "size": trade.size,
                    "symbol": self.symbol,
                    "commission": trade.commission,
                    "duration_minutes": duration_minutes,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "pnl_percentage": ((net_pnl / entry_value) * 100 if entry_value != 0 else 0),
                    "trade_type": "long" if trade.size > 0 else "short",
                    "exit_reason": self.current_exit_reason or "unknown",
                    "status": "closed"
                }

                self.trades.append(trade_record)

                # Update performance metrics
                self.total_trades += 1
                self.total_pnl += net_pnl

                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                _logger.info(
                    "Trade closed - Entry: %.4f, Exit: %.4f, PnL: %.2f (%.2f%%), Duration: %.1f min",
                    self.entry_price, self.data.close[0], net_pnl,
                    trade_record["pnl_percentage"], duration_minutes
                )

                # Reset trade tracking
                self.current_trade = None
                self.current_exit_reason = None
                self.entry_price = None
                self.highest_profit = 0.0

            else:
                # Trade opened
                self.current_trade = {
                    "entry_time": self.data.num2date(trade.dtopen),
                    "entry_price": trade.price,
                    "size": trade.size,
                    "symbol": self.symbol,
                    "status": "open",
                    "trade_type": "long" if trade.size > 0 else "short"
                }

                _logger.info("Trade opened - Price: %.4f, Size: %s", trade.price, trade.size)

        except Exception as e:
            _logger.exception("Error in notify_trade")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.

        Returns:
            Dict containing performance metrics
        """
        try:
            if not self.trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "max_drawdown": 0.0
                }

            win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
            avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0

            return {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": win_rate,
                "total_pnl": self.total_pnl,
                "avg_pnl": avg_pnl,
                "max_drawdown": self.max_drawdown,
                "peak_equity": self.peak_equity,
                "current_equity": self.broker.getvalue() if hasattr(self, 'broker') else 0
            }

        except Exception as e:
            _logger.exception("Error getting performance summary")
            return {}

    def stop(self):
        """Called when strategy stops."""
        try:
            performance = self.get_performance_summary()
            _logger.info("Strategy stopped - Performance: %s", performance)

        except Exception as e:
            _logger.exception("Error in stop")
