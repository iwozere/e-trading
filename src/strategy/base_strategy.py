"""
Base Backtrader Strategy Module

This module provides a base class for all Backtrader-based trading strategies.
It extracts common functionality like trade tracking, position management,
and performance monitoring that is shared across different strategy implementations.
"""

from datetime import datetime
from typing import Dict, Any
import backtrader as bt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from src.notification.logger import setup_logger
from src.data.data_manager import ProviderSelector

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
        ("asset_type", "crypto"),  # Asset type: "crypto" or "stock"
        ("min_order_value", 0.0),  # Minimum order value (for stocks)
    )

    def __init__(self):
        """Initialize base strategy components."""
        super().__init__()



        # Configuration
        self.config = self.p.strategy_config or {}
        self.symbol = self.p.symbol
        self.timeframe = self.p.timeframe
        self.asset_type = self.config.get('asset_type', self.p.asset_type)
        self.min_order_value = self.config.get('min_order_value', self.p.min_order_value)

        # Auto-detect asset type from symbol if not explicitly set
        if not self.asset_type or self.asset_type == "crypto":
            self.asset_type = self._detect_asset_type()

        # Position and trade tracking
        self.current_trade = None
        self.current_exit_reason = None
        self.entry_price = None
        self.current_position_size = None  # Track current position size (handles partial exits)
        self.exit_size = None  # Track the size being exited for accurate trade notifications
        self.highest_profit = 0.0

        # Database integration
        self.trade_repository = None
        self.bot_instance_id = None
        self.enable_database_logging = self.config.get('enable_database_logging', False)
        self.bot_type = self.config.get('bot_type', 'paper')  # paper, live, optimization
        self.current_position_id = None  # Track current position ID for partial exits

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

        # Initialize database if enabled
        if self.enable_database_logging:
            self._initialize_database()

    def _detect_asset_type(self) -> str:
        """
        Auto-detect asset type from symbol using ProviderSelector.

        Returns:
            str: "crypto", "stock", or "unknown"
        """
        if not self.symbol:
            return "crypto"  # Default to crypto

        try:
            # Use ProviderSelector for sophisticated symbol classification
            provider_selector = ProviderSelector()
            asset_type = provider_selector.classify_symbol(self.symbol)

            # Map 'unknown' to 'stock' for backward compatibility
            if asset_type == "unknown":
                return "stock"

            return asset_type

        except Exception as e:
            _logger.warning("Failed to classify symbol %s using ProviderSelector: %s", self.symbol, e)
            # Fallback to simple detection
            return self._simple_asset_type_detection()

    def _simple_asset_type_detection(self) -> str:
        """
        Simple fallback asset type detection.

        Returns:
            str: "crypto" or "stock"
        """
        if not self.symbol:
            return "crypto"

        symbol_upper = self.symbol.upper()

        # Common crypto patterns
        crypto_patterns = [
            "BTC", "ETH", "USDT", "USDC", "BNB", "ADA", "SOL", "DOT", "MATIC",
            "AVAX", "LINK", "UNI", "LTC", "BCH", "XRP", "DOGE", "SHIB"
        ]

        # Check if symbol contains crypto patterns
        for pattern in crypto_patterns:
            if pattern in symbol_upper:
                return "crypto"

        # Check for common crypto exchange suffixes
        if any(suffix in symbol_upper for suffix in ["USDT", "USDC", "BUSD", "BTC", "ETH"]):
            return "crypto"

        # Default to stock for unknown symbols
        return "stock"

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
                # Try different approaches to get current bar datetime
                try:
                    # Method 1: Use self.data.datetime.datetime() - get full datetime with time
                    current_date = self.data.datetime.datetime()
                    #_logger.debug("Current datetime from self.data.datetime.datetime(): %s (type: %s)",
                    #             current_date.strftime("%Y-%m-%d %H:%M:%S") if hasattr(current_date, 'strftime') else str(current_date),
                    #             type(current_date))
                except Exception as e1:
                    _logger.debug("Method 1 failed: %s", e1)
                    try:
                        # Method 2: Use num2date with current bar index
                        current_date = self.data.num2date(self.data.idx)
                        _logger.debug("Current date from num2date(idx): %s (type: %s)",
                                     current_date.strftime("%Y-%m-%d %H:%M:%S") if hasattr(current_date, 'strftime') else str(current_date),
                                     type(current_date))
                    except Exception as e2:
                        _logger.debug("Method 2 failed: %s", e2)
                        # Method 3: Use num2date(1) as fallback
                        current_date = self.data.num2date(1)
                        _logger.debug("Current date from num2date(1): %s (type: %s)",
                                     current_date.strftime("%Y-%m-%d %H:%M:%S") if hasattr(current_date, 'strftime') else str(current_date),
                                     type(current_date))

                self.equity_dates.append(current_date)
            except (ValueError, IndexError) as e:
                # Fallback to current datetime if num2date fails
                current_date = datetime.now()
                self.equity_dates.append(current_date)

                # DEBUG: Detailed analysis of why num2date failed
                _logger.debug("=== NUM2DATE FAILURE ANALYSIS ===")
                _logger.debug("Error type: %s", type(e).__name__)
                _logger.debug("Error message: %s", str(e))
                _logger.debug("Data feed type: %s", type(self.data))
                _logger.debug("Data feed name: %s", getattr(self.data, 'name', 'Unknown'))

                # Check if we can access the datetime line directly
                try:
                    _logger.debug("self.data.datetime type: %s", type(self.data.datetime))
                    _logger.debug("self.data.datetime value: %s", self.data.datetime)
                    if hasattr(self.data.datetime, 'date'):
                        _logger.debug("self.data.datetime.date(): %s", self.data.datetime.date())
                except Exception as dt_e:
                    _logger.debug("Error accessing self.data.datetime: %s", dt_e)

                # Check data feed attributes
                try:
                    _logger.debug("Data feed dataname type: %s", type(self.data.dataname))
                    if hasattr(self.data.dataname, 'columns'):
                        _logger.debug("DataFrame columns: %s", list(self.data.dataname.columns))
                    if hasattr(self.data.dataname, 'index'):
                        _logger.debug("DataFrame index type: %s", type(self.data.dataname.index))
                        _logger.debug("DataFrame index length: %s", len(self.data.dataname.index))
                        if len(self.data.dataname.index) > 0:
                            _logger.debug("First index value: %s (type: %s)", self.data.dataname.index[0], type(self.data.dataname.index[0]))
                except Exception as debug_e:
                    _logger.debug("Error accessing data feed attributes: %s", debug_e)

                # Check data feed parameters
                try:
                    _logger.debug("Data feed datetime param: %s", getattr(self.data, 'datetime', 'Not set'))
                    _logger.debug("Data feed fromdate: %s", getattr(self.data, 'fromdate', 'Not set'))
                    _logger.debug("Data feed todate: %s", getattr(self.data, 'todate', 'Not set'))
                except Exception as debug_e:
                    _logger.debug("Error accessing data feed parameters: %s", debug_e)

                # Check current bar information
                try:
                    _logger.debug("Current bar index: %s", self.data.idx)
                    _logger.debug("Data length: %s", len(self.data))
                    _logger.debug("Data buflen: %s", self.data.buflen())
                except Exception as debug_e:
                    _logger.debug("Error accessing bar information: %s", debug_e)

                _logger.debug("Using fallback date for equity curve: %s", current_date)
                _logger.debug("=== END ANALYSIS ===")

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
        Handles both crypto (fractional) and stock (whole shares) trading.

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

            # Calculate raw shares
            shares = (cash * position_size) / price

            # Apply asset type-specific rules
            if self.asset_type.lower() == "stock":
                # For stocks: round down to whole shares
                shares = int(shares)

                # Check minimum order value
                order_value = shares * price
                if self.min_order_value > 0 and order_value < self.min_order_value:
                    _logger.debug("Order value %.2f below minimum %.2f", order_value, self.min_order_value)
                    return 0.0

            elif self.asset_type.lower() == "crypto":
                # For crypto: allow fractional shares, but ensure minimum precision
                # Round to 8 decimal places (typical crypto precision)
                shares = round(shares, 8)

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

            # Validate position size based on asset type
            if not self._validate_position_size(shares):
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
                # Validate price before setting entry_price
                current_price = self.data.close[0]
                import math
                if math.isnan(current_price) or math.isinf(current_price):
                    _logger.error("Invalid price data: %s, cannot set entry price", current_price)
                    return

                self.entry_price = current_price
                self.current_position_size = abs(shares)  # Store the actual position size in shares/units
                self.highest_profit = 0.0

                # Notify exit mixin of position entry
                if hasattr(self, 'exit_mixin') and self.exit_mixin:
                    try:
                        self.exit_mixin.on_entry(
                            entry_price=self.entry_price,
                            entry_time=self.data.datetime[0],
                            position_size=abs(shares),
                            direction=direction
                        )
                    except Exception as e:
                        _logger.warning("Error notifying exit mixin of entry: %s", e)

        except Exception as e:
            _logger.exception("Error entering position")

    def _validate_position_size(self, shares: float) -> bool:
        """
        Validate position size based on asset type.

        Args:
            shares: Number of shares/units to trade

        Returns:
            bool: True if valid, False otherwise
        """
        if self.asset_type.lower() == "stock":
            # For stocks, size must be a whole number >= 1
            if shares < 1 or not shares.is_integer():
                _logger.debug("Invalid stock position size: %s (must be whole number >= 1)", shares)
                return False
        elif self.asset_type.lower() == "crypto":
            # For crypto, size must be positive (can be fractional)
            if shares <= 0:
                _logger.debug("Invalid crypto position size: %s (must be positive)", shares)
                return False
        else:
            # Default behavior for unknown asset types
            if shares < 1:
                _logger.debug("Invalid position size: %s (must be >= 1)", shares)
                return False

        return True

    def _exit_position(self, reason: str = ""):
        """
        Exit current position.

        Args:
            reason: Reason for exit
        """
        try:
            if self.position.size == 0:
                return

            # Store the size being exited BEFORE closing
            self.exit_size = abs(self.position.size)

            # Store the exit reason for trade recording
            self.current_exit_reason = reason

            self.close()
            _logger.info("Position exit - Size: %.6f, Reason: %s", self.exit_size, reason)

            # Note: Don't reset entry_price here - it will be reset in notify_trade after the trade is actually closed

        except Exception as e:
            _logger.exception("Error exiting position")

    def _exit_partial_position(self, exit_size: float, reason: str = ""):
        """
        Exit a partial position.

        Args:
            exit_size: Number of shares/units to exit
            reason: Reason for partial exit
        """
        try:
            if self.position.size == 0:
                _logger.debug("No position to exit")
                return

            # Validate exit size
            if not self._validate_position_size(exit_size):
                _logger.warning("Invalid exit size: %s", exit_size)
                return

            # Check if exit size is valid for current position
            if abs(exit_size) > abs(self.position.size):
                _logger.warning("Exit size %s exceeds position size %s", exit_size, self.position.size)
                return

            # Determine order direction based on position
            if self.position.size > 0:  # Long position
                order = self.sell(size=exit_size)
            else:  # Short position
                order = self.buy(size=exit_size)

            if order:
                # Store the size being exited BEFORE closing
                self.exit_size = exit_size

                # Store the exit reason for trade recording
                self.current_exit_reason = reason

                # Update current position size
                self.current_position_size = abs(self.position.size - exit_size)
                _logger.info("Partial exit - Size: %.6f, Remaining: %.6f, Reason: %s",
                           exit_size, self.current_position_size, reason)

        except Exception as e:
            _logger.exception("Error in partial exit")

    def _calculate_actual_trade_size(self, trade) -> float:
        """Calculate the actual trade size for closed trades."""
        if not trade.isclosed:
            return abs(trade.size)

        # Use the stored exit size if available (most reliable)
        if hasattr(self, 'exit_size') and self.exit_size is not None:
            return self.exit_size

        # Fallback: use current position size (for full closes)
        if self.current_position_size is not None:
            return self.current_position_size

        # Fallback: calculate from PnL and price difference
        if self.entry_price and self.entry_price != 0 and trade.pnl != 0:
            price_diff = abs(trade.price - self.entry_price)
            if price_diff > 0:
                return abs(trade.pnl / price_diff)

        # Final fallback
        return 1.0

    def _initialize_database(self):
        """Initialize database connection and bot instance."""
        if not self.enable_database_logging:
            return

        try:
            from src.trading.services.trading_bot_service import trading_bot_service
            self.trade_repository = trading_bot_service

            # Create or get bot instance
            bot_data = {
                'name': self.config.get('bot_instance_name', f"{self.__class__.__name__}_{self.symbol}"),
                'type': self.bot_type,  # Use the bot_type from config
                'status': 'running',
                'strategy_name': self.__class__.__name__,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'config': self.config
            }

            bot_instance = self.trade_repository.create_bot_instance(bot_data)
            self.bot_instance_id = bot_instance.get("id")

            _logger.info("Database initialized for bot instance: %s", self.bot_instance_id)

        except Exception as e:
            _logger.exception("Error initializing database: %s", e)
            self.enable_database_logging = False

    def _store_trade_in_database(self, trade_record: Dict[str, Any], is_partial_exit: bool = False):
        """Store trade in database with proper partial exit handling."""
        if not self.enable_database_logging or not self.trade_repository:
            return

        try:
            import uuid

            if is_partial_exit:
                # Store as partial exit
                trade_data = {
                    'bot_id': self.bot_instance_id,
                    'symbol': self.symbol,
                    'trade_type': self.bot_type,  # Use the bot_type from config
                    'strategy_name': self.__class__.__name__,
                    'entry_logic_name': getattr(self, 'entry_logic', {}).get('name', 'unknown'),
                    'exit_logic_name': getattr(self, 'exit_logic', {}).get('name', 'unknown'),
                    'interval': self.timeframe,
                    'entry_time': trade_record['entry_time'],
                    'exit_time': trade_record['exit_time'],
                    'entry_price': trade_record['entry_price'],
                    'exit_price': trade_record['exit_price'],
                    'size': trade_record['size'],
                    'direction': trade_record['direction'],
                    'commission': trade_record['commission'],
                    'gross_pnl': trade_record['gross_pnl'],
                    'net_pnl': trade_record['net_pnl'],
                    'pnl_percentage': trade_record['pnl_percentage'],
                    'exit_reason': trade_record['exit_reason'],
                    'status': 'closed',
                    'position_id': self.current_position_id,
                    'extra_metadata': {
                        'duration_minutes': trade_record['duration_minutes'],
                        'strategy_config': self.config
                    }
                }

                # Get the original position trade
                original_trade = self.trade_repository.get_trade_by_id(self.current_position_id)
                if original_trade:
                    self.trade_repository.create_partial_exit_trade(trade_data, original_trade.get("id"))
                else:
                    _logger.warning("Original position trade not found for partial exit")

            else:
                # Store as new position
                trade_data = {
                    'bot_id': self.bot_instance_id,
                    'symbol': self.symbol,
                    'trade_type': self.bot_type,  # Use the bot_type from config
                    'strategy_name': self.__class__.__name__,
                    'entry_logic_name': getattr(self, 'entry_logic', {}).get('name', 'unknown'),
                    'exit_logic_name': getattr(self, 'exit_logic', {}).get('name', 'unknown'),
                    'interval': self.timeframe,
                    'entry_time': trade_record['entry_time'],
                    'exit_time': trade_record['exit_time'],
                    'entry_price': trade_record['entry_price'],
                    'exit_price': trade_record['exit_price'],
                    'size': trade_record['size'],
                    'direction': trade_record['direction'],
                    'commission': trade_record['commission'],
                    'gross_pnl': trade_record['gross_pnl'],
                    'net_pnl': trade_record['net_pnl'],
                    'pnl_percentage': trade_record['pnl_percentage'],
                    'exit_reason': trade_record['exit_reason'],
                    'status': 'closed',
                    'original_position_size': trade_record['size'],
                    'remaining_position_size': 0,  # Fully closed
                    'is_partial_exit': False,
                    'extra_metadata': {
                        'duration_minutes': trade_record['duration_minutes'],
                        'strategy_config': self.config
                    }
                }

                trade = self.trade_repository.create_trade(trade_data)
                self.current_position_id = trade.get("id")

            _logger.debug("Stored trade in database: %s", trade_data.get('id', 'unknown'))

        except Exception as e:
            _logger.exception("Error storing trade in database: %s", e)

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
            # Calculate actual trade size (handles both full and partial closes)
            actual_size = self._calculate_actual_trade_size(trade)
            trade_pnl = trade.pnl if trade.pnl is not None else 0.0
            trade_price = trade.price if trade.price is not None else 0.0

            # Log trade notification with correct size
            _logger.info(
                "Trade notification - Status: %s, Size: %.6f, PnL: %s, Price: %s",
                'CLOSED' if trade.isclosed else 'OPEN',
                actual_size, trade_pnl, trade_price
            )

            if trade.isclosed:
                # Determine if this is a partial exit
                is_partial_exit = self.position.size != 0

                # Debug: Log trade details for investigation
                _logger.debug("Trade closed details - Size: %.6f, PnL: %s, Entry Price: %s, Exit Price: %s",
                             actual_size, trade_pnl, self.entry_price, self.data.close[0])

                # Calculate trade metrics
                duration_days = trade.dtclose - trade.dtopen
                duration_minutes = duration_days * 24 * 60

                # Calculate PnL
                entry_value = self.entry_price * actual_size if self.entry_price else 0
                exit_value = self.data.close[0] * actual_size

                # Determine position direction for PnL calculation
                # For closed trades, we need to determine if it was a long or short position
                # We can use the stored exit_size and current position to determine this
                if hasattr(self, 'exit_size') and self.exit_size is not None:
                    # If we have exit_size, we can determine direction from the original position
                    # For now, assume long position (this could be improved with direction tracking)
                    gross_pnl = exit_value - entry_value
                else:
                    # Fallback: use trade.pnl directly as it's already calculated correctly by Backtrader
                    gross_pnl = trade.pnl + trade.commission  # Add commission back to get gross PnL

                net_pnl = gross_pnl - trade.commission

                # Update trade record
                trade_record = {
                    "entry_time": self.data.num2date(trade.dtopen),
                    "exit_time": self.data.num2date(trade.dtclose),
                    "entry_price": self.entry_price,
                    "exit_price": self.data.close[0],
                    "size": actual_size,  # Use corrected size
                    "symbol": self.symbol,
                    "commission": trade.commission,
                    "duration_minutes": duration_minutes,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "pnl_percentage": ((net_pnl / entry_value) * 100 if entry_value != 0 else 0),
                    "direction": "long" if actual_size > 0 else "short",
                    "exit_reason": self.current_exit_reason or "unknown",
                    "status": "closed"
                }

                self.trades.append(trade_record)

                # Store in database
                self._store_trade_in_database(trade_record, is_partial_exit)

                # Update performance metrics
                self.total_trades += 1
                self.total_pnl += net_pnl

                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                _logger.info(
                    "Trade closed - Entry: %.4f, Exit: %.4f, Size: %.6f, PnL: %.2f (%.2f%%), Duration: %.1f min",
                    self.entry_price if self.entry_price is not None else 0.0,
                    self.data.close[0], actual_size, net_pnl,
                    trade_record["pnl_percentage"], duration_minutes
                )

                # Reset trade tracking only if entire position is closed
                if self.position.size == 0:
                    self.current_trade = None
                    self.current_exit_reason = None
                    self.entry_price = None
                    self.current_position_size = None
                    self.current_position_id = None  # Reset position ID
                    self.exit_size = None  # Reset exit size
                    self.highest_profit = 0.0
                else:
                    # Partial exit - update remaining position size
                    self.current_position_size = abs(self.position.size)
                    # Don't reset exit_size for partial exits as it might be needed for the next partial exit

            else:
                # Trade opened - create new position ID
                import uuid
                self.current_position_id = str(uuid.uuid4())

                self.current_trade = {
                    "entry_time": self.data.num2date(trade.dtopen),
                    "entry_price": trade_price,
                    "size": actual_size,
                    "symbol": self.symbol,
                    "status": "open",
                    "direction": "long" if actual_size > 0 else "short"
                }

                _logger.info("Trade opened - Price: %.4f, Size: %.6f, Position ID: %s",
                           trade_price, actual_size, self.current_position_id)

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
