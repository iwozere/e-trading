#!/usr/bin/env python3
"""
Paper Trading Mixin
-------------------

This mixin provides comprehensive paper trading functionality that can be used
by any broker implementation to add realistic paper trading simulation.

Features:
- Realistic execution simulation with slippage and latency
- Market impact modeling and partial fills
- Comprehensive execution quality metrics
- Position and portfolio management
- Trade history and analytics
- Integration with market data feeds

Classes:
- PaperTradingMixin: Mixin class for paper trading functionality
"""

import asyncio
import random
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

from src.trading.broker.base_broker import (
    Order, Position, Portfolio, OrderStatus, OrderSide, OrderType,
    PaperTradingConfig, PaperTradingMode, ExecutionMetrics, ExecutionQuality
)

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class PaperTradingMixin:
    """
    Mixin class that provides comprehensive paper trading functionality.

    This mixin can be used by any broker implementation to add realistic
    paper trading simulation with advanced features like:
    - Realistic execution simulation with slippage and latency
    - Market impact modeling and partial fills
    - Comprehensive execution quality metrics
    - Position and portfolio management with analytics
    - Trade history tracking and performance analysis
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize paper trading state if not already done
        if not hasattr(self, 'paper_orders'):
            self.paper_orders: Dict[str, Order] = {}
        if not hasattr(self, 'paper_positions'):
            self.paper_positions: Dict[str, Position] = {}
        if not hasattr(self, 'paper_portfolio'):
            self.paper_portfolio: Optional[Portfolio] = None
        if not hasattr(self, 'paper_trade_history'):
            self.paper_trade_history: List[Dict[str, Any]] = []
        if not hasattr(self, 'market_data_cache'):
            self.market_data_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize paper portfolio if in paper trading mode
        if hasattr(self, 'paper_trading_enabled') and self.paper_trading_enabled:
            if self.paper_portfolio is None:
                self._initialize_paper_portfolio()

    def _initialize_paper_portfolio(self) -> None:
        """Initialize paper trading portfolio with starting balance."""
        if not hasattr(self, 'paper_trading_config'):
            return

        config = self.paper_trading_config

        self.paper_portfolio = Portfolio(
            total_value=config.initial_balance,
            cash=config.initial_balance,
            positions={},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=datetime.now(timezone.utc),
            paper_trading=True,
            initial_balance=config.initial_balance
        )

        _logger.info("Initialized paper trading portfolio with $%,.2f", config.initial_balance)

    async def simulate_realistic_execution(self, order: Order, market_price: float) -> Tuple[bool, float, float, str]:
        """
        Simulate realistic order execution for paper trading.

        Args:
            order: Order to execute
            market_price: Current market price for the symbol

        Returns:
            Tuple of (success, executed_price, executed_quantity, reason)
        """
        if not hasattr(self, 'paper_trading_config'):
            return False, 0.0, 0.0, "Paper trading not configured"

        config = self.paper_trading_config

        # Simulate execution latency
        if hasattr(self, 'simulate_execution_latency'):
            await self.simulate_execution_latency()

        # Check for order rejection
        if random.random() < config.reject_probability:
            return False, 0.0, 0.0, "Order rejected by exchange simulation"

        # Calculate execution price with slippage
        slippage = 0.0
        if hasattr(self, 'calculate_slippage'):
            slippage = self.calculate_slippage(order, market_price)

        executed_price = market_price + slippage

        # Determine executed quantity (partial fills)
        executed_quantity = order.quantity
        if config.realistic_fills and random.random() < config.partial_fill_probability:
            # Simulate partial fill (50-95% of requested quantity)
            fill_ratio = random.uniform(0.5, 0.95)
            executed_quantity = order.quantity * fill_ratio
            _logger.debug("Partial fill: %.4f of %.4f", executed_quantity, order.quantity)

        # Validate execution price is reasonable
        price_deviation = abs(executed_price - market_price) / market_price
        if price_deviation > 0.05:  # 5% max deviation
            return False, 0.0, 0.0, f"Execution price deviation too high: {price_deviation:.2%}"

        return True, executed_price, executed_quantity, "Executed successfully"

    async def paper_place_order(self, order: Order, market_price: float) -> str:
        """
        Place an order in paper trading mode with realistic simulation.

        Args:
            order: Order to place
            market_price: Current market price for the symbol

        Returns:
            Order ID if successful
        """
        if not hasattr(self, 'paper_trading_enabled') or not self.paper_trading_enabled:
            raise ValueError("Not in paper trading mode")

        # Validate order
        if hasattr(self, 'validate_order'):
            is_valid, validation_message = await self.validate_order(order)
            if not is_valid:
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = validation_message
                _logger.warning("Order rejected: %s", validation_message)
                return order.order_id

        # Set paper trading flag
        order.paper_trading = True
        if hasattr(self, 'paper_trading_config'):
            order.simulation_config = self.paper_trading_config
        order.timestamp = datetime.now(timezone.utc)

        # Store order
        self.paper_orders[order.order_id] = order

        # Process order based on type
        if order.order_type == OrderType.MARKET:
            await self._execute_paper_market_order(order, market_price)
        elif order.order_type == OrderType.LIMIT:
            await self._process_paper_limit_order(order, market_price)
        elif order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            await self._process_paper_stop_order(order, market_price)
        else:
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = f"Unsupported order type: {order.order_type}"

        _logger.info("Paper order placed: %s - %s %s %s @ %s - Status: %s",
                    order.order_id, order.symbol, order.side.value, order.quantity, order.price or 'MARKET', order.status.value)

        return order.order_id

    async def _execute_paper_market_order(self, order: Order, market_price: float) -> None:
        """Execute a market order in paper trading mode."""
        start_time = datetime.now(timezone.utc)

        # Simulate realistic execution
        success, executed_price, executed_quantity, reason = await self.simulate_realistic_execution(
            order, market_price
        )

        if not success:
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = reason
            return

        # Calculate execution metrics
        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        # Execute the order
        await self._fill_paper_order(order, executed_price, executed_quantity, latency_ms)

    async def _process_paper_limit_order(self, order: Order, market_price: float) -> None:
        """Process a limit order in paper trading mode."""
        # Check if limit order can be filled immediately
        if ((order.side == OrderSide.BUY and market_price <= order.price) or
            (order.side == OrderSide.SELL and market_price >= order.price)):

            # Fill at limit price (better execution)
            start_time = datetime.now(timezone.utc)
            success, _, executed_quantity, reason = await self.simulate_realistic_execution(
                order, order.price
            )

            if success:
                latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                await self._fill_paper_order(order, order.price, executed_quantity, latency_ms)
            else:
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = reason
        else:
            # Order remains pending
            order.status = OrderStatus.PENDING
            _logger.debug("Limit order pending: %s - Market: %s, Limit: %s", order.order_id, market_price, order.price)

    async def _process_paper_stop_order(self, order: Order, market_price: float) -> None:
        """Process a stop order in paper trading mode."""
        # Stop orders remain pending until triggered
        order.status = OrderStatus.PENDING
        _logger.debug("Stop order pending: %s - Market: %s, Stop: %s", order.order_id, market_price, order.stop_price)

    async def _fill_paper_order(self, order: Order, executed_price: float,
                              executed_quantity: float, latency_ms: int) -> None:
        """Fill a paper order and update positions/portfolio."""
        if not hasattr(self, 'paper_trading_config'):
            return

        config = self.paper_trading_config

        # Calculate commission
        commission = executed_quantity * executed_price * config.commission_rate

        # Update order
        order.filled_quantity += executed_quantity
        order.average_price = executed_price
        order.commission += commission
        order.status = OrderStatus.FILLED if order.filled_quantity >= order.quantity else OrderStatus.PARTIALLY_FILLED

        # Record execution metrics
        if hasattr(self, 'record_execution_metrics'):
            metrics = self.record_execution_metrics(order, executed_price, executed_quantity, latency_ms)

        # Update position
        await self._update_paper_position(order, executed_price, executed_quantity, commission)

        # Update portfolio
        await self._update_paper_portfolio(order, executed_price, executed_quantity, commission)

        # Record trade history
        trade_record = {
            'timestamp': datetime.now(timezone.utc),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': executed_quantity,
            'price': executed_price,
            'commission': commission,
            'execution_metrics': metrics.execution_id if 'metrics' in locals() and metrics else None
        }
        self.paper_trade_history.append(trade_record)

        # Send position notification
        if hasattr(self, 'notify_position_event'):
            position_data = {
                'symbol': order.symbol,
                'side': order.side.value,
                'price': executed_price,
                'size': executed_quantity,
                'timestamp': datetime.now(timezone.utc),
                'order_id': order.order_id,
                'strategy': order.metadata.get('strategy', 'Unknown')
            }

            if order.side == OrderSide.BUY:
                await self.notify_position_event("opened", position_data)
            else:
                # For sell orders, include P&L information
                if order.symbol in self.paper_positions:
                    position = self.paper_positions[order.symbol]
                    pnl = (executed_price - position.average_price) * executed_quantity
                    pnl_percentage = (pnl / (position.average_price * executed_quantity)) * 100

                    position_data.update({
                        'entry_price': position.average_price,
                        'exit_price': executed_price,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percentage,
                        'hold_duration': self._calculate_hold_duration(position)
                    })

                await self.notify_position_event("closed", position_data)

        _logger.info("Paper order filled: %s - %.4f @ $%.4f (Commission: $%.4f)", order.order_id, executed_quantity, executed_price, commission)

    def _calculate_hold_duration(self, position: Position) -> str:
        """Calculate human-readable hold duration."""
        if not position.entry_timestamp:
            return "Unknown"

        duration = datetime.now(timezone.utc) - position.entry_timestamp

        days = duration.days
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        else:
            return f"{minutes}m {seconds}s"

    async def _update_paper_position(self, order: Order, executed_price: float,
                                   executed_quantity: float, commission: float) -> None:
        """Update paper trading position after order execution."""
        symbol = order.symbol

        if symbol not in self.paper_positions:
            # Create new position
            self.paper_positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                paper_trading=True,
                entry_timestamp=datetime.now(timezone.utc),
                entry_orders=[order.order_id],
                commission_paid=commission
            )

        position = self.paper_positions[symbol]

        # Calculate new position
        if order.side == OrderSide.BUY:
            # Buying - increase position
            total_cost = position.quantity * position.average_price + executed_quantity * executed_price
            position.quantity += executed_quantity
            position.average_price = total_cost / position.quantity if position.quantity > 0 else 0.0
        else:
            # Selling - decrease position or realize P&L
            if position.quantity > 0:
                # Calculate realized P&L for the sold quantity
                realized_pnl = executed_quantity * (executed_price - position.average_price)
                position.realized_pnl += realized_pnl
                position.quantity -= executed_quantity

                # Update portfolio realized P&L
                if self.paper_portfolio:
                    self.paper_portfolio.realized_pnl += realized_pnl

                    # Update trade statistics
                    self.paper_portfolio.total_trades += 1
                    if realized_pnl > 0:
                        self.paper_portfolio.winning_trades += 1
                        if realized_pnl > self.paper_portfolio.largest_win:
                            self.paper_portfolio.largest_win = realized_pnl
                    else:
                        self.paper_portfolio.losing_trades += 1
                        if realized_pnl < self.paper_portfolio.largest_loss:
                            self.paper_portfolio.largest_loss = realized_pnl
            else:
                # Short selling - create negative position
                total_cost = position.quantity * position.average_price - executed_quantity * executed_price
                position.quantity -= executed_quantity
                position.average_price = total_cost / abs(position.quantity) if position.quantity != 0 else 0.0

        # Update position metadata
        position.commission_paid += commission
        position.entry_orders.append(order.order_id)
        position.timestamp = datetime.now(timezone.utc)
        position.update_holding_period()

        # Remove position if quantity is zero
        if abs(position.quantity) < 1e-8:  # Floating point precision
            del self.paper_positions[symbol]

    async def _update_paper_portfolio(self, order: Order, executed_price: float,
                                    executed_quantity: float, commission: float) -> None:
        """Update paper trading portfolio after order execution."""
        if not self.paper_portfolio:
            return

        # Update cash based on trade
        if order.side == OrderSide.BUY:
            self.paper_portfolio.cash -= (executed_quantity * executed_price + commission)
        else:
            self.paper_portfolio.cash += (executed_quantity * executed_price - commission)

        # Update commission tracking
        self.paper_portfolio.total_commission += commission

        # Recalculate portfolio value and unrealized P&L
        await self._recalculate_paper_portfolio_value()

    async def _recalculate_paper_portfolio_value(self) -> None:
        """Recalculate paper portfolio total value and unrealized P&L."""
        if not self.paper_portfolio:
            return

        total_position_value = 0.0
        total_unrealized_pnl = 0.0

        for symbol, position in self.paper_positions.items():
            # Get current market price (would need market data feed in real implementation)
            current_price = await self._get_current_market_price(symbol)
            if current_price:
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = position.quantity * (current_price - position.average_price)

                total_position_value += position.market_value
                total_unrealized_pnl += position.unrealized_pnl

        # Update portfolio
        self.paper_portfolio.unrealized_pnl = total_unrealized_pnl
        self.paper_portfolio.total_value = self.paper_portfolio.cash + total_position_value
        self.paper_portfolio.timestamp = datetime.now(timezone.utc)

        # Update max portfolio value for drawdown calculation
        if self.paper_portfolio.total_value > self.paper_portfolio.max_portfolio_value:
            self.paper_portfolio.max_portfolio_value = self.paper_portfolio.total_value

    async def _get_current_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol (placeholder for market data integration)."""
        # This would integrate with the market data feed in a real implementation
        # For now, return cached price or None
        if symbol in self.market_data_cache:
            return self.market_data_cache[symbol].get('price')
        return None

    def update_market_data_cache(self, symbol: str, price: float, timestamp: datetime = None) -> None:
        """Update market data cache for paper trading simulation."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.market_data_cache[symbol] = {
            'price': price,
            'timestamp': timestamp,
            'bid': price * 0.9995,  # Simulate bid-ask spread
            'ask': price * 1.0005
        }

    async def paper_cancel_order(self, order_id: str) -> bool:
        """Cancel a paper trading order."""
        if order_id not in self.paper_orders:
            return False

        order = self.paper_orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False

        order.status = OrderStatus.CANCELLED
        order.metadata['cancellation_timestamp'] = datetime.now(timezone.utc)

        _logger.info("Paper order cancelled: %s", order_id)
        return True

    async def get_paper_positions(self) -> Dict[str, Position]:
        """Get current paper trading positions."""
        # Update market values before returning
        await self._recalculate_paper_portfolio_value()
        return self.paper_positions.copy()

    async def get_paper_portfolio(self) -> Portfolio:
        """Get current paper trading portfolio."""
        if not self.paper_portfolio:
            self._initialize_paper_portfolio()

        # Update portfolio values
        await self._recalculate_paper_portfolio_value()
        return self.paper_portfolio

    def get_paper_order_status(self, order_id: str) -> Optional[Order]:
        """Get paper trading order status."""
        return self.paper_orders.get(order_id)

    def get_paper_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get paper trading trade history."""
        history = sorted(self.paper_trade_history, key=lambda x: x['timestamp'], reverse=True)
        if limit:
            history = history[:limit]
        return history

    def get_paper_trading_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive paper trading performance report."""
        if not self.paper_portfolio:
            return {"error": "No paper trading portfolio available"}

        # Get portfolio performance metrics
        portfolio_metrics = self.paper_portfolio.calculate_performance_metrics()

        # Get execution quality report
        execution_report = {}
        if hasattr(self, 'get_execution_quality_report'):
            execution_report = self.get_execution_quality_report()

        # Calculate additional metrics
        total_trades = len(self.paper_trade_history)
        avg_trade_size = 0.0
        if total_trades > 0:
            avg_trade_size = sum(trade['quantity'] * trade['price'] for trade in self.paper_trade_history) / total_trades

        # Calculate Sharpe ratio (simplified - would need returns data for proper calculation)
        total_return = portfolio_metrics.get('total_return', 0.0)
        sharpe_ratio = total_return * math.sqrt(252) if total_return > 0 else 0.0  # Simplified

        return {
            "portfolio_metrics": portfolio_metrics,
            "execution_quality": execution_report,
            "trading_statistics": {
                "total_trades": total_trades,
                "average_trade_size": round(avg_trade_size, 2),
                "sharpe_ratio": round(sharpe_ratio, 3),
                "total_commission_paid": self.paper_portfolio.total_commission,
                "commission_as_pct_of_pnl": (
                    (self.paper_portfolio.total_commission / abs(portfolio_metrics.get('total_pnl', 1))) * 100
                    if portfolio_metrics.get('total_pnl', 0) != 0 else 0.0
                )
            },
            "current_positions": len(self.paper_positions),
            "cash_balance": self.paper_portfolio.cash,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "paper_trading_mode": self.paper_trading_config.mode.value if hasattr(self, 'paper_trading_config') else 'unknown'
        }

    def reset_paper_trading_state(self) -> None:
        """Reset paper trading state (useful for testing or restarting)."""
        self.paper_orders.clear()
        self.paper_positions.clear()
        self.paper_trade_history.clear()

        if hasattr(self, 'execution_metrics'):
            self.execution_metrics.clear()
        if hasattr(self, 'total_executions'):
            self.total_executions = 0
        if hasattr(self, 'execution_quality_stats'):
            self.execution_quality_stats = {
                ExecutionQuality.EXCELLENT: 0,
                ExecutionQuality.GOOD: 0,
                ExecutionQuality.FAIR: 0,
                ExecutionQuality.POOR: 0
            }

        self.market_data_cache.clear()

        # Reinitialize portfolio
        if hasattr(self, 'paper_trading_enabled') and self.paper_trading_enabled:
            self._initialize_paper_portfolio()

        _logger.info("Paper trading state reset")

    async def process_pending_paper_orders(self, market_data: Dict[str, float]) -> None:
        """Process pending paper orders against current market data."""
        if not hasattr(self, 'paper_trading_enabled') or not self.paper_trading_enabled:
            return

        pending_orders = [
            order for order in self.paper_orders.values()
            if order.status == OrderStatus.PENDING
        ]

        for order in pending_orders:
            if order.symbol not in market_data:
                continue

            current_price = market_data[order.symbol]

            # Update market data cache
            self.update_market_data_cache(order.symbol, current_price)

            # Check if order should be triggered/filled
            should_fill = False

            if order.order_type == OrderType.LIMIT:
                if ((order.side == OrderSide.BUY and current_price <= order.price) or
                    (order.side == OrderSide.SELL and current_price >= order.price)):
                    should_fill = True

            elif order.order_type == OrderType.STOP:
                if ((order.side == OrderSide.BUY and current_price >= order.stop_price) or
                    (order.side == OrderSide.SELL and current_price <= order.stop_price)):
                    # Convert to market order
                    order.order_type = OrderType.MARKET
                    should_fill = True

            elif order.order_type == OrderType.STOP_LIMIT:
                if ((order.side == OrderSide.BUY and current_price >= order.stop_price) or
                    (order.side == OrderSide.SELL and current_price <= order.stop_price)):
                    # Convert to limit order
                    order.order_type = OrderType.LIMIT
                    # Check if limit can be filled immediately
                    if ((order.side == OrderSide.BUY and current_price <= order.price) or
                        (order.side == OrderSide.SELL and current_price >= order.price)):
                        should_fill = True

            if should_fill:
                if order.order_type == OrderType.MARKET:
                    await self._execute_paper_market_order(order, current_price)
                else:
                    await self._process_paper_limit_order(order, current_price)