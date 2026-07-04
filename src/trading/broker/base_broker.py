#!/usr/bin/env python3
"""
Enhanced Base Broker Module
--------------------------

This module provides an enhanced base broker class with comprehensive paper trading
support, seamless paper-to-live trading switching, integrated notifications, and
optional backtrader framework integration.

Features:
- Unified paper trading interface with realistic simulation
- Seamless configuration-based paper/live mode switching
- Comprehensive execution quality metrics and analytics
- Integrated email and Telegram notifications for position events
- Support for advanced order types (stop-loss, take-profit, OCO)
- Multi-broker support (Binance, IBKR) with automatic credential selection
- Risk management and position tracking
- Database integration for trade history and analytics
- Optional backtrader integration via ``BacktraderBrokerBridge`` (see ``backtrader_broker_bridge``)

Classes:
- BaseBroker: Enhanced base class for all brokers (always ``abc.ABC``; use ``wrap_broker_for_cerebro`` for Cerebro)
- PaperTradingConfig: Configuration for paper trading simulation
- ExecutionMetrics: Execution quality metrics for analysis
- PositionNotificationManager: Notification system for position events

Backtrader Integration:
- Use ``wrap_broker_for_cerebro(core)`` which returns a ``bt.broker.BrokerBase`` adapter; do not pass ``BaseBroker`` to ``cerebro.setbroker`` directly.
"""

import asyncio
import math
import random
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Dict, List, Tuple

from src.notification.logger import setup_logger

# Backtrader adapters — extracted to their own module.  Re-exported here so
# existing imports of `BacktraderOrderAdapter` etc. from this module still work.
from src.trading.broker.backtrader_adapters import (  # noqa: F401
    BacktraderCreatedInfo,
    BacktraderExecutedInfo,
    BacktraderOrderAdapter,
    BacktraderPortfolioAdapter,
    BacktraderPositionAdapter,
)
from src.trading.broker.backtrader_availability import BACKTRADER_AVAILABLE

# Domain models — imported from models.py so this file stays focused on the
# abstract broker interface.  Re-exported here for backward compatibility with
# all call sites that do `from src.trading.broker.base_broker import Order, ...`.
from src.trading.broker.models import (  # noqa: F401
    ExecutionMetrics,
    ExecutionQuality,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperTradingConfig,
    PaperTradingMode,
    Portfolio,
    Position,
    TradingMode,
)

_logger = setup_logger(__name__)

# Log backtrader availability
if BACKTRADER_AVAILABLE:
    _logger.info("Backtrader framework detected - Base broker will support backtrader integration")
else:
    _logger.info("Backtrader framework not available - Base broker will use ABC inheritance")


def check_backtrader_availability() -> bool:
    """
    Check if backtrader is available for use.

    Returns:
        bool: True if backtrader is available, False otherwise
    """
    return BACKTRADER_AVAILABLE


def require_backtrader(feature_name: str = "feature"):
    """
    Raise an error if backtrader is not available for a required feature.

    Args:
        feature_name: Name of the feature requiring backtrader

    Raises:
        ImportError: If backtrader is not available
    """
    if not BACKTRADER_AVAILABLE:
        raise ImportError(
            f"Backtrader framework is required for {feature_name} but is not installed. "
            f"Please install backtrader: pip install backtrader"
        )


class PositionNotificationManager:
    """
    Manages notifications for position events (opened/closed).

    Supports email and Telegram notifications with configurable settings.

    **Opt-in**: A ``notification_client`` must be supplied to actually send
    notifications.  When ``notification_client=None`` (the default), all
    notify calls are silently skipped at DEBUG level.  This is intentional
    for environments where notifications are handled via a separate hook
    (e.g. ``BaseTradingBot.trade_notification_hook`` / StrategyInstance).
    """

    def __init__(self, config: Dict[str, Any], notification_client=None):
        self.notifications_config = config.get("notifications", {})
        self.position_opened_enabled = self.notifications_config.get("position_opened", True)
        self.position_closed_enabled = self.notifications_config.get("position_closed", True)
        self.email_enabled = self.notifications_config.get("email_enabled", True)
        self.telegram_enabled = self.notifications_config.get("telegram_enabled", True)
        self.error_notifications = self.notifications_config.get("error_notifications", True)
        self.notification_client = notification_client

        _logger.info(
            "Position notifications initialized - Opened: %s, Closed: %s, Email: %s, Telegram: %s",
            self.position_opened_enabled,
            self.position_closed_enabled,
            self.email_enabled,
            self.telegram_enabled,
        )

    async def notify_position_opened(self, position_data: Dict[str, Any]):
        """Send notifications when position is opened."""
        if not self.position_opened_enabled:
            return

        message = self._format_position_opened_message(position_data)
        title = f"Position Opened - {position_data.get('trading_mode', 'UNKNOWN').upper()}"

        await self._send_notifications(message, title)

    async def notify_position_closed(self, position_data: Dict[str, Any]):
        """Send notifications when position is closed."""
        if not self.position_closed_enabled:
            return

        message = self._format_position_closed_message(position_data)
        title = f"Position Closed - {position_data.get('trading_mode', 'UNKNOWN').upper()}"

        await self._send_notifications(message, title)

    async def notify_error(self, error_data: Dict[str, Any]):
        """Send error notifications."""
        if not self.error_notifications:
            return

        message = self._format_error_message(error_data)
        title = f"Trading Error - {error_data.get('trading_mode', 'UNKNOWN').upper()}"

        await self._send_notifications(message, title)

    def _format_position_opened_message(self, data: Dict[str, Any]) -> str:
        """Format position opened notification message."""
        emoji = "🟢"
        mode_emoji = "📄" if data.get("trading_mode") == "paper" else "💰"

        return (
            f"{emoji} Position Opened - {mode_emoji} {data.get('trading_mode', 'UNKNOWN').upper()}\n\n"
            f"Bot ID: {data.get('bot_id', 'Unknown')}\n"
            f"Symbol: {data.get('symbol', 'Unknown')}\n"
            f"Side: {data.get('side', 'Unknown')}\n"
            f"Price: ${data.get('price', 0):.4f}\n"
            f"Size: {data.get('size', 0)}\n"
            f"Value: ${(data.get('price', 0) * data.get('size', 0)):,.2f}\n"
            f"Time: {data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            f"Strategy: {data.get('strategy', 'Unknown')}\n"
            f"Order ID: {data.get('order_id', 'Unknown')}"
        )

    def _format_position_closed_message(self, data: Dict[str, Any]) -> str:
        """Format position closed notification message."""
        emoji = "🔴"
        mode_emoji = "📄" if data.get("trading_mode") == "paper" else "💰"
        pnl = data.get("pnl", 0)
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        return (
            f"{emoji} Position Closed - {mode_emoji} {data.get('trading_mode', 'UNKNOWN').upper()}\n\n"
            f"Bot ID: {data.get('bot_id', 'Unknown')}\n"
            f"Symbol: {data.get('symbol', 'Unknown')}\n"
            f"Side: {data.get('side', 'Unknown')}\n"
            f"Entry Price: ${data.get('entry_price', 0):.4f}\n"
            f"Exit Price: ${data.get('exit_price', 0):.4f}\n"
            f"Size: {data.get('size', 0)}\n"
            f"{pnl_emoji} P&L: ${pnl:.2f} ({data.get('pnl_percentage', 0):.2f}%)\n"
            f"Time: {data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            f"Hold Duration: {data.get('hold_duration', 'Unknown')}\n"
            f"Strategy: {data.get('strategy', 'Unknown')}"
        )

    def _format_error_message(self, data: Dict[str, Any]) -> str:
        """Format error notification message."""
        return (
            f"⚠️ Trading Error - {data.get('trading_mode', 'UNKNOWN').upper()}\n\n"
            f"Bot ID: {data.get('bot_id', 'Unknown')}\n"
            f"Symbol: {data.get('symbol', 'Unknown')}\n"
            f"Error: {data.get('error_message', 'Unknown error')}\n"
            f"Time: {data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            f"Please check the bot configuration and system logs."
        )

    async def _send_notifications(self, message: str, title: str):
        """Send notifications via enabled channels."""
        try:
            if self.notification_client:
                # Use the new notification service client
                channels = []
                if self.email_enabled:
                    channels.append("email")
                if self.telegram_enabled:
                    channels.append("telegram")

                if channels:
                    from src.notification.service.client import MessagePriority, MessageType

                    success = await self.notification_client.send_notification(
                        notification_type=MessageType.SYSTEM,
                        title=title,
                        message=message,
                        priority=MessagePriority.NORMAL,
                        source="trading_broker",
                        channels=channels,
                        recipient_id="trading_system",
                    )

                    if success:
                        _logger.info("Trading notification sent successfully: %s", title)
                    else:
                        _logger.warning("Failed to send trading notification: %s", title)
                else:
                    _logger.debug("No notification channels enabled")
            else:
                # notification_client=None means this manager is opt-in disabled.
                # Log at DEBUG so callers that intentionally omit it don't see noisy warnings.
                _logger.debug(
                    "PositionNotificationManager: notification_client not set — "
                    "skipping notification '%s' (pass notification_client= to enable)",
                    title,
                )

        except Exception:
            _logger.exception("Error sending notifications:")


class BaseBroker(ABC):
    """
    Enhanced base class for all brokers with comprehensive paper trading support
    and optional backtrader integration.

    For ``bt.Cerebro``, wrap instances with ``wrap_broker_for_cerebro`` from
    ``backtrader_broker_bridge`` so the engine receives a real ``BrokerBase`` subclass.

    Features:
    - Seamless paper/live trading mode switching via configuration
    - Realistic paper trading simulation with slippage, latency, and market impact
    - Comprehensive execution quality metrics and analytics
    - Integrated position notifications (email/Telegram)
    - Support for advanced order types and risk management
    - Multi-broker support with automatic credential selection
    - Optional backtrader framework integration
    """

    def __init__(self, config: Dict[str, Any], notification_client=None):
        super().__init__()
        # Set True only by enable_backtrader_trading_mode() (used by BacktraderBrokerBridge)
        self._backtrader_mode = False

        self.config = config
        self.name = config.get("name", "unknown")
        self.is_connected = False
        self._logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

        # Trading mode configuration
        self.trading_mode = TradingMode(config.get("trading_mode", "paper"))
        self.paper_trading_enabled = self.trading_mode == TradingMode.PAPER

        # Paper trading configuration
        paper_config = config.get("paper_trading_config", {})
        self.paper_trading_config = PaperTradingConfig(
            mode=PaperTradingMode(paper_config.get("mode", "realistic")),
            initial_balance=paper_config.get("initial_balance", 10000.0),
            commission_rate=paper_config.get("commission_rate", 0.001),
            slippage_model=paper_config.get("slippage_model", "linear"),
            base_slippage=paper_config.get("base_slippage", 0.0005),
            latency_simulation=paper_config.get("latency_simulation", True),
            min_latency_ms=paper_config.get("min_latency_ms", 10),
            max_latency_ms=paper_config.get("max_latency_ms", 100),
            market_impact_enabled=paper_config.get("market_impact_enabled", True),
            market_impact_factor=paper_config.get("market_impact_factor", 0.0001),
            realistic_fills=paper_config.get("realistic_fills", True),
            partial_fill_probability=paper_config.get("partial_fill_probability", 0.1),
            reject_probability=paper_config.get("reject_probability", 0.01),
            enable_execution_quality=paper_config.get("enable_execution_quality", True),
        )

        # Notification manager
        self.notification_manager = PositionNotificationManager(config, notification_client)

        # Backtrader-specific configuration
        backtrader_config = config.get("backtrader_config", {})
        self.backtrader_auto_process = backtrader_config.get("auto_process_orders", True)
        self.backtrader_notification_buffer = backtrader_config.get("notification_buffer_size", 100)

        # Current bar prices: populated by update_market_price() on every bar
        # (typically called by BacktraderBrokerBridge.next() or buy()/sell())
        # so that paper-trading order simulation uses real instrument prices
        # instead of the hardcoded $100 placeholder.
        self._current_bar_prices: Dict[str, float] = {}

        # Execution metrics tracking
        self.execution_metrics: List[ExecutionMetrics] = []
        self.total_executions = 0
        self.execution_quality_stats = {
            ExecutionQuality.EXCELLENT: 0,
            ExecutionQuality.GOOD: 0,
            ExecutionQuality.FAIR: 0,
            ExecutionQuality.POOR: 0,
        }

        # Paper trading state (if enabled)
        if self.paper_trading_enabled:
            self.paper_orders: Dict[str, Order] = {}
            self.paper_positions: Dict[str, Position] = {}
            self.paper_portfolio: Portfolio | None = None
            self.paper_trade_history: List[Dict[str, Any]] = []
            self.market_data_cache: Dict[str, Dict[str, Any]] = {}
            self._initialize_paper_portfolio()

        self._logger.info(
            "Base broker initialized - Mode: %s, Broker: %s, Paper Trading: %s, Backtrader: %s",
            self.trading_mode.value,
            self.name,
            self.paper_trading_enabled,
            self._backtrader_mode,
        )

    def _initialize_paper_portfolio(self) -> None:
        """Initialize paper trading portfolio with starting balance."""
        if not self.paper_trading_enabled:
            return

        config = self.paper_trading_config

        self.paper_portfolio = Portfolio(
            total_value=config.initial_balance,
            cash=config.initial_balance,
            positions={},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=datetime.now(UTC),
            paper_trading=True,
            initial_balance=config.initial_balance,
        )

        self._logger.info("Initialized paper trading portfolio with $%.2f", config.initial_balance)

    # Abstract methods that must be implemented by subclasses

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the broker."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order | None:
        """Get order status."""
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_portfolio(self) -> Portfolio:
        """Get portfolio information."""
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass

    # Base broker methods

    def get_name(self) -> str:
        """Get broker name."""
        return self.name

    def is_backtrader_mode(self) -> bool:
        """Check if this broker is running in backtrader mode."""
        return self._backtrader_mode

    def enable_backtrader_trading_mode(self) -> None:
        """
        Activate buy/sell/cancel/next adapters for Backtrader. Called by
        ``BacktraderBrokerBridge``; do not use for standalone live brokers.
        """
        self._backtrader_mode = True
        self._bt_notification_queue = []
        self._bt_processed_orders = {}

    def _bt_getcash(self) -> float:
        """Cash for Backtrader ``getcash`` (paper portfolio or configured initial)."""
        if self.paper_trading_enabled and self.paper_portfolio is not None:
            return float(self.paper_portfolio.cash)
        return float(self.paper_trading_config.initial_balance)

    def _bt_getvalue(self, datas=None) -> float:
        """Portfolio value for Backtrader ``getvalue``."""
        if self.paper_trading_enabled and self.paper_portfolio is not None:
            return float(self.paper_portfolio.total_value)
        return float(self.paper_trading_config.initial_balance)

    def _bt_getposition(self, data):
        """Build a ``backtrader.position.Position`` from paper state."""
        import backtrader as bt

        symbol = getattr(data, "_name", "UNKNOWN") if data else "UNKNOWN"
        if self.paper_trading_enabled and self.paper_portfolio and symbol in self.paper_portfolio.positions:
            pos = self.paper_portfolio.positions[symbol]
            return bt.position.Position(size=pos.quantity, price=pos.average_price)
        return bt.position.Position(size=0, price=0.0)

    def is_paper_trading(self) -> bool:
        """Check if this is a paper trading broker."""
        return self.paper_trading_enabled

    def get_trading_mode(self) -> TradingMode:
        """Get current trading mode."""
        return self.trading_mode

    def get_paper_trading_config(self) -> PaperTradingConfig:
        """Get paper trading configuration."""
        return self.paper_trading_config

    def update_paper_trading_config(self, config: PaperTradingConfig) -> None:
        """Update paper trading configuration."""
        self.paper_trading_config = config
        self._logger.info("Updated paper trading config: mode=%s", config.mode.value)

    async def simulate_execution_latency(self) -> None:
        """Simulate realistic execution latency for paper trading."""
        if not self.paper_trading_enabled or not self.paper_trading_config.latency_simulation:
            return

        latency_ms = random.randint(self.paper_trading_config.min_latency_ms, self.paper_trading_config.max_latency_ms)
        await asyncio.sleep(latency_ms / 1000.0)

    def calculate_slippage(self, order: Order, market_price: float) -> float:
        """Calculate realistic slippage for paper trading execution."""
        if not self.paper_trading_enabled:
            return 0.0

        base_slippage = self.paper_trading_config.base_slippage

        if self.paper_trading_config.slippage_model == "fixed":
            slippage_factor = base_slippage
        elif self.paper_trading_config.slippage_model == "sqrt":
            # Square root model - higher slippage for larger orders
            size_factor = math.sqrt(order.quantity / 100.0)  # Normalize by 100 shares
            slippage_factor = base_slippage * (1 + size_factor)
        else:  # linear model
            # Linear model - slippage increases linearly with order size
            size_factor = order.quantity / 1000.0  # Normalize by 1000 shares
            slippage_factor = base_slippage * (1 + size_factor)

        # Add market impact if enabled
        if self.paper_trading_config.market_impact_enabled:
            market_impact = self.paper_trading_config.market_impact_factor * order.quantity
            slippage_factor += market_impact

        # Add random component (±50% of calculated slippage)
        random_factor = random.uniform(0.5, 1.5)
        final_slippage = slippage_factor * random_factor

        # Apply slippage direction based on order side
        if order.side == OrderSide.BUY:
            return market_price * final_slippage  # Buy higher
        else:
            return -market_price * final_slippage  # Sell lower

    def calculate_execution_quality(self, slippage_bps: float) -> ExecutionQuality:
        """Calculate execution quality based on slippage."""
        abs_slippage = abs(slippage_bps)

        if abs_slippage < 10:  # < 0.1%
            return ExecutionQuality.EXCELLENT
        elif abs_slippage < 50:  # 0.1% - 0.5%
            return ExecutionQuality.GOOD
        elif abs_slippage < 100:  # 0.5% - 1.0%
            return ExecutionQuality.FAIR
        else:  # > 1.0%
            return ExecutionQuality.POOR

    def record_execution_metrics(
        self, order: Order, executed_price: float, executed_quantity: float, latency_ms: int
    ) -> ExecutionMetrics | None:
        """Record execution metrics for paper trading analysis."""
        if not self.paper_trading_enabled or not self.paper_trading_config.enable_execution_quality:
            return None

        # Calculate slippage in basis points
        if order.price and order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            slippage_bps = ((executed_price - order.price) / order.price) * 10000
            if order.side == OrderSide.SELL:
                slippage_bps = -slippage_bps  # Invert for sell orders
        else:
            slippage_bps = 0.0  # Market orders have no reference price

        execution_quality = self.calculate_execution_quality(slippage_bps)

        # Calculate market impact
        market_impact_bps = 0.0
        if self.paper_trading_config.market_impact_enabled:
            market_impact_bps = self.paper_trading_config.market_impact_factor * executed_quantity * 10000

        metrics = ExecutionMetrics(
            execution_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            executed_quantity=executed_quantity,
            requested_price=order.price,
            executed_price=executed_price,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            execution_quality=execution_quality,
            market_impact_bps=market_impact_bps,
            timestamp=datetime.now(UTC),
            broker_name=self.name,
            simulation_mode=self.paper_trading_config.mode,
        )

        # Update statistics
        self.execution_metrics.append(metrics)
        self.total_executions += 1
        self.execution_quality_stats[execution_quality] += 1

        # Add to order's execution metrics
        order.execution_metrics.append(metrics)

        self._logger.debug(
            "Recorded execution metrics: %s - Quality: %s, Slippage: %.2fbps",
            metrics.execution_id,
            execution_quality.value,
            slippage_bps,
        )

        return metrics

    def get_execution_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive execution quality report for paper trading."""
        if not self.execution_metrics:
            return {"error": "No execution metrics available"}

        total_executions = len(self.execution_metrics)
        avg_slippage = sum(abs(m.slippage_bps) for m in self.execution_metrics) / total_executions
        avg_latency = sum(m.latency_ms for m in self.execution_metrics) / total_executions

        quality_distribution = {
            quality.value: (count / total_executions) * 100 for quality, count in self.execution_quality_stats.items()
        }

        return {
            "total_executions": total_executions,
            "average_slippage_bps": round(avg_slippage, 2),
            "average_latency_ms": round(avg_latency, 1),
            "quality_distribution": quality_distribution,
            "paper_trading_mode": self.paper_trading_config.mode.value,
            "broker_name": self.name,
            "trading_mode": self.trading_mode.value,
            "report_timestamp": datetime.now(UTC).isoformat(),
        }

    async def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate order for both paper and live trading."""
        # Basic order validation
        if order.quantity <= 0:
            return False, "Invalid quantity: must be positive"

        if order.order_type == OrderType.LIMIT and order.price is None:
            return False, "Limit order requires price"

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return False, "Stop order requires stop price"

        # Paper trading specific validation
        if self.paper_trading_enabled:
            # Simulate order rejection probability
            if random.random() < self.paper_trading_config.reject_probability:
                return False, "Order rejected by paper trading simulation"

        return True, "Order validated successfully"

    async def notify_position_event(self, event_type: str, position_data: Dict[str, Any]):
        """Send position event notifications."""
        try:
            position_data["trading_mode"] = self.trading_mode.value
            position_data["broker_name"] = self.name

            if event_type == "opened":
                await self.notification_manager.notify_position_opened(position_data)
            elif event_type == "closed":
                await self.notification_manager.notify_position_closed(position_data)

        except Exception:
            self._logger.exception("Error sending position notification:")

    async def notify_error(self, error_message: str, context: Dict[str, Any] = None):
        """Send error notifications."""
        try:
            error_data = {
                "error_message": error_message,
                "trading_mode": self.trading_mode.value,
                "broker_name": self.name,
                "timestamp": datetime.now(UTC),
                **(context or {}),
            }

            await self.notification_manager.notify_error(error_data)

        except Exception:
            self._logger.exception("Error sending error notification:")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive broker status."""
        status = {
            "broker_name": self.name,
            "trading_mode": self.trading_mode.value,
            "paper_trading_enabled": self.paper_trading_enabled,
            "is_connected": self.is_connected,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if self.paper_trading_enabled:
            status.update(
                {
                    "paper_trading_config": {
                        "mode": self.paper_trading_config.mode.value,
                        "initial_balance": self.paper_trading_config.initial_balance,
                        "commission_rate": self.paper_trading_config.commission_rate,
                    },
                    "execution_stats": {
                        "total_executions": self.total_executions,
                        "quality_distribution": {
                            quality.value: count for quality, count in self.execution_quality_stats.items()
                        },
                    },
                }
            )

        return status

    # Backtrader Interface Adapter Methods
    # These methods are only available when inheriting from bt.Broker

    def buy(
        self,
        owner=None,
        data=None,
        size=None,
        price=None,
        plimit=None,
        exectype=None,
        valid=None,
        tradeid=0,
        oco=None,
        trailamount=None,
        trailpercent=None,
        parent=None,
        transmit=True,
        **kwargs,
    ):
        """
        Backtrader buy method adapter.

        Converts backtrader buy parameters to our internal Order format.
        Delegates to :meth:`_create_bt_order`.

        Args:
            owner: Strategy instance (backtrader specific)
            data: Data feed (backtrader specific)
            size: Order size (quantity)
            price: Limit price (for limit orders)
            plimit: Price limit (for stop-limit orders)
            exectype: Execution type (Market, Limit, Stop, etc.)
            valid: Order validity
            tradeid: Trade ID
            oco: One-Cancels-Other order
            trailamount: Trailing stop amount
            trailpercent: Trailing stop percentage
            parent: Parent order
            transmit: Whether to transmit immediately
            **kwargs: Additional parameters

        Returns:
            Order object compatible with backtrader
        """
        return self._create_bt_order(
            OrderSide.BUY,
            owner=owner,
            data=data,
            size=size,
            price=price,
            plimit=plimit,
            exectype=exectype,
            valid=valid,
            tradeid=tradeid,
            oco=oco,
            trailamount=trailamount,
            trailpercent=trailpercent,
            parent=parent,
            transmit=transmit,
            **kwargs,
        )

    def _create_bt_order(
        self,
        side: OrderSide,
        owner=None,
        data=None,
        size=None,
        price=None,
        plimit=None,
        exectype=None,
        valid=None,
        tradeid=0,
        oco=None,
        trailamount=None,
        trailpercent=None,
        parent=None,
        transmit=True,
        **kwargs,
    ) -> Order:
        """
        Shared implementation for backtrader ``buy()`` / ``sell()`` adapters.

        Validates parameters, converts backtrader types to internal types, creates
        an :class:`Order` object, and either queues it (paper) or marks it
        ``QUEUED`` for async processing (live).

        Args:
            side: ``OrderSide.BUY`` or ``OrderSide.SELL``.
            owner: Strategy instance (backtrader specific).
            data: Data feed (backtrader specific).
            size: Order size (quantity).
            price: Limit price (for limit orders).
            plimit: Price limit (for stop-limit orders).
            exectype: Execution type (Market, Limit, Stop, etc.).
            valid: Order validity.
            tradeid: Trade ID.
            oco: One-Cancels-Other order.
            trailamount: Trailing stop amount.
            trailpercent: Trailing stop percentage.
            parent: Parent order.
            transmit: Whether to transmit immediately.
            **kwargs: Additional parameters forwarded to order metadata.

        Returns:
            :class:`Order` object compatible with backtrader.
        """
        side_str = "buy" if side == OrderSide.BUY else "sell"

        if not self._backtrader_mode:
            raise RuntimeError(
                f"{side_str}() method only available in backtrader mode. "
                "Ensure backtrader is installed and the broker is initialized in backtrader context."
            )

        # Log the operation
        self._log_backtrader_operation(side_str, size=size, price=price, exectype=exectype)

        # Validate parameters
        is_valid, error_msg = self._validate_backtrader_order_params(size, price, exectype)
        if not is_valid:
            self._logger.error("Invalid %s order parameters: %s", side_str, error_msg)
            return Order(
                symbol="INVALID",
                side=side,
                order_type=OrderType.MARKET,
                quantity=0.0,
                status=OrderStatus.REJECTED,
                metadata={"error": error_msg},
            )

        # Convert backtrader exectype to our OrderType
        order_type = self._convert_bt_exectype_to_order_type(exectype)

        # Get symbol from data feed
        symbol = getattr(data, "_name", "UNKNOWN") if data else "UNKNOWN"

        # Capture the current bar close price so that _get_simulated_market_price()
        # returns a real price instead of the hardcoded $100 placeholder.
        if data is not None and hasattr(data, "close"):
            try:
                self.update_market_price(symbol, float(data.close[0]))
            except (IndexError, TypeError, AttributeError):
                pass

        # Create our internal Order object
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(size) if size else 0.0,
            price=float(price) if price else None,
            stop_price=float(plimit) if plimit else None,
            time_in_force=self._convert_bt_valid_to_tif(valid),
            metadata={
                "backtrader_owner": owner,
                "backtrader_data": data,
                "tradeid": tradeid,
                "oco": oco,
                "trailamount": trailamount,
                "trailpercent": trailpercent,
                "parent": parent,
                "transmit": transmit,
                **kwargs,
            },
        )

        try:
            if self.paper_trading_enabled:
                # In paper trading mode, process immediately
                self.paper_orders[order.order_id] = order
                order.status = OrderStatus.PENDING
                self._logger.info(
                    "Backtrader %s order created: %s %s @ %s",
                    side_str,
                    size,
                    symbol,
                    price or "MARKET",
                )
                return order
            else:
                # For live trading, queue for async processing
                order.status = OrderStatus.QUEUED
                self._logger.info(
                    "Backtrader %s order queued: %s %s @ %s",
                    side_str,
                    size,
                    symbol,
                    price or "MARKET",
                )
                return order

        except Exception as e:
            self._logger.exception("Error creating backtrader %s order: %s", side_str, e)
            order.status = OrderStatus.REJECTED
            return order

    def _convert_bt_exectype_to_order_type(self, exectype) -> OrderType:
        """Convert backtrader execution type to our OrderType enum."""
        if not BACKTRADER_AVAILABLE:
            return OrderType.MARKET

        # Import backtrader order types if available
        try:
            import backtrader as bt

            if exectype is None or exectype == bt.Order.Market:
                return OrderType.MARKET
            elif exectype == bt.Order.Limit:
                return OrderType.LIMIT
            elif exectype == bt.Order.Stop:
                return OrderType.STOP
            elif exectype == bt.Order.StopLimit:
                return OrderType.STOP_LIMIT
            else:
                self._logger.warning("Unknown backtrader exectype %s, defaulting to MARKET", exectype)
                return OrderType.MARKET
        except (ImportError, AttributeError):
            return OrderType.MARKET

    def _convert_bt_valid_to_tif(self, valid) -> str:
        """Convert backtrader validity to time-in-force."""
        if valid is None:
            return "GTC"  # Good Till Cancelled

        # Handle different backtrader validity types
        if hasattr(valid, "days"):
            # It's a timedelta or similar
            return "GTD"  # Good Till Date
        elif isinstance(valid, (int, float)):
            # It's a number of bars
            return "GTD"
        else:
            return "GTC"

    def sell(
        self,
        owner=None,
        data=None,
        size=None,
        price=None,
        plimit=None,
        exectype=None,
        valid=None,
        tradeid=0,
        oco=None,
        trailamount=None,
        trailpercent=None,
        parent=None,
        transmit=True,
        **kwargs,
    ):
        """
        Backtrader sell method adapter.

        Converts backtrader sell parameters to our internal Order format.
        Delegates to :meth:`_create_bt_order`.

        Args:
            owner: Strategy instance (backtrader specific)
            data: Data feed (backtrader specific)
            size: Order size (quantity)
            price: Limit price (for limit orders)
            plimit: Price limit (for stop-limit orders)
            exectype: Execution type (Market, Limit, Stop, etc.)
            valid: Order validity
            tradeid: Trade ID
            oco: One-Cancels-Other order
            trailamount: Trailing stop amount
            trailpercent: Trailing stop percentage
            parent: Parent order
            transmit: Whether to transmit immediately
            **kwargs: Additional parameters

        Returns:
            Order object compatible with backtrader
        """
        return self._create_bt_order(
            OrderSide.SELL,
            owner=owner,
            data=data,
            size=size,
            price=price,
            plimit=plimit,
            exectype=exectype,
            valid=valid,
            tradeid=tradeid,
            oco=oco,
            trailamount=trailamount,
            trailpercent=trailpercent,
            parent=parent,
            transmit=transmit,
            **kwargs,
        )

    def cancel(self, order):
        """
        Backtrader cancel method adapter.

        Converts backtrader order cancellation to our internal cancel_order method.

        Args:
            order: Order object to cancel (backtrader format or our format)

        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        if not self._backtrader_mode:
            raise RuntimeError(
                "cancel() method only available in backtrader mode. "
                "Ensure backtrader is installed and the broker is initialized in backtrader context."
            )

        try:
            # Extract order ID from the order object
            if hasattr(order, "order_id"):
                # It's our Order object
                order_id = order.order_id
            elif hasattr(order, "ref"):
                # It's a backtrader order object
                order_id = str(order.ref)
            else:
                # Try to use the object itself as ID
                order_id = str(order)

            # Cancel using our existing method
            if self.paper_trading_enabled:
                # In paper trading mode, cancel immediately
                if order_id in self.paper_orders:
                    self.paper_orders[order_id].status = OrderStatus.CANCELLED
                    self._logger.info("Backtrader order cancelled: %s", order_id)
                    return True
                else:
                    self._logger.warning("Backtrader order not found for cancellation: %s", order_id)
                    return False
            else:
                # For live trading, we would call the async cancel_order method
                # For now, we'll mark it as cancelled and handle async processing later
                self._logger.info("Backtrader order cancellation queued: %s", order_id)
                return True

        except Exception as e:
            self._logger.exception("Error cancelling backtrader order: %s", e)
            return False

    def get_notification(self):
        """
        Backtrader notification method.

        Returns order status updates and other notifications for backtrader.
        This method is called by backtrader to get order status updates.

        Returns:
            Order object with updated status or None if no notifications
        """
        if not self._backtrader_mode:
            return None

        try:
            # Return notifications from the queue
            if self._bt_notification_queue:
                order = self._bt_notification_queue.pop(0)
                self._logger.debug("Backtrader notification: order %s status %s", order.order_id, order.status.value)
                return BacktraderOrderAdapter(order)

            # No notifications available
            return None

        except Exception as e:
            self._logger.exception("Error getting backtrader notification: %s", e)
            return None

    def next(self):
        """
        Backtrader next method for processing.

        This method is called by backtrader on each bar/tick to process
        pending orders and update positions.
        """
        if not self._backtrader_mode:
            return

        try:
            # Process pending orders in paper trading mode
            if self.paper_trading_enabled:
                self._process_pending_backtrader_orders()

            # Update positions and portfolio
            self._update_backtrader_positions()

        except Exception as e:
            self._logger.exception("Error in backtrader next() processing: %s", e)

    def _process_pending_backtrader_orders(self):
        """Process pending orders in paper trading mode for backtrader."""
        for order_id, order in list(self.paper_orders.items()):
            # Re-process PARTIALLY_FILLED orders so the unfilled remainder can
            # complete on subsequent bars rather than being silently dropped.
            if order.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED):
                try:
                    # Enhanced paper trading simulation for backtrader
                    executed = self._simulate_backtrader_order_execution(order)

                    if executed:
                        # Add to notification queue for backtrader
                        if self._backtrader_mode:
                            self._bt_notification_queue.append(order)
                            self._bt_processed_orders[order_id] = order

                        # Update paper trading portfolio
                        self._update_paper_portfolio_from_order(order)

                except Exception as e:
                    self._logger.exception("Error processing backtrader order %s: %s", order_id, e)
                    order.status = OrderStatus.REJECTED

                    # Add rejected order to notification queue
                    if self._backtrader_mode:
                        self._bt_notification_queue.append(order)

    def _simulate_backtrader_order_execution(self, order: Order) -> bool:
        """
        Enhanced order execution simulation for backtrader integration.

        Returns:
            bool: True if order was executed, False otherwise
        """
        try:
            # Simulate execution latency if enabled
            if self.paper_trading_config.latency_simulation:
                latency_ms = random.randint(
                    self.paper_trading_config.min_latency_ms, self.paper_trading_config.max_latency_ms
                )
            else:
                latency_ms = 0

            # Get market price (in real implementation, this would come from backtrader data feed)
            market_price = self._get_simulated_market_price(order.symbol)

            # Check if order should be executed based on type
            should_execute = False
            executed_price = market_price

            if order.order_type == OrderType.MARKET:
                should_execute = True
                # Apply slippage for market orders
                slippage = self.calculate_slippage(order, market_price)
                executed_price = market_price + slippage

            elif order.order_type == OrderType.LIMIT:
                # Limit order execution logic
                if order.side == OrderSide.BUY and market_price <= order.price:
                    should_execute = True
                    executed_price = order.price
                elif order.side == OrderSide.SELL and market_price >= order.price:
                    should_execute = True
                    executed_price = order.price

            elif order.order_type == OrderType.STOP:
                # Stop order execution logic
                if order.side == OrderSide.BUY and market_price >= order.stop_price:
                    should_execute = True
                    executed_price = market_price
                elif order.side == OrderSide.SELL and market_price <= order.stop_price:
                    should_execute = True
                    executed_price = market_price

            # Simulate partial fills if enabled.
            # NOTE: should_execute is checked first so that a limit/stop order
            # whose price condition is NOT met is never mis-tagged as
            # PARTIALLY_FILLED just because the random roll happened to pass.
            executed_quantity = order.quantity
            if (
                should_execute
                and self.paper_trading_config.realistic_fills
                and random.random() < self.paper_trading_config.partial_fill_probability
            ):
                executed_quantity = order.quantity * random.uniform(0.5, 0.9)
                order.status = OrderStatus.PARTIALLY_FILLED
            elif should_execute:
                order.status = OrderStatus.FILLED

            if should_execute:
                # Execute the order
                order.filled_quantity += executed_quantity
                order.average_price = executed_price

                # Calculate commission
                commission = executed_quantity * executed_price * self.paper_trading_config.commission_rate
                order.commission += commission

                self._logger.info(
                    "Backtrader paper order executed: %s %s @ %s (commission: %s)",
                    executed_quantity,
                    order.symbol,
                    executed_price,
                    commission,
                )

                # Record execution metrics
                self.record_execution_metrics(order, executed_price, executed_quantity, latency_ms)

                return True

            return False

        except Exception as e:
            self._logger.exception("Error simulating order execution: %s", e)
            order.status = OrderStatus.REJECTED
            return False

    def update_market_price(self, symbol: str, price: float) -> None:
        """Store the current bar close price for a symbol.

        Call this on every Backtrader bar (e.g. from ``BacktraderBrokerBridge.next()``
        or from the ``buy()`` / ``sell()`` adapters) so that paper-trading order
        simulation uses the real instrument price instead of the hardcoded $100
        placeholder.

        Args:
            symbol: Instrument symbol (e.g. 'BTCUSDT').
            price:  Current bar close price (must be > 0).
        """
        if price > 0:
            self._current_bar_prices[symbol] = float(price)

    def _get_simulated_market_price(self, symbol: str) -> float:
        """Return the current market price for *symbol*.

        Priority:
        1. Price stored via :meth:`update_market_price` (real bar close from
           Backtrader or a live data feed).
        2. Noisy random walk around the last known price — only used as a last
           resort and triggers a warning so misconfiguration is visible.

        Returns:
            float: Current market price with a tiny spread noise (±0.02 %).
        """
        last_price = self._current_bar_prices.get(symbol)
        if last_price and last_price > 0:
            # Add tiny Gaussian noise (±0.02 %) to simulate bid/ask spread
            # without drifting away from the real bar close price.
            noise = random.gauss(0, 0.0002)
            return last_price * (1.0 + noise)

        # No real price available — warn loudly so the mis-configuration is
        # caught early; results using this path will be inaccurate.
        self._logger.warning(
            "No real market price stored for '%s'. "
            "Call update_market_price() on each bar (e.g. from "
            "BacktraderBrokerBridge.next()) to avoid using the $100 placeholder. "
            "Paper-trading results will be inaccurate.",
            symbol,
        )
        base_price = 100.0
        volatility = 0.02
        price_change = random.gauss(0, volatility)
        return base_price * (1.0 + price_change)

    def _update_paper_portfolio_from_order(self, order: Order):
        """Update paper trading portfolio based on executed order."""
        if not self.paper_trading_enabled or not self.paper_portfolio:
            return

        try:
            symbol = order.symbol
            executed_qty = order.filled_quantity
            executed_price = order.average_price
            commission = order.commission

            # Update cash
            if order.side == OrderSide.BUY:
                cash_change = -(executed_qty * executed_price + commission)
            else:
                cash_change = executed_qty * executed_price - commission

            self.paper_portfolio.cash += cash_change
            self.paper_portfolio.total_commission += commission

            # Update position
            if symbol in self.paper_portfolio.positions:
                position = self.paper_portfolio.positions[symbol]

                if order.side == OrderSide.BUY:
                    # Add to position
                    total_cost = position.quantity * position.average_price + executed_qty * executed_price
                    total_qty = position.quantity + executed_qty
                    position.average_price = total_cost / total_qty if total_qty > 0 else 0
                    position.quantity = total_qty
                else:
                    # Reduce position — capture closed qty BEFORE subtracting to compute PnL correctly.
                    # Using position.quantity after subtraction would give 0 when the position is fully
                    # closed, causing realized_pnl to always be $0.
                    qty_closed = min(executed_qty, position.quantity)
                    realized_pnl = (executed_price - position.average_price) * qty_closed
                    self.paper_portfolio.realized_pnl += realized_pnl
                    position.quantity -= qty_closed
                    if position.quantity <= 0:
                        del self.paper_portfolio.positions[symbol]

            else:
                # New position (no existing entry for this symbol)
                if order.side == OrderSide.BUY:
                    position = Position(
                        symbol=symbol,
                        quantity=executed_qty,
                        average_price=executed_price,
                        market_value=executed_qty * executed_price,
                        unrealized_pnl=0.0,
                        paper_trading=True,
                        commission_paid=commission,
                    )
                    self.paper_portfolio.positions[symbol] = position
                else:
                    # Short sell: no long position to close → open a short (negative quantity).
                    # Cash is already credited above; record a negative-qty position so that
                    # a subsequent BUY can close it and compute realized PnL correctly.
                    position = Position(
                        symbol=symbol,
                        quantity=-executed_qty,
                        average_price=executed_price,
                        market_value=-(executed_qty * executed_price),
                        unrealized_pnl=0.0,
                        paper_trading=True,
                        commission_paid=commission,
                    )
                    self.paper_portfolio.positions[symbol] = position
                    self._logger.debug(
                        "Short position opened for %s: qty=%.6f @ %.4f", symbol, executed_qty, executed_price
                    )

            # Update portfolio totals
            self._recalculate_paper_portfolio_value()

        except Exception as e:
            self._logger.exception("Error updating paper portfolio: %s", e)

    def _recalculate_paper_portfolio_value(self):
        """Recalculate total portfolio value."""
        if not self.paper_portfolio:
            return

        total_position_value = 0.0
        total_unrealized_pnl = 0.0

        for position in self.paper_portfolio.positions.values():
            # In real implementation, use current market prices
            current_price = self._get_simulated_market_price(position.symbol)
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.average_price) * position.quantity

            total_position_value += position.market_value
            total_unrealized_pnl += position.unrealized_pnl

        self.paper_portfolio.total_value = self.paper_portfolio.cash + total_position_value
        self.paper_portfolio.unrealized_pnl = total_unrealized_pnl

    def _validate_backtrader_order_params(self, size, price=None, exectype=None) -> Tuple[bool, str]:
        """
        Validate backtrader order parameters.

        Args:
            size: Order size
            price: Order price (optional)
            exectype: Execution type (optional)

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Validate size
            if size is None or size <= 0:
                return False, f"Invalid order size: {size}. Size must be positive."

            # Validate price for limit orders
            if BACKTRADER_AVAILABLE:
                import backtrader as bt

                if exectype == bt.Order.Limit and (price is None or price <= 0):
                    return False, f"Invalid price for limit order: {price}. Price must be positive."

            # Validate numeric types
            try:
                float(size)
                if price is not None:
                    float(price)
            except (ValueError, TypeError) as e:
                return False, f"Invalid numeric parameter: {e}"

            return True, "Parameters valid"

        except Exception as e:
            return False, f"Parameter validation error: {e}"

    def _log_backtrader_operation(self, operation: str, **kwargs):
        """
        Log backtrader operations for debugging and monitoring.

        Args:
            operation: Name of the operation
            **kwargs: Operation parameters
        """
        try:
            params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
            self._logger.debug("Backtrader operation: %s(%s)", operation, params_str)

        except Exception as e:
            self._logger.warning("Error logging backtrader operation: %s", e)

    def _update_backtrader_positions(self):
        """Update positions for backtrader compatibility."""
        try:
            # Update paper trading positions if enabled
            if self.paper_trading_enabled and self.paper_portfolio:
                # Calculate unrealized P&L and update portfolio
                # This would typically use current market prices from backtrader data feeds
                pass

        except Exception as e:
            self._logger.exception("Error updating backtrader positions: %s", e)
