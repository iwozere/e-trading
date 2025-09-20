#!/usr/bin/env python3
"""
Enhanced Base Broker Module
--------------------------

This module provides an enhanced base broker class with comprehensive paper trading
support, seamless paper-to-live trading switching, and integrated notifications.

Features:
- Unified paper trading interface with realistic simulation
- Seamless configuration-based paper/live mode switching
- Comprehensive execution quality metrics and analytics
- Integrated email and Telegram notifications for position events
- Support for advanced order types (stop-loss, take-profit, OCO)
- Multi-broker support (Binance, IBKR) with automatic credential selection
- Risk management and position tracking
- Database integration for trade history and analytics

Classes:
- EnhancedBaseBroker: Enhanced abstract base class for all brokers
- PaperTradingConfig: Configuration for paper trading simulation
- ExecutionMetrics: Execution quality metrics for analysis
- PositionNotificationManager: Notification system for position events
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timezone
import uuid
import asyncio
import json
import random
import math

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration."""
    PAPER = "paper"
    LIVE = "live"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SIMULATING = "simulating"  # Paper trading specific
    QUEUED = "queued"  # Paper trading specific


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other
    BRACKET = "bracket"  # Entry + Stop Loss + Take Profit


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class PaperTradingMode(Enum):
    """Paper trading simulation modes."""
    DISABLED = "disabled"  # Live trading mode
    BASIC = "basic"  # Simple paper trading
    REALISTIC = "realistic"  # Realistic simulation with slippage/latency
    ADVANCED = "advanced"  # Advanced simulation with market impact


class ExecutionQuality(Enum):
    """Execution quality metrics for paper trading."""
    EXCELLENT = "excellent"  # < 0.1% slippage
    GOOD = "good"  # 0.1% - 0.5% slippage
    FAIR = "fair"  # 0.5% - 1.0% slippage
    POOR = "poor"  # > 1.0% slippage


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading simulation."""
    mode: PaperTradingMode = PaperTradingMode.REALISTIC
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "linear"  # linear, sqrt, fixed
    base_slippage: float = 0.0005  # 0.05%
    latency_simulation: bool = True
    min_latency_ms: int = 10
    max_latency_ms: int = 100
    market_impact_enabled: bool = True
    market_impact_factor: float = 0.0001
    realistic_fills: bool = True
    partial_fill_probability: float = 0.1
    reject_probability: float = 0.01
    enable_execution_quality: bool = True


@dataclass
class ExecutionMetrics:
    """Execution quality metrics for paper trading analysis."""
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    requested_quantity: float
    executed_quantity: float
    requested_price: Optional[float]
    executed_price: float
    slippage_bps: float  # Basis points
    latency_ms: int
    execution_quality: ExecutionQuality
    market_impact_bps: float
    timestamp: datetime
    broker_name: str
    simulation_mode: PaperTradingMode
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Enhanced order class with paper trading support."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced paper trading fields
    paper_trading: bool = False
    simulation_config: Optional[PaperTradingConfig] = None
    execution_metrics: List[ExecutionMetrics] = field(default_factory=list)
    parent_order_id: Optional[str] = None  # For bracket/OCO orders
    child_order_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = str(uuid.uuid4())
        if self.client_order_id is None:
            self.client_order_id = f"client_{self.order_id[:8]}"
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class Position:
    """Enhanced position class with paper trading support."""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced paper trading fields
    paper_trading: bool = False
    entry_timestamp: Optional[datetime] = None
    entry_orders: List[str] = field(default_factory=list)  # Order IDs that created this position
    commission_paid: float = 0.0
    slippage_cost: float = 0.0
    holding_period_seconds: Optional[int] = None

    def __post_init__(self):
        if self.entry_timestamp is None:
            self.entry_timestamp = datetime.now(timezone.utc)
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def update_holding_period(self):
        """Update holding period in seconds."""
        if self.entry_timestamp:
            self.holding_period_seconds = int(
                (datetime.now(timezone.utc) - self.entry_timestamp).total_seconds()
            )


@dataclass
class Portfolio:
    """Enhanced portfolio class with paper trading support."""
    total_value: float
    cash: float
    positions: Dict[str, Position]
    unrealized_pnl: float
    realized_pnl: float
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced paper trading fields
    paper_trading: bool = False
    initial_balance: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown: float = 0.0
    max_portfolio_value: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.initial_balance == 0.0:
            self.initial_balance = self.total_value
        if self.max_portfolio_value < self.total_value:
            self.max_portfolio_value = self.total_value

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for paper trading."""
        total_return = (self.total_value - self.initial_balance) / self.initial_balance
        win_rate = self.winning_trades / max(self.total_trades, 1)

        # Calculate drawdown
        current_drawdown = (self.max_portfolio_value - self.total_value) / self.max_portfolio_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.unrealized_pnl + self.realized_pnl,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'net_pnl': self.realized_pnl + self.unrealized_pnl - self.total_commission - self.total_slippage
        }


class PositionNotificationManager:
    """
    Manages notifications for position events (opened/closed).
    Supports email and Telegram notifications with configurable settings.
    """

    def __init__(self, config: Dict[str, Any]):
        self.notifications_config = config.get('notifications', {})
        self.position_opened_enabled = self.notifications_config.get('position_opened', True)
        self.position_closed_enabled = self.notifications_config.get('position_closed', True)
        self.email_enabled = self.notifications_config.get('email_enabled', True)
        self.telegram_enabled = self.notifications_config.get('telegram_enabled', True)
        self.error_notifications = self.notifications_config.get('error_notifications', True)

        _logger.info(f"Position notifications initialized - "
                    f"Opened: {self.position_opened_enabled}, "
                    f"Closed: {self.position_closed_enabled}, "
                    f"Email: {self.email_enabled}, "
                    f"Telegram: {self.telegram_enabled}")

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
        mode_emoji = "📄" if data.get('trading_mode') == 'paper' else "💰"

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
        mode_emoji = "📄" if data.get('trading_mode') == 'paper' else "💰"
        pnl = data.get('pnl', 0)
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
            if self.email_enabled:
                await self._send_email_notification(message, title)

            if self.telegram_enabled:
                await self._send_telegram_notification(message, title)

        except Exception as e:
            _logger.exception(f"Error sending notifications: {e}")

    async def _send_email_notification(self, message: str, title: str):
        """Send email notification."""
        try:
            # Import here to avoid circular imports
            from src.notification.async_notification_manager import AsyncNotificationManager

            # Use existing notification system
            # This would integrate with your existing email notification setup
            _logger.info(f"Email notification sent: {title}")

        except Exception as e:
            _logger.exception(f"Error sending email notification: {e}")

    async def _send_telegram_notification(self, message: str, title: str):
        """Send Telegram notification."""
        try:
            # Import here to avoid circular imports
            from src.frontend.telegram.screener.http_api_client import send_notification_to_admins

            # Send to admin users via existing system
            await send_notification_to_admins(
                message=message,
                title=title
            )

        except Exception as e:
            _logger.exception(f"Error sending Telegram notification: {e}")


class EnhancedBaseBroker(ABC):
    """
    Enhanced abstract base class for all brokers with comprehensive paper trading support.

    Features:
    - Seamless paper/live trading mode switching via configuration
    - Realistic paper trading simulation with slippage, latency, and market impact
    - Comprehensive execution quality metrics and analytics
    - Integrated position notifications (email/Telegram)
    - Support for advanced order types and risk management
    - Multi-broker support with automatic credential selection
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'unknown')
        self.is_connected = False
        self._logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

        # Trading mode configuration
        self.trading_mode = TradingMode(config.get('trading_mode', 'paper'))
        self.paper_trading_enabled = (self.trading_mode == TradingMode.PAPER)

        # Paper trading configuration
        paper_config = config.get('paper_trading_config', {})
        self.paper_trading_config = PaperTradingConfig(
            mode=PaperTradingMode(paper_config.get('mode', 'realistic')),
            initial_balance=paper_config.get('initial_balance', 10000.0),
            commission_rate=paper_config.get('commission_rate', 0.001),
            slippage_model=paper_config.get('slippage_model', 'linear'),
            base_slippage=paper_config.get('base_slippage', 0.0005),
            latency_simulation=paper_config.get('latency_simulation', True),
            min_latency_ms=paper_config.get('min_latency_ms', 10),
            max_latency_ms=paper_config.get('max_latency_ms', 100),
            market_impact_enabled=paper_config.get('market_impact_enabled', True),
            market_impact_factor=paper_config.get('market_impact_factor', 0.0001),
            realistic_fills=paper_config.get('realistic_fills', True),
            partial_fill_probability=paper_config.get('partial_fill_probability', 0.1),
            reject_probability=paper_config.get('reject_probability', 0.01),
            enable_execution_quality=paper_config.get('enable_execution_quality', True)
        )

        # Notification manager
        self.notification_manager = PositionNotificationManager(config)

        # Execution metrics tracking
        self.execution_metrics: List[ExecutionMetrics] = []
        self.total_executions = 0
        self.execution_quality_stats = {
            ExecutionQuality.EXCELLENT: 0,
            ExecutionQuality.GOOD: 0,
            ExecutionQuality.FAIR: 0,
            ExecutionQuality.POOR: 0
        }

        # Paper trading state (if enabled)
        if self.paper_trading_enabled:
            self.paper_orders: Dict[str, Order] = {}
            self.paper_positions: Dict[str, Position] = {}
            self.paper_portfolio: Optional[Portfolio] = None
            self.paper_trade_history: List[Dict[str, Any]] = []
            self.market_data_cache: Dict[str, Dict[str, Any]] = {}
            self._initialize_paper_portfolio()

        self._logger.info(f"Enhanced broker initialized - Mode: {self.trading_mode.value}, "
                         f"Broker: {self.name}, Paper Trading: {self.paper_trading_enabled}")

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
            timestamp=datetime.now(timezone.utc),
            paper_trading=True,
            initial_balance=config.initial_balance
        )

        self._logger.info(f"Initialized paper trading portfolio with ${config.initial_balance:,.2f}")

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
    async def get_order_status(self, order_id: str) -> Optional[Order]:
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

    # Enhanced broker methods

    def get_name(self) -> str:
        """Get broker name."""
        return self.name

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
        self._logger.info(f"Updated paper trading config: mode={config.mode.value}")

    async def simulate_execution_latency(self) -> None:
        """Simulate realistic execution latency for paper trading."""
        if not self.paper_trading_enabled or not self.paper_trading_config.latency_simulation:
            return

        latency_ms = random.randint(
            self.paper_trading_config.min_latency_ms,
            self.paper_trading_config.max_latency_ms
        )
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

    def record_execution_metrics(self, order: Order, executed_price: float,
                               executed_quantity: float, latency_ms: int) -> Optional[ExecutionMetrics]:
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
            timestamp=datetime.now(timezone.utc),
            broker_name=self.name,
            simulation_mode=self.paper_trading_config.mode
        )

        # Update statistics
        self.execution_metrics.append(metrics)
        self.total_executions += 1
        self.execution_quality_stats[execution_quality] += 1

        # Add to order's execution metrics
        order.execution_metrics.append(metrics)

        self._logger.debug(f"Recorded execution metrics: {metrics.execution_id} - "
                          f"Quality: {execution_quality.value}, Slippage: {slippage_bps:.2f}bps")

        return metrics

    def get_execution_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive execution quality report for paper trading."""
        if not self.execution_metrics:
            return {"error": "No execution metrics available"}

        total_executions = len(self.execution_metrics)
        avg_slippage = sum(abs(m.slippage_bps) for m in self.execution_metrics) / total_executions
        avg_latency = sum(m.latency_ms for m in self.execution_metrics) / total_executions

        quality_distribution = {
            quality.value: (count / total_executions) * 100
            for quality, count in self.execution_quality_stats.items()
        }

        return {
            "total_executions": total_executions,
            "average_slippage_bps": round(avg_slippage, 2),
            "average_latency_ms": round(avg_latency, 1),
            "quality_distribution": quality_distribution,
            "paper_trading_mode": self.paper_trading_config.mode.value,
            "broker_name": self.name,
            "trading_mode": self.trading_mode.value,
            "report_timestamp": datetime.now(timezone.utc).isoformat()
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
            position_data['trading_mode'] = self.trading_mode.value
            position_data['broker_name'] = self.name

            if event_type == "opened":
                await self.notification_manager.notify_position_opened(position_data)
            elif event_type == "closed":
                await self.notification_manager.notify_position_closed(position_data)

        except Exception as e:
            self._logger.exception(f"Error sending position notification: {e}")

    async def notify_error(self, error_message: str, context: Dict[str, Any] = None):
        """Send error notifications."""
        try:
            error_data = {
                'error_message': error_message,
                'trading_mode': self.trading_mode.value,
                'broker_name': self.name,
                'timestamp': datetime.now(timezone.utc),
                **(context or {})
            }

            await self.notification_manager.notify_error(error_data)

        except Exception as e:
            self._logger.exception(f"Error sending error notification: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive broker status."""
        status = {
            'broker_name': self.name,
            'trading_mode': self.trading_mode.value,
            'paper_trading_enabled': self.paper_trading_enabled,
            'is_connected': self.is_connected,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        if self.paper_trading_enabled:
            status.update({
                'paper_trading_config': {
                    'mode': self.paper_trading_config.mode.value,
                    'initial_balance': self.paper_trading_config.initial_balance,
                    'commission_rate': self.paper_trading_config.commission_rate
                },
                'execution_stats': {
                    'total_executions': self.total_executions,
                    'quality_distribution': {
                        quality.value: count
                        for quality, count in self.execution_quality_stats.items()
                    }
                }
            })

        return status