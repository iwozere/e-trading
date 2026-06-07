"""
Trading broker domain models.

Enums and dataclasses shared across all broker implementations, the paper-trading
engine, and the Backtrader adapter layer.  Keeping them here avoids the circular
imports that arise when the adapter classes live in the same file as BaseBroker.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


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
