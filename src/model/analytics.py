"""
Models for analytics and performance tracking.

Includes:
- Trade data structure
- Performance metrics (returns, drawdown, Sharpe, win rate, etc.)
"""
from dataclasses import dataclass, field
from datetime import datetime,timedelta

@dataclass
class Trade:
    """Trade data structure"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    net_pnl: float
    exit_reason: str = "unknown"

    @property
    def duration(self) -> timedelta:
        """Calculate trade duration"""
        return self.exit_time - self.entry_time

    @property
    def return_pct(self) -> float:
        """Calculate percentage return"""
        if self.side.upper() == "BUY":
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Advanced metrics
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional Value at Risk (95%)
    kelly_criterion: float = 0.0
    expectancy: float = 0.0

    # Trade analysis
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: timedelta = timedelta()

    # Consecutive analysis
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Time analysis
    avg_trades_per_day: float = 0.0
    best_day: datetime = field(default_factory=datetime.now)
    worst_day: datetime = field(default_factory=datetime.now)

    # Additional metrics
    recovery_factor: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor_ratio: float = 0.0
