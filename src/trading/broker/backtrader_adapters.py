"""
Backtrader-compatible adapter classes for the broker domain models.

These adapters wrap Order / Position / Portfolio objects so that backtrader's
Cerebro engine can consume them without depending on backtrader being installed.
They are extracted from base_broker.py to keep the live-execution interface
small and independently reviewable.
"""

from src.trading.broker.backtrader_availability import BACKTRADER_AVAILABLE
from src.trading.broker.models import Order, OrderStatus, Portfolio, Position


class BacktraderOrderAdapter:
    """
    Adapter to make our Order objects compatible with backtrader.

    This class provides a backtrader-compatible interface for our internal Order objects,
    mapping properties and methods to match backtrader's expectations.
    """

    def __init__(self, order: Order):
        self._order = order

    @property
    def ref(self):
        """Backtrader order reference (ID)."""
        return self._order.order_id

    @property
    def status(self):
        """Backtrader order status."""
        return self._convert_status_to_bt(self._order.status)

    @property
    def size(self):
        """Order size (quantity)."""
        return self._order.quantity

    @property
    def price(self):
        """Order price."""
        return self._order.price

    @property
    def executed(self):
        """Executed information."""
        return BacktraderExecutedInfo(self._order)

    @property
    def created(self):
        """Order creation information."""
        return BacktraderCreatedInfo(self._order)

    def _convert_status_to_bt(self, status: OrderStatus):
        """Convert our OrderStatus to backtrader status."""
        if not BACKTRADER_AVAILABLE:
            return status.value

        try:
            import backtrader as bt

            status_map = {
                OrderStatus.PENDING: bt.Order.Submitted,
                OrderStatus.FILLED: bt.Order.Completed,
                OrderStatus.PARTIALLY_FILLED: bt.Order.Partial,
                OrderStatus.CANCELLED: bt.Order.Cancelled,
                OrderStatus.REJECTED: bt.Order.Rejected,
                OrderStatus.EXPIRED: bt.Order.Expired,
                OrderStatus.SIMULATING: bt.Order.Submitted,
                OrderStatus.QUEUED: bt.Order.Submitted,
            }

            return status_map.get(status, bt.Order.Created)

        except (ImportError, AttributeError):
            return status.value


class BacktraderExecutedInfo:
    """Backtrader-compatible executed information for orders."""

    def __init__(self, order: Order):
        self._order = order

    @property
    def price(self):
        """Executed price."""
        return self._order.average_price or 0.0

    @property
    def size(self):
        """Executed size."""
        return self._order.filled_quantity

    @property
    def comm(self):
        """Commission paid."""
        return self._order.commission

    @property
    def dt(self):
        """Execution datetime."""
        return self._order.timestamp


class BacktraderCreatedInfo:
    """Backtrader-compatible created information for orders."""

    def __init__(self, order: Order):
        self._order = order

    @property
    def price(self):
        """Created price."""
        return self._order.price or 0.0

    @property
    def size(self):
        """Created size."""
        return self._order.quantity

    @property
    def dt(self):
        """Creation datetime."""
        return self._order.timestamp


class BacktraderPositionAdapter:
    """
    Adapter to make our Position objects compatible with backtrader.

    This class provides a backtrader-compatible interface for our internal Position objects,
    mapping properties to match backtrader's position expectations.
    """

    def __init__(self, position: Position):
        self._position = position

    @property
    def size(self):
        """Position size (quantity)."""
        return self._position.quantity

    @property
    def price(self):
        """Average position price."""
        return self._position.average_price

    @property
    def adjbase(self):
        """Adjusted base price (same as price for simplicity)."""
        return self._position.average_price

    @property
    def upnl(self):
        """Unrealized P&L."""
        return self._position.unrealized_pnl

    @property
    def pnl(self):
        """Total P&L (realized + unrealized)."""
        return self._position.realized_pnl + self._position.unrealized_pnl

    @property
    def pnlcomm(self):
        """P&L including commission."""
        commission = getattr(self._position, "commission_paid", 0.0)
        return self.pnl - commission

    def clone(self):
        """Clone the position (backtrader compatibility)."""
        return BacktraderPositionAdapter(self._position)


class BacktraderPortfolioAdapter:
    """
    Adapter to make our Portfolio objects compatible with backtrader.

    This class provides a backtrader-compatible interface for our internal Portfolio objects,
    mapping properties to match backtrader's account/portfolio expectations.
    """

    def __init__(self, portfolio: Portfolio):
        self._portfolio = portfolio

    @property
    def cash(self):
        """Available cash."""
        return self._portfolio.cash

    @property
    def value(self):
        """Total portfolio value."""
        return self._portfolio.total_value

    @property
    def pnl(self):
        """Total P&L (realized + unrealized)."""
        return self._portfolio.realized_pnl + self._portfolio.unrealized_pnl

    @property
    def upnl(self):
        """Unrealized P&L."""
        return self._portfolio.unrealized_pnl

    @property
    def positions(self):
        """Dictionary of positions adapted for backtrader."""
        return {symbol: BacktraderPositionAdapter(position) for symbol, position in self._portfolio.positions.items()}

    def get_position(self, data):
        """Get position for a specific data feed (backtrader compatibility)."""
        symbol = getattr(data, "_name", "UNKNOWN") if data else "UNKNOWN"
        position = self._portfolio.positions.get(symbol)
        if position:
            return BacktraderPositionAdapter(position)
        empty_position = Position(
            symbol=symbol,
            quantity=0.0,
            average_price=0.0,
            market_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        )
        return BacktraderPositionAdapter(empty_position)
