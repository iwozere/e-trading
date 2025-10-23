"""
Mock broker implementation for testing and development.
Simulates order execution and portfolio management without real API calls.
"""

from src.trading.broker.base_broker import BaseBroker


class MockBroker(BaseBroker):
    """
    Mock broker for testing. Simulates order execution without real API calls.
    """

    def __init__(self, cash: float = 1000.0, config: dict = None) -> None:
        # Create a default config if none provided
        if config is None:
            config = {
                'name': 'Mock Broker',
                'type': 'mock',
                'trading_mode': 'paper',
                'cash': cash,
                'notifications': {
                    'position_opened': False,
                    'position_closed': False,
                    'email_enabled': False,
                    'telegram_enabled': False,
                    'error_notifications': False
                }
            }

        super().__init__(config)
        self.broker_name = "Mock Broker"
        self._cash = cash
        self.orders = []
        self.positions = {}

    def buy(self, symbol: str, qty: float, price: float = None) -> dict:
        """Simulate a buy order."""
        order = {
            "type": "buy",
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "status": "filled",
        }
        self.orders.append(order)
        self._cash -= (price or 1.0) * qty
        self.positions[symbol] = self.positions.get(symbol, 0) + qty
        self._notify_order(order)
        return order

    def sell(self, symbol: str, qty: float, price: float = None) -> dict:
        """Simulate a sell order."""
        order = {
            "type": "sell",
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "status": "filled",
        }
        self.orders.append(order)
        self._cash += (price or 1.0) * qty
        self.positions[symbol] = self.positions.get(symbol, 0) - qty
        self._notify_order(order)
        return order

    # Implement abstract methods from BaseBroker

    async def connect(self) -> bool:
        """Connect to the mock broker (always succeeds)."""
        self.is_connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from the mock broker."""
        self.is_connected = False
        return True

    async def place_order(self, order) -> str:
        """Place an order and return order ID."""
        from src.trading.broker.base_broker import OrderStatus

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = order.price or 100.0  # Default price for market orders

        # Update positions and cash
        if order.side.value == "buy":
            self._cash -= order.average_price * order.quantity
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        else:  # sell
            self._cash += order.average_price * order.quantity
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - order.quantity

        self.orders.append(order)
        return order.order_id

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (mock implementation)."""
        # Find and cancel the order
        for order in self.orders:
            if order.order_id == order_id:
                from src.trading.broker.base_broker import OrderStatus
                order.status = OrderStatus.CANCELLED
                return True
        return False

    async def get_order_status(self, order_id: str):
        """Get order status."""
        for order in self.orders:
            if order.order_id == order_id:
                return order
        return None

    async def get_positions(self) -> dict:
        """Get current positions."""
        from src.trading.broker.base_broker import Position
        from datetime import datetime, timezone

        positions = {}
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=100.0,  # Mock price
                    market_value=quantity * 100.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=datetime.now(timezone.utc),
                    paper_trading=True
                )
        return positions

    async def get_portfolio(self):
        """Get portfolio information."""
        from src.trading.broker.base_broker import Portfolio
        from datetime import datetime, timezone

        positions = await self.get_positions()
        total_value = self._cash + sum(pos.market_value for pos in positions.values())

        return Portfolio(
            total_value=total_value,
            cash=self._cash,
            positions=positions,
            unrealized_pnl=sum(pos.unrealized_pnl for pos in positions.values()),
            realized_pnl=0.0,
            timestamp=datetime.now(timezone.utc),
            paper_trading=True,
            initial_balance=self.config.get('cash', 1000.0)
        )

    async def get_account_info(self) -> dict:
        """Get account information."""
        return {
            'broker_name': self.broker_name,
            'account_type': 'mock',
            'trading_mode': 'paper',
            'cash': self._cash,
            'total_positions': len(self.positions),
            'is_connected': getattr(self, 'is_connected', False)
        }

    def _notify_order(self, order):
        """Notify about order execution (placeholder)."""
        # This method exists for backward compatibility
        # Actual notifications are handled by the base class
        pass
