from src.broker.base_broker import BaseBroker


class MockBroker(BaseBroker):
    """
    Mock broker for testing. Simulates order execution without real API calls.
    """

    def __init__(self, cash: float = 1000.0) -> None:
        super().__init__(cash)
        self.broker_name = "Mock Broker"

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
