from typing import Any, Optional

from binance.client import Client
from src.trading.broker.base_broker import BaseBroker


class BinanceBroker(BaseBroker):
    """
    Concrete implementation of a trading broker for Binance, handling live trading operations and API communication.
    Implements order placement and management for live trading accounts.
    """

    def __init__(self, api_key: str, api_secret: str, cash: float = 1000.0) -> None:
        super().__init__(cash)
        self.client = Client(api_key, api_secret)
        self.client.API_URL = "https://api.binance.com/api"
        self.broker_name = "Binance"

    def buy(self, symbol: str, qty: float, price: Optional[float] = None) -> Any:
        """Place a buy order on Binance."""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side="BUY",
                type="MARKET" if price is None else "LIMIT",
                quantity=qty,
                price=price,
            )
            self.orders.append(order)
            self._notify_order(order)
            return order
        except Exception as e:
            return {"error": str(e)}

    def sell(self, symbol: str, qty: float, price: Optional[float] = None) -> Any:
        """Place a sell order on Binance."""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side="SELL",
                type="MARKET" if price is None else "LIMIT",
                quantity=qty,
                price=price,
            )
            self.orders.append(order)
            self._notify_order(order)
            return order
        except Exception as e:
            return {"error": str(e)}
