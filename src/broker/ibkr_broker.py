from typing import Any, Optional

import pandas as pd
from ib_insync import IB, LimitOrder, MarketOrder, Stock
from src.broker.base_broker import BaseBroker


class IBKRBroker(BaseBroker):
    """
    Interactive Brokers (IBKR) broker implementation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        cash: float = 1000.0,
    ) -> None:
        super().__init__(cash)
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)
        self.broker_name = "IBKR"

    def buy(self, symbol: str, qty: float, price: Optional[float] = None) -> Any:
        """Place a buy order on IBKR."""
        try:
            contract = Stock(symbol, "SMART", "USD")
            order = (
                MarketOrder("BUY", qty)
                if price is None
                else LimitOrder("BUY", qty, price)
            )
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            self.orders.append(trade)
            self._notify_order(trade)
            return trade.orderStatus.status
        except Exception as e:
            return {"error": str(e)}

    def sell(self, symbol: str, qty: float, price: Optional[float] = None) -> Any:
        """Place a sell order on IBKR."""
        try:
            contract = Stock(symbol, "SMART", "USD")
            order = (
                MarketOrder("SELL", qty)
                if price is None
                else LimitOrder("SELL", qty, price)
            )
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            self.orders.append(trade)
            self._notify_order(trade)
            return trade.orderStatus.status
        except Exception as e:
            return {"error": str(e)}

    def place_order(
        self, symbol, side, quantity, order_type="market", price=None, **kwargs
    ):
        """
        Place an order on IBKR. Supports market and limit orders.
        Returns the order object.
        """
        try:
            contract = Stock(symbol, "SMART", "USD")
            if order_type.lower() == "market":
                order = MarketOrder(side.capitalize(), quantity)
            elif order_type.lower() == "limit":
                order = LimitOrder(side.capitalize(), quantity, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            return trade.orderStatus.status
        except Exception as e:
            return {"error": str(e)}

    def cancel_order(self, order_id, symbol=None):
        """
        Cancel an order on IBKR.
        """
        try:
            order = self.ib.orders()[order_id]
            self.ib.cancelOrder(order)
            return {"status": "canceled"}
        except Exception as e:
            return {"error": str(e)}

    def get_balance(self, asset=None):
        """
        Get account balance for a specific asset or all assets.
        """
        try:
            acc = self.ib.accountSummary()
            if asset:
                for row in acc:
                    if row.tag == asset:
                        return row.value
                return None
            return {row.tag: row.value for row in acc}
        except Exception as e:
            return {"error": str(e)}

    def get_open_orders(self, symbol=None):
        """
        Get open orders for a symbol or all symbols.
        """
        try:
            orders = self.ib.openOrders()
            if symbol:
                return [o for o in orders if o.contract.symbol == symbol]
            return orders
        except Exception as e:
            return {"error": str(e)}

    def get_order_status(self, order_id, symbol=None):
        """
        Get order status by order ID.
        """
        try:
            orders = self.ib.openOrders()
            for o in orders:
                if o.order.orderId == order_id:
                    return o.orderStatus.status
            return None
        except Exception as e:
            return {"error": str(e)}

    def fetch_ohlcv(self, symbol, interval, limit=100):
        """
        Fetch OHLCV data as a pandas DataFrame.
        """
        try:
            contract = Stock(symbol, "SMART", "USD")
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=f"{limit} D",
                barSizeSetting=interval,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            df = pd.DataFrame([b.__dict__ for b in bars])
            if not df.empty:
                df.set_index("date", inplace=True)
                return df[["open", "high", "low", "close", "volume"]].astype(float)
            return df
        except Exception as e:
            return {"error": str(e)}
