from unittest.mock import MagicMock, patch

import pytest
from src.broker.ibkr_broker import IBKRBroker


@pytest.fixture
def broker():
    with patch("src.trading.ibkr_broker.IB") as MockIB:
        mock_ib = MockIB.return_value
        yield IBKRBroker()


@patch("src.trading.ibkr_broker.IB")
def test_place_market_order(mock_ib_class, broker):
    mock_ib = mock_ib_class.return_value
    mock_ib.placeOrder.return_value.orderStatus.status = "Filled"
    result = broker.place_order("AAPL", "BUY", 1)
    assert result == "Filled"
    mock_ib.placeOrder.assert_called_once()


@patch("src.trading.ibkr_broker.IB")
def test_place_limit_order(mock_ib_class, broker):
    mock_ib = mock_ib_class.return_value
    mock_ib.placeOrder.return_value.orderStatus.status = "Submitted"
    result = broker.place_order("AAPL", "SELL", 1, order_type="limit", price=200)
    assert result == "Submitted"
    mock_ib.placeOrder.assert_called_once()


@patch("src.trading.ibkr_broker.IB")
def test_cancel_order(mock_ib_class, broker):
    mock_ib = mock_ib_class.return_value
    mock_ib.orders.return_value = [MagicMock(orderId=1)]
    mock_ib.cancelOrder.return_value = None
    result = broker.cancel_order(0)
    assert result["status"] == "canceled"
    mock_ib.cancelOrder.assert_called_once()


@patch("src.trading.ibkr_broker.IB")
def test_get_balance(mock_ib_class, broker):
    mock_ib = mock_ib_class.return_value
    mock_ib.accountSummary.return_value = [
        MagicMock(tag="TotalCashValue", value="1000")
    ]
    result = broker.get_balance("TotalCashValue")
    assert result == "1000"


@patch("src.trading.ibkr_broker.IB")
def test_get_open_orders(mock_ib_class, broker):
    mock_ib = mock_ib_class.return_value
    mock_ib.openOrders.return_value = [MagicMock(contract=MagicMock(symbol="AAPL"))]
    result = broker.get_open_orders("AAPL")
    assert result[0].contract.symbol == "AAPL"


@patch("src.trading.ibkr_broker.IB")
def test_get_order_status(mock_ib_class, broker):
    mock_ib = mock_ib_class.return_value
    order = MagicMock()
    order.order.orderId = 1
    order.orderStatus.status = "Filled"
    mock_ib.openOrders.return_value = [order]
    result = broker.get_order_status(1)
    assert result == "Filled"


@patch("src.trading.ibkr_broker.IB")
def test_fetch_ohlcv(mock_ib_class, broker):
    mock_ib = mock_ib_class.return_value
    bar = MagicMock()
    bar.__dict__ = {
        "date": "2024-01-01",
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume": 1000,
    }
    mock_ib.reqHistoricalData.return_value = [bar]
    df = broker.fetch_ohlcv("AAPL", "1 day", limit=1)
    assert not df.empty
    assert "open" in df.columns


@patch("src.trading.ibkr_broker.IB")
def test_api_exception_handling(mock_ib_class, broker):
    mock_ib = mock_ib_class.return_value
    mock_ib.placeOrder.side_effect = Exception("error")
    result = broker.place_order("AAPL", "BUY", 1)
    assert "error" in result
