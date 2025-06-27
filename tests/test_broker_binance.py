from unittest.mock import MagicMock, patch

import pytest
from binance.exceptions import BinanceAPIException
from src.broker.binance_broker import BinanceBroker


@pytest.fixture
def broker():
    with patch("src.trading.binance_broker.Client") as MockClient:
        mock_client = MockClient.return_value
        yield BinanceBroker("fake_key", "fake_secret")


@patch("src.trading.binance_broker.Client")
def test_place_market_order(mock_client_class, broker):
    mock_client = mock_client_class.return_value
    mock_client.create_order.return_value = {"orderId": 123}
    result = broker.place_order("BTCUSDT", "BUY", 0.01)
    assert result["orderId"] == 123
    mock_client.create_order.assert_called_once()


@patch("src.trading.binance_broker.Client")
def test_place_limit_order(mock_client_class, broker):
    mock_client = mock_client_class.return_value
    mock_client.create_order.return_value = {"orderId": 456}
    result = broker.place_order(
        "BTCUSDT", "SELL", 0.01, order_type="LIMIT", price=30000
    )
    assert result["orderId"] == 456
    mock_client.create_order.assert_called_once()


@patch("src.trading.binance_broker.Client")
def test_cancel_order(mock_client_class, broker):
    mock_client = mock_client_class.return_value
    mock_client.cancel_order.return_value = {"status": "canceled"}
    result = broker.cancel_order(123, symbol="BTCUSDT")
    assert result["status"] == "canceled"
    mock_client.cancel_order.assert_called_once()


@patch("src.trading.binance_broker.Client")
def test_get_balance(mock_client_class, broker):
    mock_client = mock_client_class.return_value
    mock_client.get_account.return_value = {
        "balances": [{"asset": "BTC", "free": "1.0", "locked": "0.0"}]
    }
    result = broker.get_balance("BTC")
    assert result["asset"] == "BTC"


@patch("src.trading.binance_broker.Client")
def test_get_open_orders(mock_client_class, broker):
    mock_client = mock_client_class.return_value
    mock_client.get_open_orders.return_value = [{"orderId": 1}]
    result = broker.get_open_orders("BTCUSDT")
    assert result[0]["orderId"] == 1


@patch("src.trading.binance_broker.Client")
def test_get_order_status(mock_client_class, broker):
    mock_client = mock_client_class.return_value
    mock_client.get_order.return_value = {"orderId": 1, "status": "FILLED"}
    result = broker.get_order_status(1, symbol="BTCUSDT")
    assert result["status"] == "FILLED"


@patch("src.trading.binance_broker.Client")
def test_fetch_ohlcv(mock_client_class, broker):
    mock_client = mock_client_class.return_value
    mock_client.get_klines.return_value = [
        [
            1625097600000,
            "34000",
            "35000",
            "33000",
            "34500",
            "100",
            1625097660000,
            "0",
            0,
            0,
            0,
            0,
        ]
    ]
    df = broker.fetch_ohlcv("BTCUSDT", "1m", limit=1)
    assert not df.empty
    assert "open" in df.columns


@patch("src.trading.binance_broker.Client")
def test_api_exception_handling(mock_client_class, broker):
    mock_client = mock_client_class.return_value
    mock_client.create_order.side_effect = BinanceAPIException(None, "error", "error")
    result = broker.place_order("BTCUSDT", "BUY", 0.01)
    assert "error" in result
