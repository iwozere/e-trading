import pytest

from src.trading.broker.binance_broker import BinanceBroker
from src.trading.broker.broker_factory import get_broker
from src.trading.broker.ibkr_broker import IBKRBroker
from src.trading.broker.mock_broker import MockBroker


@pytest.mark.parametrize(
    "broker_type,expected_class",
    [
        ("mock", MockBroker),
        ("binance", BinanceBroker),
        ("ibkr", IBKRBroker),
    ],
)
def test_get_broker_returns_correct_type(broker_type, expected_class):
    config = {"type": broker_type, "api_key": "k", "api_secret": "s", "cash": 1000.0}
    if broker_type == "ibkr":
        config.update({"host": "127.0.0.1", "port": 7497, "client_id": 1})
    broker = get_broker(config)
    assert isinstance(broker, expected_class)


@pytest.mark.skip(reason="create_trading_bot is not exported from src.trading (restore when API returns)")
def test_create_trading_bot_returns_bot():
    """Reserved: wire to trading bot factory when ``src.trading`` exposes it again."""
    pass
