import pytest
from src.broker.base_broker import MockBroker
from src.broker.binance_broker import BinanceBroker
from src.broker.binance_paper_broker import BinancePaperBroker
from src.broker.broker_factory import get_broker
from src.broker.ibkr_broker import IBKRBroker
from src.trading import create_trading_bot


class DummyStrategy:
    def get_signals(self, trading_pair):
        return []


@pytest.mark.parametrize(
    "broker_type,expected_class",
    [
        ("mock", MockBroker),
        ("binance", BinanceBroker),
        ("binance_paper", BinancePaperBroker),
        ("ibkr", IBKRBroker),
    ],
)
def test_get_broker_returns_correct_type(broker_type, expected_class):
    config = {"type": broker_type, "api_key": "k", "api_secret": "s", "cash": 1000.0}
    if broker_type == "ibkr":
        config.update({"host": "127.0.0.1", "port": 7497, "client_id": 1})
    broker = get_broker(config)
    assert isinstance(broker, expected_class)


def test_create_trading_bot_returns_bot():
    config = {
        "type": "mock",
        "bot_type": "rsi_bb_volume",
        "trading_pair": "BTCUSDT",
        "initial_balance": 1000.0,
    }
    strategy = DummyStrategy()
    bot = create_trading_bot(config, strategy)
    assert hasattr(bot, "run")
    assert hasattr(bot, "execute_trade")
    assert bot.trading_pair == "BTCUSDT"
