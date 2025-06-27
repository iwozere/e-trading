import pytest
from src.trading.bot_factory import get_bot
from src.trading.rsi_bb_volume_bot import RsiBbVolumeBot


class DummyStrategy:
    def get_signals(self, trading_pair):
        return []


def test_get_bot_returns_rsi_bb_volume_bot():
    config = {
        "bot_type": "rsi_bb_volume",
        "trading_pair": "BTCUSDT",
        "initial_balance": 1000.0,
    }
    strategy = DummyStrategy()
    broker = object()
    bot = get_bot(config, strategy, broker)
    assert isinstance(bot, RsiBbVolumeBot)
    assert bot.trading_pair == "BTCUSDT"


def test_get_bot_raises_for_unsupported_type():
    config = {"bot_type": "unknown_type"}
    strategy = DummyStrategy()
    broker = object()
    with pytest.raises(ValueError):
        get_bot(config, strategy, broker)
