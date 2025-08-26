import json
import os
from unittest.mock import MagicMock, patch

import pytest
from src.trading.base_trading_bot import BaseTradingBot


class DummyStrategy:
    def __init__(self):
        self.get_signals_calls = 0
        self.get_current_price_calls = 0

    def get_signals(self, trading_pair):
        self.get_signals_calls += 1
        # On first call, return buy, on second, return sell, then nothing
        if self.get_signals_calls == 1:
            return [{"type": "buy", "price": 100, "size": 1}]
        elif self.get_signals_calls == 2:
            return [{"type": "sell", "price": 110, "size": 1}]
        else:
            return []

    def get_current_price(self, trading_pair):
        self.get_current_price_calls += 1
        return 110


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.place_order.return_value = {"orderId": 1}
    return broker


@pytest.fixture
def clean_state_file():
    state_file = os.path.join("logs", "json", "BTCUSDT_bot_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
    yield
    if os.path.exists(state_file):
        os.remove(state_file)


@patch("src.trading.base_trading_bot.create_telegram_notifier")
@patch("src.trading.base_trading_bot.EmailNotifier")
def test_buy_and_sell_signal_triggers_order_and_trade(
    mock_email, mock_telegram, mock_broker, clean_state_file
):
    strategy = DummyStrategy()
    bot = BaseTradingBot(
        {"trading_pair": "BTCUSDT", "initial_balance": 1000},
        strategy,
        broker=mock_broker,
        paper_trading=False,
    )
    bot.telegram_notifier = MagicMock()
    bot.email_notifier = MagicMock()
    # Simulate one buy and one sell
    bot.process_signals(strategy.get_signals("BTCUSDT"))
    assert "BTCUSDT" in bot.active_positions
    bot.process_signals(strategy.get_signals("BTCUSDT"))
    assert "BTCUSDT" not in bot.active_positions
    assert len(bot.trade_history) == 1
    # Check trade log file
    trades_path = os.path.join("logs", "json", "trades.json")
    with open(trades_path) as f:
        trades = json.load(f)
        assert any(trade["pair"] == "BTCUSDT" for trade in trades)


@patch("src.trading.base_trading_bot.create_telegram_notifier")
@patch("src.trading.base_trading_bot.EmailNotifier")
def test_broker_error_handling_and_notification(
    mock_email, mock_telegram, clean_state_file
):
    strategy = DummyStrategy()
    broker = MagicMock()
    broker.place_order.side_effect = Exception("Order failed")
    bot = BaseTradingBot(
        {"trading_pair": "BTCUSDT", "initial_balance": 1000},
        strategy,
        broker=broker,
        paper_trading=False,
    )
    bot.telegram_notifier = MagicMock()
    bot.email_notifier = MagicMock()
    # Should not raise, should notify error
    bot.process_signals([{"type": "buy", "price": 100, "size": 1}])
    bot.telegram_notifier.send_error_notification.assert_called()
    bot.email_notifier.send_notification_email.assert_called()


@patch("src.trading.base_trading_bot.create_telegram_notifier")
@patch("src.trading.base_trading_bot.EmailNotifier")
def test_state_persistence_and_recovery(
    mock_email, mock_telegram, mock_broker, clean_state_file
):
    strategy = DummyStrategy()
    bot = BaseTradingBot(
        {"trading_pair": "BTCUSDT", "initial_balance": 1000},
        strategy,
        broker=mock_broker,
        paper_trading=False,
    )
    bot.active_positions["BTCUSDT"] = {
        "entry_price": 100,
        "size": 1,
        "entry_time": "2024-01-01T00:00:00",
    }
    bot.trade_history.append(
        {
            "pair": "BTCUSDT",
            "entry_price": 100,
            "exit_price": 110,
            "size": 1,
            "pl": 10,
            "time": "2024-01-01T01:00:00",
        }
    )
    bot.current_balance = 1100
    bot.total_pnl = 10
    bot.save_state()
    # New bot loads state
    bot2 = BaseTradingBot(
        {"trading_pair": "BTCUSDT", "initial_balance": 1000},
        strategy,
        broker=mock_broker,
        paper_trading=False,
    )
    assert bot2.active_positions["BTCUSDT"]["entry_price"] == 100
    assert bot2.current_balance == 1100
    assert bot2.total_pnl == 10
    assert any(trade["pair"] == "BTCUSDT" for trade in bot2.trade_history)
