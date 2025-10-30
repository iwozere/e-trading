"""
Unit tests for src.management.bot_manager

- Tests starting and stopping bots
- Tests status and trade retrieval
- Mocks bot class loading and threading
- Does not require real bot classes or threads to run

How to run:
    pytest tests/test_bot_manager.py
"""

from unittest.mock import MagicMock, patch

import pytest
import src.management.bot_manager as bot_manager


@pytest.fixture(autouse=True)
def clear_bots():
    # Clear bot registries before each test
    bot_manager.running_bots.clear()
    bot_manager.bot_threads.clear()


@patch("src.management.bot_manager.importlib.import_module")
@patch("src.management.bot_manager.threading.Thread")
def test_start_and_stop_bot(mock_thread, mock_import_module):
    # Mock bot class
    DummyBot = MagicMock()
    DummyBot.return_value.trade_history = ["trade1", "trade2"]
    mock_import_module.return_value = MagicMock()
    mock_import_module.return_value.__getattr__.side_effect = lambda name: (
        DummyBot if name == "DummyBot" else None
    )
    # Patch getattr to return DummyBot
    with patch("src.management.bot_manager.getattr", return_value=DummyBot):
        bot_id = bot_manager.start_bot("dummy", {"foo": "bar"}, bot_id="testbot")
        assert bot_id == "testbot"
        assert bot_id in bot_manager.running_bots
        # Status
        status = bot_manager.get_status()
        assert status[bot_id] == "running"
        # Trades
        trades = bot_manager.get_trades(bot_id)
        assert trades == ["trade1", "trade2"]
        # Stop
        bot_manager.stop_bot(bot_id)
        assert bot_id not in bot_manager.running_bots


@patch("src.management.bot_manager.importlib.import_module")
@patch("src.management.bot_manager.threading.Thread")
def test_start_bot_duplicate_id(mock_thread, mock_import_module):
    DummyBot = MagicMock()
    mock_import_module.return_value = MagicMock()
    with patch("src.management.bot_manager.getattr", return_value=DummyBot):
        bot_manager.start_bot("dummy", {}, bot_id="dup")
        with pytest.raises(Exception) as e:
            bot_manager.start_bot("dummy", {}, bot_id="dup")
        assert "already running" in str(e.value)


@patch("src.management.bot_manager.importlib.import_module")
@patch("src.management.bot_manager.threading.Thread")
def test_stop_bot_not_found(mock_thread, mock_import_module):
    with pytest.raises(Exception) as e:
        bot_manager.stop_bot("notfound")
    assert "No running bot" in str(e.value)


def test_get_trades_not_found():
    assert bot_manager.get_trades("notfound") == []


def test_get_running_bots():
    # Should be empty at start
    assert bot_manager.get_running_bots() == []
