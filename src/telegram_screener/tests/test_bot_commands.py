import pytest
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import Dispatcher
from aiogram.fsm.context import FSMContext
from unittest.mock import AsyncMock
import src.telegram_screener.bot as bot_module
from src.telegram_screener.business_logic import analyze_ticker_business, TickerAnalysis

@pytest.mark.asyncio
async def test_cmd_start(monkeypatch):
    message = AsyncMock(spec=Message)
    await bot_module.cmd_start(message)
    message.answer.assert_awaited_with(bot_module.HELP_TEXT, parse_mode="HTML")

@pytest.mark.asyncio
async def test_cmd_help(monkeypatch):
    message = AsyncMock(spec=Message)
    await bot_module.cmd_help(message)
    message.answer.assert_awaited_with(bot_module.HELP_TEXT, parse_mode="HTML")

@pytest.mark.asyncio
async def test_cmd_info(monkeypatch):
    message = AsyncMock(spec=Message)
    await bot_module.cmd_info(message)
    message.answer.assert_awaited_with("<b>Your info:</b>\nEmail: (not set)\nVerified: No", parse_mode="HTML")

@pytest.mark.asyncio
async def test_unknown_command(monkeypatch):
    message = AsyncMock(spec=Message)
    message.text = "/unknown"
    await bot_module.unknown_command(message)
    assert message.answer.await_count == 1
    args, kwargs = message.answer.await_args
    assert "Unknown command." in args[0]
    assert "Welcome to the Telegram Screener Bot" in args[0]
    assert kwargs["parse_mode"] == "HTML"

def test_analyze_ticker_business_valid(monkeypatch):
    class DummyDownloader:
        def is_valid_period_interval(self, period, interval):
            return True
        def get_ohlcv(self, ticker, interval, start_date, end_date):
            return 'dummy_df'
    monkeypatch.setattr('src.telegram_screener.data_provider_factory.get_downloader', lambda prov: DummyDownloader())
    result = analyze_ticker_business('AAPL', provider='yf', period='2y', interval='1d')
    assert isinstance(result, TickerAnalysis)
    assert result.ohlcv == 'dummy_df'
    assert result.error is None

def test_analyze_ticker_business_invalid_period(monkeypatch):
    class DummyDownloader:
        def is_valid_period_interval(self, period, interval):
            return False
        def get_ohlcv(self, ticker, interval, start_date, end_date):
            return 'dummy_df'
    monkeypatch.setattr('src.telegram_screener.data_provider_factory.get_downloader', lambda prov: DummyDownloader())
    result = analyze_ticker_business('AAPL', provider='yf', period='bad', interval='bad')
    assert isinstance(result, TickerAnalysis)
    assert result.ohlcv is None
    assert 'Invalid period/interval' in result.error

def test_analyze_ticker_business_exception(monkeypatch):
    class DummyDownloader:
        def is_valid_period_interval(self, period, interval):
            return True
        def get_ohlcv(self, ticker, interval, start_date, end_date):
            raise Exception('fail')
    monkeypatch.setattr('src.telegram_screener.data_provider_factory.get_downloader', lambda prov: DummyDownloader())
    result = analyze_ticker_business('AAPL', provider='yf', period='2y', interval='1d')
    assert isinstance(result, TickerAnalysis)
    assert result.ohlcv is None
    assert 'fail' in result.error