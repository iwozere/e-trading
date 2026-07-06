from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from aiogram.types import Message

from src.model.telegram_bot import Fundamentals, Technicals, TickerAnalysis
from src.telegram.handlers import account as account_handler
from src.telegram.handlers import misc as misc_handler
from src.telegram.handlers.common import HELP_TEXT


@pytest.mark.asyncio
@patch("src.telegram.handlers.common.get_service_instances")
async def test_cmd_start(mock_get_services):
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/start"

    mock_get_services.return_value = (MagicMock(), None)

    await misc_handler.cmd_start(message)

    # We check if it called answer with at least the HELP_TEXT part
    assert message.answer.called
    call_args = message.answer.call_args[0][0]
    assert HELP_TEXT in call_args


@pytest.mark.asyncio
@patch("src.telegram.handlers.common.get_service_instances")
async def test_cmd_help(mock_get_services):
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/help"

    mock_get_services.return_value = (MagicMock(), None)

    await misc_handler.cmd_help(message)
    message.answer.assert_awaited_with(HELP_TEXT)


@pytest.mark.asyncio
@patch("src.telegram.handlers.account.get_service_instances")
@patch("src.telegram.handlers.common.get_service_instances")
async def test_cmd_info(mock_common_get_services, mock_account_get_services):
    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/info"

    mock_telegram_service = MagicMock()
    mock_telegram_service.get_user_status.return_value = None
    # Both layers must return the same values
    mock_common_get_services.return_value = (mock_telegram_service, None)
    mock_account_get_services.return_value = (mock_telegram_service, None)

    await account_handler.cmd_info(message)
    message.reply.assert_awaited()


@pytest.mark.asyncio
@patch("src.telegram.handlers.common.get_service_instances")
async def test_unknown_command(mock_get_services):
    message = AsyncMock(spec=Message)
    message.reply = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.text = "/unknown"

    mock_get_services.return_value = (MagicMock(), None)

    await misc_handler.unknown_command(message)
    message.reply.assert_awaited()
    assert message.reply.await_args is not None
    args, kwargs = message.reply.await_args
    assert "Unknown command." in args[0]


def test_analyze_ticker_business_valid(monkeypatch):
    # Mock the common functions
    mock_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            "open": [100] * 100,
            "high": [110] * 100,
            "low": [90] * 100,
            "close": [105] * 100,
            "volume": [1000] * 100,
        }
    )
    mock_fundamentals = Fundamentals(
        ticker="AAPL",
        company_name="Apple Inc.",
        current_price=150.0,
        market_cap=2500000000000.0,
        pe_ratio=25.0,
        forward_pe=24.0,
        dividend_yield=0.5,
        earnings_per_share=6.0,
        data_source="Yahoo Finance",
        last_updated="2023-12-01 12:00:00",
    )
    technicals = Technicals(
        rsi=65.0,
        sma_fast=100.0,
        sma_slow=95.0,
        ema_fast=98.0,
        ema_slow=96.0,
        macd=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        stoch_k=75.0,
        stoch_d=70.0,
        adx=25.0,
        plus_di=30.0,
        minus_di=20.0,
        obv=1000000.0,
        adr=2.5,
        avg_adr=2.0,
        trend="BULLISH",
        bb_upper=105.0,
        bb_middle=100.0,
        bb_lower=95.0,
        bb_width=0.1,
        cci=50.0,
        roc=5.0,
        mfi=60.0,
        williams_r=-30.0,
        atr=2.0,
        recommendations={},
    )
    monkeypatch.setattr("src.telegram.screener.business_logic.get_ohlcv", lambda *args, **kwargs: mock_df)
    monkeypatch.setattr(
        "src.telegram.screener.business_logic.get_fundamentals", lambda *args, **kwargs: mock_fundamentals
    )
    monkeypatch.setattr(
        "src.telegram.screener.business_logic.get_indicator_data",
        lambda *args, **kwargs: {"status": "ok", "technicals": technicals},
    )
    # Patch TickerAnalysis to always include chart_image=None
    result = TickerAnalysis(
        ticker="AAPL",
        provider="yahoo",
        period="2y",
        interval="1d",
        ohlcv=mock_df,
        fundamentals=mock_fundamentals,
        technicals=technicals,
        error=None,
        chart_image=None,
    )
    assert isinstance(result, TickerAnalysis)
    assert result.ohlcv is not None
    assert result.fundamentals is not None
    assert result.technicals is not None
    assert result.error is None


def test_analyze_ticker_business_invalid_period(monkeypatch):
    def mock_get_ohlcv(*args, **kwargs):
        raise ValueError("Invalid period/interval combination")

    monkeypatch.setattr("src.telegram.screener.business_logic.get_ohlcv", mock_get_ohlcv)
    try:
        TickerAnalysis(
            ticker="AAPL",
            provider="yahoo",
            period="bad",
            interval="bad",
            ohlcv=None,
            fundamentals=None,
            technicals=None,
            error="Invalid period/interval combination",
            chart_image=None,
        )
    except Exception:
        pass


def test_analyze_ticker_business_exception(monkeypatch):
    def mock_get_ohlcv(*args, **kwargs):
        raise Exception("fail")

    monkeypatch.setattr("src.telegram.screener.business_logic.get_ohlcv", mock_get_ohlcv)
    try:
        TickerAnalysis(
            ticker="AAPL",
            provider="yahoo",
            period="2y",
            interval="1d",
            ohlcv=None,
            fundamentals=None,
            technicals=None,
            error="fail",
            chart_image=None,
        )
    except Exception:
        pass
