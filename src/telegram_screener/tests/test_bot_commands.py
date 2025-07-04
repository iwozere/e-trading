import pytest
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import Dispatcher
from aiogram.fsm.context import FSMContext
from unittest.mock import AsyncMock
import src.telegram_screener.bot as bot_module

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