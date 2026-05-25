"""
Integration tests: full trade execution path
---------------------------------------------

Covers the critical path described in P2-X1:
  1. BUY signal → position recorded + DB create_trade called
  2. SELL signal → position closed + DB update_trade called + PnL > 0
  3. Commission is calculated on *notional* (price × size), not on P&L
  4. Only ONE notification fires per trade when trade_notification_hook is set
     (no duplicate from position_notification_manager)

These tests use in-memory mock objects — no DB, no real broker, no Telegram.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from src.trading.base_trading_bot import BaseTradingBot
from src.trading.dto.created_trade import CreatedTrade


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _MockTradeRepository:
    """In-memory trade repository stub implementing the full interface used by BaseTradingBot."""

    def __init__(self) -> None:
        self.created_trades: List[Dict[str, Any]] = []
        self.updated_trades: List[Dict[str, Any]] = []
        self._counter = 0

    def create_trade(self, trade_data: Dict[str, Any]) -> CreatedTrade:
        self._counter += 1
        row = dict(trade_data, id=str(self._counter))
        self.created_trades.append(row)
        return CreatedTrade.synthetic(str(self._counter), row)

    def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> None:
        self.updated_trades.append(dict(update_data, id=trade_id))

    def update_bot_instance(self, bot_id: str, data: Dict[str, Any]) -> None:
        pass  # heartbeat — not relevant for these tests

    # --- Methods called during BaseTradingBot.__init__ ---

    def get_bot_instance(self, bot_id: str) -> Optional[Dict[str, Any]]:
        return None  # no pre-existing bot record

    def create_bot_instance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": 1}

    def get_open_trades(
        self, bot_id: str, status: str = "open"
    ) -> List[Dict[str, Any]]:
        return []  # no pre-existing open trades

    def get_open_positions_for_bot(self, bot_id: str) -> List[Dict[str, Any]]:
        return []


def _make_bot(
    *,
    pair: str = "BTCUSDT",
    hook_calls: Optional[List] = None,
    commission_rate: float = 0.001,
) -> Tuple[BaseTradingBot, _MockTradeRepository]:
    """
    Build a minimal ``BaseTradingBot`` wired to in-memory mocks.

    Returns ``(bot, repo)`` so tests can inspect both the bot state and
    recorded DB calls without monkey-patching the bot object.

    ``hook_calls`` receives every ``(side, price, size, pnl)`` tuple from
    the ``trade_notification_hook`` so tests can assert notification counts.
    """
    config: Dict[str, Any] = {
        "trading_pair": pair,
        "initial_balance": 10_000.0,
        "data": {"data_source": "file"},  # suppresses live notifications
        "paper_trading": {"commission_rate": commission_rate},
    }

    calls: List = [] if hook_calls is None else hook_calls

    def _hook(side: str, price: float, size: float, pnl: Optional[float]) -> None:
        calls.append((side, price, size, pnl))

    repo = _MockTradeRepository()

    bot = BaseTradingBot(
        config=config,
        strategy_class=MagicMock(),
        parameters={},
        broker=None,
        paper_trading=True,
        bot_id="test_bot_001",
        trade_repository=repo,
        trade_notification_hook=_hook,
    )

    # Disable the risk controller so tests are not gated by risk checks.
    # The risk controller blocks SELL when exit notional > entry notional × 1.01
    # (pre_exit_checks), which would reject a profitable +10 % trade.
    # Risk controller logic is tested separately; here we test execution paths.
    object.__setattr__(bot, "risk_controller", None)

    return bot, repo


def _repo_of(bot: BaseTradingBot) -> _MockTradeRepository:
    """Return the bot's trade_repository as a _MockTradeRepository."""
    assert isinstance(bot.trade_repository, _MockTradeRepository)
    return bot.trade_repository


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestBuySignalRecordsPosition:
    """BUY signal: position is stored and DB create_trade is called."""

    def test_position_recorded_after_buy(self) -> None:
        bot, _repo = _make_bot()
        assert bot.trading_pair not in bot.active_positions

        bot.execute_trade("buy", price=50_000.0, size=0.1)

        assert bot.trading_pair in bot.active_positions
        pos = bot.active_positions[bot.trading_pair]
        assert pos["entry_price"] == 50_000.0
        assert pos["size"] == 0.1

    def test_db_create_trade_called_on_buy(self) -> None:
        bot, repo = _make_bot()
        bot.execute_trade("buy", price=50_000.0, size=0.1)

        assert len(repo.created_trades) == 1
        record = repo.created_trades[0]
        assert record["entry_price"] == 50_000.0
        assert record["symbol"] == "BTCUSDT"
        assert record["status"] == "open"


class TestSellSignalCalculatesPnL:
    """SELL signal: position is closed, PnL is non-zero, DB update_trade called."""

    def _buy_then_sell(
        self,
        entry: float = 50_000.0,
        exit_: float = 55_000.0,
        size: float = 0.1,
    ) -> Tuple[BaseTradingBot, _MockTradeRepository]:
        bot, repo = _make_bot()
        bot.execute_trade("buy", price=entry, size=size)
        bot.execute_trade("sell", price=exit_, size=size)
        return bot, repo

    def test_position_cleared_after_sell(self) -> None:
        bot, _repo = self._buy_then_sell()
        assert bot.trading_pair not in bot.active_positions

    def test_pnl_is_positive_on_winning_trade(self) -> None:
        bot, _repo = self._buy_then_sell(entry=50_000.0, exit_=55_000.0)
        # gross PnL = (55000 - 50000) * 0.1 = 500.0 → balance > initial
        assert bot.total_pnl > 0.0

    def test_db_update_trade_called_on_sell(self) -> None:
        _bot, repo = self._buy_then_sell()
        assert len(repo.updated_trades) == 1
        update = repo.updated_trades[0]
        assert update["status"] == "closed"
        assert update["exit_price"] == 55_000.0

    def test_realized_pnl_not_zero_on_full_close(self) -> None:
        """Regression test for P1-T4 — PnL must not be $0 when position is fully closed."""
        _bot, repo = self._buy_then_sell(entry=50_000.0, exit_=55_000.0, size=0.1)
        update = repo.updated_trades[0]
        # gross_pnl = (55000 - 50000) * 0.1 = 500.0 → net_pnl must be < 500 but > 0
        assert update["gross_pnl"] == pytest.approx(500.0)
        assert update["net_pnl"] > 0.0


class TestCommissionOnNotional:
    """Commission must be charged on trade notional (price × size), not on PnL."""

    def test_commission_positive_on_losing_trade(self) -> None:
        """On a losing trade gross_pnl < 0 — commission must still be > 0."""
        bot, repo = _make_bot(commission_rate=0.001)
        bot.execute_trade("buy", price=50_000.0, size=0.1)
        bot.execute_trade("sell", price=45_000.0, size=0.1)  # loss

        update = repo.updated_trades[0]
        assert update["commission"] > 0.0, (
            "Commission must be positive even on a losing trade "
            "(it is charged on notional, not on P&L)"
        )

    def test_commission_equals_notional_times_rate(self) -> None:
        """commission = exit_price × size × rate (regression for P2-T3)."""
        exit_price = 55_000.0
        size = 0.1
        rate = 0.001
        bot, repo = _make_bot(commission_rate=rate)
        bot.execute_trade("buy", price=50_000.0, size=size)
        bot.execute_trade("sell", price=exit_price, size=size)

        update = repo.updated_trades[0]
        expected = exit_price * size * rate  # = 5.5
        assert update["commission"] == pytest.approx(expected, rel=1e-6)


class TestNotificationDeduplication:
    """When trade_notification_hook is set, only the hook must fire — not the position manager."""

    def test_only_hook_fires_on_buy(self) -> None:
        calls: List = []
        bot, _repo = _make_bot(hook_calls=calls)
        bot.execute_trade("buy", price=50_000.0, size=0.1)

        assert len(calls) == 1
        assert calls[0][0] == "buy"

    def test_only_hook_fires_on_sell(self) -> None:
        calls: List = []
        bot, _repo = _make_bot(hook_calls=calls)
        bot.execute_trade("buy", price=50_000.0, size=0.1)
        bot.execute_trade("sell", price=55_000.0, size=0.1)

        # Exactly two hook calls: one buy, one sell — no extras from position_notification_manager
        assert len(calls) == 2
        sides = [c[0] for c in calls]
        assert sides == ["buy", "sell"]

    def test_position_manager_not_called_when_hook_set(self) -> None:
        """position_notification_manager must be skipped when a hook is registered."""
        bot, _repo = _make_bot()
        mock_mgr = MagicMock()
        bot.position_notification_manager = mock_mgr

        bot.execute_trade("buy", price=50_000.0, size=0.1)

        mock_mgr.notify_position_opened.assert_not_called()

    def test_position_manager_called_without_hook(self) -> None:
        """Without a hook, position_notification_manager.notify_position_opened should fire."""
        config: Dict[str, Any] = {
            "trading_pair": "BTCUSDT",
            "initial_balance": 10_000.0,
            "data": {"data_source": "file"},
        }
        repo = _MockTradeRepository()
        bot = BaseTradingBot(
            config=config,
            strategy_class=MagicMock(),
            parameters={},
            broker=None,
            paper_trading=True,
            bot_id="test_bot_no_hook",
            trade_repository=repo,
            trade_notification_hook=None,  # no hook
        )
        object.__setattr__(bot, "risk_controller", None)
        mock_mgr = MagicMock()
        bot.position_notification_manager = mock_mgr

        # _run_async is a fire-and-forget; mock it to capture the call
        with patch("src.trading.base_trading_bot._run_async") as mock_run_async:
            bot.execute_trade("buy", price=50_000.0, size=0.1)
            assert mock_run_async.called, (
                "position_notification_manager.notify_position_opened should be scheduled "
                "via _run_async when no hook is registered"
            )
