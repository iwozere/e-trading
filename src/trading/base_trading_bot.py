"""
Base Trading Bot Module
----------------------

This module defines the BaseTradingBot class, which provides a framework for implementing trading bots with signal processing, trade execution, position management, and notification capabilities. It is designed to be subclassed by specific strategy bots and supports integration with notification systems (Telegram, email).

Main Features:
- Signal processing and trade execution logic
- Position and balance management
- Trade history tracking with database integration
- Notification via Telegram and email
- Designed for extension by concrete strategy bots

Classes:
- BaseTradingBot: Abstract base class for trading bots
"""

import asyncio
import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Optional, Any, Callable, Dict, List

from src.notification.logger import setup_logger
from src.notification.service.client import (
    MessagePriority,
    MessageType,
    NotificationServiceClient,
    NotificationServiceError,
)
from src.trading.broker.base_broker import PositionNotificationManager
from src.trading.constants import TRADING_STATE_DIR
from src.trading.dto.created_trade import CreatedTrade
from src.trading.risk.controller import RiskController
from src.trading.services.trading_bot_service import trading_bot_service

_logger = setup_logger(__name__)

from src.trading.execution_persistence import execution_persistence
from src.trading.metrics_tracker import metrics_registry
from src.trading.trading_notification_recipient import get_trading_bot_notification_recipient

# Keys that must be redacted from notification/log messages
_SENSITIVE_KEYS = {"key", "secret", "token", "password", "api_key", "api_secret"}


def is_file_based_simulation_config(config: Dict[str, Any]) -> bool:
    """True when the bot runs on CSV/file data — no owner or admin notifications."""
    data = config.get("data") or {}
    return str(data.get("data_source", "")).lower() == "file"


def _sanitize_params(params: dict) -> dict:
    """Recursively redact values whose keys match known sensitive patterns."""
    sanitized = {}
    for k, v in params.items():
        if any(s in k.lower() for s in _SENSITIVE_KEYS):
            sanitized[k] = "***REDACTED***"
        elif isinstance(v, dict):
            sanitized[k] = _sanitize_params(v)
        else:
            sanitized[k] = v
    return sanitized


def _run_async(coro) -> None:
    """
    Fire-and-forget a coroutine safely from both sync and async contexts.
    - Inside a running event loop: schedules the coroutine as a task.
    - Outside a running event loop: runs it in a background daemon thread.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(asyncio.ensure_future, coro)
    except RuntimeError:
        threading.Thread(target=lambda: asyncio.run(coro), daemon=True).start()


class BaseTradingBot:
    """
    Base class for trading bots. Handles config, strategy class, parameters, broker, notifications, and state.
    Subclasses should only override methods if custom logic is needed.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        strategy_class: Any,
        parameters: Dict[str, Any],
        broker: Optional[Any] = None,
        paper_trading: bool = True,
        bot_id: str | None = None,
        trade_repository: Optional[Any] = None,
        trade_notification_hook: Callable[[str, float, float, float | None], None] | None = None,
    ) -> None:
        """
        Initialize the trading bot with config, strategy, broker, and mode.
        Args:
            config: Configuration dictionary
            strategy_class: Strategy class
            parameters: Strategy parameters
            broker: Broker instance (optional)
            paper_trading: Whether to use paper trading mode
            bot_id: Bot identifier (config filename or optimization result filename)
            trade_repository: Optional persistence adapter; ``create_trade`` must return
                :class:`~src.trading.dto.created_trade.CreatedTrade`.
            trade_notification_hook: Optional callback after a successful paper/sim trade
                ``(side: 'buy'|'sell', price, size, pnl_or_none)`` for user-targeted alerts.
        """
        self.config = config
        self.trading_pair = config.get("trading_pair", "BTCUSDT")
        self.initial_balance = config.get("initial_balance", 1000.0)
        self.strategy_class = strategy_class
        self.parameters = parameters
        self.is_running = False
        self.active_positions: dict[Any, Any] = {}
        self._positions_lock = threading.RLock()  # P2-4: thread-safe position access
        self.trade_history: list[Any] = []
        self.current_balance = self.initial_balance
        self.total_pnl = 0.0
        self._signal_queue: list[Any] = []  # Thread-safe signal queue for decoupled strategies
        self._signal_lock = threading.Lock()
        self.broker = broker
        self.paper_trading = paper_trading
        self.trade_notification_hook = trade_notification_hook

        # Database integration
        self.bot_id = bot_id or f"bot_{uuid.uuid4().hex[:8]}"
        self.trade_type = "paper" if paper_trading else "live"
        self.trade_repository = trade_repository or trading_bot_service

        # P1-1: Derive absolute state file path from repo root
        _state_dir = TRADING_STATE_DIR / self.bot_id
        _state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = str(_state_dir / "state.json")

        self._file_based_simulation = is_file_based_simulation_config(config)

        # Enhanced notification setup (skipped for CSV/file backtests — no alerts)
        self.notification_client = None
        self.position_notification_manager = None

        if not self._file_based_simulation:
            try:
                notification_service_url = config.get("notification_service_url", "database://direct")
                self.notification_client = NotificationServiceClient(
                    service_url=notification_service_url,
                    timeout=5,
                    max_retries=1,
                )

                notification_config = config.get(
                    "notifications",
                    {
                        "position_opened": True,
                        "position_closed": True,
                        "email_enabled": True,
                        "telegram_enabled": True,
                        "error_notifications": True,
                    },
                )
                self.position_notification_manager = PositionNotificationManager(
                    {"notifications": notification_config},
                    notification_client=self.notification_client,
                )

            except Exception as e:
                _logger.exception("Notification client not initialized: %s", e)

        self.max_drawdown_pct = config.get("max_drawdown_pct", 20.0)
        self.max_exposure = config.get("max_exposure", 1.0)  # 1.0 = 100% of balance
        self.position_sizing_pct = config.get("position_sizing_pct", 0.1)  # 10% of balance per trade

        # Initialize bot instance in database
        self._initialize_bot_instance()

        # Load state (including open positions from database)
        self.load_state()

        self.risk_controller = RiskController(config.get("risk", {}))

    def _schedule_notification_to_owner(
        self,
        *,
        purpose: str,
        title: str,
        message: str,
        notification_type: MessageType,
        priority: MessagePriority,
        data: Dict[str, Any] | None = None,
        source: str = "trading_bot",
    ) -> None:
        """Queue notification to this bot's sole owner (never admins)."""
        if self._file_based_simulation:
            return
        if not self.notification_client:
            _logger.warning("Notification client not available")
            return

        async def _send() -> None:
            recipient = get_trading_bot_notification_recipient(
                self.config,
                self.bot_id,
                purpose=purpose,
                log_name=self.bot_id,
            )
            if not recipient:
                return
            tg_raw = recipient.get("telegram_user_id")
            tg_id = None
            if tg_raw is not None:
                s = str(tg_raw).strip()
                tg_id = int(s) if s.isdigit() else None

            await self.notification_client.send_notification(
                notification_type=notification_type,
                title=title,
                message=message,
                priority=priority,
                data=data or {},
                source=source,
                channels=recipient["channels"],
                recipient_id=recipient["recipient_id"],
                email_receiver=recipient.get("email"),
                telegram_chat_id=tg_id,
            )

        _run_async(_send())

    def _dispatch_trade_notification(
        self,
        *,
        side: str,
        price: float,
        size: float,
        timestamp: datetime,
        order: Any,
        trade_id: str | None,
        position: Dict[str, Any] | None = None,
        pnl: float | None = None,
        gross_pnl: float | None = None,
        net_pnl: float | None = None,
    ) -> None:
        """
        Dispatch trade notifications through all registered channels.

        Consolidates the three notification paths so each is called exactly once
        per trade event:

        1. ``notify_trade_event`` — legacy owner notification (skipped internally
           when ``trade_notification_hook`` is set).
        2. ``position_notification_manager`` — broker-level position events
           (used only on the legacy path, i.e. when no hook is set).
        3. ``trade_notification_hook`` — modern StrategyInstance hook (primary
           path when running under the strategy manager).

        Args:
            side: "buy" or "sell".
            price: Execution price.
            size: Trade size.
            timestamp: Execution timestamp (UTC-aware).
            order: Raw broker order object or ``None`` for paper trades.
            trade_id: Database trade ID (used as fallback order reference).
            position: Position dict from ``active_positions`` (SELL only).
            pnl: Percentage P&L (SELL only).
            gross_pnl: Gross P&L in base currency (SELL only).
            net_pnl: Net P&L after commission (SELL only).
        """
        upper_side = side.upper()

        # Path 1: legacy owner notification (no-op when hook is set)
        if upper_side == "BUY":
            self.notify_trade_event(upper_side, price, size, timestamp)
        else:
            self.notify_trade_event(
                upper_side,
                price,
                size,
                timestamp,
                entry_price=position["entry_price"] if position else None,
                pnl=pnl,
            )

        # Path 2: position notification manager (legacy path only)
        if self.position_notification_manager and not self.trade_notification_hook:
            strategy_name = self.strategy_class.__name__ if hasattr(self.strategy_class, "__name__") else "Unknown"
            order_ref = str(order) if order else trade_id
            trading_mode = "paper" if self.paper_trading else "live"

            if upper_side == "BUY":
                position_data: Dict[str, Any] = {
                    "symbol": self.trading_pair,
                    "side": "BUY",
                    "price": price,
                    "size": size,
                    "timestamp": timestamp,
                    "bot_id": self.bot_id,
                    "trading_mode": trading_mode,
                    "order_id": order_ref,
                    "strategy": strategy_name,
                }
                _run_async(self.position_notification_manager.notify_position_opened(position_data))
            else:
                hold_duration = "Unknown"
                if position and position.get("entry_time"):
                    duration = timestamp - position["entry_time"]
                    days = duration.days
                    hours, remainder = divmod(duration.seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    if days > 0:
                        hold_duration = f"{days}d {hours}h {minutes}m"
                    elif hours > 0:
                        hold_duration = f"{hours}h {minutes}m {seconds}s"
                    else:
                        hold_duration = f"{minutes}m {seconds}s"

                position_data = {
                    "symbol": self.trading_pair,
                    "side": "SELL",
                    "entry_price": position["entry_price"] if position else None,
                    "exit_price": price,
                    "size": size,
                    "pnl": gross_pnl,
                    "pnl_percentage": pnl,
                    "timestamp": timestamp,
                    "bot_id": self.bot_id,
                    "trading_mode": trading_mode,
                    "order_id": order_ref,
                    "strategy": strategy_name,
                    "hold_duration": hold_duration,
                }
                _run_async(self.position_notification_manager.notify_position_closed(position_data))

        # Path 3: modern StrategyInstance hook
        if self.trade_notification_hook:
            try:
                hook_pnl = net_pnl if upper_side == "SELL" else None
                self.trade_notification_hook(side, price, size, hook_pnl)
            except Exception:
                _logger.exception("trade_notification_hook failed (%s)", side)

    def _initialize_bot_instance(self):
        """Initialize bot instance in database."""
        try:
            bot_data = {
                "id": self.bot_id,
                "type": self.trade_type,
                "config_file": self.config.get("config_file", None),
                "status": "stopped",
                "current_balance": self.current_balance,
                "total_pnl": self.total_pnl,
                "extra_metadata": {
                    "trading_pair": self.trading_pair,
                    "initial_balance": self.initial_balance,
                    "strategy_class": self.strategy_class.__name__
                    if hasattr(self.strategy_class, "__name__")
                    else str(self.strategy_class),
                    "parameters": self.parameters,
                },
            }

            # Check if bot instance already exists
            existing_bot = self.trade_repository.get_bot_instance(self.bot_id)
            if existing_bot:
                # Update existing bot instance
                self.trade_repository.update_bot_instance(
                    self.bot_id,
                    {
                        "status": "stopped",
                        "current_balance": self.current_balance,
                        "total_pnl": self.total_pnl,
                        "last_heartbeat": datetime.now(UTC),
                    },
                )
            else:
                # Create new bot instance
                self.trade_repository.create_bot_instance(bot_data)

            _logger.info("Initialized bot instance: %s", self.bot_id)

        except Exception:
            _logger.exception("Error initializing bot instance: %s")

    def run(self) -> None:
        """
        Main bot loop. Handles signals, order management, error handling, and state persistence.
        """
        self.is_running = True
        _logger.info("Starting bot for %s", self.trading_pair)

        # Update bot status to running
        try:
            self.trade_repository.update_bot_instance(
                self.bot_id,
                {
                    "status": "running",
                    "started_at": datetime.now(UTC),
                    "last_heartbeat": datetime.now(UTC),
                },
            )
        except Exception:
            _logger.exception("Error updating bot status: %s")

        while self.is_running:
            try:
                signals = self.get_signals()
                self.process_signals(signals)
                self.update_positions()
                self.save_state()
                # Note (P3-X2): heartbeat / current_balance DB updates are
                # intentionally omitted here.  When BaseTradingBot is managed
                # by StrategyInstance, its _heartbeat_loop() is the single DB
                # writer for last_heartbeat and current_balance, which prevents
                # last-write-wins contention on the same row from two threads.
                time.sleep(1)
            except Exception as e:
                _logger.exception("Error in bot loop: %s")
                self.notify_error(str(e))
                time.sleep(5)

    def get_signals(self) -> List[Dict[str, Any]]:
        """
        Get trading signals from both the internal strategy class and the external signal queue.
        Returns:
            List of signal dictionaries
        """
        combined_signals = []

        # 1. Get signals from internal strategy class (if it has get_signals method)
        if hasattr(self.strategy_class, "get_signals"):
            try:
                combined_signals.extend(self.strategy_class.get_signals(self.trading_pair))
            except Exception:
                _logger.exception("Error getting signals from strategy class")

        # 2. Get signals from external queue (e.g., from Backtrader)
        with self._signal_lock:
            if self._signal_queue:
                combined_signals.extend(self._signal_queue)
                self._signal_queue.clear()

        return combined_signals

    def add_signal(self, signal: Dict[str, Any]) -> None:
        """
        Add a signal to the internal queue. Called by external signal sources.
        """
        with self._signal_lock:
            self._signal_queue.append(signal)
            _logger.debug("Added signal to bot queue: %s", signal)

    def process_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Process a list of trading signals and execute trades as needed.
        Args:
            signals: List of signal dictionaries
        """
        for signal in signals:
            side = (signal.get("type") or signal.get("side") or "").strip().lower()
            if not side:
                continue
            sym = signal.get("symbol") or signal.get("pair") or self.trading_pair
            if str(sym).upper() != str(self.trading_pair).upper():
                continue
            try:
                price = float(signal.get("price") or 0.0)
                size = float(signal.get("size") if signal.get("size") is not None else signal.get("quantity") or 0.0)
            except (TypeError, ValueError):
                continue
            if side == "buy" and self.trading_pair not in self.active_positions:
                self.execute_trade("buy", price, size)
            elif side == "sell" and self.trading_pair in self.active_positions:
                self.execute_trade("sell", price, size)

    def execute_trade(self, trade_type: str, price: float, size: float) -> None:
        """
        Generic trade execution logic for buy/sell with database integration.
        Runs pre-trade risk checks before placing any order.
        """
        # P2-3: Pre-trade risk gate — block trades that breach limits
        if trade_type == "buy" and hasattr(self, "risk_controller") and self.risk_controller is not None:
            with self._positions_lock:
                current_exposures = {
                    pair: pos["entry_price"] * pos["size"] for pair, pos in self.active_positions.items()
                }
            stop_loss_pct = self.config.get("stop_loss_pct", 0.02)
            approved_size = self.risk_controller.pre_trade_checks(
                account_equity=self.current_balance,
                stop_loss_pct=stop_loss_pct,
                current_exposures=current_exposures,
                entry_price=price,
            )
            if approved_size == 0.0:
                _logger.warning(
                    "Risk controller blocked BUY trade for %s (balance=%.2f)", self.trading_pair, self.current_balance
                )
                return
            size = min(size, approved_size)

        timestamp = datetime.now(UTC)
        order = None

        sell_position: Dict[str, Any] | None = None
        if trade_type == "sell":
            with self._positions_lock:
                sell_position = self.active_positions.get(self.trading_pair)
            if sell_position is None:
                _logger.warning("No active position found for SELL on %s", self.trading_pair)
                return
            if self.risk_controller is not None:
                with self._positions_lock:
                    current_exposures = {
                        pair: pos["entry_price"] * pos["size"] for pair, pos in self.active_positions.items()
                    }
                eff_size = min(size, sell_position["size"])
                if not self.risk_controller.pre_exit_checks(
                    account_equity=self.current_balance,
                    current_exposures=current_exposures,
                    symbol=self.trading_pair,
                    exit_size=eff_size,
                    exit_price=price,
                ):
                    _logger.warning(
                        "Risk controller blocked SELL for %s (size=%s price=%s)",
                        self.trading_pair,
                        eff_size,
                        price,
                    )
                    return

        try:
            if not self.paper_trading and self.broker:
                order = self.broker.place_order(self.trading_pair, trade_type.upper(), size, price=price)
                self.log_order(order)

            if trade_type == "buy":
                # Create new trade record in database
                trade_data = {
                    "bot_id": self.bot_id,
                    "trade_type": self.trade_type,
                    "strategy_name": self.strategy_class.__name__
                    if hasattr(self.strategy_class, "__name__")
                    else "Unknown",
                    "entry_logic_name": self.parameters.get("strategy_config", {})
                    .get("entry_logic", {})
                    .get("name", "Unknown"),
                    "exit_logic_name": self.parameters.get("strategy_config", {})
                    .get("exit_logic", {})
                    .get("name", "Unknown"),
                    "symbol": self.trading_pair,
                    "interval": self.config.get("data", {}).get("interval", "1h"),
                    "entry_time": timestamp,
                    "buy_order_created": timestamp,
                    "entry_price": price,
                    "entry_value": price * size,
                    "size": size,
                    "direction": "long",
                    "status": "open",
                    "extra_metadata": {"order_id": str(order) if order else None, "paper_trading": self.paper_trading},
                }

                # Create trade in database
                trade = self.trade_repository.create_trade(trade_data)
                if not isinstance(trade, CreatedTrade):
                    raise TypeError(
                        f"trade_repository.create_trade must return CreatedTrade, got {type(trade).__name__}"
                    )
                new_trade_id = trade.id

                # Update local state (P2-4: lock)
                with self._positions_lock:
                    self.active_positions[self.trading_pair] = {
                        "entry_price": price,
                        "size": size,
                        "entry_time": timestamp,
                        "trade_id": new_trade_id,
                    }

                self._dispatch_trade_notification(
                    side="buy",
                    price=price,
                    size=size,
                    timestamp=timestamp,
                    order=order,
                    trade_id=new_trade_id,
                )

            else:  # sell
                position = sell_position
                assert position is not None  # narrowed by sell_position guard above
                trade_id = position.get("trade_id")

                # Calculate PnL.
                # Commission is a percentage of trade *notional* (price × size), not of
                # P&L.  Using gross_pnl would produce a negative commission on losing
                # trades (i.e. a rebate), which is wrong.  The rate comes from the
                # paper_trading config block so it is configurable per-deployment.
                pnl = ((price - position["entry_price"]) / position["entry_price"]) * 100
                gross_pnl = (price - position["entry_price"]) * position["size"]
                commission_rate = self.config.get("paper_trading", {}).get("commission_rate", 0.001)
                commission = price * position["size"] * commission_rate
                net_pnl = gross_pnl - commission

                # Update trade in database
                if trade_id:
                    update_data = {
                        "exit_time": timestamp,
                        "sell_order_created": timestamp,
                        "sell_order_closed": timestamp,
                        "exit_price": price,
                        "exit_value": price * position["size"],
                        "commission": commission,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "pnl_percentage": pnl,
                        "exit_reason": "signal",
                        "status": "closed",
                        "extra_metadata": {
                            "order_id": str(order) if order else None,
                            "paper_trading": self.paper_trading,
                        },
                    }
                    self.trade_repository.update_trade(trade_id, update_data)

                # Update local state (P2-4: lock)
                trade_record = {
                    "bot_id": self.bot_id,
                    "pair": self.trading_pair,
                    "type": "long",
                    "entry_price": position["entry_price"],
                    "exit_price": price,
                    "size": position["size"],
                    "pl": pnl,
                    "time": timestamp.isoformat(),
                }
                self.trade_history.append(trade_record)

                # Legacy JSON logging (for backward compatibility)
                self.log_trade(trade_record)

                # Update balance using the actual dollar P&L (net_pnl), not
                # the entry-price % return.  Applying pnl/100 as a multiplier
                # would only be correct if the entire balance was invested in
                # this single position (100 % position sizing).  Using net_pnl
                # correctly handles partial-position sizing.
                self.current_balance += net_pnl
                self.total_pnl += pnl

                # Update metrics registry
                metrics_registry.record_trade(
                    bot_id=self.bot_id,
                    symbol=self.trading_pair,
                    trade_pnl=net_pnl,
                    trade_pnl_pct=pnl,
                    current_balance=self.current_balance,
                )

                # Remove from active positions (P2-4: lock)
                with self._positions_lock:
                    self.active_positions.pop(self.trading_pair, None)

                self._dispatch_trade_notification(
                    side="sell",
                    price=price,
                    size=size,
                    timestamp=timestamp,
                    order=order,
                    trade_id=trade_id,
                    position=position,
                    pnl=pnl,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                )

        except Exception as e:
            _logger.exception("Error executing trade: %s", e)
            self.notify_error(str(e))

    def log_order(self, order: Any) -> None:
        """
        Persist order details to orders.json using the persistence service.
        Args:
            order: Order object or dictionary
        """
        try:
            # Ensure it is a dict for the persistence service
            order_data = order if isinstance(order, dict) else vars(order)
            execution_persistence.save_order(self.bot_id, order_data)
        except Exception:
            _logger.exception("Failed to log order for bot %s", self.bot_id)

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Persist trade details to the legacy ``trades.json`` file.

        .. deprecated::
            The database (via ``trade_repository``) is the single source of truth
            for completed trades (P3-X1).  JSON file persistence is kept for
            backward compatibility only and will be removed once the DB recovery
            path (``_load_open_positions_from_db``) is confirmed stable in all
            deployments.  Do **not** add new callers of this method.
        """
        try:
            execution_persistence.save_trade(self.bot_id, trade)
        except Exception:
            _logger.exception("Failed to log trade for bot %s", self.bot_id)

    def save_state(self) -> None:
        """
        Save open positions and bot state to disk for recovery.
        """
        try:
            state = {
                "active_positions": self.active_positions,
                "trade_history": self.trade_history,
                "current_balance": self.current_balance,
                "total_pnl": self.total_pnl,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, default=str, indent=2)
        except Exception:
            _logger.exception("Failed to save bot state: %s")

    def load_state(self) -> None:
        """
        Load open positions and bot state from database and legacy files.
        """
        # First try to load from database
        self._load_open_positions_from_db()

        # Then load from legacy state file (for backward compatibility)
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    state = json.load(f)
                    # Only load trade history and balance from legacy file
                    # Active positions should come from database
                    self.trade_history = state.get("trade_history", [])
                    self.current_balance = state.get("current_balance", self.initial_balance)
                    self.total_pnl = state.get("total_pnl", 0.0)
            except Exception:
                _logger.exception("Failed to load legacy bot state: %s")

    def _load_open_positions_from_db(self) -> None:
        """
        Load open positions from database for this bot.
        Supports both ORM model instances and plain dicts returned by the service.
        """
        try:
            open_trades = self.trade_repository.get_open_trades(bot_id=self.bot_id, symbol=self.trading_pair)

            # P1-4: Support both ORM objects (attribute access) and plain dicts
            def _get(trade, key):
                return getattr(trade, key, None) if hasattr(trade, key) else trade.get(key)

            with self._positions_lock:
                for trade in open_trades:
                    symbol = _get(trade, "symbol")
                    entry_price = _get(trade, "entry_price")
                    size = _get(trade, "size")
                    entry_time = _get(trade, "entry_time")
                    trade_id = _get(trade, "id")

                    if not symbol:
                        _logger.warning("Skipping trade with missing symbol: %s", trade)
                        continue

                    self.active_positions[symbol] = {
                        "entry_price": float(entry_price) if entry_price else 0.0,
                        "size": float(size) if size else 0.0,
                        "entry_time": entry_time,
                        "trade_id": str(trade_id) if trade_id else None,
                    }

            _logger.info("Loaded %d open positions from database for %s", len(open_trades), self.bot_id)

        except Exception:
            _logger.exception("Error loading open positions from database: %s")
            with self._positions_lock:
                self.active_positions = {}

    def notify_error(self, error_msg: str) -> None:
        """
        Send error notification to admin users always, and to the bot owner when
        ``notifications.error_notifications`` is enabled.
        Args:
            error_msg: Error message string
        """
        if self._file_based_simulation:
            return
        try:
            error_message = (
                f"⚠️ Trading Bot Error\n\n"
                f"Bot ID: {self.bot_id}\n"
                f"Trading Pair: {self.trading_pair}\n"
                f"Error: {error_msg}\n\n"
                f"Please check the bot configuration and system logs."
            )

            err_data = {
                "bot_id": self.bot_id,
                "trading_pair": self.trading_pair,
                "error_type": "trading_bot_error",
            }

            self._schedule_notification_to_owner(
                purpose="error",
                title="Trading Bot Error",
                message=error_message,
                notification_type=MessageType.ERROR,
                priority=MessagePriority.HIGH,
                data=err_data,
            )

            if self.notification_client:
                _run_async(
                    self.notification_client.send_to_admins(
                        title="Trading Bot Error (admin)",
                        message=error_message,
                        notification_type=MessageType.ERROR,
                        priority=MessagePriority.HIGH,
                        data={**err_data, "admin_notification": True},
                        channels=["telegram", "email"],
                    )
                )

        except NotificationServiceError:
            _logger.exception("Failed to send error notification via service:")
        except Exception as e:
            _logger.exception("Failed to send error notification: %s", e)

    def notify_trade_event(
        self,
        side: str,
        price: float,
        size: float,
        timestamp: datetime,
        entry_price: float | None = None,
        pnl: float | None = None,
    ) -> None:
        """
        Send trade event to this bot's owner (skipped when StrategyInstance uses
        ``trade_notification_hook`` to avoid duplicate alerts).
        """
        if self._file_based_simulation:
            return
        try:
            if self.trade_notification_hook is not None:
                return

            purpose = "trade_buy" if side.upper() == "BUY" else "trade_sell"
            emoji = "🟢" if side == "BUY" else "🔴"
            pnl_text = f"PnL: {pnl:.2f}%" if pnl is not None else ""

            trade_message = (
                f"{emoji} Trade {side}\n\n"
                f"Bot ID: {self.bot_id}\n"
                f"Trading Pair: {self.trading_pair}\n"
                f"Price: ${price:.2f}\n"
                f"Size: {size}\n"
                f"{pnl_text}\n"
                f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            msg_type = MessageType.TRADE_ENTRY if side.upper() == "BUY" else MessageType.TRADE_EXIT

            self._schedule_notification_to_owner(
                purpose=purpose,
                title=f"Trading Bot - {side}",
                message=trade_message,
                notification_type=msg_type,
                priority=MessagePriority.NORMAL,
                data={
                    "bot_id": self.bot_id,
                    "trading_pair": self.trading_pair,
                    "side": side,
                    "price": price,
                    "size": size,
                    "entry_price": entry_price,
                    "pnl": pnl,
                    "timestamp": timestamp.isoformat(),
                },
            )

        except NotificationServiceError:
            _logger.exception("Failed to send trade notification via service:")
        except Exception as e:
            _logger.exception("Failed to send trade notification: %s", e)

    def update_positions(self):
        """
        Generic position update logic for stop loss/take profit. Subclasses can override for custom behavior.
        """
        with self._positions_lock:  # P2-4: snapshot under lock to avoid mutation during iteration
            positions_snapshot = list(self.active_positions.items())

        for pair, position in positions_snapshot:
            # Assumes the strategy instance is available and has get_current_price, sl_atr_mult, tp_atr_mult
            # If running in Backtrader, you may need to adapt this for live/real-time bots
            current_price = None
            if hasattr(self, "strategy") and self.strategy and hasattr(self.strategy, "get_current_price"):
                current_price = self.strategy.get_current_price(pair)
            if current_price is None:
                continue
            entry_price = position["entry_price"]
            # Prefer a stop_price recorded at entry time (absolute price level).
            # The sl_atr_mult check below is deliberately disabled because
            # multiplying an ATR multiplier by 100 gives a threshold like -150 %,
            # which is impossible for a long position and never fires (P4-T1).
            # Stop-loss is properly handled by RiskController.real_time_adjustments()
            # when a risk_controller is configured; the logic here only covers
            # explicit stop/take-profit prices stored in the position dict.
            stop_price = position.get("stop_price")
            take_profit_price = position.get("take_profit_price")
            if stop_price and current_price <= stop_price:
                _logger.info(
                    "Stop-loss triggered for %s: price %.4f <= stop %.4f",
                    pair,
                    current_price,
                    stop_price,
                )
                self.execute_trade("sell", current_price, position["size"])
            elif take_profit_price and current_price >= take_profit_price:
                _logger.info(
                    "Take-profit triggered for %s: price %.4f >= tp %.4f",
                    pair,
                    current_price,
                    take_profit_price,
                )
                self.execute_trade("sell", current_price, position["size"])

    def stop(self):
        """
        Generic stop logic: set is_running to False and close all open positions. Subclasses can override for custom behavior.
        """
        self.is_running = False
        _logger.info("Stopping bot for %s", self.trading_pair)

        # Update bot status in database
        try:
            self.trade_repository.update_bot_instance(
                self.bot_id,
                {
                    "status": "stopped",
                    "last_heartbeat": datetime.now(UTC),
                    "current_balance": self.current_balance,
                    "total_pnl": self.total_pnl,
                },
            )
        except Exception:
            _logger.exception("Error updating bot status on stop: %s")

        # Close all open positions (P2-4: snapshot under lock)
        with self._positions_lock:
            pairs_to_close = list(self.active_positions.keys())

        for pair in pairs_to_close:
            current_price = None
            if hasattr(self, "strategy") and self.strategy and hasattr(self.strategy, "get_current_price"):
                current_price = self.strategy.get_current_price(pair)
            if current_price is not None:
                with self._positions_lock:
                    pos_size = self.active_positions.get(pair, {}).get("size")
                if pos_size:
                    self.execute_trade("sell", current_price, pos_size)

    def log_bot_event(self, event: str):
        _logger.info("%s: %s", event, self.__class__.__name__)
        _logger.info("Trading pair: %s", self.trading_pair)
        _logger.info("Initial balance: %s", self.initial_balance)
        # P2-2: sanitize parameters before logging
        _logger.info("Strategy parameters: %s", _sanitize_params(getattr(self, "parameters", {})))
        _logger.info("Broker: %s", self.broker.__class__.__name__ if self.broker else "None")
        timestamp = datetime.now(UTC)
        _logger.info("Timestamp: %s", timestamp)

    def notify_bot_event(self, event: str, emoji: str):
        if self._file_based_simulation:
            return
        # P2-2: sanitize parameters before including in notifications
        safe_params = _sanitize_params(getattr(self, "parameters", {}))
        msg = (
            f"{emoji} Bot {event.lower()}\n\n"
            f"Class: {self.__class__.__name__}\n"
            f"Pair: {self.trading_pair}\n"
            f"Initial balance: {self.initial_balance}\n"
            f"Strategy: {getattr(self, 'strategy_class', type(self.strategy_class).__name__)}\n"
            f"Parameters: {safe_params}\n"
            f"Broker: {self.broker.__class__.__name__ if self.broker else 'None'}"
        )
        try:
            if not self.notification_client:
                _logger.warning("Notification client not available for bot event notification")
                return

            if event.lower() in ["error", "failed", "stopped"]:
                priority = MessagePriority.HIGH
            elif event.lower() in ["started", "resumed"]:
                priority = MessagePriority.NORMAL
            else:
                priority = MessagePriority.NORMAL

            self._schedule_notification_to_owner(
                purpose="bot_event",
                title=f"Trading Bot - {event.title()}",
                message=msg,
                notification_type=MessageType.SYSTEM,
                priority=priority,
                data={
                    "bot_id": self.bot_id,
                    "trading_pair": self.trading_pair,
                    "event": event,
                    "bot_class": self.__class__.__name__,
                },
                source="trading_bot_lifecycle",
            )

        except NotificationServiceError:
            _logger.exception("Failed to send bot event notification via service:")
        except Exception as e:
            _logger.exception("Failed to send bot event notification: %s", e)

    def pre_run(self, data_feed):
        """
        Hook for custom logic before running the Backtrader engine.
        Subclasses can override this to add analyzers, observers, or other setup.
        Args:
            data_feed: The data feed object that will be added to Cerebro.
        """

    def post_run(self, data_feed):
        """
        Hook for custom logic after running the Backtrader engine.
        Subclasses can override this to process results, save reports, etc.
        Args:
            data_feed: The data feed object that was used in Cerebro.
        """

    def run_backtrader_engine(self, data_feed):
        self.log_bot_event("started")
        self.notify_bot_event("started", "🤖")
        import backtrader as bt

        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        _logger.info("Added data feed for %s", self.trading_pair)
        if self.broker:
            from src.trading.broker.backtrader_broker_bridge import wrap_broker_for_cerebro

            cerebro.setbroker(wrap_broker_for_cerebro(self.broker))
        else:
            cerebro.broker.setcash(self.initial_balance)
        cerebro.addstrategy(getattr(self, "strategy_class"), params=getattr(self, "parameters", {}))
        self.pre_run(data_feed)
        _logger.info("Starting Backtrader engine for %s", self.trading_pair)
        cerebro.run()
        self.post_run(data_feed)
        self.log_bot_event("stopped")
        self.notify_bot_event("stopped", "🛑")
